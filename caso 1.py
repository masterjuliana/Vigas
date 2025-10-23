import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
import time
import os

# Configurações para melhor performance e estabilidade
np.seterr(divide='ignore', invalid='ignore')
dde.config.set_random_seed(42) # Para reproducibilidade

class BeamPINN:
    """ Classe otimizada para resolver o Caso 1: Viga biapoiada com propriedades constantes.
    Incorpora a lógica de treino e aprimoramento discutida."""
    
    def __init__(self, L=10.0, E0_I0=1000.0, q0=10.0):
        """Inicializa os parâmetros da viga com base nos valores fornecidos."""
        self.L = L
        # Usamos E0_I0 para representar o EI (Rigidez à Flexão)
        self.EI = E0_I0 
        self.q0 = q0
        self.start_time = None
        
    def _print_step(self, message):
        """Função auxiliar para imprimir mensagens com timestamp"""
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.1f}s] {message}")
        
    def analytical_solution(self, x):
        """
        Solução analítica para deflexão de viga biapoiada com carga uniforme.
        """
        X = x[:, 0] if len(x.shape) > 1 else x
        return (self.q0 * X / (24 * self.EI)) * (self.L**3 - 2 * self.L * X**2 + X**3)
    
    def pde(self, x, v):
        """
        Equação governante (Resíduo da EDO): EI * d⁴v/dx⁴ - q0 = 0
        """
        dv_xx = dde.grad.hessian(v, x, i=0, j=0)
        dv_xxxx = dde.grad.hessian(dv_xx, x, i=0, j=0)
        return self.EI * dv_xxxx - self.q0
    
    def setup_problem(self):
        """Configura a geometria, Condições de Contorno (BCs) e o objeto de dados do DeepXDE."""
        self._print_step("Configurando problema PDE...")
        
        geom = dde.geometry.Interval(0, self.L)
        
        def boundary_left(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)
        
        def boundary_right(x, on_boundary):
            return on_boundary and np.isclose(x[0], self.L)
        
        # 1. Condições de Contorno de Deflexão (v = 0)
        bc_v_0 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left)
        bc_v_L = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right)
        
        # 2. Condições de Contorno de Momento Fletor (v'' = 0) - Implementação via OperatorBC
        def moment_zero_bc(x, y, X):
            return dde.grad.hessian(y, x, i=0, j=0)
        
        bc_M_0 = dde.icbc.OperatorBC(geom, moment_zero_bc, boundary_left)
        bc_M_L = dde.icbc.OperatorBC(geom, moment_zero_bc, boundary_right)
        
        # Usamos os parâmetros de amostragem da sua seção 4, mas com BCs aumentados
        data = dde.data.PDE(
            geom,
            self.pde,
            [bc_v_0, bc_v_L, bc_M_0, bc_M_L], 
            num_domain=1000, 
            num_boundary=80, # Aprimoramento: Aumentado para melhor BC de 2ª ordem
            num_test=1000
        )
        
        return data
    
    def train_model_optimized(self, num_adam_iterations=5000, num_lbfgs_iterations=5000):
        """Treina o modelo PINN com estratégia otimizada (Adam + L-BFGS-B)."""
        self._print_step("Iniciando treinamento da PINN...")
        
        data = self.setup_problem()
        
        # Arquitetura mais profunda para EDO de 4ª ordem (Aprimoramento)
        layer_size = [1] + [50] * 8 + [1] 
        net = dde.nn.FNN(layer_size, "tanh", "Glorot normal")
        model = dde.Model(data, net)
        
        # Pesos de perda ajustados [EDP, v(0), v(L), v''(0), v''(L)]
        # Usamos seus pesos iniciais e aumentamos a PDE e BCs de Momento
        loss_weights = [1.5, 200, 200, 400, 400] 
        
        # --- FASE 1: Otimizador ADAM ---
        self._print_step(f"FASE 1: Otimizador ADAM ({num_adam_iterations} iterações)")
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        losshistory, train_state = model.train(
            iterations=num_adam_iterations, 
            display_every=1000
        )
        
        # --- FASE 2: Otimizador L-BFGS-B ---
        self._print_step(f"FASE 2: Otimizador L-BFGS-B ({num_lbfgs_iterations} max iter)")
        model.compile("L-BFGS-B", loss_weights=loss_weights)
        
        dde.optimizers.set_LBFGS_options(
            maxcor=100, 
            ftol=1.0e-12, # Relaxado em relação ao seu 1.0e-35 para estabilidade
            gtol=1.0e-08, # Relaxado em relação ao seu 1.0e-35 para estabilidade
            maxiter=num_lbfgs_iterations
        )
        
        try:
            losshistory, train_state = model.train()
            self._print_step("L-BFGS-B convergiu com sucesso.")
        except Exception as e:
            self._print_step(f"L-BFGS-B parou: {str(e)[:100]}...")
            
        return model, losshistory
    
    def safe_gradient(self, y, x):
        """Calcula derivada numérica de forma segura usando numpy.gradient."""
        try:
            if len(y) < 3:
                return np.zeros_like(y)
            y_flat = y.flatten() if len(y.shape) > 1 else y
            x_flat = x.flatten() if len(x.shape) > 1 else x
            return np.gradient(y_flat, x_flat)
        except Exception:
            return np.zeros_like(y.flatten() if len(y.shape) > 1 else y)
    
    def evaluate_solution_comprehensive(self, model, num_points=500):
        """Avaliação completa da solução PINN vs analítica."""
        self._print_step("Avaliando soluções...")
        
        X_test = np.linspace(0, self.L, num_points).reshape(-1, 1)
        v_pred = model.predict(X_test)
        v_true = self.analytical_solution(X_test).reshape(-1, 1) 
        
        v_pred_flat = v_pred.flatten()
        v_true_flat = v_true.flatten()
        
        # Cálculo de Erros
        absolute_error = np.abs(v_pred_flat - v_true_flat)
        
        # Cálculo de Erro Relativo (seguro)
        relative_error = np.zeros_like(absolute_error)
        mask = np.abs(v_true_flat) > 1e-12
        relative_error[mask] = (absolute_error[mask] / np.abs(v_true_flat[mask])) * 100
        
        # Cálculo de Derivadas e Momento Fletor
        X_flat = X_test.flatten()
        dv_dx_pred = self.safe_gradient(v_pred_flat, X_flat)
        dv_dx_true = self.safe_gradient(v_true_flat, X_flat)
        d2v_dx2_pred = self.safe_gradient(dv_dx_pred, X_flat)
        d2v_dx2_true = self.safe_gradient(dv_dx_true, X_flat)
        M_pred = self.EI * d2v_dx2_pred
        M_true = self.EI * d2v_dx2_true
        
        # Deflexão Máxima
        X_mid = np.array([[self.L/2]])
        v_max_pred = model.predict(X_mid)[0, 0]
        v_max_true = self.analytical_solution(X_mid)[0]
        
        # Pontos de treino para métricas
        X_train = model.data.train_x_all
        v_train_pred = model.predict(X_train)
        v_train_true = self.analytical_solution(X_train).reshape(-1, 1)
        
        results = {
            'X_test': X_test, 'X_train': X_train,
            'v_pred': v_pred, 'v_true': v_true,
            'v_train_pred': v_train_pred, 'v_train_true': v_train_true,
            'v_pred_flat': v_pred_flat, 'v_true_flat': v_true_flat,
            'absolute_error': absolute_error, 'relative_error': relative_error,
            'dv_dx_pred': dv_dx_pred, 'dv_dx_true': dv_dx_true,
            'M_pred': M_pred, 'M_true': M_true,
            'v_max_pred': v_max_pred, 'v_max_true': v_max_true
        }
        return results
    
    def calculate_comprehensive_metrics(self, results):
        """Calcula métricas de desempenho abrangentes."""
        v_pred_flat = results['v_pred_flat']
        v_true_flat = results['v_true_flat']
        v_train_pred = results['v_train_pred']
        v_train_true = results['v_train_true']
        abs_error = results['absolute_error']
        rel_error = results['relative_error']
        
        # Erros
        max_abs_error = np.max(abs_error)
        mean_abs_error = np.mean(abs_error)
        
        # Erro relativo seguro
        mask_nonzero = np.abs(v_true_flat) > 1e-12
        max_rel_error_safe = np.max(rel_error[mask_nonzero]) if np.any(mask_nonzero) else 0.0
        mean_rel_error_safe = np.mean(rel_error[mask_nonzero]) if np.any(mask_nonzero) else 0.0
        
        # Erro na deflexão máxima
        max_deflection_error = np.abs(results['v_max_pred'] - results['v_max_true'])
        
        # Prevenção de divisão por zero se v_max_true for zero
        if np.isclose(results['v_max_true'], 0.0):
             max_deflection_rel_error = 0.0
        else:
            max_deflection_rel_error = (max_deflection_error / np.abs(results['v_max_true'])) * 100
        
        # Coeficientes R²
        slope_test, intercept_test, r_value_test, _, _ = stats.linregress(v_true_flat, v_pred_flat)
        r_squared_test = r_value_test ** 2
        
        slope_train, intercept_train, r_value_train, _, _ = stats.linregress(
            v_train_true.flatten(), v_train_pred.flatten()
        )
        r_squared_train = r_value_train ** 2
        
        metrics = {
            'max_abs_error': max_abs_error, 'mean_abs_error': mean_abs_error,
            'max_rel_error': max_rel_error_safe, 'mean_rel_error': mean_rel_error_safe,
            'v_max_pred': results['v_max_pred'], 'v_max_true': results['v_max_true'],
            'max_deflection_error': max_deflection_error, 
            'max_deflection_rel_error': max_deflection_rel_error,
            'r_squared_test': r_squared_test, 'r_squared_train': r_squared_train,
            'slope_test': slope_test, 'intercept_test': intercept_test
        }
        return metrics
    
    def plot_comprehensive_comparison(self, results, metrics, losshistory):
        """Cria gráficos abrangentes da comparação."""
        self._print_step("Gerando gráficos de comparação...")
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig)
        
        # Gráfico 1: Comparação das deflexões (principal)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['X_test'], results['v_true'], 'r-', linewidth=3, label='Solução Analítica')
        ax1.plot(results['X_test'], results['v_pred'], 'b--', linewidth=2, label='PINN')
        ax1.scatter(results['X_train'], results['v_train_pred'], color='green', alpha=0.6, s=15, label='Pontos de Treino (PINN)')
        ax1.set_xlabel('Posição x (m)', fontsize=12)
        ax1.set_ylabel('Deflexão v(x) (m)', fontsize=12)
        ax1.set_title('COMPARAÇÃO: Deflexão da Viga', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Erro absoluto
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(results['X_test'], results['absolute_error'], 'r-', linewidth=2)
        ax2.set_xlabel('Posição x (m)', fontsize=12)
        ax2.set_ylabel('Erro Absoluto (m)', fontsize=12)
        ax2.set_title('Erro Absoluto', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        if np.max(results['absolute_error']) > 1e-10:
             ax2.set_yscale('log')
        
        # Gráfico 3: Erro relativo
        ax3 = fig.add_subplot(gs[0, 2])
        relative_error_plot = np.clip(results['relative_error'], 0, 100)
        ax3.plot(results['X_test'], relative_error_plot, 'b-', linewidth=2)
        ax3.set_xlabel('Posição x (m)', fontsize=12)
        ax3.set_ylabel('Erro Relativo (%)', fontsize=12)
        ax3.set_title('Erro Relativo', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Derivadas (inclinação)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(results['X_test'], results['dv_dx_true'], 'r-', linewidth=2, label='Analítica')
        ax4.plot(results['X_test'], results['dv_dx_pred'], 'b--', linewidth=2, label='PINN')
        ax4.set_xlabel('Posição x (m)', fontsize=12)
        ax4.set_ylabel("v'(x) (inclinação)", fontsize=12)
        ax4.set_title('Derivada Primeira - Inclinação', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Momento fletor
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(results['X_test'], results['M_true'], 'r-', linewidth=2, label='Analítica')
        ax5.plot(results['X_test'], results['M_pred'], 'b--', linewidth=2, label='PINN')
        ax5.set_xlabel('Posição x (m)', fontsize=12)
        ax5.set_ylabel('Momento Fletor M(x) (N·m)', fontsize=12)
        ax5.set_title('Momento Fletor', fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Gráfico de regressão (teste)
        ax6 = fig.add_subplot(gs[1, 2])
        min_val = min(results['v_true_flat'].min(), results['v_pred_flat'].min())
        max_val = max(results['v_true_flat'].max(), results['v_pred_flat'].max())
        perfect_line = np.linspace(min_val, max_val, 100)
        regression_line = metrics['slope_test'] * perfect_line + metrics['intercept_test']
        ax6.scatter(results['v_true_flat'], results['v_pred_flat'], alpha=0.6, s=20, label=f'Teste - R² = {metrics["r_squared_test"]:.6f}')
        ax6.plot(perfect_line, regression_line, 'r-', linewidth=2)
        ax6.plot(perfect_line, perfect_line, 'k--', linewidth=1, label='y = x (perfeito)')
        ax6.set_xlabel('Valores Analíticos (m)', fontsize=12)
        ax6.set_ylabel('Valores PINN (m)', fontsize=12)
        ax6.set_title('Gráfico de Regressão - Dados de Teste', fontsize=14, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_aspect('equal')
        
        # Gráfico 7: Histórico de perda
        ax7 = fig.add_subplot(gs[2, 0])
        if hasattr(losshistory, 'steps') and losshistory.steps:
            # Perda de Treinamento
            ax7.semilogy(losshistory.steps, losshistory.loss_train, 'b-', linewidth=2, label='Perda de Treino')
            
            # Perda de Teste 
            if hasattr(losshistory, 'loss_test') and losshistory.loss_test and len(losshistory.loss_test[-1]) > 0:
                loss_test_total = np.array(losshistory.loss_test).T[-1]
                num_test_points = len(loss_test_total)
                
                if len(losshistory.steps) >= num_test_points:
                    adam_steps = losshistory.steps[:num_test_points - 1]
                    final_step = [losshistory.steps[-1]]
                    steps_combined = adam_steps + final_step
                    loss_combined = loss_test_total
                    
                    if len(steps_combined) == len(loss_combined):
                        ax7.semilogy(steps_combined, loss_combined, 'r--', linewidth=2, label='Perda de Teste')
                            
            ax7.set_xlabel('Iteração', fontsize=12)
            ax7.set_ylabel('Perda', fontsize=12)
            ax7.set_title('Histórico de Treinamento', fontsize=14, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Resumo das métricas
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')
        
        textstr = '\n'.join([
            'RESUMO DAS MÉTRICAS',
            '=' * 35,
            f'Deflexão máxima analítica: {metrics["v_max_true"]:.4e} m',
            f'Deflexão máxima PINN: {metrics["v_max_pred"]:.4e} m',
            f'Erro relativo def. máx: {metrics["max_deflection_rel_error"]:.2f}%',
            '',
            f'Erro absoluto máximo: {metrics["max_abs_error"]:.2e} m',
            f'Erro absoluto médio: {metrics["mean_abs_error"]:.2e} m',
            f'Erro relativo máximo (seguro): {metrics["max_rel_error"]:.2f}%',
            f'Erro relativo médio (seguro): {metrics["mean_rel_error"]:.2f}%',
            '',
            f'R² (teste): {metrics["r_squared_test"]:.6f}',
            f'R² (treino): {metrics["r_squared_train"]:.6f}'
        ])
        ax8.text(0.05, 0.95, textstr, transform=ax8.transAxes, fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontfamily='monospace')
        
        # Gráfico 9: Informações do modelo
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        model_info = '\n'.join([
            'INFORMAÇÕES DO MODELO',
            '=' * 25,
            f'Comprimento (L): {self.L} m',
            f'Rigidez (EI): {self.EI:.2e} N·m²',
            f'Carga (q₀): {self.q0} N/m',
            '',
            'ARQUITETURA DA REDE',
            f'Camadas: 1 → [50]×8 → 1 (Aprimorado)',
            f'Ativação: tanh',
            f'Otimizador: Adam + L-BFGS-B',
            f'Pesos de Perda (EDP:BCs): 1.5:200:400 (Ajustado)',
            f'Pontos de domínio: {model.data.num_domain}',
            f'Pontos de contorno: {model.data.num_boundary} (Aprimorado)'
        ])
        ax9.text(0.05, 0.95, model_info, transform=ax9.transAxes, fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

    def print_detailed_report(self, metrics):
        """Imprime relatório detalhado."""
        print("\n" + "="*80)
        print("RELATÓRIO DETALHADO - CASO 1: VIGA BIAPOIADA (MESCLADO E APERFEIÇOADO)")
        print("="*80)
        
        print(f"\nPARÂMETROS DA VIGA (Novos Valores):")
        print(f"   Comprimento (L): {self.L} m")
        print(f"   Rigidez à Flexão (EI): {self.EI:.2e} N·m²")
        print(f"   Carga Distribuída (q₀): {self.q0} N/m")
        
        print(f"\nDEFLEXÃO MÁXIMA (no centro L/2):")
        print(f"   Solução Analítica: {metrics['v_max_true']:.6e} m")
        print(f"   Solução PINN:      {metrics['v_max_pred']:.6e} m")
        print(f"   Erro Absoluto:     {metrics['max_deflection_error']:.2e} m")
        print(f"   Erro Relativo:     {metrics['max_deflection_rel_error']:.2f}%")
        
        print(f"\nANÁLISE DE ERRO (Domínio):")
        print(f"   Erro Absoluto Máximo:  {metrics['max_abs_error']:.2e} m")
        print(f"   Erro Absoluto Médio:   {metrics['mean_abs_error']:.2e} m")
        print(f"   Erro Relativo Máximo (seguro):  {metrics['max_rel_error']:.2f}%")
        print(f"   Erro Relativo Médio (seguro):   {metrics['mean_rel_error']:.2f}%")
        
        print(f"\nQUALIDADE DO AJUSTE:")
        print(f"   Coeficiente R² (teste):  {metrics['r_squared_test']:.6f}")
        print(f"   Coeficiente R² (treino): {metrics['r_squared_train']:.6f}")
        
        print(f"\nCLASSIFICAÇÃO DA PRECISÃO:")
        if metrics['max_deflection_rel_error'] < 1.0:
            print("   ✅ PRECISÃO EXCELENTE (Erro < 1%)")
        elif metrics['max_deflection_rel_error'] < 5.0:
            print("   ✅ PRECISÃO MUITO BOA (Erro < 5%)")
        elif metrics['max_deflection_rel_error'] < 10.0:
            print("   ⚠️  PRECISÃO BOA (Erro < 10%)")
        else:
            print("   ❌ PRECISÃO INSUFICIENTE (Erro > 10%)")
        
        print("="*80)

# ----------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    start_global_time = time.time()
    
    print("=" * 80)
    print("ANÁLISE PINN - VIGA BIAPOIADA (CASO 1) - CÓDIGO MESCLADO")
    print("=" * 80)
    
    # Parâmetros fornecidos pelo usuário para a mesclagem
    L_USER = 1.0 
    EI_USER = 1000.0 
    Q0_USER = 10.0 
    
    beam = BeamPINN(
        L=L_USER, 
        E0_I0=EI_USER, 
        q0=Q0_USER
    )
    
    try:
        # 1. Treinamento (usando 5000 Adam + 5000 L-BFGS-B, conforme sua sugestão)
        model, losshistory = beam.train_model_optimized(
            num_adam_iterations=5000, 
            num_lbfgs_iterations=5000
        ) 
        
        # 2. Avaliação
        results = beam.evaluate_solution_comprehensive(model)
        metrics = beam.calculate_comprehensive_metrics(results)
        
        # 3. Visualização
        # Isso irá gerar a visualização completa de 9 gráficos
        beam.plot_comprehensive_comparison(results, metrics, losshistory)
        
        # 4. Relatório Final
        beam.print_detailed_report(metrics)
        
        total_elapsed_time = time.time() - start_global_time
        print(f"\n🎯 ANÁLISE CONCLUÍDA! (Tempo total: {total_elapsed_time:.2f} segundos)")
        
    except Exception as e:
        print(f"\n❌ ERRO FATAL DURANTE A EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)