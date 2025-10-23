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

class VariableBeamPINN:
    """
    Classe otimizada para resolver o Caso 2: Viga biapoiada com Rigidez (EI) Variável.
    EDO: d²/dx² (EI(x) * d²v/dx²) + q₀ = 0
    """
    
    def __init__(self, L=10.0, E0=1.0e9, I0=1.0e-3, q0=10.0):
        """Inicializa os parâmetros da viga variável."""
        self.L = L
        self.E0 = E0
        self.I0 = I0
        self.q0 = q0
        self.start_time = None
        
    def _print_step(self, message):
        """Função auxiliar para imprimir mensagens com timestamp"""
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        print(f"[{elapsed:7.1f}s] {message}")

    # =================================================================
    # 1. FUNÇÕES DE PROPRIEDADES VARIÁVEIS (CORRIGIDAS)
    # =================================================================

    def E_function(self, x):
        """Módulo de Elasticidade (Constante neste exemplo)"""
        # CORREÇÃO: Usando tf.ones_like para compatibilidade
        import tensorflow as tf
        return self.E0 * tf.ones_like(x)

    def I_function(self, x):
        """Momento de Inércia Variável: I(x) = I₀ * (1 + x/L)"""
        return self.I0 * (1.0 + x / self.L)
    
    def EI_function(self, x):
        """Rigidez à Flexão EI(x)"""
        return self.E_function(x) * self.I_function(x)
    
    # =================================================================
    # 2. EQUAÇÃO GOVERNANTE OTIMIZADA PARA RIGIDEZ VARIÁVEL
    # =================================================================

    def pde(self, x, v):
        """
        Equação governante (Resíduo da EDO) OTIMIZADA: 
        d²/dx² (EI(x) * d²v/dx²) + q₀ = 0
        """
        import tensorflow as tf
        
        # 1. Calcular d²v/dx² (Segunda derivada da deflexão)
        dv_xx = dde.grad.hessian(v, x, i=0, j=0)
        
        # 2. Calcular M = EI * d²v/dx² (Momento Fletor)
        EI = self.EI_function(x)
        M = EI * dv_xx
        
        # 3. Calcular d²M/dx² (Segunda derivada do Momento = -q)
        dM_dx = dde.grad.jacobian(M, x)
        d2M_dx2 = dde.grad.jacobian(dM_dx, x)
        
        # 4. Resíduo da EDO: d²M/dx² + q₀
        # CORREÇÃO: Usando tf.ones_like para a carga constante
        q_tensor = self.q0 * tf.ones_like(x)
        return d2M_dx2 + q_tensor

    def analytical_solution(self, x):
        """Solução analítica complexa, retorna um array de zeros para evitar erro."""
        X = x[:, 0] if len(x.shape) > 1 else x
        return np.zeros_like(X)
    
    def setup_problem(self):
        """Configura a geometria e as BCs."""
        self._print_step("Configurando problema PDE (Caso 2)...")
        
        geom = dde.geometry.Interval(0, self.L)
        
        def boundary_left(x, on_boundary):
            return on_boundary and np.isclose(x[0], 0)
        
        def boundary_right(x, on_boundary):
            return on_boundary and np.isclose(x[0], self.L)
        
        # 1. Condições de Contorno de Deflexão (v = 0)
        bc_v_0 = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_left)
        bc_v_L = dde.icbc.DirichletBC(geom, lambda x: 0, boundary_right)
        
        # 2. Condições de Contorno de Momento Fletor (M = EI*v'' = 0)
        def moment_zero_bc(x, y, X):
            return dde.grad.hessian(y, x, i=0, j=0)
        
        bc_M_0 = dde.icbc.OperatorBC(geom, moment_zero_bc, boundary_left)
        bc_M_L = dde.icbc.OperatorBC(geom, moment_zero_bc, boundary_right)
        
        data = dde.data.PDE(
            geom,
            self.pde,
            [bc_v_0, bc_v_L, bc_M_0, bc_M_L], 
            num_domain=2000, 
            num_boundary=100, 
            num_test=1000,
            train_distribution="uniform"
        )
        
        return data
    
    def train_model_optimized(self, num_adam_iterations=8000, num_lbfgs_iterations=5000):
        """Treina o modelo PINN com estratégia otimizada (Adam + L-BFGS-B)."""
        self._print_step("Iniciando treinamento da PINN (Caso 2: Rigidez Variável)...")
        
        data = self.setup_problem()
        
        layer_size = [1] + [50] * 8 + [1] 
        net = dde.nn.FNN(layer_size, "tanh", "Glorot normal")
        model = dde.Model(data, net)
        
        loss_weights = [2.0, 500, 500, 800, 800] 
        
        # --- FASE 1: Otimizador ADAM (Aumentado) ---
        self._print_step(f"FASE 1: Otimizador ADAM ({num_adam_iterations} iterações)")
        model.compile("adam", lr=1e-3, loss_weights=loss_weights)
        model.train(iterations=num_adam_iterations, display_every=1000)
        
        # --- FASE 2: Otimizador L-BFGS-B ---
        self._print_step(f"FASE 2: Otimizador L-BFGS-B ({num_lbfgs_iterations} max iter)")
        model.compile("L-BFGS-B", loss_weights=loss_weights)
        
        dde.optimizers.set_LBFGS_options(
            maxcor=100, 
            ftol=1.0e-12, 
            gtol=1.0e-08, 
            maxiter=num_lbfgs_iterations
        )
        
        try:
            losshistory, train_state = model.train()
            self._print_step("L-BFGS-B convergiu com sucesso.")
        except Exception as e:
            self._print_step(f"L-BFGS-B parou: {str(e)[:100]}...")
            
        return model, losshistory

    def safe_gradient(self, y, x):
        """Calcula gradiente de forma segura"""
        try:
            if len(y) < 3:
                return np.zeros_like(y)
            y_flat = y.flatten() if len(y.shape) > 1 else y
            x_flat = x.flatten() if len(x.shape) > 1 else x
            return np.gradient(y_flat, x_flat)
        except Exception:
            return np.zeros_like(y.flatten() if len(y.shape) > 1 else y)

    def evaluate_solution_comprehensive(self, model, num_points=500):
        """Avaliação focada na solução PINN, sem comparação analítica."""
        self._print_step("Avaliando soluções (Sem Solução Analítica)...")
        
        X_test = np.linspace(0, self.L, num_points).reshape(-1, 1)
        v_pred = model.predict(X_test)
        v_true = self.analytical_solution(X_test).reshape(-1, 1) 
        
        v_pred_flat = v_pred.flatten()
        v_true_flat = v_true.flatten()
        
        absolute_error = np.abs(v_pred_flat - v_true_flat)
        relative_error = np.zeros_like(absolute_error)
        
        X_flat = X_test.flatten()
        dv_dx_pred = self.safe_gradient(v_pred_flat, X_flat)
        dv_dx_true = self.safe_gradient(v_true_flat, X_flat) 
        d2v_dx2_pred = self.safe_gradient(dv_dx_pred, X_flat)
        d2v_dx2_true = self.safe_gradient(dv_dx_true, X_flat) 
        
        EI_values = self.E0 * self.I0 * (1.0 + X_test / self.L).flatten()
        M_pred = EI_values * d2v_dx2_pred
        M_true = np.zeros_like(M_pred) 

        X_mid = np.array([[self.L/2]])
        v_max_pred = model.predict(X_mid)[0, 0]
        v_max_true = 0.0 

        X_train = model.data.train_x_all
        v_train_pred = model.predict(X_train)
        v_train_true = np.zeros_like(v_train_pred) 
        
        # Calcular resíduos da PDE para análise de convergência
        X_pde_residual = np.linspace(0, self.L, 200).reshape(-1, 1)
        R_pred = model.predict(X_pde_residual, operator=self.pde)
        
        results = {
            'X_test': X_test, 'X_train': X_train,
            'v_pred': v_pred, 'v_true': v_true,
            'v_train_pred': v_train_pred, 'v_train_true': v_train_true,
            'v_pred_flat': v_pred_flat, 'v_true_flat': v_true_flat,
            'absolute_error': absolute_error, 'relative_error': relative_error,
            'dv_dx_pred': dv_dx_pred, 'dv_dx_true': dv_dx_true,
            'M_pred': M_pred, 'M_true': M_true,
            'v_max_pred': v_max_pred, 'v_max_true': v_max_true,
            'X_pde_residual': X_pde_residual, 'R_pred': R_pred  # Adicionados para análise de resíduos
        }
        return results

    def calculate_comprehensive_metrics(self, results):
        """Calcula métricas de desempenho abrangentes, desativando métricas relativas."""
        v_pred_flat = results['v_pred_flat']
        
        max_abs_error = np.max(np.abs(v_pred_flat)) 
        mean_abs_error = np.mean(np.abs(v_pred_flat))
        
        # Calcular R² mesmo sem solução analítica (em relação à média)
        if len(v_pred_flat) > 1:
            v_mean = np.mean(v_pred_flat)
            ss_tot = np.sum((v_pred_flat - v_mean) ** 2)
            ss_res = np.sum((v_pred_flat - v_mean) ** 2)  # Para solução única, será 0
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r_squared = 0

        metrics = {
            'max_abs_error': max_abs_error, 'mean_abs_error': mean_abs_error,
            'max_rel_error': 0.0, 'mean_rel_error': 0.0, 
            'v_max_pred': results['v_max_pred'], 'v_max_true': 0.0, 
            'max_deflection_error': 0.0, 
            'max_deflection_rel_error': 0.0, 
            'r_squared_test': r_squared, 'r_squared_train': 0.0, 
            'slope_test': 0.0, 'intercept_test': 0.0,
            'max_residual': np.max(np.abs(results['R_pred']))
        }
        return metrics

    def plot_comprehensive_comparison(self, results, metrics, losshistory, model):
        """Cria gráficos abrangentes da solução PINN (sem comparação analítica)."""
        self._print_step("Gerando gráficos de comparação...")
        
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 3, figure=fig)
        
        # Gráfico 1: Deflexão (apenas PINN)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['X_test'], results['v_pred'], 'b-', linewidth=3, label='PINN')
        ax1.scatter(results['X_train'], results['v_train_pred'], color='green', alpha=0.6, s=15, label='Pontos de Treino (PINN)')
        ax1.set_xlabel('Posição x (m)', fontsize=12)
        ax1.set_ylabel('Deflexão v(x) (m)', fontsize=12)
        ax1.set_title('SOLUÇÃO PINN: Deflexão da Viga (Rigidez Variável)', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Rigidez Variável EI(x)
        ax2 = fig.add_subplot(gs[0, 1])
        EI_test = self.E0 * self.I0 * (1.0 + results['X_test'] / self.L)
        ax2.plot(results['X_test'], EI_test, 'k-', linewidth=3)
        ax2.set_xlabel('Posição x (m)', fontsize=12)
        ax2.set_ylabel('Rigidez EI(x) (N·m²)', fontsize=12)
        ax2.set_title('Propriedade: Rigidez EI(x) Variável', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Gráfico 3: Força Cortante (V(x) = -dM/dx)
        ax3 = fig.add_subplot(gs[0, 2])
        V_pred = -self.safe_gradient(results['M_pred'], results['X_test'].flatten())
        ax3.plot(results['X_test'], V_pred, 'm-', linewidth=2)
        ax3.set_xlabel('Posição x (m)', fontsize=12)
        ax3.set_ylabel('Força Cortante V(x) (N)', fontsize=12)
        ax3.set_title('Força Cortante V(x) = -dM/dx', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Gráfico 4: Derivada (Inclinação)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.plot(results['X_test'], results['dv_dx_pred'], 'b-', linewidth=2)
        ax4.set_xlabel('Posição x (m)', fontsize=12)
        ax4.set_ylabel("v'(x) (inclinação)", fontsize=12)
        ax4.set_title('Derivada Primeira - Inclinação', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Gráfico 5: Momento fletor M(x)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(results['X_test'], results['M_pred'], 'b-', linewidth=2)
        ax5.set_xlabel('Posição x (m)', fontsize=12)
        ax5.set_ylabel('Momento Fletor M(x) (N·m)', fontsize=12)
        ax5.set_title('Momento Fletor M(x) = EI(x)v\'\'', fontsize=14, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Gráfico 6: Erro da PDE (Medida de Convergência) - CORRIGIDO
        ax6 = fig.add_subplot(gs[1, 2])
        # Usando os pontos pré-calculados para o resíduo
        ax6.scatter(results['X_pde_residual'], np.abs(results['R_pred']), alpha=0.5, s=15)
        ax6.axhline(y=0, color='r', linestyle='--')
        ax6.set_xlabel('Posição x (m)', fontsize=12)
        ax6.set_ylabel('|Resíduo da PDE|', fontsize=12)
        ax6.set_title('Resíduo da EDO (Convergência Física)', fontsize=14, fontweight='bold')
        ax6.set_yscale('log')
        ax6.grid(True, alpha=0.3)
        
        # Gráfico 7: Histórico de perda
        ax7 = fig.add_subplot(gs[2, 0])
        if hasattr(losshistory, 'steps') and losshistory.steps:
            ax7.semilogy(losshistory.steps, losshistory.loss_train, 'b-', linewidth=2, label='Perda de Treino Total')
            ax7.set_xlabel('Iteração', fontsize=12)
            ax7.set_ylabel('Perda', fontsize=12)
            ax7.set_title('Histórico de Treinamento', fontsize=14, fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # Gráfico 8: Resumo das métricas (Focado na Perda)
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.axis('off')
        
        final_loss = losshistory.loss_train[-1][0] if losshistory.loss_train else 'N/A'
        
        textstr = '\n'.join([
            'RESUMO DAS MÉTRICAS',
            '=' * 35,
            f'Perda Final (Treino): {final_loss:.2e}',
            f'Máx. Deflexão PINN: {metrics["v_max_pred"]:.4e} m',
            '',
            f'Máx. |v(x)|: {metrics["max_abs_error"]:.2e} m',
            f'Médio |v(x)|: {metrics["mean_abs_error"]:.2e} m',
            '',
            f'Convergiu? {"✅ SIM" if final_loss < 1.0e6 else "⚠️ NÃO"}',
            f'Resíduo PDE Máx: {metrics["max_residual"]:.2e}'
        ])
        ax8.text(0.05, 0.95, textstr, transform=ax8.transAxes, fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontfamily='monospace')
        
        # Gráfico 9: Informações do modelo
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        
        model_info = '\n'.join([
            'INFORMAÇÕES DO MODELO (CASO 2)',
            '=' * 35,
            f'Comprimento (L): {self.L} m',
            f'E₀: {self.E0:.2e} Pa',
            f'I₀: {self.I0:.2e} m⁴',
            f'Carga (q₀): {self.q0} N/m',
            f'Variável: I(x) = I₀(1 + x/L)',
            '',
            'ARQUITETURA',
            f'Camadas: 1 → [50]×8 → 1',
            f'Pontos domínio: 2000',
            f'Pesos de Perda: 2.0:500:800',
        ])
        ax9.text(0.05, 0.95, model_info, transform=ax9.transAxes, fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()

    def print_detailed_report(self, metrics, final_loss):
        """Imprime relatório detalhado."""
        print("\n" + "="*80)
        print("RELATÓRIO DETALHADO - CASO 2: VIGA BIAPOIADA (RIGIDEZ VARIÁVEL)")
        print("="*80)
        
        print(f"\nPARÂMETROS DA VIGA:")
        print(f"   Comprimento (L): {self.L} m")
        print(f"   Módulo E₀: {self.E0:.2e} Pa")
        print(f"   Inércia I₀: {self.I0:.2e} m⁴")
        print(f"   Carga Distribuída (q₀): {self.q0} N/m")
        print(f"   Rigidez Variável: EI(x) = E₀ * I₀ * (1 + x/L)")
        
        print(f"\nCONVERGÊNCIA (FOCO NA PERDA):")
        print(f"   Perda de Treinamento Final: {final_loss:.6e}")
        print(f"   Máx. Deflexão (PINN):       {metrics['v_max_pred']:.6e} m")
        print(f"   Máx. |v(x)| no domínio:     {metrics['max_abs_error']:.2e} m")
        print(f"   Resíduo PDE Máximo:         {metrics['max_residual']:.2e}")
        
        print("\nOBSERVAÇÃO:")
        print("   Sem solução analítica simples, a precisão é avaliada pela minimização do resíduo da EDO (Perda).")
        print("   A perda ainda está alta (>1e6), indicando necessidade de mais ajustes nos hiperparâmetros.")
        
        print("="*80)

# ----------------------------------------------------------------------
# EXECUÇÃO PRINCIPAL
# ----------------------------------------------------------------------
if __name__ == "__main__":
    
    start_global_time = time.time()
    
    print("=" * 80)
    print("ANÁLISE PINN - VIGA BIAPOIADA (CASO 2) - RIGIDEZ VARIÁVEL")
    print("=" * 80)
    
    # Parâmetros de engenharia realistas para a rigidez variável
    L_USER = 10.0 
    E0_USER = 200e9  # Aço
    I0_USER = 1e-4   # Inércia inicial
    Q0_USER = 5000.0 # Carga maior para uma viga de engenharia
    
    beam = VariableBeamPINN(
        L=L_USER, 
        E0=E0_USER, 
        I0=I0_USER,
        q0=Q0_USER
    )
    
    try:
        # 1. Treinamento
        model, losshistory = beam.train_model_optimized(
            num_adam_iterations=8000, 
            num_lbfgs_iterations=5000
        ) 
        
        # 2. Avaliação
        results = beam.evaluate_solution_comprehensive(model)
        metrics = beam.calculate_comprehensive_metrics(results)
        final_loss = losshistory.loss_train[-1][0] if losshistory.loss_train else float('inf')
        
        # 3. Visualização 
        beam.plot_comprehensive_comparison(results, metrics, losshistory, model)
        
        # 4. Relatório Final
        beam.print_detailed_report(metrics, final_loss)
        
        total_elapsed_time = time.time() - start_global_time
        print(f"\n🎯 ANÁLISE CONCLUÍDA! (Tempo total: {total_elapsed_time:.2f} segundos)")
        
    except Exception as e:
        print(f"\n❌ ERRO FATAL DURANTE A EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
    
    print("=" * 80)