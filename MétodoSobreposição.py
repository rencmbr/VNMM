import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# ==========================================
# 1. Parâmetros do Domínio e do Gap
# ==========================================
L = 4.0          # Comprimento do domínio
h_gap = 2.0      # Espessura do gap
H_mesh = 2.0     # Altura de cada região malhada (estator/rotor)

# Tensões de Dirichlet nas fronteiras do gap
V_bottom = 100.0 # Potencial em y = 0
V_top = 0.0      # Potencial em y = h_gap

# ==========================================
# 2. Geração das Malhas Triangulares 1 e 2
# ==========================================
# Malha 1 (Inferior) - Não conforme (resolução menor)
nx1, ny1 = 5, 4
x1 = np.linspace(0, L, nx1)
y1 = np.linspace(-H_mesh, 0, ny1)
X1, Y1 = np.meshgrid(x1, y1)
pts1 = np.column_stack([X1.ravel(), Y1.ravel()])
triang1 = tri.Triangulation(pts1[:,0], pts1[:,1])

# Malha 2 (Superior) - Não conforme (resolução maior)
nx2, ny2 = 9, 4
x2 = np.linspace(0, L, nx2)
y2 = np.linspace(h_gap, h_gap + H_mesh, ny2)
X2, Y2 = np.meshgrid(x2, y2)
pts2 = np.column_stack([X2.ravel(), Y2.ravel()])
triang2 = tri.Triangulation(pts2[:,0], pts2[:,1])

# ==========================================
# 3. Funções de Forma do Método OFEM no Gap
# ==========================================
def get_1D_shape_functions(x_eval, x_nodes):
    """Encontra o elemento 1D na fronteira e retorna os nós e as funções de forma."""
    for i in range(len(x_nodes)-1):
        if x_nodes[i] <= x_eval <= x_nodes[i+1]:
            L_elem = x_nodes[i+1] - x_nodes[i]
            N1 = (x_nodes[i+1] - x_eval) / L_elem
            N2 = (x_eval - x_nodes[i]) / L_elem
            return i, i+1, N1, N2
    return -1, -1, 0, 0

def compute_gap_potential(x, y, x_bottom, x_top, V_b, V_t, h):
    """Calcula o potencial no gap usando as funções de forma sobrepostas."""
    # Contribuição da Malha 1 (Superfície inferior y = 0)
    i1, i2, N1_b, N2_b = get_1D_shape_functions(x, x_bottom)
    weight_bottom = (h - y) / h
    V_gap_bottom = (N1_b * V_b + N2_b * V_b) * weight_bottom
    
    # Contribuição da Malha 2 (Superfície superior y = h)
    j1, j2, N1_t, N2_t = get_1D_shape_functions(x, x_top)
    weight_top = y / h
    V_gap_top = (N1_t * V_t + N2_t * V_t) * weight_top
    
    return V_gap_bottom + V_gap_top

# ==========================================
# 4. Avaliação e Comparação Analítica
# ==========================================
# Pontos de avaliação dentro do gap
n_eval = 50
x_eval = np.linspace(0.01, L-0.01, n_eval)
y_eval = np.linspace(0, h_gap, n_eval)
X_gap, Y_gap = np.meshgrid(x_eval, y_eval)
V_ofem = np.zeros_like(X_gap)
V_analytical = np.zeros_like(X_gap)

for i in range(n_eval):
    for j in range(n_eval):
        # Aproximação OFEM
        V_ofem[i, j] = compute_gap_potential(
            X_gap[i,j], Y_gap[i,j], x1, x2, V_bottom, V_top, h_gap
        )
        # Solução Analítica (Capacitor de placas paralelas)
        V_analytical[i, j] = V_bottom * (h_gap - Y_gap[i,j]) / h_gap

# Cálculo do Erro Absoluto
Erro = np.abs(V_ofem - V_analytical)

# ==========================================
# 5. Visualização dos Resultados
# ==========================================
fig = plt.figure(figsize=(15, 5))

# Plot 1: Malhas Não Conformes e Setup
ax1 = fig.add_subplot(131)
ax1.triplot(triang1, 'b-', lw=1, label='Malha 1 (S1)')
ax1.triplot(triang2, 'r-', lw=1, label='Malha 2 (S2)')
ax1.plot(x1, np.zeros_like(x1), 'bo', markersize=5, label='Nós Reais S1')
ax1.plot(x2, np.ones_like(x2)*h_gap, 'ro', markersize=5, label='Nós Reais S2')
ax1.fill_between([0, L], [0, 0], [h_gap, h_gap], color='gray', alpha=0.2, label='Gap Não Malhado')
ax1.set_title('Malhas Triangulares Não Conformes')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
ax1.legend(loc='upper right', fontsize=8)

# Plot 2: Potencial no Gap via OFEM
ax2 = fig.add_subplot(132)
cp = ax2.contourf(X_gap, Y_gap, V_ofem, levels=20, cmap='viridis')
fig.colorbar(cp, ax=ax2, label='Potencial (V)')
ax2.set_title('Aproximação do Potencial no Gap (OFEM)')
ax2.set_xlabel('x (m)')
ax2.set_ylabel('y (m)')

# Plot 3: Gráfico de Erro (Solução OFEM vs Analítica)
ax3 = fig.add_subplot(133)
cp_err = ax3.contourf(X_gap, Y_gap, Erro, levels=20, cmap='Reds')
fig.colorbar(cp_err, ax=ax3, label='Erro Absoluto (V)')
ax3.set_title('Erro (OFEM vs Analítica)')
ax3.set_xlabel('x (m)')
ax3.set_ylabel('y (m)')

plt.tight_layout()
plt.show()

# Validação Específica no Ponto de Gauss Pg(1,1) conforme material
V_pg_ofem = compute_gap_potential(1.0, 1.0, x1, x2, V_bottom, V_top, h_gap)
print(f"Potencial no ponto Pg(1,1): {V_pg_ofem:.2f} V")