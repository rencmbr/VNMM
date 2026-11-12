"""
Experimento Didático: Interpolação 2D com o Método VNMM (Vector Nodal Meshless Method)
======================================================================================

Objetivo:
---------
Demonstrar passo a passo como o método VNMM interpola um campo vetorial 2D
a partir de uma nuvem de pontos discretos com diretores arbitrários.

Conceitos Fundamentais do VNMM em 2D:
-------------------------------------
1. Cada nó 'i' possui coordenadas x_i = (x_i, y_i) e um vetor diretor unitário t_i = (t_ix, t_iy).
2. O grau de liberdade (DoF) escalar associado a cada nó é a projeção tangencial
   do campo vetorial ao longo do diretor:
       e_i = E(x_i) . t_i
3. Base polinomial vetorial linear completa (P^1, 6 termos):
       p_1 = [1, 0]^T
       p_2 = [0, 1]^T
       p_3 = [dx, 0]^T
       p_4 = [dy, 0]^T
       p_5 = [0, dx]^T
       p_6 = [0, dy]^T
   onde dx = x - x_P e dy = y - y_P são coordenadas relativas ao ponto de avaliação P.
4. Para um ponto de avaliação P, escolhe-se um suporte de 6 nós (stencil).
5. A matriz de colocação local A (6x6) é montada impondo:
       N_i(x_k) . t_k = delta_ik  =>  A * beta = I_6  =>  beta = inv(A)
   Linha k de A:
       A[k, :] = [ t_kx,  t_ky,  dx_k*t_kx,  dy_k*t_kx,  dx_k*t_ky,  dy_k*t_ky ]
6. No ponto P (onde dx=0, dy=0):
       Phi(P) = beta[0:2, :]   (matriz 2x6)
       curl_Phi = beta[4, :] - beta[3, :]   (vetor 1x6)
7. Campo interpolado e rotacional:
       E^h(P) = Phi(P) @ e_s
       (rot E^h)_z(P) = curl_Phi @ e_s
"""

import sys
from pathlib import Path
import numpy as np

# Configuração dos caminhos para importar módulos do VNMM
DIR_EXPERIMENTOS = Path(__file__).resolve().parent
DIR_2D = DIR_EXPERIMENTOS.parent
ROOT_DIR = DIR_2D.parent

for p in [str(DIR_2D), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Módulos do projeto
from scripts.vector_field_te11_2d import vector_field_te11
from scripts.build_search_tree_2d import build_search_tree
from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from scripts.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1


def main():
    print("=" * 80)
    print("      PASSO A PASSO DA INTERPOLAÇÃO VETORIAL VNMM 2D (BASE COMPLETA P^1)")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # PASSO 1: Criar a nuvem de nós aleatórios e seus vetores diretores
    # -------------------------------------------------------------------------
    print("\n--- PASSO 1: Geração da Nuvem de Pontos e Diretores ---")
    np.random.seed(42)  # Semente fixa para reprodutibilidade
    num_nodes = 80
    domain_min, domain_max = -10.0, 10.0

    # Coordenadas aleatórias (N, 2)
    coords = np.random.uniform(domain_min, domain_max, (num_nodes, 2))

    # Ângulos aleatórios para definir diretores unitários t_i = (cos theta, sin theta)
    thetas = np.random.uniform(0, 2 * np.pi, num_nodes)
    vectors = np.column_stack([np.cos(thetas), np.sin(thetas)])

    print(f"Total de nós gerados : {num_nodes}")
    print(f"Domínio              : [{domain_min}, {domain_max}] x [{domain_min}, {domain_max}]")
    print(f"Exemplo do Nó 0      : Coord = {coords[0].round(3)}, Diretor t = {vectors[0].round(3)}")

    # Construir árvore KDTree para busca espacial de vizinhos
    tree = build_search_tree(coords)

    # -------------------------------------------------------------------------
    # PASSO 2: Amostragem do Campo Exato nos Nós (Graus de Liberdade e_i)
    # -------------------------------------------------------------------------
    print("\n--- PASSO 2: Amostragem do Campo Alvo (Modo TE11 em Cavidade) ---")
    # vector_field_te11 calcula o campo elétrico E e seu rotacional analítico
    E_nodes, curl_E_nodes = vector_field_te11(coords)

    # O DoF de cada nó é a projeção escalar e_i = E(x_i) . t_i
    nodal_projections = np.sum(E_nodes * vectors, axis=1)

    print(f"Campo exato avaliado em todos os nós.")
    print(f"Exemplo do Nó 0: E exato = {E_nodes[0].round(4)}, Projeção e_0 = {nodal_projections[0]:.4f}")

    # -------------------------------------------------------------------------
    # PASSO 3: Definir os 5 Pontos de Avaliação
    # -------------------------------------------------------------------------
    print("\n--- PASSO 3: Definição dos 5 Pontos de Avaliação ---")
    eval_points = np.array([
        [ 2.0,  3.0],
        [-3.5,  1.5],
        [ 0.0, -4.0],
        [-5.0, -5.0],
        [ 4.5, -2.5]
    ])

    for idx, P in enumerate(eval_points, 1):
        print(f"  Ponto {idx}: P = [{P[0]:5.2f}, {P[1]:5.2f}]")

    # -------------------------------------------------------------------------
    # PASSO 4: Interpolação Detalhada Ponto a Ponto
    # -------------------------------------------------------------------------
    print("\n--- PASSO 4: Execução da Interpolação Passo a Passo ---")

    results_summary = []

    for idx, P in enumerate(eval_points, 1):
        print("\n" + "#" * 65)
        print(f" AVALIANDO PONTO {idx}: P = [{P[0]:.3f}, {P[1]:.3f}]")
        print("#" * 65)

        # 4.1: Valor analítico exato no ponto P
        exact_E, exact_curl_E = vector_field_te11(P)
        print(f"\n[4.1] Valor Analítico Exato em P:")
        print(f"      E_exato(P)   = [{exact_E[0]:.6f}, {exact_E[1]:.6f}]^T")
        print(f"      rot_E_exato  = {exact_curl_E:.6f} z_hat")

        # 4.2: Seleção dos 6 nós de suporte (stencil)
        selected_nodes, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
            P=P,
            nodes_coords=coords,
            nodes_vectors=vectors,
            search_tree=tree,
            K=12,
            Tol_det=1e-4,
            adaptive=True
        )

        print(f"\n[4.2] Seleção dos 6 Nós de Suporte (Stencil):")
        print(f"      Índices dos nós selecionados: {selected_nodes}")
        print(f"      Vizinhos K consultados      : {eff_k}")
        print(f"      Determinante |det(A)|       : {det_A:.6f}")
        for k_pos, n_idx in enumerate(selected_nodes):
            dist = np.linalg.norm(coords[n_idx] - P)
            print(f"        Nó {n_idx:>2d} (posição {k_pos}): Coord = {coords[n_idx].round(3)}, "
                  f"t = {vectors[n_idx].round(3)}, dist = {dist:.3f}")

        # 4.3: Matriz de Colocação A (6x6)
        print(f"\n[4.3] Matriz de Colocação Local A (6x6):")
        print("      Linha k: [tx, ty, dx*tx, dy*tx, dx*ty, dy*ty]")
        for row in matrix_A:
            print("      [" + "  ".join(f"{val:8.3f}" for val in row) + " ]")

        # 4.4: Inversão de A e Funções de Forma
        # beta = inv(A)
        # Phi(P) = beta[0:2, :] (2x6)
        # curl_Phi = beta[4, :] - beta[3, :] (1x6)
        Phi, curl_Phi, beta = shape_functions_vnmm_2d_6_p1(
            P=P,
            nodes_coords=coords,
            nodes_vectors=vectors,
            selected_nodes=selected_nodes,
            matrix_a=matrix_A
        )

        print(f"\n[4.4] Matriz de Funções de Forma Phi(P) (2 linhas x 6 nós):")
        print(f"      Linha x (Phi_x): {np.round(Phi[0], 4)}")
        print(f"      Linha y (Phi_y): {np.round(Phi[1], 4)}")
        print(f"      Rotacional das funções de forma (curl_Phi): {np.round(curl_Phi, 4)}")

        # 4.5: Projeções dos nós selecionados (e_s)
        e_s = nodal_projections[selected_nodes]
        print(f"\n[4.5] Graus de Liberdade do Suporte e_s (projeções escalares):")
        print(f"      e_s = {np.round(e_s, 5)}")

        # 4.6: Reconstrução do campo e do rotacional
        interp_E = Phi @ e_s
        interp_curl_E = float(np.dot(curl_Phi, e_s))

        # 4.7: Erros de Interpolação
        vec_error = float(np.linalg.norm(interp_E - exact_E))
        curl_error = float(abs(interp_curl_E - exact_curl_E))

        print(f"\n[4.6] Resultado da Interpolação:")
        print(f"      E interpolado     : [{interp_E[0]:.6f}, {interp_E[1]:.6f}]^T")
        print(f"      E exato           : [{exact_E[0]:.6f}, {exact_E[1]:.6f}]^T")
        print(f"      Erro vetorial ||E^h - E|| : {vec_error:.6e}")
        print(f"      rot(E) interpolado: {interp_curl_E:.6f}")
        print(f"      rot(E) exato      : {exact_curl_E:.6f}")
        print(f"      Erro de rotacional: {curl_error:.6e}")

        results_summary.append({
            'id': idx,
            'P': P,
            'selected_nodes': selected_nodes,
            'E_exact': exact_E,
            'E_interp': interp_E,
            'vec_err': vec_error,
            'curl_exact': exact_curl_E,
            'curl_interp': interp_curl_E,
            'curl_err': curl_error
        })

    # -------------------------------------------------------------------------
    # PASSO 5: Tabela Resumo dos Resultados
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("                    TABELA RESUMO DOS 5 PONTOS")
    print("=" * 80)
    header = f"{'Ponto':<6} | {'Coords P':<14} | {'||E_exato||':<11} | {'||E_interp||':<12} | {'Erro Vetorial':<14} | {'Erro Rotacional':<15}"
    print(header)
    print("-" * len(header))
    for r in results_summary:
        p_str = f"({r['P'][0]:4.1f}, {r['P'][1]:4.1f})"
        norm_exact = np.linalg.norm(r['E_exact'])
        norm_interp = np.linalg.norm(r['E_interp'])
        print(f"P{r['id']:<5} | {p_str:<14} | {norm_exact:<11.4f} | {norm_interp:<12.4f} | {r['vec_err']:<14.4e} | {r['curl_err']:<15.4e}")
    print("=" * 80)
    print("\nInterpolação concluída com sucesso!")

    # -------------------------------------------------------------------------
    # PASSO 6: Visualização Gráfica
    # -------------------------------------------------------------------------
    plot_file = DIR_EXPERIMENTOS / "interpolation_2d_plot.png"
    print(f"\n--- PASSO 6: Gerando Gráfico dos Nós, Diretores e Suportes ---")
    plot_interpolation_setup(
        coords=coords,
        vectors=vectors,
        eval_points=eval_points,
        results_summary=results_summary,
        output_path=plot_file,
        show=True
    )


def plot_interpolation_setup(
    coords,
    vectors,
    eval_points,
    results_summary,
    output_path=None,
    show=True
):
    """
    Gera um gráfico visualizando:
    1. A nuvem de nós espalhados com seus vetores diretores unitários t_i.
    2. Os 5 pontos de interpolação P_1 a P_5 (marcados com estrelas coloridas).
    3. Os 6 nós de suporte selecionados para cada ponto P, com suas direções
       e linhas tracejadas conectando o ponto de avaliação ao stencil.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 10))

    # 1. Nós da nuvem de fundo (em cinza para não poluir a visualização)
    ax.scatter(
        coords[:, 0], coords[:, 1],
        color="lightgray", edgecolor="#777777",
        s=40, alpha=0.75, zorder=2,
        label="Nós VNMM da Nuvem"
    )

    # 2. Vetores diretores unitários dos nós de fundo
    ax.quiver(
        coords[:, 0], coords[:, 1],
        vectors[:, 0], vectors[:, 1],
        angles="xy", scale_units="xy", scale=1.0,
        color="#bbbbbb", width=0.003, alpha=0.6, zorder=2
    )

    # Paleta com 5 cores distintas para os 5 pontos de interpolação
    stencil_colors = ["#D90429", "#0077B6", "#2B9348", "#7209B7", "#FF7B00"]

    for r, color in zip(results_summary, stencil_colors):
        p_id = r['id']
        P = r['P']
        supp_nodes = r['selected_nodes']

        # 3. Linhas tracejadas ligando o ponto P aos seus 6 nós de suporte
        for node_idx in supp_nodes:
            node_xy = coords[node_idx]
            ax.plot(
                [P[0], node_xy[0]], [P[1], node_xy[1]],
                linestyle="--", color=color, linewidth=1.2, alpha=0.55, zorder=3
            )

        # 4. Destacar os 6 nós de suporte
        ax.scatter(
            coords[supp_nodes, 0], coords[supp_nodes, 1],
            facecolor=color, edgecolor="black", linewidth=1.0,
            s=80, alpha=0.9, zorder=4
        )

        # 5. Destacar os vetores diretores dos 6 nós de suporte
        ax.quiver(
            coords[supp_nodes, 0], coords[supp_nodes, 1],
            vectors[supp_nodes, 0], vectors[supp_nodes, 1],
            angles="xy", scale_units="xy", scale=1.0,
            color=color, width=0.0055, zorder=5
        )

        # 6. Destacar o Ponto de Interpolação (estrela grande)
        ax.scatter(
            P[0], P[1],
            facecolor=color, edgecolor="black", linewidth=1.5,
            marker="*", s=360, zorder=6,
            label=f"P{p_id} ({P[0]:.1f}, {P[1]:.1f}) + 6 Nós Suporte"
        )

        # 7. Rótulo com o identificador do ponto
        ax.annotate(
            f"P{p_id}",
            (P[0], P[1]),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=12,
            fontweight="bold",
            color=color,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, alpha=0.9),
            zorder=7
        )

    ax.set_title(
        "Interpolação Vetorial VNMM 2D (Base Linear Completa $\\mathcal{P}^1$)\n"
        "Nuvem de Nós, Diretores Vetoriais $\\mathbf{t}_i$, 5 Pontos de Avaliação e Stencils de Suporte",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Coordenada X", fontsize=11)
    ax.set_ylabel("Coordenada Y", fontsize=11)
    ax.set_xlim(-11.5, 11.5)
    ax.set_ylim(-11.5, 11.5)
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico salvo com sucesso em: {output_path}")

    if show:
        plt.show()


if __name__ == "__main__":
    main()

