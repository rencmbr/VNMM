"""
Teste 1: Junção em Vértice Múltiplo (Quad-Junction / Four-Material Corner)
==========================================================================
Arquivo: VNMM_Quad_Junction_Test.py

Objetivo:
---------
Validar a capacidade do VNMM 2D em lidar com uma singularidade clássica de
junção quádrupla de materiais dielétricos distintos se encontrando em (0, 0):
  Domínio PEC: Omega = [-1, 1] x [-1, 1]
  - Q1 (x < 0, y > 0): eps_r1 = 1.0, mu_r = 1.0
  - Q2 (x > 0, y > 0): eps_r2 = 4.0, mu_r = 1.0
  - Q3 (x < 0, y < 0): eps_r3 = 2.0, mu_r = 1.0
  - Q4 (x > 0, y < 0): eps_r4 = 9.0, mu_r = 1.0

Metodologia:
------------
1. Referência de Alta Precisão via Elementos Finitos Vetoriais (Nédélec Q1
   em malha retangular fina 80x80 = 6400 elementos, livre de modos espúrios).
2. Discretização VNMM 2D por Nuvem de Pontos:
   - Nós internos com diretores alternados +-45 deg.
   - Nós escalonados nas 4 semi-interfaces (x=0, y=0) com diretores tangenciais,
     evitando nó exato na singularidade central (0,0) e nas quinas PEC.
   - Estênceis estritamente restritos a cada quadrante + suas 2 semi-interfaces.
3. Montagem variacional e purificação espectral (R_div < 5%).
"""

import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DIR_EXP = Path(__file__).resolve().parent
DIR_2D = DIR_EXP.parent
ROOT_DIR = DIR_2D.parent
DIR_FIG = ROOT_DIR / "paper" / "VNMM2D-Multimateriais" / "figures"
DIR_FIG.mkdir(parents=True, exist_ok=True)

for p in [str(DIR_2D), str(ROOT_DIR), str(DIR_EXP)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from src.gauss_quadrature_2d import get_gauss_points_weights_1d


# ==============================================================================
# 1. REFERÊNCIA: ELEMENTOS FINITOS ESCALARES H_z (Q1) EM MALHA FINA (LIVRE DE ESPÚRIOS)
# ==============================================================================

def solve_fem_hz_quad_junction(N=100, eps_map=None, num_modes=8):
    """
    Resolve o problema de autovalores TE_z em [-1, 1]^2 via formulação escalar em H_z:
      div( (1/eps_r) grad(H_z) ) + lambda H_z = 0,  dH_z/dn = 0 na fronteira PEC.
    Livre de modos espúrios e sem espaço nulo de dimensão infinita (apenas H_z = const, lambda = 0).
    """
    if eps_map is None:
        eps_map = {1: 1.0, 2: 4.0, 3: 2.0, 4: 9.0}

    assert N % 2 == 0, "N deve ser par para alinhar com x=0 e y=0."
    h = 2.0 / N
    num_nodes = (N + 1) * (N + 1)

    def n_idx(i, j):
        return j * (N + 1) + i

    # Matriz de rigidez local unitária Q1 (gradiente) para quadrado h x h
    K_loc_unit = (1.0 / 6.0) * np.array([
        [ 4, -1, -2, -1],
        [-1,  4, -1, -2],
        [-2, -1,  4, -1],
        [-1, -2, -1,  4]
    ], dtype=float)

    # Matriz de massa local unitária Q1 para quadrado h x h
    M_loc_unit = (h * h / 36.0) * np.array([
        [4, 2, 1, 2],
        [2, 4, 2, 1],
        [1, 2, 4, 2],
        [2, 1, 2, 4]
    ], dtype=float)

    rows, cols, data_K, data_M = [], [], [], []

    for j in range(N):
        yc = -1.0 + (j + 0.5) * h
        for i in range(N):
            xc = -1.0 + (i + 0.5) * h
            if xc < 0 and yc > 0:
                q = 1
            elif xc > 0 and yc > 0:
                q = 2
            elif xc < 0 and yc < 0:
                q = 3
            else:
                q = 4
            inv_eps = 1.0 / eps_map[q]

            # Ordem anti-horária dos nós do elemento: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
            n_ids = [n_idx(i, j), n_idx(i + 1, j), n_idx(i + 1, j + 1), n_idx(i, j + 1)]
            for a in range(4):
                ia = n_ids[a]
                for b in range(4):
                    ib = n_ids[b]
                    rows.append(ia); cols.append(ib)
                    data_K.append(inv_eps * K_loc_unit[a, b])
                    data_M.append(M_loc_unit[a, b])

    K_mat = csr_matrix((data_K, (rows, cols)), shape=(num_nodes, num_nodes))
    M_mat = csr_matrix((data_M, (rows, cols)), shape=(num_nodes, num_nodes))

    # Busca os menores autovalores próximos de sigma=0.0 (pulando o modo constante lambda=0)
    vals, _ = eigsh(K_mat, k=num_modes + 2, M=M_mat, sigma=0.05, which='LM')
    vals = np.sort(np.real(vals))
    vals_phys = vals[vals > 0.05][:num_modes]
    return vals_phys


# ==============================================================================
# 2. DISCRETIZAÇÃO VNMM 2D (NUVEM DE PONTOS) PARA QUAD-JUNCTION
# ==============================================================================

def generate_quad_junction_point_cloud(Nq=10):
    """
    Gera a nuvem de pontos nodais em [-1, 1] x [-1, 1] com 4 quadrantes.
    Nq = número de subdivisões por semi-eixo (de 0 a 1).
    Usa escalonamento (midpoints) nas interfaces x=0, y=0 e nas paredes PEC
    para evitar nós singulares em (0,0) e nas 4 quinas.
    """
    h = 1.0 / Nq
    coords = []
    vectors = []
    is_pec = []
    # tags possíveis:
    # 'Q1', 'Q2', 'Q3', 'Q4' (internos e PEC exclusivos do quadrante)
    # 'H_L' (y=0, x<0 -> compartilhado Q1 e Q3)
    # 'H_R' (y=0, x>0 -> compartilhado Q2 e Q4)
    # 'V_T' (x=0, y>0 -> compartilhado Q1 e Q2)
    # 'V_B' (x=0, y<0 -> compartilhado Q3 e Q4)
    tags = []

    # 1. Nós internos de cada quadrante (estritamente dentro de cada quadrante)
    for q_id, (sx, sy) in [(1, (-1, 1)), (2, (1, 1)), (3, (-1, -1)), (4, (1, -1))]:
        q_tag = f'Q{q_id}'
        for i in range(1, Nq):
            x = sx * i * h
            for j in range(1, Nq):
                y = sy * j * h
                ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(ang), np.sin(ang)])
                coords.append([x, y])
                vectors.append(t)
                is_pec.append(False)
                tags.append(q_tag)

    # 2. Nós alinhados nas 4 semi-interfaces (diretores puramente tangenciais)
    # Alinhados com a grade de pontos internos (k * h), evitando nó singular em (0,0)
    for k in range(1, Nq):
        pos = k * h
        # H_L: y=0, x=-pos
        coords.append([-pos, 0.0]); vectors.append([1.0, 0.0]); is_pec.append(False); tags.append('H_L')
        # H_R: y=0, x=+pos
        coords.append([+pos, 0.0]); vectors.append([1.0, 0.0]); is_pec.append(False); tags.append('H_R')
        # V_T: x=0, y=+pos
        coords.append([0.0, +pos]); vectors.append([0.0, 1.0]); is_pec.append(False); tags.append('V_T')
        # V_B: x=0, y=-pos
        coords.append([0.0, -pos]); vectors.append([0.0, 1.0]); is_pec.append(False); tags.append('V_B')

    # 3. Nós escalonados nas paredes PEC externas (sem nós nas quinas nem em +-1,0 / 0,+-1)
    for k in range(Nq):
        pos = (k + 0.5) * h
        # Topo y = +1: Q1 (x < 0) e Q2 (x > 0)
        coords.append([-pos, 1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q1')
        coords.append([+pos, 1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q2')
        # Base y = -1: Q3 (x < 0) e Q4 (x > 0)
        coords.append([-pos, -1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q3')
        coords.append([+pos, -1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q4')
        # Esquerda x = -1: Q1 (y > 0) e Q3 (y < 0)
        coords.append([-1.0, +pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q1')
        coords.append([-1.0, -pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q3')
        # Direita x = +1: Q2 (y > 0) e Q4 (y < 0)
        coords.append([1.0, +pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q2')
        coords.append([1.0, -pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q4')

    return (
        np.array(coords, dtype=float),
        np.array(vectors, dtype=float),
        np.array(is_pec, dtype=bool),
        np.array(tags)
    )


def solve_vnmm_quad_junction(Nq=12, Nc_half=12, eps_map=None, s_div=10.0, gauss_pts=3, Tol_det=1e-5):
    """
    Monta e resolve o problema espectral generalizado via VNMM 2D para o domínio
    de 4 materiais com junção quádrupla em (0,0).
    """
    if eps_map is None:
        eps_map = {1: 1.0, 2: 4.0, 3: 2.0, 4: 9.0}

    coords, vectors, is_pec, tags = generate_quad_junction_point_cloud(Nq=Nq)
    total_nodes = len(coords)

    # Mapeamento de quais tags cada quadrante pode enxergar
    quad_visible_tags = {
        1: {'Q1', 'H_L', 'V_T'},
        2: {'Q2', 'H_R', 'V_T'},
        3: {'Q3', 'H_L', 'V_B'},
        4: {'Q4', 'H_R', 'V_B'},
    }

    subsets = {}
    for q in [1, 2, 3, 4]:
        allowed = quad_visible_tags[q]
        cand = np.where(np.isin(tags, list(allowed)))[0]
        subsets[q] = {
            'indices': cand,
            'coords': coords[cand],
            'vectors': vectors[cand],
            'tree': KDTree(coords[cand])
        }

    # Células cartesianas de fundo (2*Nc_half x 2*Nc_half) alinhadas com x=0 e y=0
    Ncx = 2 * Nc_half
    Ncy = 2 * Nc_half
    x_edges = np.linspace(-1.0, 1.0, Ncx + 1)
    y_edges = np.linspace(-1.0, 1.0, Ncy + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(gauss_pts)

    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_M, cols_M, data_M = [], [], []

    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy_c = y1 - y0
        yc = 0.5 * (y0 + y1)
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx_c = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_J = 0.25 * dx_c * dy_c

            if xc < 0 and yc > 0:
                q = 1
            elif xc > 0 and yc > 0:
                q = 2
            elif xc < 0 and yc < 0:
                q = 3
            else:
                q = 4

            eps_r = eps_map[q]
            sub = subsets[q]

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx_c * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy_c * eta
                    Pg = np.array([xg, yg])
                    wg = wi * wj * det_J

                    sel_local, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
                        P=Pg,
                        nodes_coords=sub['coords'],
                        nodes_vectors=sub['vectors'],
                        search_tree=sub['tree'],
                        K=12,
                        Tol_det=Tol_det,
                        adaptive=True
                    )
                    sel_global = sub['indices'][sel_local]

                    beta = np.linalg.inv(matrix_A)
                    Phi_g = beta[0:2, :]
                    rot_Phi = beta[4, :] - beta[3, :]
                    div_Phi = beta[2, :] + beta[5, :]

                    rot_outer = np.outer(rot_Phi, rot_Phi) * wg
                    div_outer = np.outer(div_Phi, div_Phi) * wg
                    vec_dot = (Phi_g.T @ Phi_g) * (eps_r * wg)

                    for a in range(6):
                        Ia = sel_global[a]
                        for b in range(6):
                            Ib = sel_global[b]
                            rows_Kc.append(Ia); cols_Kc.append(Ib); data_Kc.append(rot_outer[a, b])
                            rows_Kd.append(Ia); cols_Kd.append(Ib); data_Kd.append(div_outer[a, b])
                            rows_M.append(Ia); cols_M.append(Ib); data_M.append(vec_dot[a, b])

    Kc = csr_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(total_nodes, total_nodes))
    Kd = csr_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(total_nodes, total_nodes))
    M = csr_matrix((data_M, (rows_M, cols_M)), shape=(total_nodes, total_nodes))

    int_idx = np.where(~is_pec)[0]
    Kc_red = 0.5 * (Kc[int_idx, :][:, int_idx].toarray() + Kc[int_idx, :][:, int_idx].toarray().T)
    Kd_red = 0.5 * (Kd[int_idx, :][:, int_idx].toarray() + Kd[int_idx, :][:, int_idx].toarray().T)
    M_red = 0.5 * (M[int_idx, :][:, int_idx].toarray() + M[int_idx, :][:, int_idx].toarray().T)

    K_red = Kc_red + s_div * Kd_red
    vals, vecs = la.eigh(K_red, M_red)

    vals_pos = vals[vals > 0.05]
    vecs_pos = vecs[:, vals > 0.05]

    physical_modes = []
    for i in range(len(vals_pos)):
        v = vecs_pos[:, i]
        e_c = v.T @ Kc_red @ v
        e_d = v.T @ Kd_red @ v
        e_m = v.T @ M_red @ v
        r_div = (e_d / e_c) * 100.0 if e_c > 1e-12 else 99999.0
        lambda_curl = e_c / e_m
        if r_div < 5.0:
            v_global = np.zeros(total_nodes)
            v_global[int_idx] = v
            physical_modes.append({
                'lambda': lambda_curl,
                'kc': np.sqrt(lambda_curl),
                'r_div': r_div,
                'eigenvector': v_global
            })

    return {
        'coords': coords,
        'vectors': vectors,
        'is_pec': is_pec,
        'tags': tags,
        'subsets': subsets,
        'physical_modes': physical_modes,
        'total_nodes': total_nodes,
        'active_nodes': int(np.sum(~is_pec)),
        'pec_nodes': int(np.sum(is_pec))
    }


# ==============================================================================
# 3. VISUALIZAÇÃO: GEOMETRIA + NUVEM E MODOS DE CAMPO ELÉTRICO
# ==============================================================================

def plot_quad_junction_geometry(res, eps_map):
    coords = res['coords']
    vectors = res['vectors']
    is_pec = res['is_pec']
    tags = res['tags']

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Regiões coloridas dos 4 quadrantes
    quads = [
        ((-1.0, 0.0), 1.0, 1.0, "#E3F2FD", f"Q1: eps_r = {eps_map[1]:.1f}"),
        (( 0.0, 0.0), 1.0, 1.0, "#FFF3E0", f"Q2: eps_r = {eps_map[2]:.1f}"),
        ((-1.0,-1.0), 1.0, 1.0, "#E8F5E9", f"Q3: eps_r = {eps_map[3]:.1f}"),
        (( 0.0,-1.0), 1.0, 1.0, "#FCE4EC", f"Q4: eps_r = {eps_map[4]:.1f}"),
    ]
    for (x0, y0), w, h, color, label in quads:
        rect = patches.Rectangle((x0, y0), w, h, facecolor=color, edgecolor="none", alpha=0.75)
        ax.add_patch(rect)
        ax.text(x0 + 0.5 * w, y0 + 0.85 * h, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color="#263238",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#B0BEC5"))

    # Paredes PEC e Interfaces internas
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k-', linewidth=2.5, label="PEC Boundary")
    ax.plot([-1, 1], [0, 0], color="#2E7D32", linestyle="--", linewidth=2.0, label="Dielectric Interfaces (x=0, y=0)")
    ax.plot([0, 0], [-1, 1], color="#2E7D32", linestyle="--", linewidth=2.0)

    # Destaque para o vértice quádruplo (0,0)
    ax.plot(0, 0, marker='*', markersize=14, color="#FF6F00", markeredgecolor="black",
            linestyle='None', label="Quad-Junction (0,0) [No Node]")

    # Vetores diretores
    is_int = np.isin(tags, ['H_L', 'H_R', 'V_T', 'V_B'])
    is_inner = (~is_pec) & (~is_int)

    ax.quiver(coords[is_inner, 0], coords[is_inner, 1], vectors[is_inner, 0], vectors[is_inner, 1],
              color="#1565C0", scale=48, width=0.0026, headwidth=3.5, headlength=4.0, alpha=0.85, label="Interior Nodes (±45°)")
    ax.quiver(coords[is_int, 0], coords[is_int, 1], vectors[is_int, 0], vectors[is_int, 1],
              color="#2E7D32", scale=42, width=0.0032, headwidth=3.5, headlength=4.0, label="Interface Nodes (Tangential)")
    ax.quiver(coords[is_pec, 0], coords[is_pec, 1], vectors[is_pec, 0], vectors[is_pec, 1],
              color="#C62828", scale=42, width=0.0032, headwidth=3.5, headlength=4.0, label="PEC Nodes (Staggered)")

    ax.set_title("Cavity with Four-Material Corner Junction\nPoint Cloud and Vector Directors",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.92)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    out_path = DIR_FIG / "Geometria_Nuvem_Quad_Junction.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Figura de geometria salva em: {out_path}")


def plot_quad_junction_modes(res, fem_vals):
    modes = res['physical_modes']
    subsets = res['subsets']

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes = axes.ravel()

    nx_eval, ny_eval = 32, 32
    xg = np.linspace(-0.96, 0.96, nx_eval)
    yg = np.linspace(-0.96, 0.96, ny_eval)
    Xg, Yg = np.meshgrid(xg, yg)

    for m_idx in range(4):
        ax = axes[m_idx]
        mode = modes[m_idx]
        v_glob = mode['eigenvector']

        Ex = np.zeros_like(Xg)
        Ey = np.zeros_like(Yg)

        for r in range(ny_eval):
            for c in range(nx_eval):
                px, py = Xg[r, c], Yg[r, c]
                if px < 0 and py > 0:
                    q = 1
                elif px > 0 and py > 0:
                    q = 2
                elif px < 0 and py < 0:
                    q = 3
                else:
                    q = 4
                sub = subsets[q]
                sel_loc, _, mat_A, _ = support_nodes_vnmm_2d_6_p1(
                    P=np.array([px, py]),
                    nodes_coords=sub['coords'],
                    nodes_vectors=sub['vectors'],
                    search_tree=sub['tree'],
                    K=12,
                    Tol_det=1e-5,
                    adaptive=True
                )
                sel_glob = sub['indices'][sel_loc]
                beta = np.linalg.inv(mat_A)
                E_vec = beta[0:2, :] @ v_glob[sel_glob]
                Ex[r, c] = E_vec[0]
                Ey[r, c] = E_vec[1]

        Emag = np.sqrt(Ex**2 + Ey**2)
        if np.max(Emag) > 0:
            Ex /= np.max(Emag)
            Ey /= np.max(Emag)
            Emag /= np.max(Emag)

        cf = ax.contourf(Xg, Yg, Emag, levels=24, cmap="viridis")
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
        ax.quiver(Xg[::2, ::2], Yg[::2, ::2], Ex[::2, ::2], Ey[::2, ::2],
                  color="white", alpha=0.85, scale=18, width=0.004)
        ax.axhline(0.0, color="white", linestyle="--", linewidth=1.2, alpha=0.9)
        ax.axvline(0.0, color="white", linestyle="--", linewidth=1.2, alpha=0.9)

        err = abs(mode['lambda'] - fem_vals[m_idx]) / fem_vals[m_idx] * 100.0
        ax.set_title(
            f"Mode {m_idx+1}: VNMM λ={mode['lambda']:.4f} | FEM λ={fem_vals[m_idx]:.4f}\n"
            f"Erro={err:.2f}% | R_div={mode['r_div']:.2f}%",
            fontsize=10, fontweight='bold'
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect('equal')

    fig.suptitle("Modes Ressonantes TEz na Cavidade com Junção Quádrupla (4 Materiais)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = DIR_FIG / "VNMM_Quad_Junction_Modes.png"
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"Figura de modos salva em: {out_path}")


# ==============================================================================
# 4. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    eps_map = {1: 1.0, 2: 4.0, 3: 2.0, 4: 9.0}

    print("=" * 75)
    print("  TESTE 1: JUNÇÃO EM VÉRTICE MÚLTIPLO (4 MATERIAIS / QUAD-JUNCTION)")
    print("  Q1: eps=1.0 | Q2: eps=4.0 | Q3: eps=2.0 | Q4: eps=9.0")
    print("=" * 75)

    print("\n1. Calculando referência de alta precisão (FEM Escalar H_z Q1 100x100)...")
    fem_vals = solve_fem_hz_quad_junction(N=100, eps_map=eps_map, num_modes=8)
    print("   Primeiros 8 autovalores FEM:", np.round(fem_vals, 6))

    print("\n2. Executando VNMM 2D (Nuvem de Pontos, Nq=12, Gauss 3x3)...")
    res = solve_vnmm_quad_junction(Nq=12, Nc_half=12, eps_map=eps_map, s_div=10.0, gauss_pts=3, Tol_det=1e-5)
    print(f"   Nós totais: {res['total_nodes']} (Active: {res['active_nodes']}, PEC: {res['pec_nodes']})")
    print(f"   Modes físicos purificados encontrados (R_div < 5%): {len(res['physical_modes'])}")

    print("\n   Primeiros 10 modos do VNMM encontrados:")
    for i, m in enumerate(res['physical_modes'][:10]):
        print(f"     VNMM[{i+1}]: lambda={m['lambda']:.6f}, R_div={m['r_div']:.2f}%")

    print("\n" + "-" * 75)
    print(f"{'Mode':<6} | {'Ref. FEM':<12} | {'VNMM 2D':<12} | {'Erro Rel. (%)':<14} | {'R_div (%)':<10}")
    print("-" * 75)
    for idx in range(min(6, len(fem_vals), len(res['physical_modes']))):
        r_f = fem_vals[idx]
        m = res['physical_modes'][idx]
        err = abs(m['lambda'] - r_f) / r_f * 100.0
        print(f"  {idx+1:<4} | {r_f:<12.6f} | {m['lambda']:<12.6f} | {err:<14.2f} | {m['r_div']:<10.2f}")
    print("-" * 75)

    plot_quad_junction_geometry(res, eps_map)
    plot_quad_junction_modes(res, fem_vals)


if __name__ == "__main__":
    main()
