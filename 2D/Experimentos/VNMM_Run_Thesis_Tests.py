"""
Simulação Completa dos Dois Testes Avançados da Tese:
- Teste 4.3.2: Guia com Saliência em "H" Homogêneo e Anisotrópico
- Teste 4.3.3: Guia Bimaterial Anisotrópico com Cilindros Circulares PEC (raio r = 1/3)
======================================================================================
"""

import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

DIR_EXP = Path(__file__).resolve().parent
DIR_2D = DIR_EXP.parent
ROOT_DIR = DIR_2D.parent

for p in [str(DIR_2D), str(ROOT_DIR), str(DIR_EXP)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from src.gauss_quadrature_2d import get_gauss_points_weights_1d


# ==============================================================================
# TESTE 4.3.2: GUIA COM SALIÊNCIA EM "H"
# ==============================================================================

def is_inside_H(x, y, tol=1e-12):
    if x < -2.5 - tol or x > 2.5 + tol or y < -1.5 - tol or y > 1.5 + tol:
        return False
    if -1.0 + tol < x < 1.0 - tol and y > 0.5 + tol:
        return False
    if -1.0 + tol < x < 1.0 - tol and y < -0.5 - tol:
        return False
    return True


def run_test_4_3_2():
    print("\n" + "=" * 80)
    print("  EXECUTANDO TESTE 4.3.2: GUIA DE ONDA COM SALIÊNCIA EM 'H' (ANISOTRÓPICO)")
    print("=" * 80)

    # Grid ajustado
    nx_side, nx_mid, ny_side, ny_mid = 11, 15, 21, 9
    x_left = np.linspace(-2.5, -1.0, nx_side)
    x_mid = np.linspace(-1.0, 1.0, nx_mid)
    x_right = np.linspace(1.0, 2.5, nx_side)
    x_all = np.unique(np.concatenate([x_left, x_mid, x_right]))
    y_all = np.linspace(-1.5, 1.5, ny_side)

    coords_list, vectors_list, is_pec_list = [], [], []

    for i, x0 in enumerate(x_all):
        for j, y0 in enumerate(y_all):
            if not is_inside_H(x0, y0):
                continue
            is_pec = False
            t_pec = np.array([0.0, 0.0])

            if abs(x0 - (-2.5)) < 1e-6 or abs(x0 - 2.5) < 1e-6:
                is_pec = True; t_pec = np.array([0.0, 1.0])
            elif (abs(y0 - 1.5) < 1e-6 or abs(y0 - (-1.5)) < 1e-6) and (x0 <= -1.0 + 1e-6 or x0 >= 1.0 - 1e-6):
                is_pec = True; t_pec = np.array([1.0, 0.0])
            elif (abs(x0 - (-1.0)) < 1e-6 or abs(x0 - 1.0) < 1e-6) and abs(y0) >= 0.5 - 1e-6:
                is_pec = True; t_pec = np.array([0.0, 1.0])
            elif (abs(y0 - 0.5) < 1e-6 or abs(y0 - (-0.5)) < 1e-6) and abs(x0) <= 1.0 + 1e-6:
                is_pec = True; t_pec = np.array([1.0, 0.0])

            if is_pec:
                t = t_pec / np.linalg.norm(t_pec)
            else:
                ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(ang), np.sin(ang)])

            coords_list.append([x0, y0])
            vectors_list.append(t)
            is_pec_list.append(is_pec)

    coords = np.array(coords_list)
    vectors = np.array(vectors_list)
    is_pec = np.array(is_pec_list, dtype=bool)
    total_nodes = len(coords)
    tree = KDTree(coords)

    # Plotar nuvem e malha cartesiana de integração
    fig, ax = plt.subplots(figsize=(8, 5))
    Ncx, Ncy = 24, 16
    x_edges = np.linspace(-2.5, 2.5, Ncx + 1)
    y_edges = np.linspace(-1.5, 1.5, Ncy + 1)

    # Desenhar células de fundo
    for j in range(Ncy):
        for i in range(Ncx):
            xc = 0.5 * (x_edges[i] + x_edges[i+1])
            yc = 0.5 * (y_edges[j] + y_edges[j+1])
            if is_inside_H(xc, yc):
                rect = plt.Rectangle((x_edges[i], y_edges[j]), x_edges[i+1]-x_edges[i], y_edges[j+1]-y_edges[j],
                                     fill=True, facecolor="#E3F2FD", edgecolor="#90CAF9", linewidth=0.8, alpha=0.5)
                ax.add_patch(rect)

    # Contorno geométrico H
    poly_H_outer = [
        [-2.5, -1.5], [2.5, -1.5], [2.5, 1.5], [-2.5, 1.5]
    ]
    ax.plot([-2.5, -1.0, -1.0, 1.0, 1.0, 2.5, 2.5, 1.0, 1.0, -1.0, -1.0, -2.5, -2.5],
            [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, -1.5],
            'k-', linewidth=2.0, label='Fronteira PEC')

    # Nós e vetores
    ax.quiver(coords[~is_pec, 0], coords[~is_pec, 1],
              vectors[~is_pec, 0], vectors[~is_pec, 1],
              color="#1565C0", scale=35, width=0.003, alpha=0.8, label="Nós Ativos (±45°)")
    ax.quiver(coords[is_pec, 0], coords[is_pec, 1],
              vectors[is_pec, 0], vectors[is_pec, 1],
              color="#D32F2F", scale=30, width=0.004, label="Nós PEC (Tangenciais)")

    ax.set_title("Teste 4.3.2: Guia com Saliência ('H') - Nuvem de Pontos e Células de Fundo", fontsize=11, fontweight='bold')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-2.7, 2.7)
    ax.set_ylim(-1.7, 1.7)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_h_path = DIR_EXP / "Verificacao_Nuvens_Malha_Guia_H.png"
    plt.savefig(fig_h_path, dpi=200)
    plt.close()
    print(f"Figura salva em: {fig_h_path}")

    # Montagem variacional
    eps_tensor = np.array([[1.0, -1j], [1j, 4.0]], dtype=complex)
    mu_r = 1.0
    xi_1d, w_1d = get_gauss_points_weights_1d(2)

    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_M, cols_M, data_M = [], [], []

    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]; dy_c = y1 - y0; yc = 0.5 * (y0 + y1)
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]; dx_c = x1 - x0; xc = 0.5 * (x0 + x1)
            if not is_inside_H(xc, yc): continue
            det_J = 0.25 * dx_c * dy_c
            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx_c * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy_c * eta
                    if not is_inside_H(xg, yg): continue
                    Pg = np.array([xg, yg]); wg = wi * wj * det_J
                    sel_local, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
                        P=Pg, nodes_coords=coords, nodes_vectors=vectors, search_tree=tree, K=14, Tol_det=1e-5, adaptive=True
                    )
                    beta = np.linalg.inv(matrix_A)
                    Phi_g = beta[0:2, :]; rot_Phi = beta[4, :] - beta[3, :]
                    rot_outer = (1.0 / mu_r) * np.outer(rot_Phi, rot_Phi) * wg
                    vec_dot = (Phi_g.T @ eps_tensor @ Phi_g) * wg
                    for a in range(6):
                        Ia = sel_local[a]
                        for b in range(6):
                            Ib = sel_local[b]
                            rows_Kc.append(Ia); cols_Kc.append(Ib); data_Kc.append(rot_outer[a, b])
                            rows_M.append(Ia); cols_M.append(Ib); data_M.append(vec_dot[a, b])

    Kc = csr_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(total_nodes, total_nodes), dtype=complex)
    M = csr_matrix((data_M, (rows_M, cols_M)), shape=(total_nodes, total_nodes), dtype=complex)

    int_idx = np.where(~is_pec)[0]
    Kc_red = 0.5 * (Kc[int_idx, :][:, int_idx].toarray() + Kc[int_idx, :][:, int_idx].toarray().conj().T)
    M_red = 0.5 * (M[int_idx, :][:, int_idx].toarray() + M[int_idx, :][:, int_idx].toarray().conj().T)

    vals, vecs = la.eigh(Kc_red, M_red)
    ref_lambdas = [0.041212, 0.400641, 0.786745, 0.998524, 1.638579]
    res_modes = []
    for r in ref_lambdas:
        diffs = np.abs(vals - r)
        best_idx = np.argmin(diffs)
        val = vals[best_idx]
        err = abs(val - r) / r * 100.0
        res_modes.append((r, val, err))

    return res_modes


# ==============================================================================
# TESTE 4.3.3: GUIA BIMATERIAL ANISOTRÓPICO COM CONDUTORES CIRCULARES PEC
# ==============================================================================

def is_inside_circ_guide(x, y, r_hole=1.0/3.0, tol=1e-12):
    if x < -2.0 - tol or x > 2.0 + tol or y < -1.0 - tol or y > 1.0 + tol:
        return False
    # Buraco 1 centrado em (-1, 0)
    if (x - (-1.0))**2 + y**2 < r_hole**2 - tol:
        return False
    # Buraco 2 centrado em (1, 0)
    if (x - 1.0)**2 + y**2 < r_hole**2 - tol:
        return False
    return True


def run_test_4_3_3():
    print("\n" + "=" * 80)
    print("  EXECUTANDO TESTE 4.3.3: GUIA BIMATERIAL ANISOTRÓPICO COM FUROS PEC")
    print("=" * 80)

    # Subdomínio 1: [-2, 0] x [-1, 1] com furo em (-1, 0), r = 1/3
    # Subdomínio 2: [ 0, 2] x [-1, 1] com furo em ( 1, 0), r = 1/3
    r_hole = 1.0 / 3.0
    Nx1, Nx2, Ny = 17, 17, 17

    x1_grid = np.linspace(-2.0, 0.0, Nx1)
    x2_grid = np.linspace(0.0, 2.0, Nx2)
    x_all = np.unique(np.concatenate([x1_grid, x2_grid]))
    y_all = np.linspace(-1.0, 1.0, Ny)

    dx_nom = x_all[1] - x_all[0]
    dy_nom = y_all[1] - y_all[0]

    coords_list = []
    vectors_list = []
    is_pec_list = []
    mat_ids_list = []

    # 1. Nós regulares nos meios 1 e 2
    for i, x0 in enumerate(x_all):
        for j, y0 in enumerate(y_all):
            if not is_inside_circ_guide(x0, y0, r_hole=r_hole):
                continue

            is_int = (abs(x0) < 1e-6)
            is_pec = False
            t = np.array([0.0, 1.0])

            # Contornos externos
            if abs(x0 - (-2.0)) < 1e-6 or abs(x0 - 2.0) < 1e-6:
                is_pec = True; t = np.array([0.0, 1.0])
            elif abs(y0 - (-1.0)) < 1e-6 or abs(y0 - 1.0) < 1e-6:
                is_pec = True; t = np.array([1.0, 0.0])
            elif is_int:
                # Interface material interna x = 0
                is_pec = False
                t = np.array([0.0, 1.0])
            else:
                ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(ang), np.sin(ang)])

            # Material ID
            if is_int:
                m_id = 0
            elif x0 < 0.0:
                m_id = 1
            else:
                m_id = 2

            coords_list.append([x0, y0])
            vectors_list.append(t)
            is_pec_list.append(is_pec)
            mat_ids_list.append(m_id)

    # 2. Nós sobre os anéis circulares PEC (para discretizar bem a fronteira curva)
    N_circle = 20
    thetas = np.linspace(0, 2 * np.pi, N_circle, endpoint=False)
    for xc, m_id in [(-1.0, 1), (1.0, 2)]:
        for th in thetas:
            xb = xc + r_hole * np.cos(th)
            yb = r_hole * np.sin(th)
            t_circ = np.array([-np.sin(th), np.cos(th)]) # tangente ao círculo
            coords_list.append([xb, yb])
            vectors_list.append(t_circ)
            is_pec_list.append(True)
            mat_ids_list.append(m_id)

    coords = np.array(coords_list)
    vectors = np.array(vectors_list)
    is_pec = np.array(is_pec_list, dtype=bool)
    mat_ids = np.array(mat_ids_list, dtype=int)
    total_nodes = len(coords)

    # Subconjuntos por material
    subsets = {}
    for m in [1, 2]:
        cand = np.where((mat_ids == m) | (mat_ids == 0))[0]
        subsets[m] = {
            'indices': cand,
            'coords': coords[cand],
            'vectors': vectors[cand],
            'tree': KDTree(coords[cand])
        }

    # Plotar nuvem e malha cartesiana de integração
    fig, ax = plt.subplots(figsize=(8, 4.5))
    Ncx, Ncy = 32, 16
    x_edges = np.linspace(-2.0, 2.0, Ncx + 1)
    y_edges = np.linspace(-1.0, 1.0, Ncy + 1)

    for j in range(Ncy):
        for i in range(Ncx):
            xc = 0.5 * (x_edges[i] + x_edges[i+1])
            yc = 0.5 * (y_edges[j] + y_edges[j+1])
            if is_inside_circ_guide(xc, yc, r_hole=r_hole):
                color = "#E8F5E9" if xc < 0 else "#FFF3E0"
                rect = plt.Rectangle((x_edges[i], y_edges[j]), x_edges[i+1]-x_edges[i], y_edges[j+1]-y_edges[j],
                                     fill=True, facecolor=color, edgecolor="#CFD8DC", linewidth=0.6, alpha=0.6)
                ax.add_patch(rect)

    # Fronteira externa e círculos
    ax.plot([-2, 2, 2, -2, -2], [-1, -1, 1, 1, -1], 'k-', linewidth=2.0, label='Paredes PEC Externas')
    ax.axvline(x=0.0, color="#1B5E20", linestyle="--", linewidth=2.0, label="Interface Material (x=0)")

    c1 = plt.Circle((-1.0, 0.0), r_hole, color="#D32F2F", fill=True, alpha=0.3)
    c2 = plt.Circle(( 1.0, 0.0), r_hole, color="#D32F2F", fill=True, alpha=0.3)
    c1_line = plt.Circle((-1.0, 0.0), r_hole, color="#D32F2F", fill=False, linewidth=2.0, label="Cilindros PEC Internos")
    c2_line = plt.Circle(( 1.0, 0.0), r_hole, color="#D32F2F", fill=False, linewidth=2.0)
    ax.add_patch(c1); ax.add_patch(c2); ax.add_patch(c1_line); ax.add_patch(c2_line)

    # Quiver dos nós
    ax.quiver(coords[~is_pec, 0], coords[~is_pec, 1],
              vectors[~is_pec, 0], vectors[~is_pec, 1],
              color="#0D47A1", scale=30, width=0.003, alpha=0.8, label="Nós Ativos")
    ax.quiver(coords[is_pec, 0], coords[is_pec, 1],
              vectors[is_pec, 0], vectors[is_pec, 1],
              color="#B71C1C", scale=25, width=0.004, label="Nós PEC")

    ax.set_title("Teste 4.3.3: Guia Bimaterial Anisotrópico - Nuvem de Pontos e Células de Fundo", fontsize=11, fontweight='bold')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig_circ_path = DIR_EXP / "Verificacao_Nuvens_Malha_Guia_Bimaterial_Circ.png"
    plt.savefig(fig_circ_path, dpi=200)
    plt.close()
    print(f"Figura salva em: {fig_circ_path}")

    # Montagem variacional bimaterial
    # Meio 1: mu1 = 1, eps1 = [[1, 0], [0, 2]]
    # Meio 2: mu2 = 3, eps2 = [[3, -2j], [2j, 8]]
    mu_map = {1: 1.0, 2: 3.0}
    eps_map = {
        1: np.array([[1.0, 0.0], [0.0, 2.0]], dtype=complex),
        2: np.array([[3.0, -2j], [2j,  8.0]], dtype=complex)
    }

    xi_1d, w_1d = get_gauss_points_weights_1d(2)
    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_M, cols_M, data_M = [], [], []

    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]; dy_c = y1 - y0; yc = 0.5 * (y0 + y1)
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]; dx_c = x1 - x0; xc = 0.5 * (x0 + x1)
            if not is_inside_circ_guide(xc, yc, r_hole=r_hole): continue

            target_mat = 1 if xc < 0.0 else 2
            sub = subsets[target_mat]
            mu_r = mu_map[target_mat]
            eps_tensor = eps_map[target_mat]

            det_J = 0.25 * dx_c * dy_c

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx_c * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy_c * eta
                    if not is_inside_circ_guide(xg, yg, r_hole=r_hole): continue
                    Pg = np.array([xg, yg]); wg = wi * wj * det_J

                    sel_local, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
                        P=Pg,
                        nodes_coords=sub['coords'],
                        nodes_vectors=sub['vectors'],
                        search_tree=sub['tree'],
                        K=14,
                        Tol_det=1e-5,
                        adaptive=True
                    )
                    sel_global = sub['indices'][sel_local]

                    beta = np.linalg.inv(matrix_A)
                    Phi_g = beta[0:2, :]; rot_Phi = beta[4, :] - beta[3, :]

                    rot_outer = (1.0 / mu_r) * np.outer(rot_Phi, rot_Phi) * wg
                    vec_dot = (Phi_g.T @ eps_tensor @ Phi_g) * wg

                    for a in range(6):
                        Ia = sel_global[a]
                        for b in range(6):
                            Ib = sel_global[b]
                            rows_Kc.append(Ia); cols_Kc.append(Ib); data_Kc.append(rot_outer[a, b])
                            rows_M.append(Ia); cols_M.append(Ib); data_M.append(vec_dot[a, b])

    Kc = csr_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(total_nodes, total_nodes), dtype=complex)
    M = csr_matrix((data_M, (rows_M, cols_M)), shape=(total_nodes, total_nodes), dtype=complex)

    int_idx = np.where(~is_pec)[0]
    Kc_red = 0.5 * (Kc[int_idx, :][:, int_idx].toarray() + Kc[int_idx, :][:, int_idx].toarray().conj().T)
    M_red = 0.5 * (M[int_idx, :][:, int_idx].toarray() + M[int_idx, :][:, int_idx].toarray().conj().T)

    vals, vecs = la.eigh(Kc_red, M_red)
    ref_lambdas = [0.051876, 0.205316, 0.247201, 0.423569, 0.625603]
    res_modes = []
    for r in ref_lambdas:
        diffs = np.abs(vals - r)
        best_idx = np.argmin(diffs)
        val = vals[best_idx]
        err = abs(val - r) / r * 100.0
        res_modes.append((r, val, err))

    return res_modes


def main():
    res_432 = run_test_4_3_2()
    res_433 = run_test_4_3_3()

    print("\n" + "=" * 80)
    print("  RESULTADOS CONSOLIDADOS (VNMM 2D vs FEM DE REFERÊNCIA DA TESE)")
    print("=" * 80)
    print("1. Guia com Saliência ('H') Homogêneo e Anisotrópico (Teste 4.3.2):")
    print("-" * 65)
    print(f"{'Modo':<6} | {'Ref. FEM':<12} | {'VNMM 2D':<12} | {'Erro Relativo (%)':<18}")
    print("-" * 65)
    for i, (r, v, e) in enumerate(res_432):
        print(f"  {i+1:<4} | {r:<12.6f} | {v:<12.6f} | {e:<18.2f}")

    print("\n2. Guia Bimaterial Anisotrópico com Cilindros PEC (Teste 4.3.3):")
    print("-" * 65)
    print(f"{'Modo':<6} | {'Ref. FEM':<12} | {'VNMM 2D':<12} | {'Erro Relativo (%)':<18}")
    print("-" * 65)
    for i, (r, v, e) in enumerate(res_433):
        print(f"  {i+1:<4} | {r:<12.6f} | {v:<12.6f} | {e:<18.2f}")
    print("-" * 65)


if __name__ == "__main__":
    main()
