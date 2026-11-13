"""
Teste 4.3.2 da Tese: Guia de Onda com Saliência ("H") e Meio Homogêneo Anisotrópico
==================================================================================
Geometria confirmada pelo autor:
- x in [-2.5, 2.5], y in [-1.5, 1.5]
- Saliências reentrantes:
  Corte superior: de x in [-1.0, 1.0], descendo de y = 1.5 até y = 0.5
  Corte inferior: de x in [-1.0, 1.0], subindo de y = -1.5 até y = -0.5
  Região central conectora (pescoço): x in [-1.0, 1.0], y in [-0.5, 0.5]
  Braços laterais esquerdos: x in [-2.5, -1.0], y in [-1.5, 1.5]
  Braços laterais direitos: x in [1.0, 2.5], y in [-1.5, 1.5]

Propriedades físicas:
mu_r = 1.0
eps_r = [[1.0, -1j],
         [1j,   4.0]]

Valores de referência (Tese, Tabela 4-6):
Modo 1: 0.041212
Modo 2: 0.400641
Modo 3: 0.786745
Modo 4: 0.998524
Modo 5: 1.638579
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


def is_inside_H(x, y, tol=1e-12):
    """Verifica se (x, y) está dentro do domínio em forma de 'H'."""
    # Caixa envolvente
    if x < -2.5 - tol or x > 2.5 + tol or y < -1.5 - tol or y > 1.5 + tol:
        return False
    # Ranhura superior: [-1, 1] x [0.5, 1.5]
    if -1.0 + tol < x < 1.0 - tol and y > 0.5 + tol:
        return False
    # Ranhura inferior: [-1, 1] x [-1.5, -0.5]
    if -1.0 + tol < x < 1.0 - tol and y < -0.5 - tol:
        return False
    return True


def generate_H_waveguide_nodes(nx_side=11, nx_mid=13, ny_side=21, ny_mid=9, jitter_frac=0.0, seed=42):
    """
    Gera nós com vetores diretores no guia de onda com saliência ("H").
    As paredes de contorno recebem vetores tangenciais unitários (PEC).
    Os nós internos recebem diretores alternados +/- 45 graus.
    """
    if seed is not None:
        np.random.seed(seed)

    coords_list = []
    vectors_list = []
    is_pec_list = []

    # Geometria dividida em 3 blocos retangulares:
    # Bloco 1 (esquerda): [-2.5, -1.0] x [-1.5, 1.5]
    # Bloco 2 (centro):   [-1.0,  1.0] x [-0.5, 0.5]
    # Bloco 3 (direita):  [ 1.0,  2.5] x [-1.5, 1.5]

    # Grid global regular refinado com filtragem
    x_left = np.linspace(-2.5, -1.0, nx_side)
    x_mid = np.linspace(-1.0, 1.0, nx_mid)
    x_right = np.linspace(1.0, 2.5, nx_side)

    # Coordenadas em x únicas
    x_all = np.unique(np.concatenate([x_left, x_mid, x_right]))
    y_all = np.linspace(-1.5, 1.5, ny_side)

    dx_nom = (x_all[1] - x_all[0])
    dy_nom = (y_all[1] - y_all[0])

    for i, x0 in enumerate(x_all):
        for j, y0 in enumerate(y_all):
            if not is_inside_H(x0, y0):
                continue

            # Determinar se é fronteira PEC
            # Fronteiras externas e reentrâncias:
            is_pec = False
            t_pec = np.array([0.0, 0.0])

            # Parede externa esquerda x = -2.5
            if abs(x0 - (-2.5)) < 1e-6:
                is_pec = True
                t_pec = np.array([0.0, 1.0])
            # Parede externa direita x = 2.5
            elif abs(x0 - 2.5) < 1e-6:
                is_pec = True
                t_pec = np.array([0.0, 1.0])
            # Paredes de topo dos braços: y = 1.5 em x in [-2.5, -1.0] ou [1.0, 2.5]
            elif abs(y0 - 1.5) < 1e-6 and (x0 <= -1.0 + 1e-6 or x0 >= 1.0 - 1e-6):
                is_pec = True
                t_pec = np.array([1.0, 0.0])
            # Paredes de base dos braços: y = -1.5 em x in [-2.5, -1.0] ou [1.0, 2.5]
            elif abs(y0 - (-1.5)) < 1e-6 and (x0 <= -1.0 + 1e-6 or x0 >= 1.0 - 1e-6):
                is_pec = True
                t_pec = np.array([1.0, 0.0])
            # Paredes verticais das saliências: x = -1.0 ou x = 1.0 para |y| >= 0.5
            elif (abs(x0 - (-1.0)) < 1e-6 or abs(x0 - 1.0) < 1e-6) and abs(y0) >= 0.5 - 1e-6:
                is_pec = True
                t_pec = np.array([0.0, 1.0])
            # Paredes horizontais do pescoço: y = 0.5 ou y = -0.5 para x in [-1.0, 1.0]
            elif (abs(y0 - 0.5) < 1e-6 or abs(y0 - (-0.5)) < 1e-6) and abs(x0) <= 1.0 + 1e-6:
                is_pec = True
                t_pec = np.array([1.0, 0.0])

            # Aplicar jitter somente em nós internos
            x, y = x0, y0
            if not is_pec and jitter_frac > 0:
                jx = (np.random.rand() - 0.5) * 2.0 * jitter_frac * dx_nom
                jy = (np.random.rand() - 0.5) * 2.0 * jitter_frac * dy_nom
                if is_inside_H(x0 + jx, y0 + jy, tol=0.1 * min(dx_nom, dy_nom)):
                    x += jx
                    y += jy

            if is_pec:
                t = t_pec / np.linalg.norm(t_pec)
            else:
                ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(ang), np.sin(ang)])

            coords_list.append([x, y])
            vectors_list.append(t)
            is_pec_list.append(is_pec)

    return np.array(coords_list), np.array(vectors_list), np.array(is_pec_list, dtype=bool)


def solve_H_anisotropic_eigenvalues(
    coords, vectors, is_pec,
    Ncx=32, Ncy=24, points_per_dir=2, s_div=4.0, Tol_det=1e-5
):
    """
    Muda para problema espectral generalizado com tensor hermitiano complexo.
    """
    total_nodes = len(coords)
    tree = KDTree(coords)

    # Tensor de permissividade da Tese (Eq. 4-33)
    eps_tensor = np.array([[1.0, -1j],
                           [1j,   4.0]], dtype=complex)
    mu_r = 1.0

    # Malha de fundo cartesiana sobre [-2.5, 2.5] x [-1.5, 1.5]
    x_edges = np.linspace(-2.5, 2.5, Ncx + 1)
    y_edges = np.linspace(-1.5, 1.5, Ncy + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)

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

            # Verificar se o centro da célula está no domínio
            if not is_inside_H(xc, yc):
                continue

            det_J = 0.25 * dx_c * dy_c

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx_c * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy_c * eta
                    if not is_inside_H(xg, yg):
                        continue

                    Pg = np.array([xg, yg])
                    wg = wi * wj * det_J

                    sel_local, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
                        P=Pg,
                        nodes_coords=coords,
                        nodes_vectors=vectors,
                        search_tree=tree,
                        K=14,
                        Tol_det=Tol_det,
                        adaptive=True
                    )

                    beta = np.linalg.inv(matrix_A)
                    Phi_g = beta[0:2, :]  # 2 x 6 (real)
                    rot_Phi = beta[4, :] - beta[3, :]  # 6,
                    div_Phi = beta[2, :] + beta[5, :]  # 6,

                    rot_outer = (1.0 / mu_r) * np.outer(rot_Phi, rot_Phi) * wg
                    div_outer = (1.0 / mu_r) * np.outer(div_Phi, div_Phi) * wg
                    
                    # Tensor hermitiano complexo
                    # M_ab = (Phi_g[:, a])^T @ eps_tensor @ (Phi_g[:, b]) * wg
                    vec_dot = (Phi_g.T @ eps_tensor @ Phi_g) * wg

                    for a in range(6):
                        Ia = sel_local[a]
                        for b in range(6):
                            Ib = sel_local[b]
                            rows_Kc.append(Ia); cols_Kc.append(Ib); data_Kc.append(rot_outer[a, b])
                            rows_Kd.append(Ia); cols_Kd.append(Ib); data_Kd.append(div_outer[a, b])
                            rows_M.append(Ia); cols_M.append(Ib); data_M.append(vec_dot[a, b])

    Kc = csr_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(total_nodes, total_nodes), dtype=complex)
    Kd = csr_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(total_nodes, total_nodes), dtype=complex)
    M = csr_matrix((data_M, (rows_M, cols_M)), shape=(total_nodes, total_nodes), dtype=complex)

    # Imposição da condição essencial de PEC eliminando os nós de contorno
    int_idx = np.where(~is_pec)[0]

    Kc_red = 0.5 * (Kc[int_idx, :][:, int_idx].toarray() + Kc[int_idx, :][:, int_idx].toarray().conj().T)
    Kd_red = 0.5 * (Kd[int_idx, :][:, int_idx].toarray() + Kd[int_idx, :][:, int_idx].toarray().conj().T)
    M_red = 0.5 * (M[int_idx, :][:, int_idx].toarray() + M[int_idx, :][:, int_idx].toarray().conj().T)

    K_red = Kc_red + s_div * Kd_red

    # Resolução do problema generalizado de autovalores hermitiano
    vals, vecs = la.eigh(K_red, M_red)

    # Filtrar modos físicos
    pos_mask = vals > 0.005
    vals_pos = vals[pos_mask]
    vecs_pos = vecs[:, pos_mask]

    phys_modes = []
    for idx_m in range(len(vals_pos)):
        v = vecs_pos[:, idx_m]
        ec = np.real(v.conj().T @ Kc_red @ v)
        ed = np.real(v.conj().T @ Kd_red @ v)
        em = np.real(v.conj().T @ M_red @ v)

        rd = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
        lam = ec / em if em > 0 else 0.0
        if rd < 5.0 and lam > 0:
            phys_modes.append({
                'lambda': lam,
                'r_div': rd,
                'raw_val': vals_pos[idx_m]
            })

    phys_modes.sort(key=lambda x: x['lambda'])
    return coords, vectors, is_pec, phys_modes


def main():
    print("=" * 80)
    print("  TESTE 4.3.2 DA TESE: GUIA COM SALIÊNCIA ('H') HOMOGÊNEO ANISOTRÓPICO")
    print("=" * 80)

    # Geração dos nós
    coords, vectors, is_pec = generate_H_waveguide_nodes(
        nx_side=11, nx_mid=13, ny_side=21, ny_mid=9, jitter_frac=0.0
    )
    print(f"Total de nós gerados: {len(coords)} (Nós PEC: {np.sum(is_pec)}, Nós Ativos: {np.sum(~is_pec)})")

    # Resolução
    coords, vectors, is_pec, phys_modes = solve_H_anisotropic_eigenvalues(
        coords, vectors, is_pec,
        Ncx=36, Ncy=24, points_per_dir=2, s_div=4.0
    )

    ref_lambdas = [0.041212, 0.400641, 0.786745, 0.998524, 1.638579]

    print("\nResultados dos Primeiros 5 Autovalores (VNMM vs Referência FEM):")
    print("-" * 80)
    print(f"{'Modo':<6} | {'Ref. FEM':<12} | {'VNMM 2D':<12} | {'Erro Relativo (%)':<18} | {'R_div (%)':<10}")
    print("-" * 80)

    for i in range(min(5, len(phys_modes))):
        lam_vnmm = phys_modes[i]['lambda']
        lam_ref = ref_lambdas[i]
        err = abs(lam_vnmm - lam_ref) / lam_ref * 100.0
        r_div = phys_modes[i]['r_div']
        print(f"  {i+1:<4} | {lam_ref:<12.6f} | {lam_vnmm:<12.6f} | {err:<18.2f} | {r_div:<10.2f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
