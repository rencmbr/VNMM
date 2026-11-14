"""
VNMM 3D Global Matrix Assembler.
Assembles regularized sparse stiffness matrix K and mass matrix M for 3D vector electromagnetic problems
supporting:
- Complete linear vector polynomial basis R^1 = (P1)^3 (12 support nodes)
- Nédélec 1st-kind vector polynomial basis R^1 = (P0)^3 + S^1 (6 support nodes)
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(DIR_3D)
SCRIPTS_DIR = os.path.join(DIR_3D, "scripts")
for p in [DIR_3D, ROOT_DIR, SCRIPTS_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_3d_12_p1 import support_nodes_vnmm_3d_12_p1
from scripts.support_nodes_vnmm_3d_6_l1 import support_nodes_vnmm_3d_6_l1
from src.gauss_quadrature_3d import get_gauss_points_weights_1d


def assemble_vnmm_matrices_3d(
    coords,
    vectors,
    basis="P1",
    det_tolerance=None,
    mu_r=1.0,
    epsilon_r=1.0,
    s_div=6.0,
    floor_tolerance=None,
    Ncx=None,
    Ncy=None,
    Ncz=None,
    Lx=1.00,
    Ly=0.50,
    Lz=0.75,
    points_per_dir=2,
    support_mode="gauss_point",
    return_support_details=False,
    sri_div=True,
    return_components=False,
    # Compatibility aliases
    base=None,
    tolerancia_det=None,
    tol_piso=None,
    pontos_por_dir=None,
    modo_suporte=None
):
    """
    Assembles global CSR stiffness matrix K (curl-curl + s_div * div-div) and mass matrix M for 3D:
        K = K_curl + s_div * K_div
        K_curl_ij = integral_Omega (1 / mu_r) * (curl N_i) . (curl N_j) dOmega
        K_div_ij  = integral_Omega (1 / mu_r) * (div N_i) * (div N_j) dOmega
        M_ij      = integral_Omega epsilon_r * (N_i . N_j) dOmega
        
    Parameters:
    - basis: 'P1' (12 support nodes, complete linear) or 'L1' (6 support nodes, Nédélec 1st order).
    """
    if base is not None: basis = base
    if tolerancia_det is not None: det_tolerance = tolerancia_det
    if tol_piso is not None: floor_tolerance = tol_piso
    if pontos_por_dir is not None: points_per_dir = pontos_por_dir
    if modo_suporte is not None: support_mode = modo_suporte

    total_nodes = len(coords)

    inv_mu = 1.0 / mu_r
    eps = epsilon_r

    is_P1 = basis.upper() in ["P1", "12_P1", "12NOS"]
    n_supp = 12 if is_P1 else 6

    # Estimate number of distinct grid divisions per axis
    unique_x = len(np.unique(np.round(coords[:, 0], 7)))
    side_nodes = unique_x if unique_x >= 3 else int(np.round(total_nodes ** (1.0 / 3.0)))
    if Ncx is None: Ncx = max(3, side_nodes - 1)
    if Ncy is None: Ncy = max(3, side_nodes - 1)
    if Ncz is None: Ncz = max(3, side_nodes - 1)

    scales = np.array([Lx / max(Ncx, 1), Ly / max(Ncy, 1), Lz / max(Ncz, 1)], dtype=float)
    tree = KDTree(coords / scales) if is_P1 else KDTree(coords)

    h_char = max(Lx / max(side_nodes - 1, 1), Ly / max(side_nodes - 1, 1), Lz / max(side_nodes - 1, 1))
    h_ref = np.pi / 10.0
    if det_tolerance is None:
        tol_base = 1e-6 * (h_char / h_ref)**9 if is_P1 else 1e-6 * (h_char / h_ref)**3
        det_tolerance = max(tol_base, floor_tolerance) if floor_tolerance is not None else tol_base

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    z_edges = np.linspace(0.0, Lz, Ncz + 1)

    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)

    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_M, cols_M, data_M = [], [], []

    is_cell_center = support_mode in ["cell_center", "centro_celula"]

    for k in range(Ncz):
        z0, z1 = z_edges[k], z_edges[k + 1]
        dz = z1 - z0
        zc = 0.5 * (z0 + z1)

        for j in range(Ncy):
            y0, y1 = y_edges[j], y_edges[j + 1]
            dy = y1 - y0
            yc = 0.5 * (y0 + y1)

            for i in range(Ncx):
                x0, x1 = x_edges[i], x_edges[i + 1]
                dx = x1 - x0
                xc = 0.5 * (x0 + x1)

                det_J = 0.125 * dx * dy * dz
                vol_cell = dx * dy * dz

                if is_cell_center:
                    Pc = np.array([xc, yc, zc])
                    if is_P1:
                        supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_3d_12_p1(
                            P=Pc, nodes_coords=coords, nodes_vectors=vectors,
                            search_tree=tree, K=24, Tol_det=det_tolerance, adaptive=True, step_K=6, scales=scales
                        )
                        beta = np.linalg.inv(A_mat)
                        curl_Phi = np.empty((3, 12), dtype=float)
                        curl_Phi[0, :] = beta[8, :] - beta[10, :]
                        curl_Phi[1, :] = beta[9, :] - beta[5, :]
                        curl_Phi[2, :] = beta[4, :] - beta[6, :]
                        div_Phi = beta[3, :] + beta[7, :] + beta[11, :]
                    else:
                        supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_3d_6_l1(
                            P=Pc, nodes_coords=coords, nodes_vectors=vectors,
                            search_tree=tree, K=14, Tol_det=det_tolerance, adaptive=True, step_K=4
                        )
                        beta = np.linalg.inv(A_mat)
                        curl_Phi = 2.0 * beta[3:6, :].copy()
                        div_Phi = np.zeros(6, dtype=float)

                    r_idx = np.repeat(supp_nodes, n_supp)
                    c_idx = np.tile(supp_nodes, n_supp)

                    for wk, zeta in zip(w_1d, xi_1d):
                        zg = zc + 0.5 * dz * zeta
                        dz_g = zg - zc
                        for wj, eta in zip(w_1d, xi_1d):
                            yg = yc + 0.5 * dy * eta
                            dy_g = yg - yc
                            for wi, xi in zip(w_1d, xi_1d):
                                xg = xc + 0.5 * dx * xi
                                dx_g = xg - xc
                                wg = wi * wj * wk * det_J

                                if is_P1:
                                    Phi_x = beta[0, :] + beta[3, :] * dx_g + beta[6, :] * dy_g + beta[9, :] * dz_g
                                    Phi_y = beta[1, :] + beta[4, :] * dx_g + beta[7, :] * dy_g + beta[10, :] * dz_g
                                    Phi_z = beta[2, :] + beta[5, :] * dx_g + beta[8, :] * dy_g + beta[11, :] * dz_g
                                else:
                                    Phi_x = beta[0, :] + beta[4, :] * dz_g - beta[5, :] * dy_g
                                    Phi_y = beta[1, :] - beta[3, :] * dz_g + beta[5, :] * dx_g
                                    Phi_z = beta[2, :] + beta[3, :] * dy_g - beta[4, :] * dx_g

                                Phi_g = np.vstack([Phi_x, Phi_y, Phi_z])

                                rot_mat = (curl_Phi.T @ curl_Phi) * (inv_mu * wg)
                                div_mat = np.outer(div_Phi, div_Phi) * (inv_mu * wg)
                                mass_mat = (Phi_g.T @ Phi_g) * (eps * wg)

                                rows_Kc.append(r_idx); cols_Kc.append(c_idx); data_Kc.append(rot_mat.ravel())
                                rows_Kd.append(r_idx); cols_Kd.append(c_idx); data_Kd.append(div_mat.ravel())
                                rows_M.append(r_idx); cols_M.append(c_idx); data_M.append(mass_mat.ravel())
                else:
                    # support_mode == 'gauss_point'
                    cell_div_nodes = {}
                    for wk, zeta in zip(w_1d, xi_1d):
                        zg = zc + 0.5 * dz * zeta
                        for wj, eta in zip(w_1d, xi_1d):
                            yg = yc + 0.5 * dy * eta
                            for wi, xi in zip(w_1d, xi_1d):
                                xg = xc + 0.5 * dx * xi
                                Pg = np.array([xg, yg, zg])
                                wg = wi * wj * wk * det_J

                                if is_P1:
                                    supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_3d_12_p1(
                                        P=Pg, nodes_coords=coords, nodes_vectors=vectors,
                                        search_tree=tree, K=24, Tol_det=det_tolerance, adaptive=True, step_K=6, scales=scales
                                    )
                                    beta = np.linalg.inv(A_mat)
                                    Phi_g = beta[0:3, :]
                                    curl_Phi = np.empty((3, 12), dtype=float)
                                    curl_Phi[0, :] = beta[8, :] - beta[10, :]
                                    curl_Phi[1, :] = beta[9, :] - beta[5, :]
                                    curl_Phi[2, :] = beta[4, :] - beta[6, :]
                                    div_Phi = beta[3, :] + beta[7, :] + beta[11, :]
                                else:
                                    supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_3d_6_l1(
                                        P=Pg, nodes_coords=coords, nodes_vectors=vectors,
                                        search_tree=tree, K=14, Tol_det=det_tolerance, adaptive=True, step_K=4
                                    )
                                    beta = np.linalg.inv(A_mat)
                                    Phi_g = beta[0:3, :]
                                    curl_Phi = 2.0 * beta[3:6, :].copy()
                                    div_Phi = np.zeros(6, dtype=float)

                                r_idx = np.repeat(supp_nodes, n_supp)
                                c_idx = np.tile(supp_nodes, n_supp)

                                rot_mat = (curl_Phi.T @ curl_Phi) * (inv_mu * wg)
                                mass_mat = (Phi_g.T @ Phi_g) * (eps * wg)

                                rows_Kc.append(r_idx); cols_Kc.append(c_idx); data_Kc.append(rot_mat.ravel())
                                rows_M.append(r_idx); cols_M.append(c_idx); data_M.append(mass_mat.ravel())

                                if sri_div and is_P1:
                                    for a in range(n_supp):
                                        Ia = supp_nodes[a]
                                        cell_div_nodes[Ia] = cell_div_nodes.get(Ia, 0.0) + div_Phi[a] * (wg / vol_cell)
                                else:
                                    div_mat = np.outer(div_Phi, div_Phi) * (inv_mu * wg)
                                    rows_Kd.append(r_idx); cols_Kd.append(c_idx); data_Kd.append(div_mat.ravel())

                    if sri_div and is_P1 and cell_div_nodes:
                        c_nodes = np.array(list(cell_div_nodes.keys()), dtype=int)
                        c_vals = np.array(list(cell_div_nodes.values()), dtype=float)
                        div_cell_mat = np.outer(c_vals, c_vals) * (inv_mu * vol_cell)
                        rows_Kd.append(np.repeat(c_nodes, len(c_nodes)))
                        cols_Kd.append(np.tile(c_nodes, len(c_nodes)))
                        data_Kd.append(div_cell_mat.ravel())

    rows_Kc_arr = np.concatenate(rows_Kc) if rows_Kc else np.array([], dtype=int)
    cols_Kc_arr = np.concatenate(cols_Kc) if cols_Kc else np.array([], dtype=int)
    data_Kc_arr = np.concatenate(data_Kc) if data_Kc else np.array([], dtype=float)

    rows_Kd_arr = np.concatenate(rows_Kd) if rows_Kd else np.array([], dtype=int)
    cols_Kd_arr = np.concatenate(cols_Kd) if cols_Kd else np.array([], dtype=int)
    data_Kd_arr = np.concatenate(data_Kd) if data_Kd else np.array([], dtype=float)

    rows_M_arr = np.concatenate(rows_M) if rows_M else np.array([], dtype=int)
    cols_M_arr = np.concatenate(cols_M) if cols_M else np.array([], dtype=int)
    data_M_arr = np.concatenate(data_M) if data_M else np.array([], dtype=float)

    K_curl = coo_matrix((data_Kc_arr, (rows_Kc_arr, cols_Kc_arr)), shape=(total_nodes, total_nodes)).tocsr()
    K_div = coo_matrix((data_Kd_arr, (rows_Kd_arr, cols_Kd_arr)), shape=(total_nodes, total_nodes)).tocsr()
    M = coo_matrix((data_M_arr, (rows_M_arr, cols_M_arr)), shape=(total_nodes, total_nodes)).tocsr()

    K_curl = 0.5 * (K_curl + K_curl.T)
    K_div = 0.5 * (K_div + K_div.T)
    M = 0.5 * (M + M.T)

    K = K_curl + s_div * K_div

    if return_components:
        return K, M, K_curl, K_div
    return K, M


# Backward compatibility aliases
montar_matrizes_vnmm_3d = assemble_vnmm_matrices_3d
assemble_vnmm_matrices = assemble_vnmm_matrices_3d
montar_matrizes_vnmm = assemble_vnmm_matrices_3d
