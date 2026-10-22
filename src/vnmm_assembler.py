import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix

# Add root directory and codigo to sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
CODIGO_DIR = os.path.join(ROOT_DIR, "codigo")
for p in [ROOT_DIR, CODIGO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from codigo.support_nodes_vnmm_2d_3_l1 import support_nodes_vnmm_2d_3_l1
from codigo.shape_functions_vnmm_2d_3_l1 import shape_functions_vnmm_2d_3_l1
from codigo.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from codigo.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1
from src.gauss_quadrature import get_gauss_points_weights_1d


def assemble_vnmm_matrices_2d(
    coords, 
    vectors, 
    gauss_points=None, 
    gauss_weights=None, 
    basis="P1", 
    det_tolerance=None, 
    mu_r=1.0, 
    epsilon_r=1.0,
    s_div=4.0,
    floor_tolerance=None,
    Ncx=None,
    Ncy=None,
    Lx=np.pi,
    Ly=np.pi,
    points_per_dir=3,
    support_mode="gauss_point",
    return_support_details=False,
    # Backward compatibility arguments
    base=None,
    tolerancia_det=None,
    tol_piso=None,
    pontos_por_dir=None,
    modo_suporte=None,
    retornar_detalhes_suporte=None
):
    """
    Assembles global CSR stiffness matrix K (with div-curl penalty regularization)
    and mass matrix M for 2D electromagnetic vector eigenvalue problems (TEz):
    
        K = K_curl + s_div * K_div
        K_curl_ij = integral_Omega (1 / mu_r) * (curl N_i)_z * (curl N_j)_z dOmega
        K_div_ij  = integral_Omega (1 / mu_r) * (div N_i) * (div N_j) dOmega
        M_ij      = integral_Omega epsilon_r * (N_i . N_j) dOmega
        
    Parameters:
    - coords: Numpy array (N, 2) of nodal coordinates.
    - vectors: Numpy array (N, 2) of unit director components.
    - basis: 'P1' (6 nodes/terms, complete linear) or 'L1' (3 nodes/terms, Nedelec 1st order).
    - det_tolerance: Determinant tolerance |det(A)| for adaptive node selection (default: adaptive O(h^4)).
    - mu_r: Relative magnetic permeability (default: 1.0).
    - epsilon_r: Relative electric permittivity (default: 1.0).
    - s_div: Divergence penalty parameter to shift gradient spurious modes (default: 4.0 or 6.0).
    - Ncx: Number of background integration cells along x.
    - Ncy: Number of background integration cells along y.
    - Lx: Domain length in x (default: pi).
    - Ly: Domain length in y (default: pi).
    - points_per_dir: Gauss points per direction per cell (default: 3 for 3x3 = 9 points/cell).
    - support_mode: 'gauss_point' (EFG-style, per integration point) or 'cell_center'.
    - return_support_details: Whether to return diagnostic support statistics.
    
    Returns:
    - K: Global regularized stiffness matrix (csr_matrix, shape N x N).
    - M: Global mass matrix (csr_matrix, shape N x N).
    - support_stats: (Optional) Dictionary of support geometry metrics.
    """
    if base is not None:
        basis = base
    if tolerancia_det is not None:
        det_tolerance = tolerancia_det
    if tol_piso is not None:
        floor_tolerance = tol_piso
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    if modo_suporte is not None:
        support_mode = modo_suporte
    if retornar_detalhes_suporte is not None:
        return_support_details = retornar_detalhes_suporte

    total_nodes = len(coords)
    tree = KDTree(coords)
    
    inv_mu = 1.0 / mu_r
    eps = epsilon_r
    
    basis_type = basis.upper()
    if basis_type in ["P1", "6_P1", "6NOS"]:
        is_P1 = True
    elif basis_type in ["L1", "3_L1", "3NOS"]:
        is_P1 = False
    else:
        raise ValueError(f"Basis '{basis}' not supported. Options: 'P1' or 'L1'.")
        
    side_nodes = int(np.round(np.sqrt(total_nodes)))
    if Ncx is None:
        Ncx = max(4, int(np.round(0.6 * side_nodes)))
    if Ncy is None:
        Ncy = max(4, int(np.round(0.6 * side_nodes)))
        
    h_char = max(Lx / max(side_nodes - 1, 1), Ly / max(side_nodes - 1, 1))
    h_ref = np.pi / 20.0
    if det_tolerance is None:
        tol_base = 1e-4 * (h_char / h_ref)**4 if is_P1 else 1e-4 * (h_char / h_ref)
        det_tolerance = max(tol_base, floor_tolerance) if floor_tolerance is not None else tol_base
        
    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)
    
    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_M, cols_M, data_M = [], [], []
    
    effective_ks = []
    determinants = []
    
    is_cell_center = support_mode in ["cell_center", "centro_celula"]
    
    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        yc = 0.5 * (y0 + y1)
        
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_J = 0.25 * dx * dy
            
            if is_cell_center:
                Pc = np.array([xc, yc])
                if is_P1:
                    supp_nodes, det_val, A_mat, eff_k = support_nodes_vnmm_2d_6_p1(
                        P=Pc,
                        nodes_coords=coords,
                        nodes_vectors=vectors,
                        search_tree=tree,
                        K=12,
                        Tol_det=det_tolerance,
                        adaptive=True,
                        step_K=4
                    )
                    beta = np.linalg.inv(A_mat)
                    rot_Phi = beta[4, :] - beta[3, :]
                    div_Phi = beta[2, :] + beta[5, :]
                    n_supp = 6
                else:
                    supp_nodes, det_val, A_mat, eff_k = support_nodes_vnmm_2d_3_l1(
                        P=Pc,
                        nodes_coords=coords,
                        nodes_vectors=vectors,
                        search_tree=tree,
                        K=8,
                        Tol_det=det_tolerance,
                        adaptive=True,
                        step_K=2
                    )
                    beta = np.linalg.inv(A_mat)
                    rot_Phi = -2.0 * beta[2, :]
                    div_Phi = np.zeros(3, dtype=float)
                    n_supp = 3
                
                for wi, xi in zip(w_1d, xi_1d):
                    xg = xc + 0.5 * dx * xi
                    dx_g = xg - xc
                    for wj, eta in zip(w_1d, xi_1d):
                        yg = yc + 0.5 * dy * eta
                        dy_g = yg - yc
                        
                        wg = wi * wj * det_J
                        
                        if is_P1:
                            Phi_x = beta[0, :] + beta[2, :] * dx_g + beta[3, :] * dy_g
                            Phi_y = beta[1, :] + beta[4, :] * dx_g + beta[5, :] * dy_g
                        else:
                            Phi_x = beta[0, :] + beta[2, :] * dy_g
                            Phi_y = beta[1, :] - beta[2, :] * dx_g
                            
                        Phi_g = np.vstack([Phi_x, Phi_y])
                        
                        rot_outer = np.outer(rot_Phi, rot_Phi) * (inv_mu * wg)
                        div_outer = np.outer(div_Phi, div_Phi) * (inv_mu * wg)
                        vec_dot = (Phi_g.T @ Phi_g) * (eps * wg)
                        
                        for a in range(n_supp):
                            Ia = supp_nodes[a]
                            for b in range(n_supp):
                                Ib = supp_nodes[b]
                                
                                val_kc = rot_outer[a, b]
                                val_kd = div_outer[a, b]
                                val_m = vec_dot[a, b]
                                
                                if abs(val_kc) > 1e-18:
                                    rows_Kc.append(Ia)
                                    cols_Kc.append(Ib)
                                    data_Kc.append(val_kc)
                                    
                                if abs(val_kd) > 1e-18:
                                    rows_Kd.append(Ia)
                                    cols_Kd.append(Ib)
                                    data_Kd.append(val_kd)
                                    
                                if abs(val_m) > 1e-18:
                                    rows_M.append(Ia)
                                    cols_M.append(Ib)
                                    data_M.append(val_m)
            else:
                # support_mode == 'gauss_point' (EFG-style)
                for wi, xi in zip(w_1d, xi_1d):
                    xg = xc + 0.5 * dx * xi
                    for wj, eta in zip(w_1d, xi_1d):
                        yg = yc + 0.5 * dy * eta
                        Pg = np.array([xg, yg])
                        wg = wi * wj * det_J
                        
                        if is_P1:
                            supp_nodes, det_val, A_mat, eff_k = support_nodes_vnmm_2d_6_p1(
                                P=Pg,
                                nodes_coords=coords,
                                nodes_vectors=vectors,
                                search_tree=tree,
                                K=12,
                                Tol_det=det_tolerance,
                                adaptive=True,
                                step_K=4
                            )
                            beta = np.linalg.inv(A_mat)
                            Phi_g = beta[0:2, :]
                            rot_Phi = beta[4, :] - beta[3, :]
                            div_Phi = beta[2, :] + beta[5, :]
                            n_supp = 6
                        else:
                            supp_nodes, det_val, A_mat, eff_k = support_nodes_vnmm_2d_3_l1(
                                P=Pg,
                                nodes_coords=coords,
                                nodes_vectors=vectors,
                                search_tree=tree,
                                K=8,
                                Tol_det=det_tolerance,
                                adaptive=True,
                                step_K=2
                            )
                            beta = np.linalg.inv(A_mat)
                            Phi_g = beta[0:2, :]
                            rot_Phi = -2.0 * beta[2, :]
                            div_Phi = np.zeros(3, dtype=float)
                            n_supp = 3
                            
                        effective_ks.append(eff_k)
                        determinants.append(det_val)
                        
                        rot_outer = np.outer(rot_Phi, rot_Phi) * (inv_mu * wg)
                        div_outer = np.outer(div_Phi, div_Phi) * (inv_mu * wg)
                        vec_dot = (Phi_g.T @ Phi_g) * (eps * wg)
                        
                        for a in range(n_supp):
                            Ia = supp_nodes[a]
                            for b in range(n_supp):
                                Ib = supp_nodes[b]
                                
                                val_kc = rot_outer[a, b]
                                val_kd = div_outer[a, b]
                                val_m = vec_dot[a, b]
                                
                                if abs(val_kc) > 1e-18:
                                    rows_Kc.append(Ia)
                                    cols_Kc.append(Ib)
                                    data_Kc.append(val_kc)
                                    
                                if abs(val_kd) > 1e-18:
                                    rows_Kd.append(Ia)
                                    cols_Kd.append(Ib)
                                    data_Kd.append(val_kd)
                                    
                                if abs(val_m) > 1e-18:
                                    rows_M.append(Ia)
                                    cols_M.append(Ib)
                                    data_M.append(val_m)
                                    
    # Assembly and symmetrization
    Kc = coo_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(total_nodes, total_nodes)).tocsr()
    Kd = coo_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(total_nodes, total_nodes)).tocsr()
    M = coo_matrix((data_M, (rows_M, cols_M)), shape=(total_nodes, total_nodes)).tocsr()
    
    Kc = 0.5 * (Kc + Kc.T)
    Kd = 0.5 * (Kd + Kd.T)
    M = 0.5 * (M + M.T)
    
    K = Kc + s_div * Kd
    
    if return_support_details:
        ks_arr = np.array(effective_ks) if effective_ks else np.array([0])
        dets_arr = np.array(determinants) if determinants else np.array([0.0])
        support_stats = {
            'mean_k': float(np.mean(ks_arr)),
            'max_k': int(np.max(ks_arr)),
            'min_k': int(np.min(ks_arr)),
            'mean_det': float(np.mean(dets_arr)),
            'min_det': float(np.min(dets_arr)),
            'max_det': float(np.max(dets_arr)),
            'num_integration_points': len(effective_ks),
            'n_supp': 6 if is_P1 else 3,
            # Portuguese backward compatibility keys
            'k_medio': float(np.mean(ks_arr)),
            'k_max': int(np.max(ks_arr)),
            'k_min': int(np.min(ks_arr)),
            'det_medio': float(np.mean(dets_arr)),
            'det_min': float(np.min(dets_arr)),
            'det_max': float(np.max(dets_arr)),
            'num_pontos_integracao': len(effective_ks)
        }
        return K, M, support_stats
        
    return K, M


# Backwards compatibility alias
montar_matrizes_vnmm_2d = assemble_vnmm_matrices_2d
