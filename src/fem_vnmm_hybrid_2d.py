import os
import sys
import time
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fem_edge_2d import assemble_edge_fem_matrices_2d
from src.gauss_quadrature import get_gauss_points_weights_1d
from codigo.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from src.eigen_solver_cavity import ANALYTICAL_REFERENCE_MODES, MODOS_ANALITICOS_REF


def generate_hybrid_cavity_mesh(
    Lx=np.pi, 
    Ly=np.pi, 
    frac_fem=0.5, 
    Nex_fem=8, 
    Ney=12, 
    Nx_vnmm=9, 
    Ny_vnmm=13,
    interior_type_vnmm="alternating",
    jitter_frac_fem=0.0,
    jitter_frac_vnmm=0.0,
    seed=42,
    # Backward compatibility
    tipo_interior_vnmm=None,
    lx=None,
    ly=None,
    nex_fem=None,
    ney=None,
    nx_vnmm=None,
    ny_vnmm=None
):
    if lx is not None:
        Lx = lx
    if ly is not None:
        Ly = ly
    if nex_fem is not None:
        Nex_fem = nex_fem
    if ney is not None:
        Ney = ney
    if nx_vnmm is not None:
        Nx_vnmm = nx_vnmm
    if ny_vnmm is not None:
        Ny_vnmm = ny_vnmm
    """
    Generates partitioned meshes for hybrid Edge FEM + VNMM coupling:
    - Subdomain 1 (FEM): [0, x_int] x [0, Ly] with triangular mesh (Nex_fem x Ney cells).
    - Subdomain 2 (VNMM): [x_int, Lx] x [0, Ly] with nodal cloud.
      At interface x = x_int, boundary nodes of VNMM are placed exactly at midpoints
      of the Ney vertical interface edges of the FEM mesh, with tangent t = [0, 1]^T.
    Supports controlled stochastic perturbation (jitter) for internal nodes of both subdomains.
    
    Returns:
    - fem_data: Dictionary with nodes, elements, edges, and interface mapping.
    - vnmm_data: Dictionary with nodal coordinates, director vectors, and boundary/interface flags.
    """
    if tipo_interior_vnmm is not None:
        interior_type_vnmm = tipo_interior_vnmm

    if seed is not None:
        np.random.seed(seed)
        
    x_int = frac_fem * Lx
    
    # =========================================================================
    # 1. FEM SUBDOMAIN [0, x_int] x [0, Ly]
    # =========================================================================
    dx_fem = x_int / Nex_fem
    dy_fem = Ly / Ney
    x_lin_fem = np.linspace(0.0, x_int, Nex_fem + 1)
    y_lin_fem = np.linspace(0.0, Ly, Ney + 1)
    
    nodes_fem = []
    node_grid_fem = np.zeros((Ney + 1, Nex_fem + 1), dtype=int)
    node_id = 0
    for j in range(Ney + 1):
        for i in range(Nex_fem + 1):
            x, y = x_lin_fem[i], y_lin_fem[j]
            # Stochastic perturbation only for strictly interior FEM nodes
            if jitter_frac_fem > 0.0 and 0 < i < Nex_fem and 0 < j < Ney:
                x += np.random.uniform(-jitter_frac_fem * dx_fem, jitter_frac_fem * dx_fem)
                y += np.random.uniform(-jitter_frac_fem * dy_fem, jitter_frac_fem * dy_fem)
            nodes_fem.append([x, y])
            node_grid_fem[j, i] = node_id
            node_id += 1
    nodes_fem = np.array(nodes_fem, dtype=float)
    
    elements_fem = []
    for j in range(Ney):
        for i in range(Nex_fem):
            n_bl = node_grid_fem[j, i]
            n_br = node_grid_fem[j, i + 1]
            n_tl = node_grid_fem[j + 1, i]
            n_tr = node_grid_fem[j + 1, i + 1]
            
            elements_fem.append([n_bl, n_br, n_tr])
            elements_fem.append([n_bl, n_tr, n_tl])
    elements_fem = np.array(elements_fem, dtype=int)
    
    # Identify FEM edges
    edge_dict_fem = {}
    elem_edges_fem = np.zeros((len(elements_fem), 3), dtype=int)
    elem_edge_signs_fem = np.zeros((len(elements_fem), 3), dtype=float)
    
    edge_id = 0
    for elem_idx, elem in enumerate(elements_fem):
        local_edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        for local_idx, (n1, n2) in enumerate(local_edges):
            ordered = (min(n1, n2), max(n1, n2))
            if ordered not in edge_dict_fem:
                edge_dict_fem[ordered] = edge_id
                global_eid = edge_id
                edge_id += 1
            else:
                global_eid = edge_dict_fem[ordered]
                
            elem_edges_fem[elem_idx, local_idx] = global_eid
            elem_edge_signs_fem[elem_idx, local_idx] = 1.0 if n1 < n2 else -1.0
            
    edges_fem = np.zeros((edge_id, 2), dtype=int)
    for (n1, n2), eid in edge_dict_fem.items():
        edges_fem[eid] = [n1, n2]
        
    # Classify FEM edges: external PEC vs Interface vs Interior
    tol = 1e-7
    is_pec_edge_fem = np.zeros(edge_id, dtype=bool)
    is_interface_edge_fem = np.zeros(edge_id, dtype=bool)
    interface_edge_ids = []
    interface_edge_midpoints = []
    
    for eid in range(edge_id):
        n1, n2 = edges_fem[eid]
        x1, y1 = nodes_fem[n1]
        x2, y2 = nodes_fem[n2]
        
        # External left (x=0), bottom (y=0), or top (y=Ly) PEC boundary
        if (abs(x1) < tol and abs(x2) < tol) or (abs(y1) < tol and abs(y2) < tol) or (abs(y1 - Ly) < tol and abs(y2 - Ly) < tol):
            is_pec_edge_fem[eid] = True
        # Vertical interface edge at x = x_int
        elif (abs(x1 - x_int) < tol and abs(x2 - x_int) < tol):
            is_interface_edge_fem[eid] = True
            interface_edge_ids.append(eid)
            pm = 0.5 * (np.array([x1, y1]) + np.array([x2, y2]))
            interface_edge_midpoints.append((pm[1], eid, pm))
            
    # Sort interface edges by ascending y
    interface_edge_midpoints.sort(key=lambda item: item[0])
    ordered_interface_eids = [item[1] for item in interface_edge_midpoints]
    ordered_interface_pms = np.array([item[2] for item in interface_edge_midpoints])
    
    fem_data = {
        'nodes': nodes_fem,
        'elements': elements_fem,
        'edges': edges_fem,
        'elem_edges': elem_edges_fem,
        'elem_edge_signs': elem_edge_signs_fem,
        'is_pec_edge': is_pec_edge_fem,
        'is_interface_edge': is_interface_edge_fem,
        'interface_eids': ordered_interface_eids,
        'interface_midpoints': ordered_interface_pms,
        'x_int': x_int,
        'Lx': Lx,
        'Ly': Ly
    }
    
    # =========================================================================
    # 2. VNMM SUBDOMAIN [x_int, Lx] x [0, Ly]
    # =========================================================================
    x_lin_vnmm = np.linspace(x_int, Lx, Nx_vnmm)
    y_lin_vnmm = np.linspace(0.0, Ly, Ny_vnmm)
    
    nodes_vnmm = []
    vectors_vnmm = []
    is_pec_vnmm = []
    is_interface_vnmm = []
    interface_node_map = {}
    
    # 2.1 Interface nodes (x = x_int) placed exactly at FEM edge midpoints
    # with unit vector t = [0, 1] aligned with the vertical edge
    for int_idx, pm in enumerate(ordered_interface_pms):
        vnmm_nid = len(nodes_vnmm)
        nodes_vnmm.append(pm)
        vectors_vnmm.append([0.0, 1.0])
        is_pec_vnmm.append(False)
        is_interface_vnmm.append(True)
        interface_node_map[int_idx] = vnmm_nid
        
    # 2.2 Corner nodes at interface: (x_int, 0) and (x_int, Ly) with horizontal vector [1, 0] (PEC)
    nodes_vnmm.append([x_int, 0.0])
    vectors_vnmm.append([1.0, 0.0])
    is_pec_vnmm.append(True)
    is_interface_vnmm.append(False)
    
    nodes_vnmm.append([x_int, Ly])
    vectors_vnmm.append([1.0, 0.0])
    is_pec_vnmm.append(True)
    is_interface_vnmm.append(False)
    
    # 2.3 Interior and external PEC nodes for VNMM subdomain
    dx_vnmm = (Lx - x_int) / max(Nx_vnmm - 1, 1)
    dy_vnmm = Ly / max(Ny_vnmm - 1, 1)
    
    mode = interior_type_vnmm.lower()
    
    for i in range(1, Nx_vnmm):
        x_base = x_lin_vnmm[i]
        for j in range(Ny_vnmm):
            y_base = y_lin_vnmm[j]
            
            is_right = abs(x_base - Lx) < tol
            is_bottom = abs(y_base) < tol
            is_top = abs(y_base - Ly) < tol
            is_pec = is_right or is_bottom or is_top
            
            if is_pec:
                pos = [x_base, y_base]
                if is_right:
                    vec = [0.0, 1.0]
                else:
                    vec = [1.0, 0.0]
            else:
                pos_x = x_base
                pos_y = y_base
                if jitter_frac_vnmm > 0.0:
                    pos_x += np.random.uniform(-jitter_frac_vnmm * dx_vnmm, jitter_frac_vnmm * dx_vnmm)
                    pos_y += np.random.uniform(-jitter_frac_vnmm * dy_vnmm, jitter_frac_vnmm * dy_vnmm)
                pos = [pos_x, pos_y]
                
                if mode in ["random", "aleatorio"]:
                    theta = np.random.uniform(0.0, 2.0 * np.pi)
                    vec = [np.cos(theta), np.sin(theta)]
                elif mode in ["alternating", "alternado"]:
                    ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                    vec = [np.cos(ang), np.sin(ang)]
                else:
                    vec = [1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]
                    
            nodes_vnmm.append(pos)
            vectors_vnmm.append(vec)
            is_pec_vnmm.append(is_pec)
            is_interface_vnmm.append(False)
            
    nodes_vnmm = np.array(nodes_vnmm, dtype=float)
    vectors_vnmm = np.array(vectors_vnmm, dtype=float)
    is_pec_vnmm = np.array(is_pec_vnmm, dtype=bool)
    is_interface_vnmm = np.array(is_interface_vnmm, dtype=bool)
    
    vnmm_data = {
        'coords': nodes_vnmm,
        'vectors': vectors_vnmm,
        'is_pec': is_pec_vnmm,
        'is_interface': is_interface_vnmm,
        'interface_node_map': interface_node_map,
        'x_int': x_int,
        'Lx': Lx,
        'Ly': Ly,
        'Nx': Nx_vnmm,
        'Ny': Ny_vnmm
    }
    
    return fem_data, vnmm_data


def assemble_hybrid_fem_vnmm_matrices(
    fem_data, 
    vnmm_data=None, 
    Ncx_vnmm=None, 
    Ncy_vnmm=None, 
    points_per_dir=3, 
    s_div_vnmm=4.0, 
    floor_tolerance_vnmm=None,
    det_tolerance_vnmm=None,
    mu_r=1.0, 
    epsilon_r=1.0,
    # Backward compatibility aliases
    pontos_por_dir=None,
    tol_piso_vnmm=None,
    tolerancia_det_vnmm=None,
    Ncx=None,
    Ncy=None,
    ncx=None,
    ncy=None,
    s_div=None
):
    if isinstance(fem_data, (tuple, list)) and len(fem_data) == 2 and (vnmm_data is None or not isinstance(vnmm_data, dict)):
        fem_data, vnmm_data = fem_data[0], fem_data[1]

    if Ncx is not None:
        Ncx_vnmm = Ncx
    elif ncx is not None:
        Ncx_vnmm = ncx
    if Ncy is not None:
        Ncy_vnmm = Ncy
    elif ncy is not None:
        Ncy_vnmm = ncy
    if s_div is not None:
        s_div_vnmm = s_div
    """
    Assembles coupled stiffness matrix K_hybrid and mass matrix M_hybrid via direct conformal coupling:
    - Master Degrees of Freedom:
      1. e_fem_int: internal edges of FEM subdomain.
      2. e_gamma: interface edges on Gamma_int (circulation [V]).
      3. c_vnmm_int: internal nodes of VNMM subdomain (field [V/m]).
    - Dimensional Interface Relation:
      c_gamma_k = (1 / delta_y_k) * e_gamma_k, with unit vector t aligned to the oriented vertical edge.
    """
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    if tol_piso_vnmm is not None:
        floor_tolerance_vnmm = tol_piso_vnmm
    if tolerancia_det_vnmm is not None:
        det_tolerance_vnmm = tolerancia_det_vnmm

    inv_mu = 1.0 / mu_r
    eps = epsilon_r
    
    # -------------------------------------------------------------------------
    # 1. FEM SUBDOMAIN ASSEMBLY
    # -------------------------------------------------------------------------
    K_fem_full, M_fem_full = assemble_edge_fem_matrices_2d(
        fem_data['nodes'],
        fem_data['elements'],
        fem_data['edges'],
        fem_data['elem_edges'],
        fem_data['elem_edge_signs'],
        mu_r=mu_r,
        epsilon_r=epsilon_r
    )
    
    idx_fem_int = np.where(~fem_data['is_pec_edge'] & ~fem_data['is_interface_edge'])[0]
    idx_fem_gamma = np.array(fem_data['interface_eids'], dtype=int)
    
    N_fem_int = len(idx_fem_int)
    N_gamma = len(idx_fem_gamma)
    
    # -------------------------------------------------------------------------
    # 2. VNMM SUBDOMAIN ASSEMBLY OVER [x_int, Lx] x [0, Ly]
    # -------------------------------------------------------------------------
    coords_vnmm = vnmm_data['coords']
    vectors_vnmm = vnmm_data['vectors']
    N_vnmm_total = len(coords_vnmm)
    
    tree_vnmm = KDTree(coords_vnmm)
    x_int = vnmm_data['x_int']
    Lx = vnmm_data['Lx']
    Ly = vnmm_data['Ly']
    
    if Ncx_vnmm is None:
        Ncx_vnmm = max(4, int(np.round(0.6 * vnmm_data['Nx'])))
    if Ncy_vnmm is None:
        Ncy_vnmm = max(4, int(np.round(0.6 * vnmm_data['Ny'])))
        
    x_edges_vnmm = np.linspace(x_int, Lx, Ncx_vnmm + 1)
    y_edges_vnmm = np.linspace(0.0, Ly, Ncy_vnmm + 1)
    
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)
    
    h_ref = np.pi / 20.0
    h_vnmm = max((Lx - x_int) / (vnmm_data['Nx'] - 1), Ly / (vnmm_data['Ny'] - 1))
    if det_tolerance_vnmm is None:
        tol_base = 1e-4 * (h_vnmm / h_ref)**4
        tol_det_vnmm = max(tol_base, floor_tolerance_vnmm) if floor_tolerance_vnmm is not None else tol_base
    else:
        tol_det_vnmm = det_tolerance_vnmm
    
    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_Mv, cols_Mv, data_Mv = [], [], []
    
    for j in range(Ncy_vnmm):
        y0, y1 = y_edges_vnmm[j], y_edges_vnmm[j + 1]
        dy = y1 - y0
        y_mid = 0.5 * (y0 + y1)
        for i in range(Ncx_vnmm):
            x0, x1 = x_edges_vnmm[i], x_edges_vnmm[i + 1]
            dx = x1 - x0
            x_mid = 0.5 * (x0 + x1)
            det_J = 0.25 * dx * dy
            
            for wi, xi in zip(w_1d, xi_1d):
                xg = x_mid + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = y_mid + 0.5 * dy * eta
                    Pg = np.array([xg, yg])
                    
                    supp_nodes, _, A_mat, _ = support_nodes_vnmm_2d_6_p1(
                        P=Pg, 
                        nodes_coords=coords_vnmm, 
                        nodes_vectors=vectors_vnmm, 
                        search_tree=tree_vnmm, 
                        K=12, 
                        Tol_det=tol_det_vnmm, 
                        adaptive=True
                    )
                    beta = la.inv(A_mat)
                    
                    Phi_x = beta[0, :]
                    Phi_y = beta[1, :]
                    rot_Phi = beta[4, :] - beta[3, :]
                    div_Phi = beta[2, :] + beta[5, :]
                    
                    weight = wi * wj * det_J
                    Kc_elem = inv_mu * weight * np.outer(rot_Phi, rot_Phi)
                    Kd_elem = s_div_vnmm * inv_mu * weight * np.outer(div_Phi, div_Phi)
                    M_elem = eps * weight * (np.outer(Phi_x, Phi_x) + np.outer(Phi_y, Phi_y))
                    
                    for a in range(6):
                        na = supp_nodes[a]
                        for b in range(6):
                            nb = supp_nodes[b]
                            rows_Kc.append(na); cols_Kc.append(nb); data_Kc.append(Kc_elem[a, b])
                            rows_Kd.append(na); cols_Kd.append(nb); data_Kd.append(Kd_elem[a, b])
                            rows_Mv.append(na); cols_Mv.append(nb); data_Mv.append(M_elem[a, b])
                            
    K_vnmm_full = coo_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(N_vnmm_total, N_vnmm_total)).tocsr()
    if s_div_vnmm > 0.0:
        Kd_vnmm = coo_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(N_vnmm_total, N_vnmm_total)).tocsr()
        K_vnmm_full = K_vnmm_full + Kd_vnmm
    M_vnmm_full = coo_matrix((data_Mv, (rows_Mv, cols_Mv)), shape=(N_vnmm_total, N_vnmm_total)).tocsr()
    
    idx_vnmm_int = np.where(~vnmm_data['is_pec'] & ~vnmm_data['is_interface'])[0]
    idx_vnmm_gamma = np.array([vnmm_data['interface_node_map'][k] for k in range(N_gamma)], dtype=int)
    
    N_vnmm_int = len(idx_vnmm_int)
    
    delta_y_gamma = Ly / N_gamma
    T_val = 1.0 / delta_y_gamma
    
    # -------------------------------------------------------------------------
    # 3. GLOBAL BLOCK COUPLING
    # -------------------------------------------------------------------------
    N_global = N_fem_int + N_gamma + N_vnmm_int
    
    K_glob = np.zeros((N_global, N_global), dtype=float)
    M_glob = np.zeros((N_global, N_global), dtype=float)
    
    s_f_int = slice(0, N_fem_int)
    s_gamma = slice(N_fem_int, N_fem_int + N_gamma)
    s_v_int = slice(N_fem_int + N_gamma, N_global)
    
    # FEM contribution
    K_f_dense = K_fem_full.toarray()
    M_f_dense = M_fem_full.toarray()
    
    K_glob[s_f_int, s_f_int] += K_f_dense[idx_fem_int, :][:, idx_fem_int]
    K_glob[s_f_int, s_gamma] += K_f_dense[idx_fem_int, :][:, idx_fem_gamma]
    K_glob[s_gamma, s_f_int] += K_f_dense[idx_fem_gamma, :][:, idx_fem_int]
    K_glob[s_gamma, s_gamma] += K_f_dense[idx_fem_gamma, :][:, idx_fem_gamma]
    
    M_glob[s_f_int, s_f_int] += M_f_dense[idx_fem_int, :][:, idx_fem_int]
    M_glob[s_f_int, s_gamma] += M_f_dense[idx_fem_int, :][:, idx_fem_gamma]
    M_glob[s_gamma, s_f_int] += M_f_dense[idx_fem_gamma, :][:, idx_fem_int]
    M_glob[s_gamma, s_gamma] += M_f_dense[idx_fem_gamma, :][:, idx_fem_gamma]
    
    # VNMM contribution with transformation T_gamma
    K_v_dense = K_vnmm_full.toarray()
    M_v_dense = M_vnmm_full.toarray()
    
    K_glob[s_gamma, s_gamma] += (T_val**2) * K_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_gamma]
    K_glob[s_gamma, s_v_int] += T_val * K_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_int]
    K_glob[s_v_int, s_gamma] += T_val * K_v_dense[idx_vnmm_int, :][:, idx_vnmm_gamma]
    K_glob[s_v_int, s_v_int] += K_v_dense[idx_vnmm_int, :][:, idx_vnmm_int]
    
    M_glob[s_gamma, s_gamma] += (T_val**2) * M_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_gamma]
    M_glob[s_gamma, s_v_int] += T_val * M_v_dense[idx_vnmm_gamma, :][:, idx_vnmm_int]
    M_glob[s_v_int, s_gamma] += T_val * M_v_dense[idx_vnmm_int, :][:, idx_vnmm_gamma]
    M_glob[s_v_int, s_v_int] += M_v_dense[idx_vnmm_int, :][:, idx_vnmm_int]
    
    # Symmetrization
    K_glob = 0.5 * (K_glob + K_glob.T)
    M_glob = 0.5 * (M_glob + M_glob.T)
    
    info_dofs = {
        'N_global': N_global,
        'N_fem_int': N_fem_int,
        'N_gamma': N_gamma,
        'N_vnmm_int': N_vnmm_int,
        'delta_y_gamma': delta_y_gamma
    }
    
    return csr_matrix(K_glob), csr_matrix(M_glob), info_dofs


def solve_hybrid_fem_vnmm_eigenvalues(
    Lx=np.pi, 
    Ly=np.pi, 
    frac_fem=0.5, 
    Nex_fem=8, 
    Ney=12, 
    Nx_vnmm=9, 
    Ny_vnmm=13,
    Ncx_vnmm=None, 
    Ncy_vnmm=None, 
    points_per_dir=3, 
    s_div_vnmm=4.0, 
    floor_tolerance_vnmm=None,
    det_tolerance_vnmm=None,
    num_eigenvalues=10, 
    tol_zero=0.1,
    interior_type_vnmm="alternating",
    jitter_frac_fem=0.0,
    jitter_frac_vnmm=0.0,
    seed=42,
    # Backward compatibility aliases
    pontos_por_dir=None,
    tol_piso_vnmm=None,
    tolerancia_det_vnmm=None,
    num_autovalores=None,
    tipo_interior_vnmm=None,
    num_modes=None
):
    """
    Complete hybrid Edge FEM + 2D VNMM coupled eigenvalue solver pipeline.
    """
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    if tol_piso_vnmm is not None:
        floor_tolerance_vnmm = tol_piso_vnmm
    if tolerancia_det_vnmm is not None:
        det_tolerance_vnmm = tolerancia_det_vnmm
    if num_autovalores is not None:
        num_eigenvalues = num_autovalores
    elif num_modes is not None:
        num_eigenvalues = num_modes
    if tipo_interior_vnmm is not None:
        interior_type_vnmm = tipo_interior_vnmm

    if isinstance(Lx, (tuple, list)) and len(Lx) == 2 and isinstance(Lx[0], dict) and 'elements' in Lx[0]:
        fem_data, vnmm_data = Lx[0], Lx[1]
        if isinstance(Ly, (tuple, list)) and len(Ly) >= 2:
            K_glob, M_glob = Ly[0], Ly[1]
            info_dofs = Ly[2] if len(Ly) > 2 else {}
        else:
            K_glob, M_glob, info_dofs = assemble_hybrid_fem_vnmm_matrices(
                fem_data, vnmm_data, Ncx_vnmm=Ncx_vnmm, Ncy_vnmm=Ncy_vnmm,
                points_per_dir=points_per_dir, s_div_vnmm=s_div_vnmm
            )
    else:
        fem_data, vnmm_data = generate_hybrid_cavity_mesh(
            Lx=Lx, Ly=Ly, frac_fem=frac_fem, Nex_fem=Nex_fem, Ney=Ney,
            Nx_vnmm=Nx_vnmm, Ny_vnmm=Ny_vnmm, interior_type_vnmm=interior_type_vnmm,
            jitter_frac_fem=jitter_frac_fem, jitter_frac_vnmm=jitter_frac_vnmm, seed=seed
        )
        K_glob, M_glob, info_dofs = assemble_hybrid_fem_vnmm_matrices(
            fem_data, vnmm_data, Ncx_vnmm=Ncx_vnmm, Ncy_vnmm=Ncy_vnmm,
            points_per_dir=points_per_dir, s_div_vnmm=s_div_vnmm,
            floor_tolerance_vnmm=floor_tolerance_vnmm, det_tolerance_vnmm=det_tolerance_vnmm
        )
    
    K_dense = K_glob.toarray()
    M_dense = M_glob.toarray()
    
    try:
        vals, vecs = la.eigh(K_dense, M_dense)
    except Exception:
        vals, vecs = la.eig(K_dense, M_dense)
        vals = np.real(vals)
        vecs = np.real(vecs)
        
    positive_mask = vals > tol_zero
    positive_vals = np.sort(vals[positive_mask])
    n_null = np.sum(~positive_mask)
    
    ref_vals = np.array([item[2] for item in ANALYTICAL_REFERENCE_MODES[:num_eigenvalues]])
    ref_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:num_eigenvalues]])
    
    # Identify modes matching reference values
    num_eigenvals = []
    for l_ref in ref_vals:
        idx_prox = np.argmin(np.abs(positive_vals - l_ref))
        num_eigenvals.append(positive_vals[idx_prox])
    num_eigenvals = np.array(num_eigenvals)
    
    kc_num = np.sqrt(np.maximum(num_eigenvals, 0.0))
    errors_lambda = np.abs(num_eigenvals - ref_vals) / ref_vals * 100.0
    errors_kc = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    edges_fem = fem_data['edges']
    nodes_fem = fem_data['nodes']
    len_fem = [np.linalg.norm(nodes_fem[e[0]] - nodes_fem[e[1]]) for e in edges_fem]
    h_max_fem = float(np.max(len_fem))
    
    from scipy.spatial import Delaunay
    tri_v = Delaunay(vnmm_data['coords'])
    edges_v = set()
    for s in tri_v.simplices:
        for i in range(3):
            edges_v.add(tuple(sorted([s[i], s[(i + 1) % 3]])))
    len_v = [np.linalg.norm(vnmm_data['coords'][e[0]] - vnmm_data['coords'][e[1]]) for e in edges_v]
    h_max_vnmm = float(np.max(len_v))
    h_max_global = max(h_max_fem, h_max_vnmm)
    
    return {
        'info_dofs': info_dofs,
        'h_max': h_max_global,
        'h_max_fem': h_max_fem,
        'h_max_vnmm': h_max_vnmm,
        'fem_data': fem_data,
        'vnmm_data': vnmm_data,
        'all_eigenvalues': vals,
        'numerical_eigenvalues': num_eigenvals,
        'analytical_eigenvalues': ref_vals,
        'numerical_kc': kc_num,
        'analytical_kc': ref_kc,
        'rel_errors_lambda_pct': errors_lambda,
        'rel_errors_kc_pct': errors_kc,
        'mean_error_lambda_pct': float(np.mean(errors_lambda)),
        'mean_error_kc_pct': float(np.mean(errors_kc)),
        'max_error_kc_pct': float(np.max(errors_kc)),
        'discarded_null_modes': int(n_null),
        'K_glob': K_glob,
        'M_glob': M_glob,
        # Backward compatibility aliases
        'dados_fem': fem_data,
        'dados_vnmm': vnmm_data,
        'autovalores_todos': vals,
        'autovalores_numericos': num_eigenvals,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': errors_lambda,
        'erros_kc_pct': errors_kc,
        'erro_medio_lambda_pct': float(np.mean(errors_lambda)),
        'erro_medio_kc_pct': float(np.mean(errors_kc)),
        'erro_max_kc_pct': float(np.max(errors_kc)),
        'n_nulos_descartados': int(n_null)
    }


# Backward compatibility aliases
gerar_malha_hibrida_cavidade = generate_hybrid_cavity_mesh
montar_matrizes_hibridas_fem_vnmm = assemble_hybrid_fem_vnmm_matrices
resolver_autovalores_hibrido_fem_vnmm = solve_hybrid_fem_vnmm_eigenvalues
