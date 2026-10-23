import os
import sys
import numpy as np
import scipy.linalg as la
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigs


def generate_triangular_cavity_mesh(
    Nex=11, 
    Ney=11, 
    Lx=np.pi, 
    Ly=np.pi, 
    jitter_frac=0.0, 
    seed=42,
    nex=None,
    ney=None,
    nx=None,
    ny=None,
    lx=None,
    ly=None
):
    if nex is not None:
        Nex = nex
    elif nx is not None:
        Nex = nx
    if ney is not None:
        Ney = ney
    elif ny is not None:
        Ney = ny
    if lx is not None:
        Lx = lx
    if ly is not None:
        Ly = ly
    """
    Generates a 2D triangular mesh covering domain [0, Lx] x [0, Ly].
    Each rectangular cell is subdivided into two triangles.
    Supports controlled stochastic perturbation (jitter) for internal vertices.
    
    Returns:
    - nodes: Numpy array (N_nodes, 2) of Cartesian coordinates (x, y).
    - elements: Numpy array (N_elem, 3) of vertex indices for each triangle (CCW).
    - edges: Numpy array (N_edges, 2) of vertex pairs (n1, n2) per oriented edge (n1 < n2).
    - elem_edges: Numpy array (N_elem, 3) of global edge indices per element.
    - elem_edge_signs: Numpy array (N_elem, 3) with +1 or -1 for relative edge orientation.
    - is_boundary_edge: Boolean array (N_edges,) True for PEC boundary edges.
    """
    if seed is not None:
        np.random.seed(seed)
        
    x_lin = np.linspace(0.0, Lx, Nex + 1)
    y_lin = np.linspace(0.0, Ly, Ney + 1)
    dx = Lx / Nex
    dy = Ly / Ney
    
    # 1. Create nodes
    nodes_list = []
    node_grid = np.zeros((Ney + 1, Nex + 1), dtype=int)
    node_id = 0
    for j in range(Ney + 1):
        for i in range(Nex + 1):
            x, y = x_lin[i], y_lin[j]
            # Apply stochastic perturbation only to strictly interior nodes
            if jitter_frac > 0.0 and 0 < i < Nex and 0 < j < Ney:
                x += np.random.uniform(-jitter_frac * dx, jitter_frac * dx)
                y += np.random.uniform(-jitter_frac * dy, jitter_frac * dy)
            nodes_list.append([x, y])
            node_grid[j, i] = node_id
            node_id += 1
    nodes = np.array(nodes_list, dtype=float)
    
    # 2. Create triangular elements (counter-clockwise)
    elements_list = []
    for j in range(Ney):
        for i in range(Nex):
            n_bl = node_grid[j, i]       # bottom-left
            n_br = node_grid[j, i + 1]   # bottom-right
            n_tl = node_grid[j + 1, i]   # top-left
            n_tr = node_grid[j + 1, i + 1] # top-right
            
            # Lower triangle (bl, br, tr)
            elements_list.append([n_bl, n_br, n_tr])
            # Upper triangle (bl, tr, tl)
            elements_list.append([n_bl, n_tr, n_tl])
            
    elements = np.array(elements_list, dtype=int)
    
    # 3. Unique edges and connectivity
    edge_dict = {}
    edges_list = []
    elem_edges = np.zeros((len(elements), 3), dtype=int)
    elem_edge_signs = np.zeros((len(elements), 3), dtype=int)
    
    edge_id = 0
    for e_idx, elem in enumerate(elements):
        local_edges = [
            (elem[0], elem[1]),
            (elem[1], elem[2]),
            (elem[2], elem[0])
        ]
        
        for k, (u, v) in enumerate(local_edges):
            canonical = (min(u, v), max(u, v))
            sign = 1 if u < v else -1
            
            if canonical not in edge_dict:
                edge_dict[canonical] = edge_id
                edges_list.append(canonical)
                elem_edges[e_idx, k] = edge_id
                elem_edge_signs[e_idx, k] = sign
                edge_id += 1
            else:
                eid = edge_dict[canonical]
                elem_edges[e_idx, k] = eid
                elem_edge_signs[e_idx, k] = sign
                
    edges = np.array(edges_list, dtype=int)
    
    # 4. Identify PEC boundary edges
    is_boundary_edge = np.zeros(edge_id, dtype=bool)
    tol = 1e-7
    for eid in range(edge_id):
        n1, n2 = edges[eid]
        x1, y1 = nodes[n1]
        x2, y2 = nodes[n2]
        
        # Bottom or top boundary
        if (abs(y1) < tol and abs(y2) < tol) or (abs(y1 - Ly) < tol and abs(y2 - Ly) < tol):
            is_boundary_edge[eid] = True
        # Left or right boundary
        elif (abs(x1) < tol and abs(x2) < tol) or (abs(x1 - Lx) < tol and abs(x2 - Lx) < tol):
            is_boundary_edge[eid] = True
            
    return nodes, elements, edges, elem_edges, elem_edge_signs, is_boundary_edge


def assemble_edge_fem_matrices_2d(
    nodes, 
    elements=None, 
    edges=None, 
    elem_edges=None, 
    elem_edge_signs=None, 
    mu_r=1.0, 
    epsilon_r=1.0
):
    """
    Assembles global sparse curl stiffness matrix K_curl and mass matrix M
    for 1st-order Nédélec triangular edge elements (Whitney 1-forms).
    
    K_ij = integral_Omega (1/mu_r) * (curl W_i)_z * (curl W_j)_z dOmega
    M_ij = integral_Omega epsilon_r * (W_i . W_j) dOmega
    """
    if isinstance(nodes, (tuple, list)) and len(nodes) >= 5 and elements is None:
        nodes, elements, edges, elem_edges, elem_edge_signs = nodes[:5]
    elif isinstance(nodes, dict) and elements is None:
        elements = nodes['elements']
        edges = nodes['edges']
        elem_edges = nodes['elem_edges']
        elem_edge_signs = nodes['elem_edge_signs']
        nodes = nodes['nodes']
    N_edges = len(edges)
    inv_mu = 1.0 / mu_r
    eps = epsilon_r
    
    rows_K, cols_K, data_K = [], [], []
    rows_M, cols_M, data_M = [], [], []
    
    for e_idx, elem in enumerate(elements):
        p1, p2, p3 = nodes[elem[0]], nodes[elem[1]], nodes[elem[2]]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # Triangle area (2 * Area = (x2-x1)(y3-y1) - (x3-x1)(y2-y1))
        det2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * det2
        if area <= 0:
            raise ValueError(f"Triangle {e_idx} has non-positive area: {area}")
            
        # Barycentric coordinate gradients grad(L1), grad(L2), grad(L3)
        grad_L = np.zeros((3, 2), dtype=float)
        grad_L[0] = [(y2 - y3) / det2, (x3 - x2) / det2]
        grad_L[1] = [(y3 - y1) / det2, (x1 - x3) / det2]
        grad_L[2] = [(y1 - y2) / det2, (x2 - x1) / det2]
        
        # 3 local edges with node pairs: (0, 1), (1, 2), (2, 0)
        local_nodes = [(0, 1), (1, 2), (2, 0)]
        signs = elem_edge_signs[e_idx]
        global_eids = elem_edges[e_idx]
        
        # 1. Element Stiffness Matrix K_e (constant curl in triangle)
        rot_W = np.zeros(3, dtype=float)
        for k in range(3):
            rot_W[k] = signs[k] / area
            
        K_e = (inv_mu * area) * np.outer(rot_W, rot_W)
        
        # 2. Element Mass Matrix M_e (exact barycentric integration)
        M_e = np.zeros((3, 3), dtype=float)
        for a in range(3):
            ia, ja = local_nodes[a]
            sa = signs[a]
            for b in range(3):
                ib, jb = local_nodes[b]
                sb = signs[b]
                
                def int_LL(p, q):
                    return area / 6.0 if p == q else area / 12.0
                    
                term1 = int_LL(ia, ib) * np.dot(grad_L[ja], grad_L[jb])
                term2 = int_LL(ia, jb) * np.dot(grad_L[ja], grad_L[ib])
                term3 = int_LL(ja, ib) * np.dot(grad_L[ia], grad_L[jb])
                term4 = int_LL(ja, jb) * np.dot(grad_L[ia], grad_L[ib])
                
                val_m = eps * sa * sb * (term1 - term2 - term3 + term4)
                M_e[a, b] = val_m
                
        # Accumulate into global matrices
        for a in range(3):
            ea = global_eids[a]
            for b in range(3):
                eb = global_eids[b]
                
                rows_K.append(ea)
                cols_K.append(eb)
                data_K.append(K_e[a, b])
                
                rows_M.append(ea)
                cols_M.append(eb)
                data_M.append(M_e[a, b])
                
    K_glob = coo_matrix((data_K, (rows_K, cols_K)), shape=(N_edges, N_edges)).tocsr()
    M_glob = coo_matrix((data_M, (rows_M, cols_M)), shape=(N_edges, N_edges)).tocsr()
    
    K_glob = 0.5 * (K_glob + K_glob.T)
    M_glob = 0.5 * (M_glob + M_glob.T)
    
    return K_glob, M_glob


def solve_edge_fem_cavity_eigenvalues(
    Nex=11, 
    Ney=11, 
    Lx=np.pi, 
    Ly=np.pi, 
    mu_r=1.0, 
    epsilon_r=1.0, 
    num_eigenvalues=10, 
    tol_zero=1e-3,
    jitter_frac=0.0,
    seed=42,
    # Backward compatibility
    num_autovalores=None,
    num_modes=None,
    n_modes=None,
    nex=None,
    ney=None,
    nx=None,
    ny=None,
    lx=None,
    ly=None
):
    """
    Complete eigenvalue solver pipeline for 2D triangular Nédélec edge elements.
    
    Returns dictionary with numerical eigenvalues, percentage errors, and mesh metadata.
    """
    if num_autovalores is not None:
        num_eigenvalues = num_autovalores
    elif num_modes is not None:
        num_eigenvalues = num_modes
    elif n_modes is not None:
        num_eigenvalues = n_modes

    if nex is not None:
        Nex = nex
    elif nx is not None:
        Nex = nx
    if ney is not None:
        Ney = ney
    elif ny is not None:
        Ney = ny
    if lx is not None:
        Lx = lx
    if ly is not None:
        Ly = ly

    if isinstance(Nex, (tuple, list)) and len(Nex) >= 6:
        nodes, elements, edges, elem_edges, elem_edge_signs, is_boundary_edge = Nex[:6]
        Nex = max(int(np.sqrt(len(elements) // 2)), 1)
        Ney = Nex
    elif isinstance(Nex, dict) and 'nodes' in Nex:
        nodes = Nex['nodes']
        elements = Nex['elements']
        edges = Nex['edges']
        elem_edges = Nex['elem_edges']
        elem_edge_signs = Nex['elem_edge_signs']
        is_boundary_edge = Nex['is_boundary_edge']
        Nex = max(int(np.sqrt(len(elements) // 2)), 1)
        Ney = Nex
    else:
        nodes, elements, edges, elem_edges, elem_edge_signs, is_boundary_edge = generate_triangular_cavity_mesh(
            Nex=Nex, Ney=Ney, Lx=Lx, Ly=Ly, jitter_frac=jitter_frac, seed=seed
        )
    
    if isinstance(Ly, (tuple, list)) and len(Ly) >= 2:
        K_glob, M_glob = Ly[0], Ly[1]
    else:
        K_glob, M_glob = assemble_edge_fem_matrices_2d(
            nodes, elements, edges, elem_edges, elem_edge_signs, mu_r=mu_r, epsilon_r=epsilon_r
        )
    
    # Enforce PEC boundary conditions (Dirichlet E_t = 0 on boundary edges)
    interior_indices = np.where(~is_boundary_edge)[0]
    K_red = K_glob[interior_indices, :][:, interior_indices]
    M_red = M_glob[interior_indices, :][:, interior_indices]
    
    # Generalized eigenvalue solve
    try:
        vals, vecs = la.eigh(K_red.toarray(), M_red.toarray())
    except Exception:
        vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
        vals = np.real(vals)
        vecs = np.real(vecs)
        
    # Filter and discard gradient nullspace eigenvalues
    positive_mask = vals > tol_zero
    positive_vals = np.sort(vals[positive_mask])
    n_null = np.sum(~positive_mask)
    
    num_eigenvals = positive_vals[:num_eigenvalues]
    
    from src.eigen_solver_cavity_2d import ANALYTICAL_REFERENCE_MODES
    ref_vals = np.array([item[2] for item in ANALYTICAL_REFERENCE_MODES[:len(num_eigenvals)]])
    ref_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:len(num_eigenvals)]])
    
    kc_num = np.sqrt(np.maximum(num_eigenvals, 0.0))
    errors_lambda = np.abs(num_eigenvals - ref_vals) / ref_vals * 100.0
    errors_kc = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    h_max = float(max(np.linalg.norm(nodes[e[0]] - nodes[e[1]]) for e in edges))
    
    return {
        'Nex': Nex,
        'Ney': Ney,
        'N_nodes': len(nodes),
        'N_elements': len(elements),
        'N_edges_total': len(edges),
        'active_dofs': len(interior_indices),
        'boundary_edges': int(np.sum(is_boundary_edge)),
        'discarded_null_modes': int(n_null),
        'h_max': h_max,
        'numerical_eigenvalues': num_eigenvals,
        'analytical_eigenvalues': ref_vals,
        'numerical_kc': kc_num,
        'analytical_kc': ref_kc,
        'rel_errors_lambda_pct': errors_lambda,
        'rel_errors_kc_pct': errors_kc,
        'mean_error_lambda_pct': float(np.mean(errors_lambda)),
        'mean_error_kc_pct': float(np.mean(errors_kc)),
        'max_error_kc_pct': float(np.max(errors_kc)),
        'K_red': K_red,
        'M_red': M_red,
        # Backward compatibility aliases
        'N_incognitas': len(interior_indices),
        'N_fronteira': int(np.sum(is_boundary_edge)),
        'n_nulos_descartados': int(n_null),
        'autovalores_numericos': num_eigenvals,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': errors_lambda,
        'erros_kc_pct': errors_kc,
        'erro_medio_lambda_pct': float(np.mean(errors_lambda)),
        'erro_medio_kc_pct': float(np.mean(errors_kc)),
        'erro_max_kc_pct': float(np.max(errors_kc))
    }


# Backward compatibility aliases
gerar_malha_triangular_cavidade = generate_triangular_cavity_mesh
montar_matrizes_fem_aresta_2d = assemble_edge_fem_matrices_2d
resolver_autovalores_fem_aresta_2d = solve_edge_fem_cavity_eigenvalues
