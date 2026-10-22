import itertools
import numpy as np


def support_nodes_vnmm_2d_6_p1(
    P, 
    nodes_coords, 
    nodes_vectors, 
    search_tree=None, 
    K=12, 
    Tol_det=1e-4, 
    adaptive=True, 
    step_K=4, 
    K_max=None,
    # Compatibility aliases
    arvore_busca=None,
    adaptativo=None,
    passo_K=None,
    k=None,
    tol_det=None,
    step_k=None,
    k_max=None
):
    if k is not None:
        K = k
    if tol_det is not None:
        Tol_det = tol_det
    if step_k is not None:
        step_K = step_k
    if k_max is not None:
        K_max = k_max
    """
    Incremental heuristic algorithm for selecting 6 support nodes with adaptive K expansion,
    using the complete linear polynomial basis P1 (6 terms).
    
    Mathematical Formulation:
    - Complete linear polynomial basis:
      P1 = <[1,0]^T, [0,1]^T, [x,0]^T, [y,0]^T, [0,x]^T, [0,y]^T>
    - Collocation matrix A (6x6) for each candidate sextet in local coordinates (x_k - x_P, y_k - y_P):
      Row k: [t_kx, t_ky, dx_k * t_kx, dy_k * t_kx, dx_k * t_ky, dy_k * t_ky]
    - Dimensional scaling: 2 columns O(1) and 4 columns O(h) => det(A) ~ O(h^4)
    
    Parameters:
    - P: Evaluation point coordinates (x, y).
    - nodes_coords: Numpy array (N, 2) of global nodal coordinates.
    - nodes_vectors: Numpy array (N, 2) of unit director components (tx, ty).
    - search_tree: Pre-built scipy.spatial.KDTree over nodes_coords.
    - K: Initial number of nearest neighbors retrieved from KDTree (default: 12).
    - Tol_det: Minimum acceptable absolute determinant |det(A)|.
    - adaptive: Boolean indicating whether neighborhood K expands dynamically if needed.
    - step_K: Neighbor increment at each expansion step (default: 4).
    - K_max: Maximum neighbor search cap (default: total number of nodes).
    
    Returns:
    - best_sextet: List of 6 global nodal indices.
    - best_det: Absolute determinant of the selected collocation matrix A.
    - best_A: Approved 6x6 collocation matrix.
    - best_effective_k: Number of nearest neighbors utilized in the search.
    """
    if search_tree is None and arvore_busca is not None:
        search_tree = arvore_busca
    elif search_tree is None:
        search_tree = KDTree(nodes_coords)
    if adaptativo is not None:
        adaptive = adaptativo
    if passo_K is not None:
        step_K = passo_K

    P = np.asarray(P, dtype=float)
    total_nodes = len(nodes_coords)
    limit_K = total_nodes if K_max is None else min(K_max, total_nodes)
    current_K = min(max(K, 6), total_nodes)
    
    best_sextet = None
    best_det = 0.0
    best_A = None
    best_effective_k = 0
    
    while True:
        # Phase 1: Retrieve K nearest neighbors via KDTree
        distances, neighbor_indices = search_tree.query(P, k=current_K)
        
        if current_K == 1:
            neighbor_indices = [neighbor_indices]
        else:
            neighbor_indices = list(neighbor_indices)
            
        # Phase 2: Combinatorial search with progressive anchor fixing
        # Gives geometric priority to nodes closest to point P
        for local_anchor_idx in range(current_K - 5):
            global_anchor = neighbor_indices[local_anchor_idx]
            remaining_indices = range(local_anchor_idx + 1, current_K)
            
            # 5-combinations among subsequent neighbors
            for comb in itertools.combinations(remaining_indices, 5):
                candidate_sextet = [global_anchor] + [neighbor_indices[c] for c in comb]
                candidate_effective_k = comb[-1] + 1
                
                # Phase 3: Vectorized construction of collocation matrix A (6x6)
                local_coords = nodes_coords[candidate_sextet] - P
                vecs = nodes_vectors[candidate_sextet]
                
                dx = local_coords[:, 0]
                dy = local_coords[:, 1]
                tx = vecs[:, 0]
                ty = vecs[:, 1]
                
                A = np.empty((6, 6), dtype=float)
                A[:, 0] = tx
                A[:, 1] = ty
                A[:, 2] = dx * tx
                A[:, 3] = dy * tx
                A[:, 4] = dx * ty
                A[:, 5] = dy * ty
                
                # Phase 4: Determinant evaluation
                det_A = float(np.abs(np.linalg.det(A)))
                
                if det_A > best_det:
                    best_det = det_A
                    best_sextet = candidate_sextet
                    best_A = A
                    best_effective_k = candidate_effective_k
                    
                # Early stopping criterion
                if det_A >= Tol_det:
                    return candidate_sextet, det_A, A, candidate_effective_k
                    
        # Stop if non-adaptive or reached maximum allowed neighbors
        if not adaptive or current_K >= limit_K:
            break
            
        # Dynamic neighborhood expansion
        current_K = min(current_K + step_K, limit_K)
        
    # Return best found configuration
    if best_sextet is not None and (best_det >= Tol_det or adaptive):
        return best_sextet, best_det, best_A, best_effective_k
    else:
        return None, 0.0, None, 0


# Compatibility aliases
nos_suporte_vnmm_2d_6_P1 = support_nodes_vnmm_2d_6_p1
find_support_nodes_vnmm_2d_6_p1 = support_nodes_vnmm_2d_6_p1
