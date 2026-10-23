import itertools
import numpy as np


def support_nodes_vnmm_2d_3_l1(
    P, 
    nodes_coords, 
    nodes_vectors, 
    search_tree=None, 
    K=15, 
    Tol_det=1e-3, 
    adaptive=True, 
    step_K=5, 
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
    Incremental heuristic algorithm for selecting 3 support nodes with adaptive K expansion (L1 basis).
    
    Parameters:
    - P: Evaluation point coordinate (x, y).
    - nodes_coords: Numpy array of all global nodal coordinates.
    - nodes_vectors: Numpy array of unit vector components (tx, ty).
    - search_tree: Pre-built KDTree object on nodes_coords.
    - K: Initial number of nearest neighbors retrieved by KDTree.
    - Tol_det: Minimum acceptable absolute determinant of A.
    - adaptive: Boolean indicating whether neighborhood size K expands automatically.
    - step_K: Neighbor increment at each expansion step.
    - K_max: Maximum neighbor search cap.
    
    Returns:
    - trio_indices: Original global indices of the 3 selected nodes.
    - det_A: Absolute determinant of the selected A matrix.
    - A_mat: Approved 3x3 directional moment matrix.
    - effective_k: Effective number of nearest neighbors evaluated.
    """
    from scipy.spatial import KDTree
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
    current_K = min(K, total_nodes)
    
    best_trio = None
    best_det = 0.0
    best_A = None
    best_effective_k = 0
    
    while True:
        # Phase 1: Retrieve nearest neighbors using KDTree
        distances, neighbor_indices = search_tree.query(P, k=current_K)
        
        if current_K == 1:
            neighbor_indices = [neighbor_indices]
            
        # Phase 2: Incremental search with anchor fallback
        for local_anchor_idx in range(current_K - 2):
            global_anchor = neighbor_indices[local_anchor_idx]
            remaining_indices = range(local_anchor_idx + 1, current_K)
            
            # 2-combinations of remaining nodes
            for comb in itertools.combinations(remaining_indices, 2):
                global_idx2 = neighbor_indices[comb[0]]
                global_idx3 = neighbor_indices[comb[1]]
                candidate_effective_k = comb[1] + 1
                
                candidate_trio = [global_anchor, global_idx2, global_idx3]
                
                # Phase 3: Construct interpolation matrix A (3x3)
                local_coords = nodes_coords[candidate_trio] - P
                vecs = nodes_vectors[candidate_trio]
                
                local_x = local_coords[:, 0]
                local_y = local_coords[:, 1]
                tx = vecs[:, 0]
                ty = vecs[:, 1]
                
                A = np.empty((3, 3), dtype=float)
                A[:, 0] = tx
                A[:, 1] = ty
                A[:, 2] = local_y * tx - local_x * ty
                
                # Phase 4: Algebraic evaluation
                det_A = np.abs(np.linalg.det(A))
                
                if det_A > best_det:
                    best_det = det_A
                    best_trio = candidate_trio
                    best_A = A
                    best_effective_k = candidate_effective_k
                
                # Early stopping criterion
                if det_A >= Tol_det:
                    return candidate_trio, det_A, A, candidate_effective_k
                    
        # If not adaptive or reached limit
        if not adaptive or current_K >= limit_K:
            break
            
        # Expand K for next iteration
        current_K = min(current_K + step_K, limit_K)
        
    if best_trio is not None and (best_det >= Tol_det or adaptive):
        return best_trio, best_det, best_A, best_effective_k
    else:
        return None, 0.0, None, 0


# Compatibility aliases
nos_suporte_vnmm_2d_3_L1 = support_nodes_vnmm_2d_3_l1
find_support_nodes_vnmm_2d_3_l1 = support_nodes_vnmm_2d_3_l1
