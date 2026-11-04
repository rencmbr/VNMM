"""
Support Node Selection for 3D Vector Nodal Meshless Method with Nédélec L1 Basis (6 Support Nodes).
Selects 6 non-degenerate support nodes spanning the 3D Nédélec 1st-kind polynomial space:
    R^1 = (P0)^3 + S^1, dim(R^1) = 6
Collocation matrix alpha (6x6) is built according to Eq. (3.16) of the attached formulation.
"""

import itertools
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree


def build_placement_matrix_3d_l1(local_coords, vectors):
    """
    Constructs the 6x6 placement matrix alpha for 6 support nodes.
    Eq. (3.16):
    Row k: [tkx, tky, tkz,  -zk*tky + yk*tkz,  zk*tkx - xk*tkz,  -yk*tkx + xk*tky]
    """
    N = len(local_coords)
    A = np.empty((N, 6), dtype=float)

    dx = local_coords[:, 0]
    dy = local_coords[:, 1]
    dz = local_coords[:, 2]

    tx = vectors[:, 0]
    ty = vectors[:, 1]
    tz = vectors[:, 2]

    # Columns 0, 1, 2: constant vectors [1,0,0], [0,1,0], [0,0,1]
    A[:, 0] = tx
    A[:, 1] = ty
    A[:, 2] = tz

    # Columns 3, 4, 5: cross product terms dx x t
    A[:, 3] = -dz * ty + dy * tz
    A[:, 4] =  dz * tx - dx * tz
    A[:, 5] = -dy * tx + dx * ty

    return A


def support_nodes_vnmm_3d_6_l1(
    P,
    nodes_coords,
    nodes_vectors,
    search_tree=None,
    K=14,
    Tol_det=1e-6,
    adaptive=True,
    step_K=4,
    K_max=None,
    method="pivoted_qr",
    # Compatibility aliases
    arvore_busca=None,
    adaptativo=None,
    passo_K=None,
    k=None,
    tol_det=None,
    step_k=None,
    k_max=None
):
    """
    Selects 6 support nodes in 3D for Nédélec L1 basis using KDTree and pivoted QR or combinatorics.
    
    Parameters:
    - P: Evaluation point (x, y, z).
    - nodes_coords: Global nodal coordinates (N, 3).
    - nodes_vectors: Global unit director vectors (N, 3).
    - search_tree: Pre-built scipy.spatial.KDTree over nodes_coords.
    - K: Initial number of nearest neighbors retrieved from KDTree (default: 14).
    - Tol_det: Minimum acceptable absolute determinant |det(A)|.
    - adaptive: Boolean indicating dynamic neighborhood expansion.
    - step_K: Neighbor increment at each expansion step (default: 4).
    
    Returns:
    - best_sextet: List of 6 global nodal indices.
    - best_det: Absolute determinant of the selected collocation matrix alpha.
    - best_A: Approved 6x6 collocation matrix alpha.
    - best_effective_k: Number of nearest neighbors utilized.
    """
    if k is not None: K = k
    if tol_det is not None: Tol_det = tol_det
    if step_k is not None: step_K = step_k
    if k_max is not None: K_max = k_max
    if adaptativo is not None: adaptive = adaptativo
    if passo_K is not None: step_K = passo_K

    if search_tree is None and arvore_busca is not None:
        search_tree = arvore_busca
    elif search_tree is None:
        search_tree = KDTree(nodes_coords)

    P = np.asarray(P, dtype=float)
    total_nodes = len(nodes_coords)
    limit_K = total_nodes if K_max is None else min(K_max, total_nodes)
    current_K = min(max(K, 6), total_nodes)

    best_sextet = None
    best_det = 0.0
    best_A = None
    best_effective_k = 0

    while True:
        distances, neighbor_indices = search_tree.query(P, k=current_K)
        if current_K == 1:
            neighbor_indices = [neighbor_indices]
        else:
            neighbor_indices = list(neighbor_indices)

        local_coords = nodes_coords[neighbor_indices] - P
        vecs = nodes_vectors[neighbor_indices]

        M = build_placement_matrix_3d_l1(local_coords, vecs)

        if method == "pivoted_qr" or current_K > 12:
            try:
                Q, R, piv = la.qr(M.T, pivoting=True)
                candidate_local = piv[:6]
                candidate_sextet = [neighbor_indices[idx] for idx in candidate_local]
                A = M[candidate_local, :]
                det_A = float(np.abs(np.linalg.det(A)))

                if det_A > best_det:
                    best_det = det_A
                    best_sextet = candidate_sextet
                    best_A = A
                    best_effective_k = int(np.max(candidate_local)) + 1

                if det_A >= Tol_det:
                    return candidate_sextet, det_A, A, best_effective_k
            except Exception:
                pass
        else:
            for comb in itertools.combinations(range(current_K), 6):
                candidate_sextet = [neighbor_indices[c] for c in comb]
                A = M[list(comb), :]
                det_A = float(np.abs(np.linalg.det(A)))

                if det_A > best_det:
                    best_det = det_A
                    best_sextet = candidate_sextet
                    best_A = A
                    best_effective_k = comb[-1] + 1

                if det_A >= Tol_det:
                    return candidate_sextet, det_A, A, comb[-1] + 1

        if not adaptive or current_K >= limit_K:
            break
        current_K = min(current_K + step_K, limit_K)

    if best_sextet is not None:
        return best_sextet, best_det, best_A, best_effective_k
    return None, 0.0, None, 0


# Compatibility aliases
nos_suporte_vnmm_3d_6_l1 = support_nodes_vnmm_3d_6_l1
find_support_nodes_vnmm_3d_6_l1 = support_nodes_vnmm_3d_6_l1
support_nodes_vnmm_3d_l1 = support_nodes_vnmm_3d_6_l1

