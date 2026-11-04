"""
Support Node Selection for 3D Vector Nodal Meshless Method (VNMM 3D).
Selects 12 non-degenerate support nodes spanning the complete 3D linear polynomial vector space R^1 = (P1)^3.
Collocation matrix alpha (12x12) is built according to Eq. (3.26) of the theoretical formulation.
"""

import itertools
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree


def build_placement_matrix_3d(local_coords, vectors):
    """
    Constructs the 12x12 placement matrix alpha for 12 support nodes.
    Eq. (3.26):
    Row k: [tkx, tky, tkz,  xk*tkx, xk*tky, xk*tkz,  yk*tkx, yk*tky, yk*tkz,  zk*tkx, zk*tky, zk*tkz]
    """
    N = len(local_coords)
    A = np.empty((N, 12), dtype=float)

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

    # Columns 3, 4, 5: x-linear terms [x,0,0], [0,x,0], [0,0,x]
    A[:, 3] = dx * tx
    A[:, 4] = dx * ty
    A[:, 5] = dx * tz

    # Columns 6, 7, 8: y-linear terms [y,0,0], [0,y,0], [0,0,y]
    A[:, 6] = dy * tx
    A[:, 7] = dy * ty
    A[:, 8] = dy * tz

    # Columns 9, 10, 11: z-linear terms [z,0,0], [0,z,0], [0,0,z]
    A[:, 9] = dz * tx
    A[:, 10] = dz * ty
    A[:, 11] = dz * tz

    return A


def support_nodes_vnmm_3d_12_p1(
    P,
    nodes_coords,
    nodes_vectors,
    search_tree=None,
    K=24,
    Tol_det=1e-6,
    adaptive=True,
    step_K=6,
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
    Selects 12 support nodes in 3D using KDTree neighborhood query and rank-revealing QR / combinatorial search.
    
    Parameters:
    - P: Evaluation point coordinates (x, y, z).
    - nodes_coords: Global nodal coordinates (N, 3).
    - nodes_vectors: Global unit director vectors (tx, ty, tz) (N, 3).
    - search_tree: Pre-built scipy.spatial.KDTree over nodes_coords.
    - K: Initial number of nearest neighbors retrieved from KDTree (default: 24).
    - Tol_det: Minimum acceptable absolute determinant |det(A)|.
    - adaptive: Boolean indicating whether neighborhood K expands dynamically if needed.
    - step_K: Neighbor increment at each expansion step (default: 6).
    - K_max: Maximum neighbor search cap (default: total number of nodes).
    - method: 'pivoted_qr' (default, fast & well-conditioned) or 'combinatorial'.
    
    Returns:
    - best_dodecet: List of 12 global nodal indices.
    - best_det: Absolute determinant of the selected collocation matrix alpha.
    - best_A: Approved 12x12 collocation matrix alpha.
    - best_effective_k: Number of nearest neighbors utilized in the search.
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
    current_K = min(max(K, 12), total_nodes)

    best_dodecet = None
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

        M = build_placement_matrix_3d(local_coords, vecs)

        if method == "pivoted_qr" or current_K > 14:
            # Rank-revealing column-pivoted QR on M^T (shape 12 x current_K)
            try:
                Q, R, piv = la.qr(M.T, pivoting=True)
                candidate_local = piv[:12]
                candidate_dodecet = [neighbor_indices[idx] for idx in candidate_local]
                A = M[candidate_local, :]
                det_A = float(np.abs(np.linalg.det(A)))

                if det_A > best_det:
                    best_det = det_A
                    best_dodecet = candidate_dodecet
                    best_A = A
                    best_effective_k = int(np.max(candidate_local)) + 1

                if det_A >= Tol_det:
                    return candidate_dodecet, det_A, A, best_effective_k
            except Exception:
                pass
        else:
            # Combinatorial search among closest neighbors (for small K)
            for comb in itertools.combinations(range(current_K), 12):
                candidate_dodecet = [neighbor_indices[c] for c in comb]
                A = M[list(comb), :]
                det_A = float(np.abs(np.linalg.det(A)))

                if det_A > best_det:
                    best_det = det_A
                    best_dodecet = candidate_dodecet
                    best_A = A
                    best_effective_k = comb[-1] + 1

                if det_A >= Tol_det:
                    return candidate_dodecet, det_A, A, comb[-1] + 1

        if not adaptive or current_K >= limit_K:
            break

        current_K = min(current_K + step_K, limit_K)

    if best_dodecet is not None and (best_det >= Tol_det or adaptive):
        return best_dodecet, best_det, best_A, best_effective_k
    else:
        return None, 0.0, None, 0


# Compatibility aliases
nos_suporte_vnmm_3d_12_p1 = support_nodes_vnmm_3d_12_p1
find_support_nodes_vnmm_3d_12_p1 = support_nodes_vnmm_3d_12_p1
nos_suporte_vnmm_3d = support_nodes_vnmm_3d_12_p1
support_nodes_vnmm_3d = support_nodes_vnmm_3d_12_p1

