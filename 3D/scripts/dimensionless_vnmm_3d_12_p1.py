"""
Dimensionless Formulation for 3D Vector Nodal Meshless Method (VNMM 3D).
Normalizes coordinates by local characteristic spacing h_local:
    alpha = alpha_hat @ diag(1, 1, 1, h, h, h, h, h, h, h, h, h)
    |det(alpha_hat)| = |det(alpha)| / h_local^9 >= Tol_hat > 0
Eliminates artificial O(h^9) determinant decay under mesh refinement.
"""

import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scripts.support_nodes_vnmm_3d_12_p1 import build_placement_matrix_3d


def support_nodes_vnmm_3d_12_p1_dimensionless(
    point,
    node_coords,
    node_vectors,
    search_tree=None,
    h_local=None,
    num_neighbors=24,
    tol_hat=1e-3,
    adaptive=True,
    step_k=6,
    max_k=None,
    return_dimensional_matrix=False
):
    """
    Selects 12 support nodes based on scale-invariant dimensionless matrix alpha_hat.
    """
    if search_tree is None:
        search_tree = KDTree(node_coords)

    point = np.asarray(point, dtype=float)
    total_nodes = len(node_coords)
    limit_k = total_nodes if max_k is None else min(max_k, total_nodes)
    current_k = min(max(num_neighbors, 12), total_nodes)

    best_dodecet = None
    best_det_hat = 0.0
    best_a_hat = None
    best_effective_k = 0

    while True:
        distances, neighbor_indices = search_tree.query(point, k=current_k)
        if current_k == 1:
            neighbor_indices = [neighbor_indices]
            distances = [distances]
        else:
            neighbor_indices = list(neighbor_indices)
            distances = list(distances)

        if h_local is None:
            eff_h = np.mean(distances) if np.mean(distances) > 1e-12 else 1.0
        else:
            eff_h = h_local

        local_coords = (node_coords[neighbor_indices] - point) / eff_h
        vecs = node_vectors[neighbor_indices]

        M_hat = build_placement_matrix_3d(local_coords, vecs)

        try:
            Q, R, piv = la.qr(M_hat.T, pivoting=True)
            candidate_local = piv[:12]
            candidate_dodecet = [neighbor_indices[idx] for idx in candidate_local]
            A_hat = M_hat[candidate_local, :]
            det_hat = float(np.abs(np.linalg.det(A_hat)))

            if det_hat > best_det_hat:
                best_det_hat = det_hat
                best_dodecet = candidate_dodecet
                best_a_hat = A_hat
                best_effective_k = int(np.max(candidate_local)) + 1

            if det_hat >= tol_hat:
                if return_dimensional_matrix:
                    coords_dim = node_coords[candidate_dodecet] - point
                    A_dim = build_placement_matrix_3d(coords_dim, node_vectors[candidate_dodecet])
                    return candidate_dodecet, det_hat, A_dim, best_effective_k
                return candidate_dodecet, det_hat, A_hat, best_effective_k
        except Exception:
            pass

        if not adaptive or current_k >= limit_k:
            break
        current_k = min(current_k + step_k, limit_k)

    if best_dodecet is not None:
        if return_dimensional_matrix:
            coords_dim = node_coords[best_dodecet] - point
            A_dim = build_placement_matrix_3d(coords_dim, node_vectors[best_dodecet])
            return best_dodecet, best_det_hat, A_dim, best_effective_k
        return best_dodecet, best_det_hat, best_a_hat, best_effective_k
    return None, 0.0, None, 0


# Compatibility aliases
support_nodes_dimensionless_3d = support_nodes_vnmm_3d_12_p1_dimensionless
nos_suporte_adimensional_3d = support_nodes_vnmm_3d_12_p1_dimensionless

