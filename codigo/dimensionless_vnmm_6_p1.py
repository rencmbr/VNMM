"""
Module for Dimensionless Formulation of VNMM 2D with Complete Linear Basis P1 (6 nodes).

Eliminates dimensional tolerance floors (Tol_floor) and artificial scaling ill-conditioning
by normalizing local coordinates with respect to local characteristic spacing h_local:
    A = A_hat @ diag(1, 1, h, h, h, h)
    |det(A_hat)| = |det(A)| / h^4 >= Tol_hat > 0
"""

import itertools
import numpy as np


def support_nodes_vnmm_2d_6_p1_dimensionless(
    point,
    node_coords,
    node_vectors,
    search_tree,
    h_local=None,
    num_neighbors=12,
    tol_hat=1e-2,
    adaptive=True,
    step_k=4,
    max_k=None,
    return_dimensional_matrix=False,
    # Backward compatibility
    P=None,
    nodes_coords=None,
    nodes_vectors=None,
    arvore_busca=None,
    K=None,
    Tol_hat=None,
    adaptativo=None,
    passo_K=None,
    K_max=None,
    retornar_matriz_dimensional=None
):
    """
    Incremental heuristic selection of 6 support nodes based on the dimensionless matrix A_hat.
    """
    if P is not None: point = P
    if nodes_coords is not None: node_coords = nodes_coords
    if nodes_vectors is not None: node_vectors = nodes_vectors
    if arvore_busca is not None: search_tree = arvore_busca
    if K is not None: num_neighbors = K
    if Tol_hat is not None: tol_hat = Tol_hat
    if adaptativo is not None: adaptive = adaptativo
    if passo_K is not None: step_k = passo_K
    if K_max is not None: max_k = K_max
    if retornar_matriz_dimensional is not None: return_dimensional_matrix = retornar_matriz_dimensional

    point = np.asarray(point, dtype=float)
    total_nodes = len(node_coords)
    limit_k = total_nodes if max_k is None else min(max_k, total_nodes)
    current_k = min(max(num_neighbors, 6), total_nodes)

    best_sextet = None
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

        if h_local is not None and h_local > 0:
            h_eff = float(h_local)
        else:
            d6 = distances[:min(6, len(distances))]
            h_eff = float(np.mean(d6)) if len(d6) > 0 and np.mean(d6) > 1e-14 else 1.0

        inv_h = 1.0 / h_eff

        for idx_anchor_local in range(current_k - 5):
            anchor_global = neighbor_indices[idx_anchor_local]
            remaining_indices = range(idx_anchor_local + 1, current_k)

            for comb in itertools.combinations(remaining_indices, 5):
                candidate_sextet = [anchor_global] + [neighbor_indices[c] for c in comb]
                candidate_effective_k = comb[-1] + 1

                local_coords = node_coords[candidate_sextet] - point
                vecs = node_vectors[candidate_sextet]

                hat_dx = local_coords[:, 0] * inv_h
                hat_dy = local_coords[:, 1] * inv_h
                tx = vecs[:, 0]
                ty = vecs[:, 1]

                a_hat = np.empty((6, 6), dtype=float)
                a_hat[:, 0] = tx
                a_hat[:, 1] = ty
                a_hat[:, 2] = hat_dx * tx
                a_hat[:, 3] = hat_dy * tx
                a_hat[:, 4] = hat_dx * ty
                a_hat[:, 5] = hat_dy * ty

                det_hat = float(np.abs(np.linalg.det(a_hat)))

                if det_hat > best_det_hat:
                    best_det_hat = det_hat
                    best_sextet = candidate_sextet
                    best_a_hat = a_hat
                    best_effective_k = candidate_effective_k

                if det_hat >= tol_hat:
                    if return_dimensional_matrix:
                        d_h = np.diag([1.0, 1.0, h_eff, h_eff, h_eff, h_eff])
                        a_dim = a_hat @ d_h
                        return candidate_sextet, det_hat, a_hat, h_eff, candidate_effective_k, a_dim
                    return candidate_sextet, det_hat, a_hat, h_eff, candidate_effective_k

        if not adaptive or current_k >= limit_k:
            break

        current_k = min(current_k + step_k, limit_k)

    if best_sextet is not None and (best_det_hat >= tol_hat or adaptive):
        if return_dimensional_matrix:
            d_h = np.diag([1.0, 1.0, h_eff, h_eff, h_eff, h_eff])
            a_dim = best_a_hat @ d_h if best_a_hat is not None else None
            return best_sextet, best_det_hat, best_a_hat, h_eff, best_effective_k, a_dim
        return best_sextet, best_det_hat, best_a_hat, h_eff, best_effective_k
    else:
        if return_dimensional_matrix:
            return None, 0.0, None, h_eff, 0, None
        return None, 0.0, None, h_eff, 0


def shape_functions_vnmm_2d_6_p1_dimensionless(
    point,
    node_coords,
    node_vectors,
    selected_nodes,
    h_local=None,
    matrix_a_hat=None,
    # Backward compatibility
    P=None,
    nodes_coords=None,
    nodes_vectors=None,
    nos_selecionados=None,
    matriz_a_hat=None,
    delta_ref=None
):
    """
    Compute vector shape functions N_i, curl (curl N_i)_z and divergence (div N_i)
    at evaluation point using the dimensionless matrix A_hat.
    """
    if P is not None: point = P
    if nodes_coords is not None: node_coords = nodes_coords
    if nodes_vectors is not None: node_vectors = nodes_vectors
    if nos_selecionados is not None: selected_nodes = nos_selecionados
    if matriz_a_hat is not None: matrix_a_hat = matriz_a_hat
    if h_local is None and delta_ref is not None:
        h_local = delta_ref
    elif h_local is None:
        h_local = 1.0

    point = np.asarray(point, dtype=float)
    h_eff = float(h_local)
    inv_h = 1.0 / h_eff

    if matrix_a_hat is None:
        local_coords = node_coords[selected_nodes] - point
        vecs = node_vectors[selected_nodes]

        hat_dx = local_coords[:, 0] * inv_h
        hat_dy = local_coords[:, 1] * inv_h
        tx = vecs[:, 0]
        ty = vecs[:, 1]

        a_hat = np.empty((6, 6), dtype=float)
        a_hat[:, 0] = tx
        a_hat[:, 1] = ty
        a_hat[:, 2] = hat_dx * tx
        a_hat[:, 3] = hat_dy * tx
        a_hat[:, 4] = hat_dx * ty
        a_hat[:, 5] = hat_dy * ty
    else:
        a_hat = np.asarray(matrix_a_hat, dtype=float)

    beta_hat = np.linalg.inv(a_hat)

    phi = beta_hat[0:2, :]
    rot_phi = (beta_hat[4, :] - beta_hat[3, :]) * inv_h
    div_phi = (beta_hat[2, :] + beta_hat[5, :]) * inv_h

    return phi, rot_phi, div_phi, beta_hat


# Backward compatibility and English aliases
nos_suporte_vnmm_2d_6_P1_adimensional = support_nodes_vnmm_2d_6_p1_dimensionless
funcoes_forma_vnmm_2d_6_P1_adimensional = shape_functions_vnmm_2d_6_p1_dimensionless
dimensionless_support_nodes_6_p1 = support_nodes_vnmm_2d_6_p1_dimensionless
dimensionless_shape_functions_6_p1 = shape_functions_vnmm_2d_6_p1_dimensionless
