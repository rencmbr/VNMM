"""
3D Vector Shape Functions and Derivatives for VNMM 3D with Nédélec L1 Basis (6 Support Nodes).
Implements equations (3.11) - (3.21) of the mathematical formulation.
"""

import numpy as np


def shape_functions_vnmm_3d_6_l1(
    P,
    nodes_coords,
    nodes_vectors,
    selected_nodes,
    matrix_a=None,
    dx_eval=0.0,
    dy_eval=0.0,
    dz_eval=0.0,
    # Compatibility aliases
    nos_selecionados=None,
    matriz_a=None
):
    """
    Computes vector shape functions N_i(P), curl(N_i), and div(N_i) for 6 support nodes in 3D.
    
    Mathematical Formulation:
    - Shape functions (Eq. 3.12):
        N_ix = beta_1i + beta_5i * dz - beta_6i * dy
        N_iy = beta_2i - beta_4i * dz + beta_6i * dx
        N_iz = beta_3i + beta_4i * dy - beta_5i * dx
    - Collocation:
        alpha * beta = I_6  ==>  beta = inv(alpha)
    - At evaluation point (dx = 0, dy = 0, dz = 0):
        Phi = beta[0:3, :]  (shape 3 x 6)
    - Curl:
        curl N_i = 2 * [beta_4i, beta_5i, beta_6i]^T  (shape 3 x 6)
    - Divergence:
        div N_i = 0  (identically divergence-free, shape 6,)
    """
    if nos_selecionados is not None:
        selected_nodes = nos_selecionados
    if matriz_a is not None:
        matrix_a = matriz_a

    P = np.asarray(P, dtype=float)

    if matrix_a is None:
        coords_supp = nodes_coords[selected_nodes] - P
        vecs_supp = nodes_vectors[selected_nodes]
        from scripts.support_nodes_vnmm_3d_6_l1 import build_placement_matrix_3d_l1
        matrix_a = build_placement_matrix_3d_l1(coords_supp, vecs_supp)

    beta = np.linalg.inv(matrix_a)

    if dx_eval == 0.0 and dy_eval == 0.0 and dz_eval == 0.0:
        Phi = beta[0:3, :].copy()
    else:
        Phi_x = beta[0, :] + beta[4, :] * dz_eval - beta[5, :] * dy_eval
        Phi_y = beta[1, :] - beta[3, :] * dz_eval + beta[5, :] * dx_eval
        Phi_z = beta[2, :] + beta[3, :] * dy_eval - beta[4, :] * dx_eval
        Phi = np.vstack([Phi_x, Phi_y, Phi_z])

    # curl(N_i) = 2 * [beta_4i, beta_5i, beta_6i]^T
    curl_Phi = 2.0 * beta[3:6, :].copy()

    # div(N_i) = 0
    div_Phi = np.zeros(6, dtype=float)

    return Phi, curl_Phi, div_Phi, beta


# Compatibility aliases
funcoes_forma_vnmm_3d_6_l1 = shape_functions_vnmm_3d_6_l1
funcoes_forma_vnmm_3d_l1 = shape_functions_vnmm_3d_6_l1
shape_functions_vnmm_3d_l1 = shape_functions_vnmm_3d_6_l1

