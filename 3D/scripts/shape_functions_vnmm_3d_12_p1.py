"""
3D Vector Shape Functions and Derivatives for VNMM 3D with Complete Linear Basis (12 Support Nodes).
Implements equations (3.24) - (3.28) of the mathematical formulation.
"""

import numpy as np


def shape_functions_vnmm_3d_12_p1(
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
    Computes vector shape functions N_i(P), their curl (curl N_i), and divergence (div N_i)
    for 12 support nodes in 3D.
    
    Mathematical Formulation:
    - Shape function components:
        N_ix = beta_1i + beta_4i * dx + beta_7i * dy + beta_10i * dz
        N_iy = beta_2i + beta_5i * dx + beta_8i * dy + beta_11i * dz
        N_iz = beta_3i + beta_6i * dx + beta_9i * dy + beta_12i * dz
    - Matrix relation:
        alpha * beta = I_12  ==>  beta = inv(alpha)
    - At evaluation point (dx = 0, dy = 0, dz = 0):
        Phi = beta[0:3, :]  (shape 3 x 12)
    - Curl:
        (curl N_i)_x = beta_9i  - beta_11i   (index 8 - index 10)
        (curl N_i)_y = beta_10i - beta_6i    (index 9 - index 5)
        (curl N_i)_z = beta_5i  - beta_7i    (index 4 - index 6)
    - Divergence:
        div N_i = beta_4i + beta_8i + beta_12i (index 3 + index 7 + index 11)
        
    Parameters:
    - P: Evaluation point (x, y, z).
    - nodes_coords: Global nodal coordinates (N, 3).
    - nodes_vectors: Global unit director vectors (N, 3).
    - selected_nodes: List/array of 12 selected global node indices.
    - matrix_a: (Optional) Precomputed 12x12 collocation matrix alpha.
    - dx_eval, dy_eval, dz_eval: Optional local offsets if evaluated away from stencil center.
    
    Returns:
    - Phi: Matrix (3, 12) of vector shape functions at evaluation point.
    - curl_Phi: Matrix (3, 12) with 3D curl components for each shape function.
    - div_Phi: Array (12,) with 3D divergence for each shape function.
    - beta: Matrix (12, 12) of expansion coefficients inv(alpha).
    """
    if nos_selecionados is not None:
        selected_nodes = nos_selecionados
    if matriz_a is not None:
        matrix_a = matriz_a

    P = np.asarray(P, dtype=float)

    if matrix_a is None:
        coords_supp = nodes_coords[selected_nodes] - P
        vecs_supp = nodes_vectors[selected_nodes]
        from scripts.support_nodes_vnmm_3d_12_p1 import build_placement_matrix_3d
        matrix_a = build_placement_matrix_3d(coords_supp, vecs_supp)

    beta = np.linalg.inv(matrix_a)

    if dx_eval == 0.0 and dy_eval == 0.0 and dz_eval == 0.0:
        Phi = beta[0:3, :].copy()
    else:
        Phi_x = beta[0, :] + beta[3, :] * dx_eval + beta[6, :] * dy_eval + beta[9, :] * dz_eval
        Phi_y = beta[1, :] + beta[4, :] * dx_eval + beta[7, :] * dy_eval + beta[10, :] * dz_eval
        Phi_z = beta[2, :] + beta[5, :] * dx_eval + beta[8, :] * dy_eval + beta[11, :] * dz_eval
        Phi = np.vstack([Phi_x, Phi_y, Phi_z])

    curl_Phi = np.empty((3, 12), dtype=float)
    curl_Phi[0, :] = beta[8, :] - beta[10, :]
    curl_Phi[1, :] = beta[9, :] - beta[5, :]
    curl_Phi[2, :] = beta[4, :] - beta[6, :]

    div_Phi = beta[3, :] + beta[7, :] + beta[11, :]

    return Phi, curl_Phi, div_Phi, beta


# Compatibility aliases
funcoes_forma_vnmm_3d_12_p1 = shape_functions_vnmm_3d_12_p1
funcoes_forma_vnmm_3d = shape_functions_vnmm_3d_12_p1
shape_functions_vnmm_3d = shape_functions_vnmm_3d_12_p1

