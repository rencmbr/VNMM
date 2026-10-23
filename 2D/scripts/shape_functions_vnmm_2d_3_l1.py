import numpy as np


def shape_functions_vnmm_2d_3_l1(
    P, 
    nodes_coords, 
    nodes_vectors, 
    selected_nodes, 
    matrix_a=None,
    # Compatibility aliases
    nos_selecionados=None,
    matriz_a=None
):
    """
    Computes vector shape functions N_i and their curl (curl N_i) 
    at evaluation point P for 2D VNMM with 3 support nodes (L1 basis).
    
    Mathematical Formulation:
    - 1st-order polynomial basis: L^1 = <[1, 0]^T, [0, 1]^T, [y, -x]^T>
    - Shape function N_i(x, y) = beta_1i * [1, 0]^T + beta_2i * [0, 1]^T + beta_3i * [y, -x]^T
    - Collocation condition: N_i(x_k, y_k) . t_k = delta_ik  =>  A * beta_i = L_i
    - Coefficient matrix: beta = [beta_1, beta_2, beta_3] = inv(A)
    - At evaluation point P (origin of local system x=0, y=0):
        N_i(P) = [beta_1i, beta_2i]^T
        Phi(P) = [N_1(P), N_2(P), N_3(P)] (2x3 matrix)
    - Curl of N_i:
        curl(N_i) = [0, 0, -2 * beta_3i]^T  (z-component: -2 * beta_3i)
        curl_Phi = [-2 * beta_31, -2 * beta_32, -2 * beta_33] (length 3 array)
    
    Parameters:
    - P: Evaluation point coordinates (x, y).
    - nodes_coords: Global nodal coordinates (N, 2).
    - nodes_vectors: Global unit direction vectors (tx, ty) (N, 2).
    - selected_nodes: 3 selected global support node indices.
    - matrix_a: (Optional) Precomputed 3x3 collocation matrix A.
    
    Returns:
    - Phi: Matrix (2, 3) of vector shape functions at evaluation point P.
    - curl_Phi: Array (3,) containing the z-component of curl for each shape function.
    - beta: Matrix (3, 3) of local expansion coefficients (inv(A)).
    """
    if nos_selecionados is not None:
        selected_nodes = nos_selecionados
    if matriz_a is not None:
        matrix_a = matriz_a

    P = np.asarray(P, dtype=float)
    
    # If matrix A is not provided, compute from selected nodes
    if matrix_a is None:
        local_coords = nodes_coords[selected_nodes] - P
        vecs = nodes_vectors[selected_nodes]
        
        local_x = local_coords[:, 0]
        local_y = local_coords[:, 1]
        tx = vecs[:, 0]
        ty = vecs[:, 1]
        
        A = np.empty((3, 3), dtype=float)
        A[:, 0] = tx
        A[:, 1] = ty
        A[:, 2] = local_y * tx - local_x * ty
    else:
        A = np.asarray(matrix_a, dtype=float)
        
    # Solve A * beta = I_3 => beta = inv(A)
    beta = np.linalg.inv(A)
    
    # Shape functions at P (where local_x=0, local_y=0)
    Phi = beta[0:2, :]  # shape (2, 3)
    
    # Curl of shape functions: curl(N_i) = -2 * beta_3i in z-direction
    curl_Phi = -2.0 * beta[2, :]  # shape (3,)
    
    return Phi, curl_Phi, beta


# Backwards compatibility alias
funcoes_forma_vnmm_2d_3_L1 = shape_functions_vnmm_2d_3_l1
