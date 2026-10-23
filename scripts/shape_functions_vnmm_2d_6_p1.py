import numpy as np


def shape_functions_vnmm_2d_6_p1(
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
    at evaluation point P for 2D VNMM with complete linear basis P1 (6 terms).
    
    Mathematical Formulation:
    - Complete linear polynomial vector basis:
        P1 = <[1, 0]^T, [0, 1]^T, [x, 0]^T, [y, 0]^T, [0, x]^T, [0, y]^T>
    - Local shape function (dx = x - x_P, dy = y - y_P):
        N_i(dx, dy) = beta_1i [1, 0]^T + beta_2i [0, 1]^T + beta_3i [dx, 0]^T 
                    + beta_4i [dy, 0]^T + beta_5i [0, dx]^T + beta_6i [0, dy]^T
    - Collocation condition: N_i(x_k) . t_k = delta_ik => A * beta = I_6 => beta = inv(A)
    - At evaluation point P (dx = 0, dy = 0):
        N_i(P) = [beta_1i, beta_2i]^T
        Phi(P) = [N_1(P), ..., N_6(P)] = beta[0:2, :] (2x6 matrix)
    - Curl of shape functions:
        curl(N_i) = (beta_5i - beta_4i) z_hat
        curl_Phi = beta[4, :] - beta[3, :] (length-6 array)
    
    Parameters:
    - P: Evaluation point coordinates (x, y).
    - nodes_coords: Global nodal coordinates (N, 2).
    - nodes_vectors: Global unit direction vectors (tx, ty) (N, 2).
    - selected_nodes: List/array of 6 selected global node indices.
    - matrix_a: (Optional) Precomputed 6x6 collocation matrix A.
    
    Returns:
    - Phi: Matrix (2, 6) of vector shape functions at evaluation point P.
    - curl_Phi: Array (6,) with the z-component of curl for each shape function.
    - beta: Matrix (6, 6) of local expansion coefficients (inv(A)).
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
    else:
        A = np.asarray(matrix_a, dtype=float)
        
    # Solve A * beta = I_6 => beta = inv(A)
    beta = np.linalg.inv(A)
    
    # Shape functions at P (where dx=0, dy=0)
    Phi = beta[0:2, :]  # shape (2, 6)
    
    # Curl of shape functions: curl(N_i) = (beta_5i - beta_4i) z_hat
    curl_Phi = beta[4, :] - beta[3, :]  # shape (6,)
    
    return Phi, curl_Phi, beta


# Backwards compatibility alias
funcoes_forma_vnmm_2d_6_P1 = shape_functions_vnmm_2d_6_p1
