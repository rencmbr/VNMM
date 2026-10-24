import os
import sys
import numpy as np
from typing import List, Optional, Tuple, Callable, Union
from scipy.sparse import csr_matrix, lil_matrix

# Ensure 2D root and scripts directories are accessible
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
SCRIPTS_DIR = os.path.join(DIR_2D, "scripts")
for p in [DIR_2D, ROOT_DIR, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.mlpg_subdomains import MLPGSubdomain2D, build_mlpg_subdomains_2d
from src.gauss_quadrature_2d import get_gauss_points_weights_1d


def get_disk_quadrature_points_2d(
    center: np.ndarray,
    radius: float,
    n_r: int = 4,
    n_theta: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates polar Gauss-Legendre / trapezoidal quadrature points and weights
    for integrating over a circular disk Omega_s of given center and radius:
        integral_{Omega_s} f(x, y) dA = sum_{q} w_q * f(x_q, y_q)

    Parameters:
    - center: Center of disk [x0, y0].
    - radius: Radius of disk r_s.
    - n_r: Number of radial Gauss-Legendre points in [0, r_s].
    - n_theta: Number of uniformly spaced angular points in [0, 2*pi).

    Returns:
    - points: Array (n_r * n_theta, 2) of Cartesian quadrature coordinates.
    - weights: Array (n_r * n_theta,) of quadrature weights (including Jacobian r).
    """
    # 1D Gauss-Legendre on [-1, 1] mapped to r in [0, radius]
    xi_r, w_xi = get_gauss_points_weights_1d(n_r)
    # Mapping: r = (radius / 2) * (1 + xi), dr = (radius / 2) * dxi
    r_vals = 0.5 * radius * (1.0 + xi_r)
    w_r = 0.5 * radius * w_xi

    # Uniform angular points in [0, 2*pi)
    theta_vals = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    w_theta = (2.0 * np.pi) / float(n_theta)

    points = []
    weights = []

    for r, wr in zip(r_vals, w_r):
        # Jacobian determinant is r
        w_circle = wr * w_theta * r
        for th in theta_vals:
            qx = center[0] + r * np.cos(th)
            qy = center[1] + r * np.sin(th)
            points.append([qx, qy])
            weights.append(w_circle)

    return np.array(points, dtype=float), np.array(weights, dtype=float)


def compute_subdomain_matrices_analytic(
    subdomain: MLPGSubdomain2D,
    mu_r: float = 1.0,
    epsilon_r: float = 1.0,
    s_div: float = 4.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes local stiffness K_local (1, 6) and mass M_local (1, 6) vectors
    using exact analytical integration over the circular subdomain Omega_s^I.

    Mathematical Formulation (Local Galerkin MLPG6 with complete linear basis P1):
    - Test function: W_I(x) = N_{local_I}^I(x) (local shape function associated with node I).
    - Subdomain area: Area = pi * r_s^2.
    - Second moment of disk: I_2 = pi * r_s^4 / 4.
    - In local coordinates (dx = x - x_I, dy = y - y_I):
        * curl(N_k)_z = beta[4, k] - beta[3, k]  (strictly constant on disk)
        * div(N_k)    = beta[2, k] + beta[5, k]  (strictly constant on disk)
        * K_curl_k    = (1 / mu_r) * curl(N_k)_z * curl(N_{local_I})_z * Area
        * K_div_k     = (s_div / mu_r) * div(N_k) * div(N_{local_I}) * Area
        * M_k         = epsilon_r * [ (beta[0,k]*beta[0,i0] + beta[1,k]*beta[1,i0]) * Area
                                    + (beta[2,k]*beta[2,i0] + beta[3,k]*beta[3,i0]
                                     + beta[4,k]*beta[4,i0] + beta[5,k]*beta[5,i0]) * I_2 ]

    Returns:
    - K_local: Array (6,) of stiffness contributions.
    - M_local: Array (6,) of mass contributions.
    """
    r_s = subdomain.radius_s
    area = np.pi * (r_s ** 2)
    i2 = 0.25 * np.pi * (r_s ** 4)

    # Local index of central node I within its own 6-node stencil
    stencil = subdomain.stencil_nodes
    matches = np.where(stencil == subdomain.node_idx)[0]
    if len(matches) == 0:
        raise ValueError(
            f"Node {subdomain.node_idx} is not in its own stencil: {stencil}"
        )
    i0 = matches[0]

    beta = subdomain.beta

    # 1. Curl and Div vectors of all 6 shape functions (strictly constant on disk)
    curls = subdomain.eval_shape_curl()  # shape (6,)
    divs = subdomain.eval_shape_div()    # shape (6,)

    curl_i0 = curls[i0]
    div_i0 = divs[i0]

    # Stiffness vector (1x6)
    k_curl = (1.0 / mu_r) * (curls * curl_i0) * area
    k_div = (s_div / mu_r) * (divs * div_i0) * area
    k_local = k_curl + k_div

    # 2. Mass vector (1x6)
    c0 = beta[0, :] * beta[0, i0] + beta[1, :] * beta[1, i0]
    c1 = (
        beta[2, :] * beta[2, i0] + beta[3, :] * beta[3, i0] +
        beta[4, :] * beta[4, i0] + beta[5, :] * beta[5, i0]
    )
    m_local = epsilon_r * (c0 * area + c1 * i2)

    return k_local, m_local


def compute_subdomain_matrices_numeric(
    subdomain: MLPGSubdomain2D,
    mu_r: float = 1.0,
    epsilon_r: float = 1.0,
    s_div: float = 4.0,
    n_r: int = 4,
    n_theta: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes local stiffness K_local (6,) and mass M_local (6,) vectors
    via numerical polar Gauss quadrature over Omega_s^I (used for validation).
    """
    pts, weights = get_disk_quadrature_points_2d(
        subdomain.center, subdomain.radius_s, n_r=n_r, n_theta=n_theta
    )

    stencil = subdomain.stencil_nodes
    matches = np.where(stencil == subdomain.node_idx)[0]
    i0 = matches[0]

    # Shape functions at all quadrature points: shape (M, 2, 6)
    N_all = subdomain.eval_shape_functions(pts)
    N_test = N_all[:, :, i0]  # shape (M, 2)

    # Constant curl and div
    curls = subdomain.eval_shape_curl()
    divs = subdomain.eval_shape_div()
    curl_i0 = curls[i0]
    div_i0 = divs[i0]

    total_area = float(np.sum(weights))

    # K_curl and K_div
    k_curl = (1.0 / mu_r) * (curls * curl_i0) * total_area
    k_div = (s_div / mu_r) * (divs * div_i0) * total_area
    k_local = k_curl + k_div

    # Mass matrix via quadrature sum: sum_q w_q * (N_k(x_q) . N_test(x_q))
    # N_all: (M, 2, 6), N_test: (M, 2)
    # Dot product for each k: dot_k = sum_c N_all[:, c, k] * N_test[:, c] -> (M, 6)
    dot_products = np.einsum("mqk,mq->mk", N_all, N_test)
    m_local = epsilon_r * np.sum(weights[:, np.newaxis] * dot_products, axis=0)

    return k_local, m_local


def compute_subdomain_source_vector(
    subdomain: MLPGSubdomain2D,
    source_func: Callable[[np.ndarray], np.ndarray],
    n_r: int = 4,
    n_theta: int = 12
) -> float:
    """
    Computes the source term integral F_I for node I:
        F_I = integral_{Omega_s^I} J_s(x) . N_{local_I}^I(x) dA

    Parameters:
    - subdomain: MLPGSubdomain2D instance.
    - source_func: Function J_s(points) accepting (M, 2) and returning (M, 2).
    - n_r, n_theta: Number of polar quadrature points.

    Returns:
    - F_I: Scalar load value for row I.
    """
    pts, weights = get_disk_quadrature_points_2d(
        subdomain.center, subdomain.radius_s, n_r=n_r, n_theta=n_theta
    )

    stencil = subdomain.stencil_nodes
    matches = np.where(stencil == subdomain.node_idx)[0]
    i0 = matches[0]

    N_all = subdomain.eval_shape_functions(pts)  # (M, 2, 6)
    N_test = N_all[:, :, i0]                     # (M, 2)

    J_vals = np.asarray(source_func(pts), dtype=float)
    if J_vals.ndim == 1:
        J_vals = np.tile(J_vals, (len(pts), 1))

    # Dot product J_s(x_q) . N_test(x_q)
    integrand = np.sum(J_vals * N_test, axis=1)
    F_I = float(np.sum(weights * integrand))
    return F_I


def assemble_mlpg_matrices_2d(
    subdomains: List[MLPGSubdomain2D],
    mu_r: float = 1.0,
    epsilon_r: float = 1.0,
    s_div: float = 4.0,
    integration_mode: str = "analytic",
    source_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_r: int = 4,
    n_theta: int = 12,
    apply_pec: bool = False
) -> Tuple[csr_matrix, csr_matrix, Optional[np.ndarray]]:
    """
    Assembles global sparse stiffness matrix K and mass matrix M (and optional force vector F)
    for the 2D Vector Nodal Meshless Method via Meshless Local Petrov-Galerkin (VNMM-MLPG 2D).

    Each subdomain Omega_s^I produces exactly one equation (row I of the global system),
    coupling node I with the 6 nodes in its fixed support stencil:
        K(I, stencil_nodes[k]) += K_local[k]
        M(I, stencil_nodes[k]) += M_local[k]

    Parameters:
    - subdomains: List of N MLPGSubdomain2D objects.
    - mu_r: Relative magnetic permeability (default: 1.0).
    - epsilon_r: Relative electric permittivity (default: 1.0).
    - s_div: Divergence penalty stabilization parameter (default: 4.0).
    - integration_mode: 'analytic' (exact, O(1) closed-form) or 'numeric' (polar Gauss).
    - source_func: Optional external source current density J_s(points) -> (M, 2).
    - n_r, n_theta: Polar quadrature subdivisions (used if integration_mode='numeric' or for F).
    - apply_pec: If True, sets identity rows for boundary nodes with is_pec=True.

    Returns:
    - K: Global sparse stiffness matrix in CSR format (N, N).
    - M: Global sparse mass matrix in CSR format (N, N).
    - F: Global force vector (N,) if source_func is provided, else None.
    """
    N = len(subdomains)

    # Pre-allocate COO lists for high assembly performance
    rows = []
    cols = []
    data_k = []
    data_m = []

    F = np.zeros(N, dtype=float) if source_func is not None else None

    for i, sub in enumerate(subdomains):
        if apply_pec and sub.is_pec:
            # Identity row for PEC boundary node: K(i, i) = 1, M(i, :) = 0, F(i) = 0
            rows.append(i)
            cols.append(i)
            data_k.append(1.0)
            data_m.append(0.0)
            if F is not None:
                F[i] = 0.0
            continue

        # Compute local 1x6 contributions
        if integration_mode == "analytic":
            k_loc, m_loc = compute_subdomain_matrices_analytic(
                subdomain=sub,
                mu_r=mu_r,
                epsilon_r=epsilon_r,
                s_div=s_div
            )
        elif integration_mode == "numeric":
            k_loc, m_loc = compute_subdomain_matrices_numeric(
                subdomain=sub,
                mu_r=mu_r,
                epsilon_r=epsilon_r,
                s_div=s_div,
                n_r=n_r,
                n_theta=n_theta
            )
        else:
            raise ValueError(f"Unknown integration_mode: '{integration_mode}'")

        # Scatter to global COO arrays
        for k_idx, global_col in enumerate(sub.stencil_nodes):
            rows.append(i)
            cols.append(global_col)
            data_k.append(k_loc[k_idx])
            data_m.append(m_loc[k_idx])

        # Evaluate source term if present
        if source_func is not None:
            F[i] = compute_subdomain_source_vector(
                subdomain=sub,
                source_func=source_func,
                n_r=n_r,
                n_theta=n_theta
            )

    # Construct CSR sparse matrices
    from scipy.sparse import coo_matrix
    K_coo = coo_matrix((data_k, (rows, cols)), shape=(N, N), dtype=float)
    M_coo = coo_matrix((data_m, (rows, cols)), shape=(N, N), dtype=float)

    K = K_coo.tocsr()
    M = M_coo.tocsr()

    return K, M, F
