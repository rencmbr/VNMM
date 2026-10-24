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
    s_div: float = 4.0,
    matrix_format: str = "1x6"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes local stiffness K_local and mass M_local using exact analytical integration
    over the circular subdomain Omega_s^I.

    Parameters:
    - subdomain: MLPGSubdomain2D instance.
    - mu_r, epsilon_r, s_div: Physical/stabilization parameters.
    - matrix_format: '1x6' (row associated with central node test function)
                     or '6x6' (full local Galerkin interaction matrix).

    Returns:
    - K_local: Array shape (6,) if matrix_format='1x6', or (6, 6) if matrix_format='6x6'.
    - M_local: Array shape (6,) if matrix_format='1x6', or (6, 6) if matrix_format='6x6'.
    """
    r_s = subdomain.radius_s
    area = np.pi * (r_s ** 2)
    i2 = 0.25 * np.pi * (r_s ** 4)

    stencil = subdomain.stencil_nodes
    matches = np.where(stencil == subdomain.node_idx)[0]
    if len(matches) == 0:
        raise ValueError(
            f"Node {subdomain.node_idx} is not in its own stencil: {stencil}"
        )
    i0 = matches[0]

    beta = subdomain.beta

    # Curl and Div vectors of all 6 shape functions (strictly constant on disk)
    curls = subdomain.eval_shape_curl()  # shape (6,)
    divs = subdomain.eval_shape_div()    # shape (6,)

    if matrix_format == "6x6":
        # Full 6x6 symmetric positive-semidefinite stiffness and mass matrices
        k_curl = (1.0 / mu_r) * np.outer(curls, curls) * area
        k_div = (s_div / mu_r) * np.outer(divs, divs) * area
        k_local = k_curl + k_div

        c0 = beta[0:2, :].T @ beta[0:2, :]
        c1 = beta[2:6, :].T @ beta[2:6, :]
        m_local = epsilon_r * (c0 * area + c1 * i2)
    else:
        # 1x6 row associated with central node test function W_I = N_{i0}^I
        curl_i0 = curls[i0]
        div_i0 = divs[i0]

        k_curl = (1.0 / mu_r) * (curls * curl_i0) * area
        k_div = (s_div / mu_r) * (divs * div_i0) * area
        k_local = k_curl + k_div

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
    matrix_format: str = "1x6",
    n_r: int = 4,
    n_theta: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes local stiffness K_local and mass M_local via numerical polar Gauss quadrature
    over Omega_s^I (used for validation).
    """
    pts, weights = get_disk_quadrature_points_2d(
        subdomain.center, subdomain.radius_s, n_r=n_r, n_theta=n_theta
    )

    stencil = subdomain.stencil_nodes
    matches = np.where(stencil == subdomain.node_idx)[0]
    i0 = matches[0]

    # Shape functions at all quadrature points: shape (M, 2, 6)
    N_all = subdomain.eval_shape_functions(pts)
    curls = subdomain.eval_shape_curl()
    divs = subdomain.eval_shape_div()
    total_area = float(np.sum(weights))

    if matrix_format == "6x6":
        k_curl = (1.0 / mu_r) * np.outer(curls, curls) * total_area
        k_div = (s_div / mu_r) * np.outer(divs, divs) * total_area
        k_local = k_curl + k_div

        # N_all: (M, 2, 6) -> dot product between all pairs of shape functions
        dot_ab = np.einsum("mqa,mqb->mab", N_all, N_all)
        m_local = epsilon_r * np.sum(weights[:, np.newaxis, np.newaxis] * dot_ab, axis=0)
    else:
        N_test = N_all[:, :, i0]  # shape (M, 2)
        curl_i0 = curls[i0]
        div_i0 = divs[i0]

        k_curl = (1.0 / mu_r) * (curls * curl_i0) * total_area
        k_div = (s_div / mu_r) * (divs * div_i0) * total_area
        k_local = k_curl + k_div

        dot_products = np.einsum("mqk,mq->mk", N_all, N_test)
        m_local = epsilon_r * np.sum(weights[:, np.newaxis] * dot_products, axis=0)

    return k_local, m_local


def compute_subdomain_source_vector(
    subdomain: MLPGSubdomain2D,
    source_func: Callable[[np.ndarray], np.ndarray],
    n_r: int = 4,
    n_theta: int = 12
) -> Union[float, np.ndarray]:
    """
    Computes the source term integral for node I:
        F_I = integral_{Omega_s^I} J_s(x) . N_{local_I}^I(x) dA
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
    assembly_mode: str = "row_petrov_galerkin",
    source_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    n_r: int = 4,
    n_theta: int = 12,
    apply_pec: bool = False
) -> Tuple[csr_matrix, csr_matrix, Optional[np.ndarray]]:
    """
    Assembles global sparse stiffness matrix K and mass matrix M (and optional force vector F)
    for the 2D Vector Nodal Meshless Method via Meshless Local Petrov-Galerkin (VNMM-MLPG 2D).

    Parameters:
    - subdomains: List of N MLPGSubdomain2D objects.
    - mu_r: Relative magnetic permeability (default: 1.0).
    - epsilon_r: Relative electric permittivity (default: 1.0).
    - s_div: Divergence penalty stabilization parameter (default: 4.0).
    - integration_mode: 'analytic' (exact closed-form) or 'numeric' (polar Gauss).
    - assembly_mode:
        * 'row_petrov_galerkin' (default): each node I generates row I from Omega_s^I.
        * 'subdomain_galerkin': sums 6x6 symmetric positive-semidefinite matrices over Omega_s^I.
    - source_func: Optional external source current density J_s(points) -> (M, 2).
    - n_r, n_theta: Polar quadrature subdivisions.
    - apply_pec: If True, sets identity rows for boundary nodes with is_pec=True.

    Returns:
    - K: Global sparse stiffness matrix in CSR format (N, N).
    - M: Global sparse mass matrix in CSR format (N, N).
    - F: Global force vector (N,) if source_func is provided, else None.
    """
    N = len(subdomains)

    rows = []
    cols = []
    data_k = []
    data_m = []

    F = np.zeros(N, dtype=float) if source_func is not None else None
    mat_format = "6x6" if assembly_mode == "subdomain_galerkin" else "1x6"

    for i, sub in enumerate(subdomains):
        if apply_pec and sub.is_pec:
            rows.append(i)
            cols.append(i)
            data_k.append(1.0)
            data_m.append(0.0)
            if F is not None:
                F[i] = 0.0
            continue

        if integration_mode == "analytic":
            k_loc, m_loc = compute_subdomain_matrices_analytic(
                subdomain=sub,
                mu_r=mu_r,
                epsilon_r=epsilon_r,
                s_div=s_div,
                matrix_format=mat_format
            )
        elif integration_mode == "numeric":
            k_loc, m_loc = compute_subdomain_matrices_numeric(
                subdomain=sub,
                mu_r=mu_r,
                epsilon_r=epsilon_r,
                s_div=s_div,
                matrix_format=mat_format,
                n_r=n_r,
                n_theta=n_theta
            )
        else:
            raise ValueError(f"Unknown integration_mode: '{integration_mode}'")

        if assembly_mode == "subdomain_galerkin":
            st = sub.stencil_nodes
            for a in range(6):
                Ia = st[a]
                for b in range(6):
                    Ib = st[b]
                    rows.append(Ia)
                    cols.append(Ib)
                    data_k.append(k_loc[a, b])
                    data_m.append(m_loc[a, b])
            if source_func is not None:
                # 6-vector source contribution: integral J_s . N_a dA
                pts, wts = get_disk_quadrature_points_2d(sub.center, sub.radius_s, n_r=n_r, n_theta=n_theta)
                N_all = sub.eval_shape_functions(pts)
                J_vals = np.asarray(source_func(pts), dtype=float)
                if J_vals.ndim == 1:
                    J_vals = np.tile(J_vals, (len(pts), 1))
                J_dot_N = np.einsum("mq,mqa->ma", J_vals, N_all)
                f_6 = np.sum(wts[:, np.newaxis] * J_dot_N, axis=0)
                for a in range(6):
                    F[st[a]] += f_6[a]
        else:
            for k_idx, global_col in enumerate(sub.stencil_nodes):
                rows.append(i)
                cols.append(global_col)
                data_k.append(k_loc[k_idx])
                data_m.append(m_loc[k_idx])

            if source_func is not None:
                F[i] = compute_subdomain_source_vector(
                    subdomain=sub,
                    source_func=source_func,
                    n_r=n_r,
                    n_theta=n_theta
                )

    from scipy.sparse import coo_matrix
    K = coo_matrix((data_k, (rows, cols)), shape=(N, N), dtype=float).tocsr()
    M = coo_matrix((data_m, (rows, cols)), shape=(N, N), dtype=float).tocsr()

    if assembly_mode == "subdomain_galerkin":
        K = 0.5 * (K + K.T)
        M = 0.5 * (M + M.T)

    return K, M, F
