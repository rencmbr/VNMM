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
    n_theta: int = 12,
    domain_bounds: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Generates polar Gauss-Legendre / trapezoidal quadrature points and weights
    for integrating over a circular disk Omega_s of given center and radius
    (or truncated subdomain Omega_s \cap Omega if domain_bounds is provided):
        integral_{Omega_s \cap Omega} f(x, y) dA = sum_{q} w_q * f(x_q, y_q)

    Parameters:
    - center: Center of disk [x0, y0].
    - radius: Radius of disk r_s.
    - n_r: Number of radial Gauss-Legendre points in [0, r_s].
    - n_theta: Number of uniformly spaced angular points in [0, 2*pi).
    - domain_bounds: Optional bounding box (xmin, xmax, ymin, ymax) to truncate points.

    Returns:
    - points: Array (M, 2) of Cartesian quadrature coordinates.
    - weights: Array (M,) of quadrature weights (including Jacobian r).
    """
    # 1D Gauss-Legendre on [-1, 1] mapped to r in [0, radius]
    xi_r, w_xi = get_gauss_points_weights_1d(n_r)
    # Mapping: r = (radius / 2) * (1 + xi), dr = (radius / 2) * dxi
    r_vals = 0.5 * radius * (1.0 + xi_r)
    w_r = 0.5 * radius * w_xi

    # Uniform midpoint angular points in [0, 2*pi) (avoids placing points on coordinate axes)
    theta_vals = (np.arange(n_theta) + 0.5) * (2.0 * np.pi / float(n_theta))
    w_theta = (2.0 * np.pi) / float(n_theta)

    points = []
    weights = []

    for r, wr in zip(r_vals, w_r):
        # Jacobian determinant is r
        w_circle = wr * w_theta * r
        for th in theta_vals:
            qx = center[0] + r * np.cos(th)
            qy = center[1] + r * np.sin(th)
            if domain_bounds is not None:
                xmin, xmax, ymin, ymax = domain_bounds
                if not ((xmin - 1e-14 <= qx <= xmax + 1e-14) and (ymin - 1e-14 <= qy <= ymax + 1e-14)):
                    continue
            points.append([qx, qy])
            weights.append(w_circle)

    return np.array(points, dtype=float), np.array(weights, dtype=float)


def compute_subdomain_geometric_moments(
    subdomain: MLPGSubdomain2D
) -> Tuple[float, float, float, float, float, float]:
    r"""
    Computes the geometric moments of the active integration domain Omega_s^I \cap Omega
    relative to the subdomain evaluation center x_I:
        S_0  = \iint d\Omega = A_active
        S_x  = \iint (x - x_I) d\Omega
        S_y  = \iint (y - y_I) d\Omega
        S_xx = \iint (x - x_I)^2 d\Omega
        S_yy = \iint (y - y_I)^2 d\Omega
        S_xy = \iint (x - x_I)(y - y_I) d\Omega
    """
    r_s = float(subdomain.radius_s)
    xc, yc = float(subdomain.center[0]), float(subdomain.center[1])
    full_area = np.pi * (r_s ** 2)
    i2_full = 0.25 * np.pi * (r_s ** 4)

    bounds = getattr(subdomain, 'domain_bounds', None)
    if bounds is None:
        return full_area, 0.0, 0.0, i2_full, i2_full, 0.0

    xmin, xmax, ymin, ymax = bounds
    dx_left = xc - xmin
    dx_right = xmax - xc
    dy_bot = yc - ymin
    dy_top = ymax - yc

    # Case 1: Uncut full circular disk
    if (dx_left >= r_s - 1e-12 and dx_right >= r_s - 1e-12 and
        dy_bot >= r_s - 1e-12 and dy_top >= r_s - 1e-12):
        return full_area, 0.0, 0.0, i2_full, i2_full, 0.0

    tol = 1e-11
    is_left = abs(dx_left) < tol
    is_right = abs(dx_right) < tol
    is_bot = abs(dy_bot) < tol
    is_top = abs(dy_top) < tol

    # Case 3 at vertex (orthogonal corner quarter-disk, eta = 1/4)
    if (is_left or is_right) and (is_bot or is_top):
        sign_x = 1.0 if is_left else -1.0
        sign_y = 1.0 if is_bot else -1.0
        S0 = 0.25 * full_area
        Sx = sign_x * (1.0 / 3.0) * (r_s ** 3)
        Sy = sign_y * (1.0 / 3.0) * (r_s ** 3)
        Sxx = 0.0625 * np.pi * (r_s ** 4)
        Syy = Sxx
        Sxy = sign_x * sign_y * 0.125 * (r_s ** 4)
        return S0, Sx, Sy, Sxx, Syy, Sxy

    # Case 3 on flat wall far from corners (exact half-disk, eta = 1/2)
    if is_bot and dx_left >= r_s - 1e-12 and dx_right >= r_s - 1e-12:
        return 0.5 * full_area, 0.0, (2.0 / 3.0) * (r_s ** 3), 0.125 * np.pi * (r_s ** 4), 0.125 * np.pi * (r_s ** 4), 0.0
    if is_top and dx_left >= r_s - 1e-12 and dx_right >= r_s - 1e-12:
        return 0.5 * full_area, 0.0, -(2.0 / 3.0) * (r_s ** 3), 0.125 * np.pi * (r_s ** 4), 0.125 * np.pi * (r_s ** 4), 0.0
    if is_left and dy_bot >= r_s - 1e-12 and dy_top >= r_s - 1e-12:
        return 0.5 * full_area, (2.0 / 3.0) * (r_s ** 3), 0.0, 0.125 * np.pi * (r_s ** 4), 0.125 * np.pi * (r_s ** 4), 0.0
    if is_right and dy_bot >= r_s - 1e-12 and dy_top >= r_s - 1e-12:
        return 0.5 * full_area, -(2.0 / 3.0) * (r_s ** 3), 0.0, 0.125 * np.pi * (r_s ** 4), 0.125 * np.pi * (r_s ** 4), 0.0

    # Case 2: Interior node intersecting single boundary wall
    cross_left = dx_left < r_s - tol and dx_right >= r_s and dy_bot >= r_s and dy_top >= r_s
    cross_right = dx_right < r_s - tol and dx_left >= r_s and dy_bot >= r_s and dy_top >= r_s
    cross_bot = dy_bot < r_s - tol and dy_top >= r_s and dx_left >= r_s and dx_right >= r_s
    cross_top = dy_top < r_s - tol and dy_bot >= r_s and dx_left >= r_s and dx_right >= r_s

    def cap_moments(d: float) -> Tuple[float, float, float, float]:
        d_c = max(0.0, min(r_s, d))
        th = np.arccos(d_c / r_s)
        a_cap = (r_s ** 2) * (th - (d_c / r_s) * np.sqrt(max(0.0, 1.0 - (d_c / r_s) ** 2)))
        x0 = np.sqrt(max(0.0, r_s ** 2 - d_c ** 2))
        s1_cap = (2.0 / 3.0) * (x0 ** 3)
        s2_par = i2_full - (0.25 * (r_s ** 4) * th - (d_c * x0 / 12.0) * (5.0 * (r_s ** 2) - 2.0 * (d_c ** 2)))
        s2_perp = i2_full - (0.25 * (r_s ** 4) * th + (d_c * x0 / 4.0) * ((r_s ** 2) - 2.0 * (d_c ** 2)))
        return a_cap, s1_cap, s2_par, s2_perp

    if cross_bot:
        a_cap, s1_cap, s2_par, s2_perp = cap_moments(dy_bot)
        s0 = full_area - a_cap
        return s0, 0.0, s1_cap, s2_par, s2_perp, 0.0
    if cross_top:
        a_cap, s1_cap, s2_par, s2_perp = cap_moments(dy_top)
        s0 = full_area - a_cap
        return s0, 0.0, -s1_cap, s2_par, s2_perp, 0.0
    if cross_left:
        a_cap, s1_cap, s2_par, s2_perp = cap_moments(dx_left)
        s0 = full_area - a_cap
        return s0, s1_cap, 0.0, s2_perp, s2_par, 0.0
    if cross_right:
        a_cap, s1_cap, s2_par, s2_perp = cap_moments(dx_right)
        s0 = full_area - a_cap
        return s0, -s1_cap, 0.0, s2_perp, s2_par, 0.0

    # General truncated subdomains (Case 3B, Case 2B near corners):
    # Evaluated via Gauss-Legendre quadrature clipped to domain bounds
    n_r = 16
    n_th = 64
    r_gl, w_r_gl = np.polynomial.legendre.leggauss(n_r)
    r_pts = 0.5 * r_s * (r_gl + 1.0)
    w_r = 0.5 * r_s * w_r_gl

    th_gl, w_th_gl = np.polynomial.legendre.leggauss(n_th)
    th_pts = np.pi * (th_gl + 1.0)
    w_th = np.pi * w_th_gl

    S0 = 0.0
    Sx = 0.0
    Sy = 0.0
    Sxx = 0.0
    Syy = 0.0
    Sxy = 0.0

    for r, wr in zip(r_pts, w_r):
        w_circle = wr * r
        for th, wt in zip(th_pts, w_th):
            dx = r * np.cos(th)
            dy = r * np.sin(th)
            x = xc + dx
            y = yc + dy
            if (xmin - 1e-13 <= x <= xmax + 1e-13) and (ymin - 1e-13 <= y <= ymax + 1e-13):
                w = w_circle * wt
                S0 += w
                Sx += w * dx
                Sy += w * dy
                Sxx += w * (dx ** 2)
                Syy += w * (dy ** 2)
                Sxy += w * (dx * dy)

    a_act = getattr(subdomain, 'area_active', S0)
    if S0 > 0 and a_act > 0:
        scale = a_act / S0
        S0 = a_act
        Sx *= scale
        Sy *= scale
        Sxx *= scale
        Syy *= scale
        Sxy *= scale

    return float(S0), float(Sx), float(Sy), float(Sxx), float(Syy), float(Sxy)


def compute_subdomain_matrices_analytic(
    subdomain: MLPGSubdomain2D,
    mu_r: float = 1.0,
    epsilon_r: float = 1.0,
    s_div: float = 4.0,
    alpha_stab: float = 0.0,
    matrix_format: str = "1x6"
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Computes local stiffness K_local and mass M_local using exact closed-form analytical
    integration (including full moments for truncated subdomains) over Omega_s^I \cap Omega.

    Parameters:
    - subdomain: MLPGSubdomain2D instance.
    - mu_r, epsilon_r, s_div: Physical/stabilization parameters.
    - alpha_stab: Orthogonal energy stabilization parameter for zero-curl zero-div strain modes.
    - matrix_format: '1x6' (row associated with central node test function)
                     or '6x6' (full local Galerkin interaction matrix).

    Returns:
    - K_local: Array shape (6,) if matrix_format='1x6', or (6, 6) if matrix_format='6x6'.
    - M_local: Array shape (6,) if matrix_format='1x6', or (6, 6) if matrix_format='6x6'.
    """
    stencil = subdomain.stencil_nodes
    matches = np.where(stencil == subdomain.node_idx)[0]
    if len(matches) == 0:
        raise ValueError(
            f"Node {subdomain.node_idx} is not in its own stencil: {stencil}"
        )
    i0 = matches[0]

    beta = subdomain.beta

    # Geometric moments of active domain Omega_s^I \cap Omega
    S0, Sx, Sy, Sxx, Syy, Sxy = compute_subdomain_geometric_moments(subdomain)

    # Curl and Div vectors of all 6 shape functions (strictly constant on disk)
    curls = subdomain.eval_shape_curl()  # shape (6,)
    divs = subdomain.eval_shape_div()    # shape (6,)

    # 6x6 local stiffness matrix
    k_curl = (1.0 / mu_r) * np.outer(curls, curls) * S0
    k_div = (s_div / mu_r) * np.outer(divs, divs) * S0
    k_local_6x6 = k_curl + k_div

    # Orthogonal energy stabilization (penalizes shear and deviatoric strain modes)
    if alpha_stab > 0.0:
        shear = beta[4, :] + beta[3, :]
        hyp = beta[2, :] - beta[5, :]
        k_stab = (alpha_stab / mu_r) * (np.outer(shear, shear) + np.outer(hyp, hyp)) * S0
        k_local_6x6 += k_stab

    # 6x6 local mass matrix with full geometric moments
    b0 = beta[0, :]
    b1 = beta[1, :]
    b2 = beta[2, :]
    b3 = beta[3, :]
    b4 = beta[4, :]
    b5 = beta[5, :]

    P0 = np.outer(b0, b0) + np.outer(b1, b1)
    Px = np.outer(b0, b2) + np.outer(b2, b0) + np.outer(b1, b4) + np.outer(b4, b1)
    Py = np.outer(b0, b3) + np.outer(b3, b0) + np.outer(b1, b5) + np.outer(b5, b1)
    Pxx = np.outer(b2, b2) + np.outer(b4, b4)
    Pyy = np.outer(b3, b3) + np.outer(b5, b5)
    Pxy = np.outer(b2, b3) + np.outer(b3, b2) + np.outer(b4, b5) + np.outer(b5, b4)

    m_local_6x6 = epsilon_r * (
        P0 * S0 + Px * Sx + Py * Sy + Pxx * Sxx + Pyy * Syy + Pxy * Sxy
    )

    if matrix_format == "6x6":
        return k_local_6x6, m_local_6x6
    else:
        # Row format associated with central node test function W_I = N_{i0}^I
        return k_local_6x6[i0, :], m_local_6x6[i0, :]


def compute_subdomain_matrices_numeric(
    subdomain: MLPGSubdomain2D,
    mu_r: float = 1.0,
    epsilon_r: float = 1.0,
    s_div: float = 4.0,
    matrix_format: str = "1x6",
    n_r: int = 4,
    n_theta: int = 12
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Computes local stiffness K_local and mass M_local via numerical polar Gauss quadrature
    over Omega_s^I \cap Omega (used for validation).
    """
    pts, weights = get_disk_quadrature_points_2d(
        subdomain.center,
        subdomain.radius_s,
        n_r=n_r,
        n_theta=n_theta,
        domain_bounds=getattr(subdomain, 'domain_bounds', None)
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
        subdomain.center,
        subdomain.radius_s,
        n_r=n_r,
        n_theta=n_theta,
        domain_bounds=getattr(subdomain, 'domain_bounds', None)
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
    alpha_stab: float = 0.0,
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
    - alpha_stab: Orthogonal energy stabilization parameter for zero-curl zero-div strain modes (default: 0.0).
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
                alpha_stab=alpha_stab,
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
                pts, wts = get_disk_quadrature_points_2d(
                    sub.center,
                    sub.radius_s,
                    n_r=n_r,
                    n_theta=n_theta,
                    domain_bounds=getattr(sub, 'domain_bounds', None)
                )
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
