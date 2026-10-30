import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.spatial import KDTree

# Ensure 2D root and scripts directories are accessible
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
SCRIPTS_DIR = os.path.join(DIR_2D, "scripts")
for p in [DIR_2D, ROOT_DIR, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1


def quartic_spline_weight(r: np.ndarray, r_s: float) -> np.ndarray:
    """
    Computes the standard C1 quartic spline radial weight function with compact support in [0, r_s]:
        w(r) = 1 - 6*(r/r_s)^2 + 8*(r/r_s)^3 - 3*(r/r_s)^4,   for r <= r_s
        w(r) = 0,                                            for r > r_s

    Properties:
        w(0) = 1,  w(r_s) = 0,  w'(0) = 0,  w'(r_s) = 0.
    """
    r_arr = np.asarray(r, dtype=float)
    s = r_arr / r_s
    mask = (s >= 0.0) & (s <= 1.0)

    # Use Horner's rule on valid mask
    s_m = s[mask]
    w_vals = np.zeros_like(r_arr)
    w_vals[mask] = 1.0 + s_m * s_m * (-6.0 + s_m * (8.0 - 3.0 * s_m))
    return w_vals


def quartic_spline_derivative(r: np.ndarray, r_s: float) -> np.ndarray:
    """
    Computes the first radial derivative w'(r) of the C1 quartic spline:
        w'(r) = (-12*s + 24*s^2 - 12*s^3) / r_s = -12*s*(1 - s)^2 / r_s,   for 0 <= r <= r_s
        w'(r) = 0,                                                         for r > r_s
    where s = r / r_s.
    """
    r_arr = np.asarray(r, dtype=float)
    s = r_arr / r_s
    mask = (s >= 0.0) & (s <= 1.0)

    s_m = s[mask]
    dw_vals = np.zeros_like(r_arr)
    dw_vals[mask] = (-12.0 * s_m * (1.0 - s_m) ** 2) / r_s
    return dw_vals


def eval_test_function_2d(
    points: np.ndarray, 
    center: np.ndarray, 
    director: np.ndarray, 
    r_s: float
) -> np.ndarray:
    """
    Evaluates the vector test function W_I(x) = w_I(||x - x_I||) * t_I at multiple points.

    Parameters:
    - points: Coordinates (M, 2) or (2,) where the test function is evaluated.
    - center: Center coordinates of node I, (2,).
    - director: Unit director vector t_I = [tx, ty], (2,).
    - r_s: Radius of the local integration subdomain Omega_s^I.

    Returns:
    - W: Array of vector values, shape (M, 2) or (2,).
    """
    pts = np.atleast_2d(points)
    diff = pts - center
    r = np.linalg.norm(diff, axis=1)
    w = quartic_spline_weight(r, r_s)
    W = w[:, np.newaxis] * director[np.newaxis, :]
    return W if np.ndim(points) == 2 else W[0]


def eval_test_function_curl_2d(
    points: np.ndarray, 
    center: np.ndarray, 
    director: np.ndarray, 
    r_s: float
) -> np.ndarray:
    """
    Evaluates the z-component of curl of the vector test function:
        (curl W_I)_z = dW_y/dx - dW_x/dy = w'(r) * (dx * ty - dy * tx) / r

    For r = 0 or r >= r_s, (curl W_I)_z = 0.
    """
    pts = np.atleast_2d(points)
    diff = pts - center
    dx = diff[:, 0]
    dy = diff[:, 1]
    r = np.linalg.norm(diff, axis=1)

    dw = quartic_spline_derivative(r, r_s)
    curl_W = np.zeros(len(pts), dtype=float)

    # Avoid division by zero at r = 0
    nonzero = (r > 1e-15) & (r <= r_s)
    if np.any(nonzero):
        r_nz = r[nonzero]
        curl_W[nonzero] = dw[nonzero] * (dx[nonzero] * director[1] - dy[nonzero] * director[0]) / r_nz

    return curl_W if np.ndim(points) == 2 else curl_W[0]


def eval_test_function_div_2d(
    points: np.ndarray, 
    center: np.ndarray, 
    director: np.ndarray, 
    r_s: float
) -> np.ndarray:
    """
    Evaluates the 2D divergence of the vector test function:
        div(W_I) = dW_x/dx + dW_y/dy = w'(r) * (dx * tx + dy * ty) / r

    For r = 0 or r >= r_s, div(W_I) = 0.
    """
    pts = np.atleast_2d(points)
    diff = pts - center
    dx = diff[:, 0]
    dy = diff[:, 1]
    r = np.linalg.norm(diff, axis=1)

    dw = quartic_spline_derivative(r, r_s)
    div_W = np.zeros(len(pts), dtype=float)

    nonzero = (r > 1e-15) & (r <= r_s)
    if np.any(nonzero):
        r_nz = r[nonzero]
        div_W[nonzero] = dw[nonzero] * (dx[nonzero] * director[0] + dy[nonzero] * director[1]) / r_nz

    return div_W if np.ndim(points) == 2 else div_W[0]


@dataclass
class MLPGSubdomain2D:
    """
    Represents an independent local integration subdomain Omega_s^I (disk of radius r_s)
    centered at node x_I, with a strictly fixed support stencil of 6 nodes (Omega_u^I).
    """
    node_idx: int
    center: np.ndarray
    director: np.ndarray
    radius_s: float
    radius_u: float
    stencil_nodes: np.ndarray
    matrix_a: np.ndarray
    beta: np.ndarray
    det_a: float
    is_boundary: bool = False
    is_pec: bool = False

    def contains_point(self, point: np.ndarray) -> bool:
        """Returns True if point is inside Omega_s^I."""
        return np.linalg.norm(point - self.center) <= self.radius_s

    def eval_test_function(self, points: np.ndarray) -> np.ndarray:
        """Evaluates vector test function W_I(x)."""
        return eval_test_function_2d(points, self.center, self.director, self.radius_s)

    def eval_test_curl(self, points: np.ndarray) -> np.ndarray:
        """Evaluates z-component of curl of test function, (curl W_I)_z."""
        return eval_test_function_curl_2d(points, self.center, self.director, self.radius_s)

    def eval_test_div(self, points: np.ndarray) -> np.ndarray:
        """Evaluates 2D divergence of test function, div(W_I)."""
        return eval_test_function_div_2d(points, self.center, self.director, self.radius_s)

    def eval_shape_functions(self, points: np.ndarray) -> np.ndarray:
        """
        Evaluates the 6 vector shape functions N_k^I(x) at points in local coordinates:
            dx = x - center_x, dy = y - center_y
            N_k(x, y) = [beta[0, k] + beta[2, k]*dx + beta[3, k]*dy,
                         beta[1, k] + beta[4, k]*dx + beta[5, k]*dy]^T

        Returns:
        - shape (M, 2, 6) or (2, 6) if single point.
        """
        pts = np.atleast_2d(points)
        diff = pts - self.center
        dx = diff[:, 0]
        dy = diff[:, 1]
        M = len(pts)

        # Vectorized evaluation for all 6 shape functions
        # N_x: (M, 6), N_y: (M, 6)
        N_x = self.beta[0, :][np.newaxis, :] + np.outer(dx, self.beta[2, :]) + np.outer(dy, self.beta[3, :])
        N_y = self.beta[1, :][np.newaxis, :] + np.outer(dx, self.beta[4, :]) + np.outer(dy, self.beta[5, :])

        # Stack into (M, 2, 6)
        N_all = np.stack([N_x, N_y], axis=1)
        return N_all if np.ndim(points) == 2 else N_all[0]

    def eval_shape_curl(self) -> np.ndarray:
        """
        Evaluates the curl of the 6 shape functions inside Omega_s^I.
        Since the basis is complete linear (P1), curl is strictly CONSTANT:
            (curl N_k^I)_z = beta[4, k] - beta[3, k]

        Returns:
        - Array of length 6.
        """
        return self.beta[4, :] - self.beta[3, :]

    def eval_shape_div(self) -> np.ndarray:
        """
        Evaluates the divergence of the 6 shape functions inside Omega_s^I.
        Since the basis is complete linear (P1), divergence is strictly CONSTANT:
            div(N_k^I) = beta[2, k] + beta[5, k]

        Returns:
        - Array of length 6.
        """
        return self.beta[2, :] + self.beta[5, :]


def build_mlpg_subdomains_2d(
    coords: np.ndarray,
    vectors: np.ndarray,
    is_boundary: Optional[np.ndarray] = None,
    alpha_s: float = 0.8,
    r_s_mode: str = "adaptive",
    det_tolerance: Optional[float] = None,
    tol_ref: Optional[float] = None,
    tol_hat: float = 0.80,
    h_ref: float = np.pi / 20.0,
    search_tree: Optional[KDTree] = None,
    K: int = 12,
    step_K: int = 4,
    K_max: Optional[int] = None
) -> List[MLPGSubdomain2D]:
    """
    Constructs the collection of MLPG local subdomains Omega_s^I for all nodes in the cloud.

    Mathematical Formulation:
    - For each node x_I:
        * Local nodal spacing h_I: estimated from nearest neighbor distance
          (in direct correspondence with h_local(x_I) in the EABE paper).
        * Quadrature radius: r_s^I = alpha_s * h_I (if adaptive) or alpha_s * h_mean (if uniform).
        * Scale-invariant determinant qualification: |det(A_hat)| >= Tol_hat,
          which corresponds to |det(A_I)| >= Tol_hat * h^4, using the calibrated baseline
          Tol_hat = 0.80 established in the EABE paper.
        * Support stencil Omega_u^I: 6 nodes selected with admissible det(A_I) >= Tol_det(h).
        * Collocation matrix A_I is inverted to precompute beta_I = inv(A_I).
        * Interpolation radius: r_u^I = max_{k in stencil} ||x_k - x_I||.
        * Guaranteed property: r_u^I > r_s^I across all subdomains.

    Parameters:
    - coords: Array (N, 2) of global nodal coordinates.
    - vectors: Array (N, 2) of unit direction vectors (tx, ty).
    - is_boundary: Boolean array (N,) indicating physical boundary nodes (default: all False).
    - alpha_s: Subdomain radius factor r_s / h (default: 0.8).
    - r_s_mode: 'adaptive' (individual h_I) or 'uniform' (domain-wide mean h).
    - det_tolerance: Fixed determinant threshold; if None, uses quartic scaling.
    - tol_ref: Reference tolerance at h_ref; if specified, tol_det = tol_ref * (mean_h / h_ref)^4.
    - tol_hat: Universal dimensionless determinant threshold (default: 0.80, as in EABE).
    - h_ref: Reference nodal spacing (default: pi / 20).
    - search_tree: Pre-built scipy.spatial.KDTree over coords.
    - K: Initial nearest neighbor candidate pool for support selection (default: 12).
    - step_K: Neighbor expansion step (default: 4).
    - K_max: Maximum neighbor expansion limit.

    Returns:
    - subdomains: List of MLPGSubdomain2D objects of length N.
    """
    coords = np.asarray(coords, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    N = len(coords)

    if is_boundary is None:
        is_boundary = np.zeros(N, dtype=bool)
    else:
        is_boundary = np.asarray(is_boundary, dtype=bool)

    if search_tree is None:
        search_tree = KDTree(coords)

    # Compute distances to nearest neighbor (k=2: index 0 is self, index 1 is closest neighbor)
    nn_dists, _ = search_tree.query(coords, k=2)
    local_h = nn_dists[:, 1]
    mean_h = float(np.mean(local_h))

    # Scale-invariant determinant tolerance: |det(A)| >= Tol_hat * h^4 (or via tol_ref)
    if det_tolerance is not None:
        tol_det_effective = float(det_tolerance)
    elif tol_ref is not None:
        tol_det_effective = float(tol_ref) * ((mean_h / h_ref) ** 4)
    else:
        tol_det_effective = float(tol_hat) * (mean_h ** 4)

    subdomains: List[MLPGSubdomain2D] = []

    for i in range(N):
        center_i = coords[i]
        vector_i = vectors[i]
        bnd_i = bool(is_boundary[i])

        # Subdomain radius r_s
        if r_s_mode == "adaptive":
            r_s = alpha_s * local_h[i]
        else:
            r_s = alpha_s * mean_h

        # Select admissible 6-node stencil using P1 basis
        best_sextet, best_det, best_A, _ = support_nodes_vnmm_2d_6_p1(
            P=center_i,
            nodes_coords=coords,
            nodes_vectors=vectors,
            search_tree=search_tree,
            K=K,
            Tol_det=tol_det_effective,
            adaptive=True,
            step_K=step_K,
            K_max=K_max
        )

        if best_sextet is None or best_A is None:
            raise RuntimeError(
                f"Failed to find non-singular 6-node stencil for node {i} at {center_i}."
            )

        # Invert collocation matrix to get shape function expansion coefficients
        beta_i = np.linalg.inv(best_A)

        # Interpolation support radius r_u
        stencil_coords = coords[best_sextet]
        dists_to_stencil = np.linalg.norm(stencil_coords - center_i, axis=1)
        r_u = float(np.max(dists_to_stencil))

        subdomain = MLPGSubdomain2D(
            node_idx=i,
            center=center_i,
            director=vector_i,
            radius_s=float(r_s),
            radius_u=r_u,
            stencil_nodes=np.array(best_sextet, dtype=int),
            matrix_a=best_A,
            beta=beta_i,
            det_a=float(best_det),
            is_boundary=bnd_i,
            is_pec=bnd_i
        )
        subdomains.append(subdomain)

    return subdomains


def verify_mlpg_coverage_2d(
    subdomains: List[MLPGSubdomain2D],
    Lx: float = np.pi,
    Ly: float = np.pi,
    sample_density: int = 50
) -> Tuple[float, np.ndarray]:
    """
    Verifies that the union of subdomains covers the computational domain [0, Lx] x [0, Ly]:
        Omega_union = bigcup_{I} Omega_s^I  supseteql  Omega

    Parameters:
    - subdomains: List of MLPGSubdomain2D objects.
    - Lx, Ly: Dimensions of the domain.
    - sample_density: Number of sample points per Cartesian direction.

    Returns:
    - coverage_ratio: Fraction of domain sample points covered by at least one subdomain.
    - coverage_count: 2D array of shape (sample_density, sample_density) indicating
      how many subdomains cover each sample point.
    """
    gx = np.linspace(0.0, Lx, sample_density)
    gy = np.linspace(0.0, Ly, sample_density)
    XX, YY = np.meshgrid(gx, gy)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])

    # Build KDTree on subdomain centers for rapid proximity lookup
    centers = np.array([sub.center for sub in subdomains])
    max_rs = max(sub.radius_s for sub in subdomains)
    tree_centers = KDTree(centers)

    counts = np.zeros(len(grid_pts), dtype=int)

    # For each sample point, find candidate centers within max_rs
    candidates_list = tree_centers.query_ball_point(grid_pts, r=max_rs)
    for idx, cand_indices in enumerate(candidates_list):
        pt = grid_pts[idx]
        c = 0
        for ci in cand_indices:
            if subdomains[ci].contains_point(pt):
                c += 1
        counts[idx] = c

    coverage_ratio = float(np.count_nonzero(counts > 0)) / float(len(grid_pts))
    coverage_count = counts.reshape(sample_density, sample_density)
    return coverage_ratio, coverage_count
