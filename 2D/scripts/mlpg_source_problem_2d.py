"""
Manufactured Solution (MMS) Convergence Analysis for VNMM-MLPG 2D.
Solves the inhomogeneous vector Helmholtz equation with volumetric divergence penalty regularization:
    curl(curl E) - s_div * grad(div E) + kappa * E = f(x, y) on [0, pi]^2
using the truly cell-free Meshless Local Petrov-Galerkin (MLPG) formulation.

Analyzes asymptotic convergence orders in L2, H(curl), and discrete DoF norms under spatial refinement:
    Rate L2:      O(h^2)
    Rate H(curl): O(h)
"""

import os
import sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Any, List, Optional, Tuple

# Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
SRC_DIR = os.path.join(DIR_2D, "src")
for p in [DIR_2D, ROOT_DIR, CURRENT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d, MLPGSubdomain2D
from src.mlpg_assembler_2d import (
    compute_subdomain_matrices_analytic,
    get_disk_quadrature_points_2d
)


def exact_solution_mms(x: float, y: float) -> np.ndarray:
    """Exact manufactured vector field E*(x, y)."""
    return np.array([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)], dtype=float)


def exact_curl_mms(x: float, y: float) -> float:
    """Exact z-component of curl: (curl E*)_z = 2 * sin(x) * sin(y)."""
    return float(2.0 * np.sin(x) * np.sin(y))


def exact_divergence_mms(x: float, y: float) -> float:
    """Exact divergence: div(E*) = 0 identically (solenoidal)."""
    return 0.0


def source_mms(x: float, y: float, kappa: float = 1.0) -> np.ndarray:
    """Inhomogeneous volumetric source: f(x, y) = (2 + kappa) * E*(x, y)."""
    return (2.0 + kappa) * exact_solution_mms(x, y)


def assemble_and_solve_mlpg_mms(
    coords: np.ndarray,
    vectors: np.ndarray,
    is_boundary: np.ndarray,
    subdomains: Optional[List[MLPGSubdomain2D]] = None,
    alpha_s: float = 0.8,
    s_div: float = 6.0,
    kappa: float = 1.0,
    mu_r: float = 1.0,
    n_r: int = 4,
    n_theta: int = 12
) -> Dict[str, Any]:
    """
    Assembles and solves the MLPG MMS boundary value problem:
        (K + kappa * M) * e = F
    under essential Dirichlet boundary conditions prescribed by E*(x_k) . t_k.
    """
    coords = np.asarray(coords, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    is_boundary = np.asarray(is_boundary, dtype=bool)
    N = len(coords)

    if subdomains is None:
        subdomains = build_mlpg_subdomains_2d(
            coords=coords,
            vectors=vectors,
            is_boundary=is_boundary,
            alpha_s=alpha_s
        )

    rows, cols, data_k, data_m = [], [], [], []
    F = np.zeros(N, dtype=float)

    for sub in subdomains:
        # Analytical 6x6 local stiffness and mass matrices
        k_loc, m_loc = compute_subdomain_matrices_analytic(
            subdomain=sub,
            mu_r=mu_r,
            epsilon_r=1.0,
            s_div=s_div,
            matrix_format="6x6"
        )

        # Consistent source load vector via disk quadrature:
        # F_curl_a = curls[a] * integral (curl E*) dA
        # F_mass_a = kappa * integral (E* . N_a) dA
        pts, wts = get_disk_quadrature_points_2d(sub.center, sub.radius_s, n_r=n_r, n_theta=n_theta)
        curls = sub.eval_shape_curl()
        curls_ex = np.array([exact_curl_mms(px, py) for px, py in pts])
        f_curl = curls * float(np.sum(wts * curls_ex))

        N_all = sub.eval_shape_functions(pts)  # (M, 2, 6)
        E_ex_pts = np.array([exact_solution_mms(px, py) for px, py in pts])
        dot_E_N = np.einsum("mq,mqa->ma", E_ex_pts, N_all)
        f_mass = kappa * np.sum(wts[:, np.newaxis] * dot_E_N, axis=0)

        f_local = f_curl + f_mass

        st = sub.stencil_nodes
        for a in range(6):
            Ia = st[a]
            F[Ia] += f_local[a]
            for b in range(6):
                Ib = st[b]
                rows.append(Ia)
                cols.append(Ib)
                data_k.append(k_loc[a, b])
                data_m.append(kappa * m_loc[a, b])

    A = coo_matrix(
        (np.array(data_k) + np.array(data_m), (rows, cols)),
        shape=(N, N),
        dtype=float
    ).tocsr()
    A = 0.5 * (A + A.T)

    # Exact DoFs on all nodes
    e_exact = np.array([np.dot(exact_solution_mms(x, y), t) for (x, y), t in zip(coords, vectors)])

    idx_bnd = np.where(is_boundary)[0]
    idx_int = np.where(~is_boundary)[0]

    A_ii = A[idx_int, :][:, idx_int]
    A_ib = A[idx_int, :][:, idx_bnd]

    e_bnd = e_exact[idx_bnd]
    rhs_int = F[idx_int] - A_ib @ e_bnd

    e_sol_int = spsolve(A_ii, rhs_int)

    e_sol = np.zeros(N, dtype=float)
    e_sol[idx_bnd] = e_bnd
    e_sol[idx_int] = e_sol_int

    # 1. Discrete DoF error
    err_dof_abs = float(np.max(np.abs(e_sol_int - e_exact[idx_int])))
    norm_dof_ref = float(np.max(np.abs(e_exact[idx_int])))
    err_dof_rel = err_dof_abs / max(norm_dof_ref, 1e-14)

    # 2. Continuous field and curl errors across all subdomain disks
    err_E_sq = 0.0
    norm_E_sq = 0.0
    err_curl_sq = 0.0
    norm_curl_sq = 0.0
    err_div_sq = 0.0
    total_integrated_area = 0.0

    for sub in subdomains:
        pts, wts = get_disk_quadrature_points_2d(sub.center, sub.radius_s, n_r=4, n_theta=12)
        N_all = sub.eval_shape_functions(pts)
        e_st = e_sol[sub.stencil_nodes]
        E_h = np.einsum("mqk,k->mq", N_all, e_st)
        E_ex = np.array([exact_solution_mms(px, py) for px, py in pts])

        err_E_sq += float(np.sum(wts * np.sum((E_h - E_ex) ** 2, axis=1)))
        norm_E_sq += float(np.sum(wts * np.sum(E_ex ** 2, axis=1)))

        c_h = float(np.dot(sub.eval_shape_curl(), e_st))
        c_ex = np.array([exact_curl_mms(px, py) for px, py in pts])
        err_curl_sq += float(np.sum(wts * (c_h - c_ex) ** 2))
        norm_curl_sq += float(np.sum(wts * (c_ex ** 2)))

        d_h = float(np.dot(sub.eval_shape_div(), e_st))
        err_div_sq += float(np.sum(wts * (d_h ** 2)))
        total_integrated_area += float(np.sum(wts))

    err_L2_E = float(np.sqrt(err_E_sq / max(norm_E_sq, 1e-14)))
    err_L2_curl = float(np.sqrt(err_curl_sq / max(norm_curl_sq, 1e-14)))
    err_L2_div = float(np.sqrt(err_div_sq / max(total_integrated_area, 1e-14)))

    return {
        "err_dof_rel": err_dof_rel,
        "err_dof_abs": err_dof_abs,
        "err_L2_E": err_L2_E,
        "err_L2_curl": err_L2_curl,
        "err_L2_div": err_L2_div,
        "e_sol": e_sol,
        "e_exact": e_exact,
        "num_dofs": N,
        "num_internal": len(idx_int)
    }


def run_mlpg_mms_convergence_study(
    grid_sizes: Optional[List[int]] = None,
    s_div: float = 6.0,
    kappa: float = 1.0,
    alpha_s: float = 0.8
) -> Dict[str, Any]:
    """
    Executes asymptotic convergence analysis across a sequence of grid refinements.
    """
    if grid_sizes is None:
        grid_sizes = [9, 13, 17, 21, 25]

    print("=" * 88)
    print("      VNMM-MLPG 2D CONVERGENCE STUDY: MANUFACTURED SOLUTION (MMS)")
    print("=" * 88)
    print(f"Parameters: s_div = {s_div:.1f}, kappa = {kappa:.1f}, alpha_s = {alpha_s:.2f} (cell-free MLPG)")
    print("Exact Field: E* = [sin(x)cos(y), -cos(x)sin(y)]^T (curl = 2*sin(x)sin(y), div = 0)")
    print("-" * 88)
    print(f"{'Mesh':<8} | {'h':<8} | {'DoFs':<6} | {'Err L2(E)':<12} | {'Err H(curl)':<12} | {'Err DoF':<12} | {'Div Res':<10}")
    print("-" * 88)

    h_values = []
    errors_L2 = []
    errors_curl = []
    errors_dof = []

    for N in grid_sizes:
        coords, vectors, is_bnd = generate_cavity_mesh(Nx=N, Ny=N, jitter_frac=0.0)
        h = np.pi / float(N - 1)

        res = assemble_and_solve_mlpg_mms(
            coords=coords,
            vectors=vectors,
            is_boundary=is_bnd,
            alpha_s=alpha_s,
            s_div=s_div,
            kappa=kappa
        )

        h_values.append(h)
        errors_L2.append(res["err_L2_E"])
        errors_curl.append(res["err_L2_curl"])
        errors_dof.append(res["err_dof_rel"])

        str_L2 = f"{res['err_L2_E'] * 100:.3f}%"
        str_curl = f"{res['err_L2_curl'] * 100:.3f}%"
        str_dof = f"{res['err_dof_rel'] * 100:.3f}%"
        str_div = f"{res['err_L2_div']:.2e}"

        print(f"{N}x{N:<5} | {h:<8.4f} | {len(coords):<6} | {str_L2:<12} | {str_curl:<12} | {str_dof:<12} | {str_div:<10}")

    # Compute asymptotic rates
    rate_L2, _ = np.polyfit(np.log(h_values), np.log(errors_L2), 1)
    rate_curl, _ = np.polyfit(np.log(h_values), np.log(errors_curl), 1)
    rate_dof, _ = np.polyfit(np.log(h_values), np.log(errors_dof), 1)

    print("-" * 88)
    print(f"Fitted Asymptotic Convergence Rate in L2(E):    O(h^{rate_L2:.2f})  [Theoretical: ~ O(h^2)]")
    print(f"Fitted Asymptotic Convergence Rate in H(curl):   O(h^{rate_curl:.2f})  [Theoretical: ~ O(h^1)]")
    print(f"Fitted Asymptotic Convergence Rate in DoFs:      O(h^{rate_dof:.2f})")
    print("=" * 88)

    return {
        "h_values": h_values,
        "errors_L2": errors_L2,
        "errors_curl": errors_curl,
        "errors_dof": errors_dof,
        "rate_L2": rate_L2,
        "rate_curl": rate_curl,
        "rate_dof": rate_dof
    }


if __name__ == "__main__":
    run_mlpg_mms_convergence_study()
