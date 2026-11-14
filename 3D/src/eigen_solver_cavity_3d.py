"""
Eigenvalue Solver for 3D Electromagnetic Resonant Cavities.
Solves the generalized eigenvalue problem K * e = lambda * M * e under PEC boundary conditions
and compares numerical resonant frequencies with analytical 3D cavity modes.
"""

import os
import sys
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh, eigs

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(DIR_3D)
for p in [DIR_3D, ROOT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_3d import generate_cavity_mesh_3d
from src.vnmm_assembler_3d import assemble_vnmm_matrices_3d

# Analytical 3D resonant cavity modes for rectangular cavity (Section 6.2):
# Dimensions: a = 1.00 cm, b = 0.50 cm, c = 0.75 cm
# lambda = (m*pi/a)^2 + (n*pi/b)^2 + (p*pi/c)^2, kc = sqrt(lambda)
def compute_analytical_cavity_modes_3d(Lx=1.00, Ly=0.50, Lz=0.75, max_order=4):
    """
    Computes exact analytical resonant modes (m, n, p, lambda, kc) for a 3D PEC rectangular cavity
    of dimensions Lx x Ly x Lz according to Eq. (6.10):
      - TE_mnp: m >= 0, n >= 0, p >= 1 (except m = n = 0)
      - TM_mnp: m >= 1, n >= 1, p >= 0
    """
    modes = []
    for m in range(0, max_order + 1):
        for n in range(0, max_order + 1):
            for p in range(0, max_order + 1):
                lam = (m * np.pi / Lx) ** 2 + (n * np.pi / Ly) ** 2 + (p * np.pi / Lz) ** 2
                kc = np.sqrt(lam)
                # TE_mnp condition: p >= 1 and not (m == 0 and n == 0)
                if p >= 1 and not (m == 0 and n == 0):
                    modes.append((m, n, p, float(lam), float(kc)))
                # TM_mnp condition: m >= 1, n >= 1, p >= 0
                if m >= 1 and n >= 1 and p >= 0:
                    modes.append((m, n, p, float(lam), float(kc)))
    modes.sort(key=lambda item: (round(item[3], 8), item[0], item[1], item[2]))
    return modes


ANALYTICAL_REFERENCE_MODES_3D = compute_analytical_cavity_modes_3d(1.00, 0.50, 0.75)

ANALYTICAL_REFERENCE_MODES = ANALYTICAL_REFERENCE_MODES_3D
MODOS_ANALITICOS_REF = ANALYTICAL_REFERENCE_MODES_3D


def apply_pec_boundary_conditions_3d(K, M, is_boundary):
    """
    Applies homogeneous Dirichlet PEC boundary conditions by eliminating boundary DoFs (E_t = 0).
    """
    interior_indices = np.where(~is_boundary)[0]
    K_red = K[interior_indices, :][:, interior_indices]
    M_red = M[interior_indices, :][:, interior_indices]
    return K_red, M_red, interior_indices


def solve_generalized_eigenvalue_problem_3d(
    K_red,
    M_red,
    num_eigenvalues=10,
    sigma=0.5,
    method="eigh",
    Kc_red=None,
    Kd_red=None,
    div_curl_tol=0.15
):
    """
    Solves the generalized eigenvalue problem K_red * e = lambda * M_red * e
    and filters out static/irrotational spurious modes when Kc_red and Kd_red are provided.
    """
    n_dof = K_red.shape[0]
    if n_dof < num_eigenvalues:
        num_eigenvalues = max(1, n_dof - 1)

    if method == "eigh" or n_dof <= 600:
        K_dense = K_red.toarray() if hasattr(K_red, "toarray") else np.array(K_red)
        M_dense = M_red.toarray() if hasattr(M_red, "toarray") else np.array(M_red)

        # Regularize M slightly if near singular
        evals_M = np.linalg.eigvalsh(0.5 * (M_dense + M_dense.T))
        min_ev = np.min(evals_M)
        if min_ev < 1e-12:
            M_dense += (abs(min_ev) + 1e-8) * np.eye(n_dof)

        vals, vecs = la.eigh(K_dense, M_dense)
    else:
        k_search = min(max(num_eigenvalues * 4, 30), n_dof - 2)
        try:
            vals, vecs = eigsh(K_red, k=k_search, M=M_red, sigma=sigma, which="LM")
        except Exception:
            vals, vecs = eigs(K_red, k=k_search, M=M_red, sigma=sigma, which="LM")
            vals = np.real(vals)
            vecs = np.real(vecs)

    vals = np.real(vals)
    idx_sort = np.argsort(vals)
    vals = vals[idx_sort]
    vecs = vecs[:, idx_sort]

    # Filter physical solenoidal modes
    if Kc_red is not None and Kd_red is not None:
        Kc_mat = Kc_red.toarray() if hasattr(Kc_red, "toarray") else np.asarray(Kc_red)
        Kd_mat = Kd_red.toarray() if hasattr(Kd_red, "toarray") else np.asarray(Kd_red)
        phys_indices = []
        for idx_v in range(len(vals)):
            if vals[idx_v] > 0.5:
                v = vecs[:, idx_v]
                e_curl = float(v.T @ Kc_mat @ v)
                e_div = float(v.T @ Kd_mat @ v)
                if e_curl > 1e-10 and e_div < div_curl_tol * e_curl:
                    phys_indices.append(idx_v)
                    if len(phys_indices) >= num_eigenvalues:
                        break
        if len(phys_indices) >= num_eigenvalues:
            vals = vals[phys_indices]
            vecs = vecs[:, phys_indices]
            return vals, vecs

    # Fallback: Filter out near-zero or negative values below physical spectrum
    pos_mask = vals > 0.5
    if np.sum(pos_mask) >= num_eigenvalues:
        vals = vals[pos_mask][:num_eigenvalues]
        vecs = vecs[:, pos_mask][:, :num_eigenvalues]
    else:
        vals = vals[:num_eigenvalues]
        vecs = vecs[:, :num_eigenvalues]

    return vals, vecs


def solve_cavity_eigenvalues_3d(
    Nx=7,
    Ny=7,
    Nz=7,
    Lx=1.00,
    Ly=0.50,
    Lz=0.75,
    Ncx=None,
    Ncy=None,
    Ncz=None,
    basis="P1",
    interior_type="triad",
    jitter_frac=0.0,
    seed=42,
    num_eigenvalues=10,
    s_div=2.0,
    points_per_dir=2,
    support_mode="gauss_point"
):
    """
    Executes full 3D cavity solver pipeline:
    1. Generates 3D cavity mesh with tangential PEC directors.
    2. Assembles 3D VNMM stiffness and mass matrices (with SRI divergence stabilization).
    3. Imposes PEC boundary conditions.
    4. Computes generalized eigenvalues and filters solenoidal resonant modes.
    5. Evaluates error relative to analytical resonant cavity modes.
    """
    if interior_type.lower() in ["triad", "triade", "orthogonal_triad"] and support_mode in ["cell_center", "centro_celula"]:
        support_mode = "gauss_point"

    coords, vectors, is_bnd = generate_cavity_mesh_3d(
        Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz,
        interior_type=interior_type, jitter_frac=jitter_frac, seed=seed
    )

    K, M, K_curl, K_div = assemble_vnmm_matrices_3d(
        coords=coords,
        vectors=vectors,
        basis=basis,
        s_div=s_div,
        Ncx=Ncx,
        Ncy=Ncy,
        Ncz=Ncz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        points_per_dir=points_per_dir,
        support_mode=support_mode,
        sri_div=True,
        return_components=True
    )

    K_red, M_red, interior_indices = apply_pec_boundary_conditions_3d(K, M, is_bnd)
    Kc_red = K_curl[interior_indices, :][:, interior_indices]
    Kd_red = K_div[interior_indices, :][:, interior_indices]

    vals_num, vecs_num = solve_generalized_eigenvalue_problem_3d(
        K_red, M_red, num_eigenvalues=num_eigenvalues, Kc_red=Kc_red, Kd_red=Kd_red
    )

    ref_modes = compute_analytical_cavity_modes_3d(Lx=Lx, Ly=Ly, Lz=Lz)[:len(vals_num)]
    vals_ref = np.array([m[3] for m in ref_modes])
    kc_ref = np.array([m[4] for m in ref_modes])
    mode_indices = [(m[0], m[1], m[2]) for m in ref_modes]
    kc_num = np.sqrt(np.maximum(vals_num, 0.0))

    errors_lambda = np.abs(vals_num - vals_ref) / vals_ref * 100.0
    errors_kc = np.abs(kc_num - kc_ref) / kc_ref * 100.0

    return {
        "mode_indices": mode_indices,
        "numerical_eigenvalues": vals_num,
        "analytical_eigenvalues": vals_ref,
        "numerical_kc": kc_num,
        "analytical_kc": kc_ref,
        "error_lambda_pct": errors_lambda,
        "error_kc_pct": errors_kc,
        "mean_error_kc_pct": float(np.mean(errors_kc)),
        "max_error_kc_pct": float(np.max(errors_kc)),
        "eigenvectors": vecs_num,
        "interior_indices": interior_indices,
        "total_nodes": len(coords),
        "boundary_nodes": int(np.sum(is_bnd)),
        "active_dofs": len(interior_indices)
    }


# Compatibility aliases
apply_pec_boundary_conditions = apply_pec_boundary_conditions_3d
solve_generalized_eigenvalue_problem = solve_generalized_eigenvalue_problem_3d
solve_cavity_eigenvalues = solve_cavity_eigenvalues_3d
resolver_autovalores_cavidade_3d = solve_cavity_eigenvalues_3d

