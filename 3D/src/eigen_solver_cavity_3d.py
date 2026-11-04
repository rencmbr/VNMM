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

# Analytical 3D resonant cavity modes in [0, pi]^3:
# lambda = m^2 + n^2 + p^2, kc = sqrt(lambda)
ANALYTICAL_REFERENCE_MODES_3D = [
    (1, 0, 1, 2.0, np.sqrt(2.0)),   # TE101
    (0, 1, 1, 2.0, np.sqrt(2.0)),   # TE011
    (1, 1, 0, 2.0, np.sqrt(2.0)),   # TM110
    (1, 1, 1, 3.0, np.sqrt(3.0)),   # TE111
    (1, 1, 1, 3.0, np.sqrt(3.0)),   # TM111
    (2, 0, 1, 5.0, np.sqrt(5.0)),   # TE201
    (0, 2, 1, 5.0, np.sqrt(5.0)),   # TE021
    (1, 0, 2, 5.0, np.sqrt(5.0)),   # TE102
    (0, 1, 2, 5.0, np.sqrt(5.0)),   # TE012
    (2, 1, 0, 5.0, np.sqrt(5.0)),   # TM210
    (1, 2, 0, 5.0, np.sqrt(5.0)),   # TM120
    (2, 1, 1, 6.0, np.sqrt(6.0)),   # TE211
    (2, 1, 1, 6.0, np.sqrt(6.0)),   # TM211
    (1, 2, 1, 6.0, np.sqrt(6.0)),   # TE121
    (1, 2, 1, 6.0, np.sqrt(6.0)),   # TM121
    (1, 1, 2, 6.0, np.sqrt(6.0)),   # TE112
    (1, 1, 2, 6.0, np.sqrt(6.0)),   # TM112
    (2, 0, 2, 8.0, np.sqrt(8.0)),   # TE202
    (0, 2, 2, 8.0, np.sqrt(8.0)),   # TE022
    (2, 2, 0, 8.0, np.sqrt(8.0))    # TM220
]

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
    method="eigh"
):
    """
    Solves the generalized eigenvalue problem K_red * e = lambda * M_red * e.
    """
    n_dof = K_red.shape[0]
    if n_dof < num_eigenvalues:
        num_eigenvalues = max(1, n_dof - 1)

    if method == "eigh" or n_dof <= 300:
        K_dense = K_red.toarray() if hasattr(K_red, "toarray") else np.array(K_red)
        M_dense = M_red.toarray() if hasattr(M_red, "toarray") else np.array(M_red)

        # Regularize M slightly if near singular
        evals_M = np.linalg.eigvalsh(0.5 * (M_dense + M_dense.T))
        min_ev = np.min(evals_M)
        if min_ev < 1e-12:
            M_dense += (abs(min_ev) + 1e-8) * np.eye(n_dof)

        vals, vecs = la.eigh(K_dense, M_dense)
    else:
        try:
            vals, vecs = eigsh(K_red, k=num_eigenvalues, M=M_red, sigma=sigma, which="LM")
        except Exception:
            vals, vecs = eigs(K_red, k=num_eigenvalues, M=M_red, sigma=sigma, which="LM")
            vals = np.real(vals)
            vecs = np.real(vecs)

    vals = np.real(vals)
    idx_sort = np.argsort(vals)
    vals = vals[idx_sort]
    vecs = vecs[:, idx_sort]

    # Filter out near-zero or negative values below physical spectrum
    pos_mask = vals > 0.1
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
    Lx=np.pi,
    Ly=np.pi,
    Lz=np.pi,
    Ncx=None,
    Ncy=None,
    Ncz=None,
    basis="P1",
    interior_type="alternating",
    jitter_frac=0.0,
    seed=42,
    num_eigenvalues=10,
    s_div=6.0,
    points_per_dir=2,
    support_mode="gauss_point"
):
    """
    Executes full 3D cavity solver pipeline:
    1. Generates 3D cavity mesh with tangential PEC directors.
    2. Assembles 3D VNMM stiffness and mass matrices.
    3. Imposes PEC boundary conditions.
    4. Computes generalized eigenvalues.
    5. Evaluates error relative to analytical resonant cavity modes.
    """
    coords, vectors, is_bnd = generate_cavity_mesh_3d(
        Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz,
        interior_type=interior_type, jitter_frac=jitter_frac, seed=seed
    )

    K, M = assemble_vnmm_matrices_3d(
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
        support_mode=support_mode
    )

    K_red, M_red, interior_indices = apply_pec_boundary_conditions_3d(K, M, is_bnd)

    vals_num, vecs_num = solve_generalized_eigenvalue_problem_3d(
        K_red, M_red, num_eigenvalues=num_eigenvalues
    )

    ref_modes = ANALYTICAL_REFERENCE_MODES_3D[:len(vals_num)]
    vals_ref = np.array([m[3] for m in ref_modes])
    kc_ref = np.array([m[4] for m in ref_modes])
    kc_num = np.sqrt(np.maximum(vals_num, 0.0))

    errors_lambda = np.abs(vals_num - vals_ref) / vals_ref * 100.0
    errors_kc = np.abs(kc_num - kc_ref) / kc_ref * 100.0

    return {
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

