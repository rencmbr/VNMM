"""
Galerkin Patch Test for Vector Nodal Meshless Method via Meshless Local Petrov-Galerkin (VNMM-MLPG 2D).
Verifies exact machine-precision reproduction of affine vector fields by solving the
partitioned discrete system K_int,int * e_int = F_int - K_int,bnd * e_bnd with essential Dirichlet conditions.

Fields tested:
1. Pure translation along X: E = [1, 0]^T (Harmonic: curl=0, div=0)
2. Pure translation along Y: E = [0, 1]^T (Harmonic: curl=0, div=0)
3. Normal strain (longitudinal): E = [x, -y]^T (Harmonic: curl=0, div=0)
4. Shear strain: E = [y, x]^T (Harmonic: curl=0, div=0)
5. Rigid body rotation (Pure Curl): E = [-y, x]^T (curl=2, div=0)
6. Volumetric expansion (Pure Div): E = [x, y]^T (curl=0, div=2)
7. Full general affine field: E = [a1 + b1*x + b2*y, a2 + b3*x + b4*y]^T
"""

import os
import sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Any, List, Optional

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
from src.mlpg_assembler_2d import compute_subdomain_matrices_analytic


AFFINE_MODES = [
    {
        "name": "1. Translation X: E = [1, 0]^T",
        "func": lambda x, y: np.array([1.0, 0.0]),
        "rot": 0.0,
        "div": 0.0,
        "harmonic": True
    },
    {
        "name": "2. Translation Y: E = [0, 1]^T",
        "func": lambda x, y: np.array([0.0, 1.0]),
        "rot": 0.0,
        "div": 0.0,
        "harmonic": True
    },
    {
        "name": "3. Normal Strain (Harmonic): E = [x, -y]^T",
        "func": lambda x, y: np.array([x, -y]),
        "rot": 0.0,
        "div": 0.0,
        "harmonic": True
    },
    {
        "name": "4. Shear Strain (Harmonic): E = [y, x]^T",
        "func": lambda x, y: np.array([y, x]),
        "rot": 0.0,
        "div": 0.0,
        "harmonic": True
    },
    {
        "name": "5. Rigid Body Rotation (Pure Curl): E = [-y, x]^T",
        "func": lambda x, y: np.array([-y, x]),
        "rot": 2.0,
        "div": 0.0,
        "harmonic": False
    },
    {
        "name": "6. Volumetric Expansion (Pure Div): E = [x, y]^T",
        "func": lambda x, y: np.array([x, y]),
        "rot": 0.0,
        "div": 2.0,
        "harmonic": False
    },
    {
        "name": "7. Full Arbitrary Affine Field",
        "func": lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y]),
        "rot": 0.4 - (-0.3),
        "div": 0.8 + 1.2,
        "harmonic": False
    }
]


def assemble_mlpg_patch_test_system(
    subdomains: List[MLPGSubdomain2D],
    target_field: Dict[str, Any],
    s_div: float = 6.0,
    mu_r: float = 1.0
) -> Tuple[csr_matrix, np.ndarray]:
    """
    Assembles MLPG global stiffness matrix K and consistent patch test load vector F.

    For each subdomain Omega_s^I (disk of radius r_s):
    - K_I: Analytical closed-form local stiffness vector (1, 6).
    - F_I: Consistent load from exact constant curl and divergence:
        F_I = [ (1 / mu_r) * rot_exact * curl(W_I)_z + (s_div / mu_r) * div_exact * div(W_I) ] * Area
      For harmonic affine fields (rot_exact = 0, div_exact = 0), F_I = 0 identically.
    """
    N = len(subdomains)
    rows, cols, data = [], [], []
    F = np.zeros(N, dtype=float)

    rot_exact = float(target_field["rot"])
    div_exact = float(target_field["div"])

    for i, sub in enumerate(subdomains):
        k_loc, _ = compute_subdomain_matrices_analytic(
            subdomain=sub,
            mu_r=mu_r,
            epsilon_r=1.0,
            s_div=s_div
        )

        for k_idx, g_col in enumerate(sub.stencil_nodes):
            rows.append(i)
            cols.append(g_col)
            data.append(k_loc[k_idx])

        # Compute consistent force vector F_i
        stencil = sub.stencil_nodes
        i0 = np.where(stencil == sub.node_idx)[0][0]
        curls = sub.eval_shape_curl()
        divs = sub.eval_shape_div()
        area = np.pi * (sub.radius_s ** 2)

        F[i] = (
            (1.0 / mu_r) * rot_exact * curls[i0] +
            (s_div / mu_r) * div_exact * divs[i0]
        ) * area

    K = coo_matrix((data, (rows, cols)), shape=(N, N), dtype=float).tocsr()
    return K, F


def run_mlpg_patch_test_mode(
    coords: np.ndarray,
    vectors: np.ndarray,
    is_boundary: np.ndarray,
    target_field: Dict[str, Any],
    subdomains: Optional[List[MLPGSubdomain2D]] = None,
    alpha_s: float = 0.8,
    s_div: float = 6.0,
    mu_r: float = 1.0
) -> Dict[str, Any]:
    """
    Executes the Galerkin Patch Test for a single affine target field:
    1. Builds subdomains (if not precomputed).
    2. Assembles K and F analytically.
    3. Prescribes exact values on boundary degrees of freedom: e_bnd = E_target(x_k) . t_k.
    4. Solves partitioned system: K_int,int * e_int = F_int - K_int,bnd * e_bnd.
    5. Evaluates relative infinity-norm error on free interior degrees of freedom.
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

    # Exact degree of freedom values
    E_func = target_field["func"]
    e_exact = np.zeros(N, dtype=float)
    for k in range(N):
        E_pt = E_func(coords[k, 0], coords[k, 1])
        e_exact[k] = np.dot(E_pt, vectors[k])

    idx_bnd = np.where(is_boundary)[0]
    idx_int = np.where(~is_boundary)[0]

    K, F = assemble_mlpg_patch_test_system(
        subdomains=subdomains,
        target_field=target_field,
        s_div=s_div,
        mu_r=mu_r
    )

    # Partitioned system
    K_ii = K[idx_int, :][:, idx_int]
    K_ib = K[idx_int, :][:, idx_bnd]

    e_bnd = e_exact[idx_bnd]
    rhs_int = F[idx_int] - K_ib @ e_bnd

    e_sol_int = spsolve(K_ii, rhs_int)

    # Absolute and relative error in free interior DoFs
    error_dof_abs = float(np.max(np.abs(e_sol_int - e_exact[idx_int])))
    norm_dof_ref = float(np.max(np.abs(e_exact[idx_int])))
    error_dof_rel = error_dof_abs / max(norm_dof_ref, 1e-14)

    return {
        "error_dof_abs": error_dof_abs,
        "error_dof_rel": error_dof_rel,
        "num_internal_dofs": len(idx_int),
        "num_boundary_dofs": len(idx_bnd),
        "e_sol_int": e_sol_int,
        "e_exact_int": e_exact[idx_int]
    }


def run_mlpg_patch_test_suite(Nx: int = 11, Ny: int = 11) -> bool:
    """
    Executes full patch test suite across all 7 affine states on two mesh topologies:
    1. Regular Cartesian grid (alternating directors 45°/135°)
    2. Irregular cloud with 25% jitter and random directors
    """
    topologies = [
        {
            "name": f"REGULAR MESH ({Nx}x{Ny}, Alternating Directors 45°/135°)",
            "jitter": 0.0,
            "interior_type": "alternating"
        },
        {
            "name": f"IRREGULAR CLOUD ({Nx}x{Ny}, 25% Jitter, Random Directors)",
            "jitter": 0.25,
            "interior_type": "random"
        }
    ]

    print("=" * 85)
    print("      VNMM-MLPG 2D GALERKIN PATCH TEST SUITE (EXACT LOCAL GALERKIN MLPG6)")
    print("=" * 85)

    all_passed = True

    for topo in topologies:
        coords, vectors, is_bnd = generate_cavity_mesh(
            Nx=Nx,
            Ny=Ny,
            Lx=np.pi,
            Ly=np.pi,
            interior_type=topo["interior_type"],
            jitter_frac=topo["jitter"],
            seed=42
        )

        subdomains = build_mlpg_subdomains_2d(
            coords=coords,
            vectors=vectors,
            is_boundary=is_bnd,
            alpha_s=0.8
        )

        n_tot = len(coords)
        n_bnd = int(np.sum(is_bnd))
        n_int = n_tot - n_bnd

        print(f"\n>>> TOPOLOGY: {topo['name']}")
        print(f"    Total nodes: {n_tot} | Boundary nodes (Dirichlet): {n_bnd} | Internal nodes (free): {n_int}")
        print("-" * 85)
        print(f"{'Tested Affine State':<44} | {'Rel DoF Error':<16} | {'Verdict':<10}")
        print("-" * 85)

        for mode in AFFINE_MODES:
            res = run_mlpg_patch_test_mode(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                target_field=mode,
                subdomains=subdomains,
                s_div=6.0
            )

            rel_err = res["error_dof_rel"]
            passed = rel_err < 1e-10
            if not passed:
                all_passed = False

            verdict = "PASS" if passed else "FAIL"
            print(f"{mode['name']:<44} | {rel_err:<16.2e} | {verdict:<10}")

    print("\n" + "=" * 85)
    final_verdict = "ALL PATCH TESTS PASSED TO MACHINE PRECISION!" if all_passed else "SOME PATCH TESTS FAILED!"
    print(f"OVERALL STATUS: {final_verdict}")
    print("=" * 85)
    return all_passed


if __name__ == "__main__":
    run_mlpg_patch_test_suite()
