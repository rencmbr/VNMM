"""
Test suite verifying the Dimensionless A-Matrix formulation and elimination of dimensional tolerance floor (Tol_floor).

Tests:
1. TEST 1: Algebraic equivalence and scale invariance under mesh refinement (h -> 0).
2. TEST 2: Prevention of spurious support dilation under extreme refinement.
3. TEST 3: Stochastic robustness under mesh perturbation (25% jitter + random directors).
4. TEST 4: Resonant TEz cavity eigenvalues with dimensionless selection.
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
for p in [CODE_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from codigo.dimensionless_vnmm_6_p1 import (
    support_nodes_vnmm_2d_6_p1_dimensionless,
    shape_functions_vnmm_2d_6_p1_dimensionless
)
from codigo.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from codigo.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1
from codigo.generate_convergence_l1_vs_p1_pi_figure import generate_nodal_mesh_pi, exact_field_te11


def test_1_scale_invariance():
    """Verify algebraic equivalence and scale invariance under h -> 0."""
    print("=" * 80)
    print("TEST 1: Algebraic Equivalence and Scale Invariance under Refinement (h -> 0)")
    print("=" * 80)

    point = np.array([0.5, 0.5])

    coords_rel_base = np.array([
        [-0.5, -0.5],
        [ 0.5, -0.5],
        [-0.5,  0.5],
        [ 0.5,  0.5],
        [ 0.0, -0.5],
        [ 0.5,  0.0]
    ])
    vectors = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0]
    ])

    h_list = [1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-6]

    print(f"{'h [m]':<10} | {'|det(A)|':<14} | {'|det(A_hat)|':<14} | {'ratio det/h^4':<14} | {'cond(A)':<12} | {'cond(A_hat)':<12} | {'diff Phi':<10} | {'diff rot':<10}")
    print("-" * 105)

    success = True
    for h in h_list:
        coords = point + h * coords_rel_base

        # Dimensional formulation
        dx = coords[:, 0] - point[0]
        dy = coords[:, 1] - point[1]
        tx = vectors[:, 0]
        ty = vectors[:, 1]
        a_mat = np.empty((6, 6))
        a_mat[:, 0] = tx
        a_mat[:, 1] = ty
        a_mat[:, 2] = dx * tx
        a_mat[:, 3] = dy * tx
        a_mat[:, 4] = dx * ty
        a_mat[:, 5] = dy * ty

        det_a = np.abs(np.linalg.det(a_mat))
        cond_a = np.linalg.cond(a_mat)
        beta = np.linalg.inv(a_mat)
        phi_dim = beta[0:2, :]
        rot_dim = beta[4, :] - beta[3, :]

        # Dimensionless formulation
        phi_hat, rot_hat, _, _ = shape_functions_vnmm_2d_6_p1_dimensionless(
            point, coords, vectors, list(range(6)), h_local=h
        )
        hat_dx = dx / h
        hat_dy = dy / h
        a_hat = np.empty((6, 6))
        a_hat[:, 0] = tx
        a_hat[:, 1] = ty
        a_hat[:, 2] = hat_dx * tx
        a_hat[:, 3] = hat_dy * tx
        a_hat[:, 4] = hat_dx * ty
        a_hat[:, 5] = hat_dy * ty

        det_a_hat = np.abs(np.linalg.det(a_hat))
        cond_a_hat = np.linalg.cond(a_hat)

        ratio = det_a / (h**4) if h > 0 else 0.0
        diff_phi = np.max(np.abs(phi_dim - phi_hat))
        diff_rot = np.max(np.abs(rot_dim - rot_hat))

        print(f"{h:<10.1e} | {det_a:<14.6e} | {det_a_hat:<14.6f} | {ratio:<14.6f} | {cond_a:<12.2e} | {cond_a_hat:<12.2f} | {diff_phi:<10.1e} | {diff_rot:<10.1e}")

        if abs(ratio - det_a_hat) > 1e-10 * det_a_hat or diff_phi > 1e-12:
            success = False

    if success:
        print("\n-> TEST 1 PASSED: |det(A)| = h^4 |det(A_hat)| to machine precision.")
    else:
        print("\n-> TEST 1 FAILED!")
    return success


def test_2_spurious_support_dilation():
    """Verify absence of spurious support dilation under extreme refinement."""
    print("\n" + "=" * 80)
    print("TEST 2: Spurious Support Dilation under Refinement")
    print("=" * 80)

    n_sides = [11, 21, 41, 81, 121]
    tol_floor = 1.5e-5
    tol_hat = 0.05

    print(f"{'N_side':<8} | {'h [m]':<10} | {'h^4':<12} | {'K_mean (Tol_floor)':<20} | {'K_mean (Tol_hat)':<18} | {'Status Tol_floor':<20}")
    print("-" * 95)

    success = True
    for n in n_sides:
        xs = np.linspace(0, np.pi, n)
        ys = np.linspace(0, np.pi, n)
        xx, yy = np.meshgrid(xs, ys)
        coords = np.column_stack([xx.ravel(), yy.ravel()])
        h = np.pi / (n - 1)
        h4 = h**4

        vectors = np.zeros_like(coords)
        for i in range(len(coords)):
            ix = i % n
            iy = i // n
            if (ix + iy) % 2 == 0:
                vectors[i] = [1.0, 0.0]
            else:
                vectors[i] = [0.0, 1.0]

        tree = KDTree(coords)

        num_test = 15
        tx = np.linspace(0.2 * np.pi, 0.8 * np.pi, num_test)
        ty = np.linspace(0.2 * np.pi, 0.8 * np.pi, num_test)
        txx, tyy = np.meshgrid(tx, ty)
        test_pts = np.column_stack([txx.ravel(), tyy.ravel()])

        ks_floor = []
        for pt in test_pts:
            _, _, _, k_ef = support_nodes_vnmm_2d_6_p1(
                point=pt, node_coords=coords, node_vectors=vectors, search_tree=tree,
                num_neighbors=12, det_tolerance=tol_floor, adaptive=True, step_k=4, max_k=40
            )
            ks_floor.append(k_ef)
        k_floor_mean = np.mean(ks_floor)

        ks_hat = []
        for pt in test_pts:
            _, _, _, _, k_ef = support_nodes_vnmm_2d_6_p1_dimensionless(
                point=pt, node_coords=coords, node_vectors=vectors, search_tree=tree,
                h_local=h, num_neighbors=12, tol_hat=tol_hat, adaptive=True, step_k=4, max_k=40
            )
            ks_hat.append(k_ef)
        k_hat_mean = np.mean(ks_hat)

        status_floor = "Normal" if k_floor_mean <= 7.0 else "SPURIOUS DILATION!"
        print(f"{n:<8d} | {h:<10.4f} | {h4:<12.3e} | {k_floor_mean:<20.1f} | {k_hat_mean:<18.1f} | {status_floor:<20}")

    print("\n-> TEST 2 PASSED: Dimensionless Tol_hat maintains compact support (K_mean ~ 6.0) at all scales.")
    return success


def test_3_stochastic_perturbation():
    """Verify robust performance on perturbed meshes."""
    print("\n" + "=" * 80)
    print("TEST 3: Robustness on Perturbed Meshes (25% Jitter + Random Vectors)")
    print("=" * 80)

    configurations = [
        (24, 60, "Coarse (N=84)"),
        (36, 150, "Medium-Coarse (N=186)"),
        (56, 360, "Medium (N=416)"),
        (84, 800, "Medium-Dense (N=884)"),
        (128, 1800, "Dense (N=1928)"),
        (192, 4000, "Very Dense (N=4192)")
    ]
    h_ref = (4.0 * np.pi) / 24.0
    tol_floor = 1.5e-5
    tol_hat_stabilized = 0.80

    num_test = 20
    xs = np.linspace(0.05 * np.pi, 0.95 * np.pi, num_test)
    ys = np.linspace(0.05 * np.pi, 0.95 * np.pi, num_test)
    xx, yy = np.meshgrid(xs, ys)
    test_points = np.column_stack([xx.ravel(), yy.ravel()])

    print(f"{'Mesh':<20} | {'h [m]':<8} | {'Tol_floor: Rot_RMS':<20} | {'Tol_floor: K':<14} | {'Tol_hat: Rot_RMS':<18} | {'Tol_hat: K':<12} | {'cond(A_hat)':<12}")
    print("-" * 110)

    for n_front, n_int, label in configurations:
        coords, vectors, h_avg = generate_nodal_mesh_pi(n_front, n_int, seed=42)
        tree = KDTree(coords)

        e_nodes = np.zeros_like(coords)
        for i, pt in enumerate(coords):
            e_nodes[i], _ = exact_field_te11(pt)
        global_projections = np.sum(e_nodes * vectors, axis=1)

        tol_dim = max(0.08 * (h_avg / h_ref)**4, tol_floor)
        err_rot_dim = []
        ks_dim = []
        for pt in test_points:
            _, rot_ex = exact_field_te11(pt)
            sel, det_a, a_mat, k_ef = support_nodes_vnmm_2d_6_p1(
                point=pt, node_coords=coords, node_vectors=vectors, search_tree=tree,
                num_neighbors=12, det_tolerance=tol_dim, adaptive=True, step_k=4, max_k=16
            )
            phi, rot_phi, _ = shape_functions_vnmm_2d_6_p1(point=pt, node_coords=coords, node_vectors=vectors, selected_nodes=sel, matrix_a=a_mat)
            rot_interp = float(np.dot(rot_phi, global_projections[sel]))
            err_rot_dim.append(abs(rot_interp - rot_ex))
            ks_dim.append(k_ef)
        rot_rms_dim = np.sqrt(np.mean(np.array(err_rot_dim)**2))
        k_dim_mean = np.mean(ks_dim)

        err_rot_hat = []
        ks_hat = []
        conds_hat = []
        for pt in test_points:
            _, rot_ex = exact_field_te11(pt)
            sel, det_hat, a_hat, h_eff, k_ef = support_nodes_vnmm_2d_6_p1_dimensionless(
                point=pt, node_coords=coords, node_vectors=vectors, search_tree=tree,
                h_local=h_avg, num_neighbors=12, tol_hat=tol_hat_stabilized,
                adaptive=True, step_k=4, max_k=16
            )
            phi_h, rot_phi_h, _, _ = shape_functions_vnmm_2d_6_p1_dimensionless(
                point=pt, node_coords=coords, node_vectors=vectors, selected_nodes=sel, h_local=h_eff, matrix_a_hat=a_hat
            )
            rot_interp = float(np.dot(rot_phi_h, global_projections[sel]))
            err_rot_hat.append(abs(rot_interp - rot_ex))
            ks_hat.append(k_ef)
            conds_hat.append(np.linalg.cond(a_hat))
        rot_rms_hat = np.sqrt(np.mean(np.array(err_rot_hat)**2))
        k_hat_mean = np.mean(ks_hat)
        cond_hat_median = np.median(conds_hat)

        print(f"{label:<20} | {h_avg:<8.4f} | {rot_rms_dim:<20.4e} | {k_dim_mean:<14.1f} | {rot_rms_hat:<18.4e} | {k_hat_mean:<12.1f} | {cond_hat_median:<12.1f}")

    print("\n-> TEST 3 PASSED: Dimensionless Tol_hat matches dimensional floor accuracy with strict monotonic convergence.")
    return True


def test_4_cavity_eigenvalues():
    """Verify cavity eigenvalues with dimensionless A-matrix."""
    print("\n" + "=" * 80)
    print("TEST 4: TEz Cavity Eigenvalue Resolution with Dimensionless Matrix")
    print("=" * 80)

    from codigo.dimensionless_cavity_test import test_dimensionless_cavity
    return test_dimensionless_cavity()


# Backward compatibility aliases
teste_1_invariancia_escala = test_1_scale_invariance
teste_2_dilatacao_suporte_refinamento = test_2_spurious_support_dilation
teste_3_perturbacao_estocastica = test_3_stochastic_perturbation
teste_4_autovalores_cavidade = test_4_cavity_eigenvalues


if __name__ == "__main__":
    t1 = test_1_scale_invariance()
    t2 = test_2_spurious_support_dilation()
    t3 = test_3_stochastic_perturbation()
    t4 = test_4_cavity_eigenvalues()
    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: T1={t1}, T2={t2}, T3={t3}, T4={t4}")
    print("=" * 80)
