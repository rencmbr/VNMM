"""
Parametric sweep of Tol_hat between 0.05 and 6.0
on a fine perturbed mesh (N = 4192, h = 0.0654 m) over Omega = [0, pi]^2
with 1600 test points using purely dimensionless formulation.
"""

import os
import sys
import time
import numpy as np
from scipy.spatial import KDTree

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from scripts.dimensionless_vnmm_6_p1 import (
    support_nodes_vnmm_2d_6_p1_dimensionless,
    shape_functions_vnmm_2d_6_p1_dimensionless
)
from scripts.generate_convergence_l1_vs_p1_pi_figure import generate_nodal_mesh_pi, exact_field_te11


def run_dimensionless_sweep(tol_hat_vals=None):
    """Run parametric sweep of dimensionless threshold Tol_hat."""
    print("=" * 80)
    print("PARAMETRIC SWEEP OF TOL_HAT (N = 4192, 1600 TEST POINTS)")
    print("=" * 80)

    # 1. Generate fine perturbed mesh
    coords_dense, vectors_dense, h_dense = generate_nodal_mesh_pi(192, 4000, seed=42)
    tree_dense = KDTree(coords_dense)
    total_nodes = len(coords_dense)
    print(f"Mesh generated: N = {total_nodes} nodes, h_avg = {h_dense:.4f} m")

    # 2. Analytical TE11 projections
    e_nodes_d = np.zeros_like(coords_dense)
    for i, pt in enumerate(coords_dense):
        e_nodes_d[i], _ = exact_field_te11(pt)
    proj_dense = np.sum(e_nodes_d * vectors_dense, axis=1)

    # 3. Test grid of 1600 points (40x40)
    num_side_pts = 40
    xs = np.linspace(0.04 * np.pi, 0.96 * np.pi, num_side_pts)
    ys = np.linspace(0.04 * np.pi, 0.96 * np.pi, num_side_pts)
    xx, yy = np.meshgrid(xs, ys)
    test_points = np.column_stack([xx.ravel(), yy.ravel()])
    num_pts = len(test_points)
    print(f"Test points: {num_pts} points")

    if tol_hat_vals is None:
        tol_hat_vals = [0.05, 0.10, 0.25, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 4.50, 6.00]
    
    results = []

    for tol_hat in tol_hat_vals:
        t0 = time.time()
        err_e = []
        err_rot = []
        ks = []
        conds_hat = []
        conds_dim = []
        successes = 0

        for pt in test_points:
            e_exact, rot_exact = exact_field_te11(pt)
            
            res_sel = support_nodes_vnmm_2d_6_p1_dimensionless(
                point=pt, node_coords=coords_dense, node_vectors=vectors_dense, search_tree=tree_dense,
                h_local=h_dense, num_neighbors=12, tol_hat=tol_hat,
                adaptive=True, step_k=4, max_k=40,
                return_dimensional_matrix=True
            )
            sel, det_hat, a_hat, h_eff, k_ef, a_dim = res_sel

            if sel is not None and len(sel) == 6 and det_hat > 0:
                successes += 1
                phi_hat, rot_phi_hat, div_phi_hat, _ = shape_functions_vnmm_2d_6_p1_dimensionless(
                    point=pt, node_coords=coords_dense, node_vectors=vectors_dense, selected_nodes=sel, h_local=h_eff, matrix_a_hat=a_hat
                )
                e_sel = proj_dense[sel]
                e_interp = phi_hat @ e_sel
                rot_interp = float(np.dot(rot_phi_hat, e_sel))

                err_e.append(np.linalg.norm(e_interp - e_exact))
                err_rot.append(abs(rot_interp - rot_exact))
                ks.append(k_ef)

                try:
                    c_hat = np.linalg.cond(a_hat)
                except Exception:
                    c_hat = 1e6
                conds_hat.append(c_hat)

                try:
                    c_dim = np.linalg.cond(a_dim)
                except Exception:
                    c_dim = 1e6
                conds_dim.append(c_dim)
            else:
                err_e.append(10.0)
                err_rot.append(50.0)
                ks.append(k_ef if k_ef > 0 else 12)
                conds_hat.append(1e6)
                conds_dim.append(1e6)

        dt = time.time() - t0
        res = {
            'tol_hat': tol_hat,
            'success_rate': (successes / num_pts) * 100.0,
            'k_mean': float(np.mean(ks)),
            'err_e_rms': float(np.sqrt(np.mean(np.array(err_e)**2))),
            'err_rot_rms': float(np.sqrt(np.mean(np.array(err_rot)**2))),
            'cond_hat_median': float(np.median(conds_hat)),
            'cond_dim_median': float(np.median(conds_dim)),
            'elapsed_seconds': dt
        }
        results.append(res)

        print(f"Tol_hat = {tol_hat:5.2f} | Success: {res['success_rate']:5.1f}% | K_mean: {res['k_mean']:4.1f} | "
              f"E Err: {res['err_e_rms']:.2e} | Rot Err: {res['err_rot_rms']:.2e} | "
              f"cond(A_hat): {res['cond_hat_median']:4.1f} | cond(A): {res['cond_dim_median']:4.0f} | Time: {dt:.2f}s")

    return results


def main():
    run_dimensionless_sweep()


# Backward compatibility aliases
executar_varredura_adimensional = run_dimensionless_sweep


if __name__ == "__main__":
    main()
