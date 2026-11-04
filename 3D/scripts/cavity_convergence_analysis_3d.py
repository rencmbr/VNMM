"""
Cavity Convergence Analysis in Three Dimensions (VNMM 3D).
Evaluates convergence of the first numerical resonant modes under grid refinement.
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_3D)
SRC_DIR = os.path.join(DIR_3D, "src")
for p in [DIR_3D, ROOT_DIR, CURRENT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.eigen_solver_cavity_3d import solve_cavity_eigenvalues_3d


def run_cavity_convergence_analysis_3d(grid_resolutions=[5, 7], num_modes=6):
    """
    Runs cavity eigenvalue convergence sweeps across multiple 3D resolutions.
    """
    print("=" * 75)
    print("      VNMM 3D CAVITY RESONANT FREQUENCY CONVERGENCE STUDY")
    print("=" * 75)

    results = []
    for N in grid_resolutions:
        print(f"\n--- Running resolution {N}x{N}x{N} ---")
        res = solve_cavity_eigenvalues_3d(
            Nx=N, Ny=N, Nz=N,
            num_eigenvalues=num_modes,
            s_div=6.0,
            points_per_dir=2
        )
        results.append(res)
        print(f"Total Nodes: {res['total_nodes']} | Active DoFs: {res['active_dofs']}")
        print(f"Mean kc Error: {res['mean_error_kc_pct']:.2f}% | Max kc Error: {res['max_error_kc_pct']:.2f}%")
        print("Mode Comparison (kc_num vs kc_ref):")
        for i in range(len(res['numerical_kc'])):
            print(f"  Mode {i+1}: num = {res['numerical_kc'][i]:.4f}, ref = {res['analytical_kc'][i]:.4f} (err = {res['error_kc_pct'][i]:.2f}%)")

    return results


if __name__ == "__main__":
    run_cavity_convergence_analysis_3d(grid_resolutions=[5, 6], num_modes=5)

