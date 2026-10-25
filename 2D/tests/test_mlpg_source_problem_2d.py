import os
import sys
import unittest
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
SCRIPTS_DIR = os.path.join(DIR_2D, "scripts")
for p in [DIR_2D, ROOT_DIR, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from scripts.mlpg_source_problem_2d import (
    assemble_and_solve_mlpg_mms,
    run_mlpg_mms_convergence_study
)


class TestMLPGSourceProblem2D(unittest.TestCase):
    def test_mlpg_mms_monotonic_convergence(self):
        """Verifies monotonic error decrease and asymptotic convergence rate for VNMM-MLPG 2D."""
        grid_sizes = [9, 13, 17, 21]
        errors_L2 = []
        errors_curl = []
        h_values = []

        for N in grid_sizes:
            coords, vectors, is_bnd = generate_cavity_mesh(Nx=N, Ny=N, jitter_frac=0.0)
            h = np.pi / float(N - 1)

            res = assemble_and_solve_mlpg_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                alpha_s=0.8,
                s_div=6.0,
                kappa=1.0
            )

            h_values.append(h)
            errors_L2.append(res["err_L2_E"])
            errors_curl.append(res["err_L2_curl"])

        # Monotonic decrease in L2 error
        for i in range(len(errors_L2) - 1):
            self.assertLess(
                errors_L2[i + 1],
                errors_L2[i],
                f"L2 error did not decrease monotonically: {errors_L2[i]} -> {errors_L2[i+1]}"
            )

        # Monotonic decrease in H(curl) error
        for i in range(len(errors_curl) - 1):
            self.assertLess(
                errors_curl[i + 1],
                errors_curl[i],
                f"H(curl) error did not decrease monotonically: {errors_curl[i]} -> {errors_curl[i+1]}"
            )

        # Asymptotic convergence rate in L2 must be approximately 2.0 (>= 1.8)
        rate_L2, _ = np.polyfit(np.log(h_values), np.log(errors_L2), 1)
        self.assertGreater(
            rate_L2,
            1.8,
            f"Fitted L2 rate should be ~ 2.0, obtained {rate_L2:.2f}"
        )

        # Asymptotic convergence rate in H(curl) must be >= 1.0
        rate_curl, _ = np.polyfit(np.log(h_values), np.log(errors_curl), 1)
        self.assertGreater(
            rate_curl,
            1.0,
            f"Fitted H(curl) rate should be >= 1.0, obtained {rate_curl:.2f}"
        )

    def test_mlpg_mms_mixed_neumann_convergence(self):
        """Verifies monotonic error decrease and asymptotic convergence rate for Mixed Neumann BCs."""
        grid_sizes = [9, 13, 17, 21]
        errors_L2 = []
        errors_curl = []
        h_values = []

        for N in grid_sizes:
            coords, vectors, is_bnd = generate_cavity_mesh(Nx=N, Ny=N, jitter_frac=0.0)
            h = np.pi / float(N - 1)

            res = assemble_and_solve_mlpg_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                alpha_s=0.8,
                s_div=6.0,
                kappa=1.0,
                boundary_condition="mixed_neumann"
            )

            h_values.append(h)
            errors_L2.append(res["err_L2_E"])
            errors_curl.append(res["err_L2_curl"])

        # Monotonic decrease in L2 error
        for i in range(len(errors_L2) - 1):
            self.assertLess(
                errors_L2[i + 1],
                errors_L2[i],
                f"Neumann L2 error did not decrease monotonically: {errors_L2[i]} -> {errors_L2[i+1]}"
            )

        # Monotonic decrease in H(curl) error
        for i in range(len(errors_curl) - 1):
            self.assertLess(
                errors_curl[i + 1],
                errors_curl[i],
                f"Neumann H(curl) error did not decrease monotonically: {errors_curl[i]} -> {errors_curl[i+1]}"
            )

        # Asymptotic convergence rate in L2 must be approximately 2.0 (>= 1.8)
        rate_L2, _ = np.polyfit(np.log(h_values), np.log(errors_L2), 1)
        self.assertGreater(
            rate_L2,
            1.8,
            f"Fitted Neumann L2 rate should be ~ 2.0, obtained {rate_L2:.2f}"
        )

        # Asymptotic convergence rate in H(curl) must be >= 1.0
        rate_curl, _ = np.polyfit(np.log(h_values), np.log(errors_curl), 1)
        self.assertGreater(
            rate_curl,
            1.0,
            f"Fitted Neumann H(curl) rate should be >= 1.0, obtained {rate_curl:.2f}"
        )


if __name__ == "__main__":
    unittest.main()
