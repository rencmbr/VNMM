import unittest
import numpy as np
from codigo.galerkin_source_problem import (
    assemble_and_solve_mms,
    generate_cavity_mesh
)


class TestGalerkinSourceProblem(unittest.TestCase):
    def test_mms_convergence_P1(self):
        """Verifies monotonic convergence and O(h^2) asymptotic rate for complete basis P1."""
        grid_sizes = [9, 13, 17, 21]
        errors_L2 = []
        errors_curl = []
        h_values = []

        for N in grid_sizes:
            coords, vectors, is_bnd = generate_cavity_mesh(Nx=N, Ny=N, jitter_frac=0.0)
            res = assemble_and_solve_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                basis="P1",
                s_div=6.0,
                kappa=1.0,
                Ncx=N - 1,
                Ncy=N - 1,
                points_per_dir=3
            )
            errors_L2.append(res['err_L2_E_rel'])
            errors_curl.append(res['err_L2_curl_rel'])
            h_values.append(res['h'])

        # Strict monotonic decrease in L2 error
        for i in range(len(errors_L2) - 1):
            self.assertLess(errors_L2[i + 1], errors_L2[i], "L2 error must decrease under refinement for P1")

        # Asymptotic convergence rate in L2: must be >= 1.8
        rate_L2, _ = np.polyfit(np.log(h_values), np.log(errors_L2), 1)
        self.assertGreater(rate_L2, 1.8, f"Asymptotic L2 rate for P1 should be ~ 2.0, obtained {rate_L2:.2f}")

    def test_mms_stagnation_L1(self):
        """Verifies that incomplete basis L1 stagnates and fails to converge (error > 20%)."""
        grid_sizes = [9, 17]
        errors_L2 = []

        for N in grid_sizes:
            coords, vectors, is_bnd = generate_cavity_mesh(Nx=N, Ny=N, jitter_frac=0.0)
            res = assemble_and_solve_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                basis="L1",
                s_div=6.0,
                kappa=1.0,
                Ncx=N - 1,
                Ncy=N - 1,
                points_per_dir=3
            )
            errors_L2.append(res['err_L2_E_rel'])

        # L1 exhibits substantial stagnation error (> 20%)
        self.assertGreater(errors_L2[0], 0.20)
        self.assertGreater(errors_L2[1], 0.20)


if __name__ == "__main__":
    unittest.main()
