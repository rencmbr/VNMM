import os
import sys
import unittest
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import (
    quartic_spline_weight,
    quartic_spline_derivative,
    eval_test_function_2d,
    eval_test_function_curl_2d,
    eval_test_function_div_2d,
    build_mlpg_subdomains_2d,
    verify_mlpg_coverage_2d,
    MLPGSubdomain2D
)


class TestMLPGSubdomains2D(unittest.TestCase):
    def setUp(self):
        self.r_s = 0.25
        self.center = np.array([1.0, 1.0])
        self.director = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)])

    def test_quartic_spline_properties(self):
        """Verifies boundary values, compact support, and C1 smoothness of quartic spline."""
        # Value at center: w(0) = 1
        self.assertAlmostEqual(quartic_spline_weight(0.0, self.r_s), 1.0, places=14)

        # Value at boundary: w(r_s) = 0
        self.assertAlmostEqual(quartic_spline_weight(self.r_s, self.r_s), 0.0, places=14)

        # Outside boundary: w(r > r_s) = 0
        self.assertAlmostEqual(quartic_spline_weight(1.5 * self.r_s, self.r_s), 0.0, places=14)

        # Derivatives at extremities: w'(0) = 0, w'(r_s) = 0
        self.assertAlmostEqual(quartic_spline_derivative(0.0, self.r_s), 0.0, places=14)
        self.assertAlmostEqual(quartic_spline_derivative(self.r_s, self.r_s), 0.0, places=14)
        self.assertAlmostEqual(quartic_spline_derivative(1.5 * self.r_s, self.r_s), 0.0, places=14)

        # Finite difference check for w'(r) inside the support
        r_test = 0.12
        eps = 1e-7
        w_plus = quartic_spline_weight(r_test + eps, self.r_s)
        w_minus = quartic_spline_weight(r_test - eps, self.r_s)
        dw_num = (w_plus - w_minus) / (2.0 * eps)
        dw_ana = quartic_spline_derivative(r_test, self.r_s)
        self.assertAlmostEqual(dw_ana, dw_num, places=6)

    def test_vector_test_function_and_derivatives(self):
        """Verifies vector test function values, curl, and div against numerical differentiation."""
        # Value at node center must match director vector exactly
        W_center = eval_test_function_2d(self.center, self.center, self.director, self.r_s)
        np.testing.assert_allclose(W_center, self.director, atol=1e-14)

        # Outside support must be strictly zero
        outside_pt = self.center + np.array([2.0 * self.r_s, 0.0])
        W_out = eval_test_function_2d(outside_pt, self.center, self.director, self.r_s)
        np.testing.assert_allclose(W_out, np.zeros(2), atol=1e-14)

        # Test point strictly inside Omega_s
        test_pt = self.center + np.array([0.08, -0.06])
        eps = 1e-6

        # Numerical curl: dW_y/dx - dW_x/dy
        W_x_plus = eval_test_function_2d(test_pt + [eps, 0.0], self.center, self.director, self.r_s)
        W_x_minus = eval_test_function_2d(test_pt - [eps, 0.0], self.center, self.director, self.r_s)
        dW_dx = (W_x_plus - W_x_minus) / (2.0 * eps)

        W_y_plus = eval_test_function_2d(test_pt + [0.0, eps], self.center, self.director, self.r_s)
        W_y_minus = eval_test_function_2d(test_pt - [0.0, eps], self.center, self.director, self.r_s)
        dW_dy = (W_y_plus - W_y_minus) / (2.0 * eps)

        curl_num = dW_dx[1] - dW_dy[0]
        curl_ana = eval_test_function_curl_2d(test_pt, self.center, self.director, self.r_s)
        self.assertAlmostEqual(curl_ana, curl_num, places=5)

        # Numerical divergence: dW_x/dx + dW_y/dy
        div_num = dW_dx[0] + dW_dy[1]
        div_ana = eval_test_function_div_2d(test_pt, self.center, self.director, self.r_s)
        self.assertAlmostEqual(div_ana, div_num, places=5)

    def test_build_mlpg_subdomains_regular_mesh(self):
        """Verifies construction of subdomains and mathematical properties on regular mesh."""
        coords, vectors, is_bnd = generate_cavity_mesh(Nx=7, Ny=7, jitter_frac=0.0)
        N = len(coords)

        subdomains = build_mlpg_subdomains_2d(
            coords=coords,
            vectors=vectors,
            is_boundary=is_bnd,
            alpha_s=0.8
        )

        self.assertEqual(len(subdomains), N)

        for sub in subdomains:
            # Stencil must contain 6 distinct nodes
            self.assertEqual(len(sub.stencil_nodes), 6)
            self.assertEqual(len(np.unique(sub.stencil_nodes)), 6)

            # Central node must belong to its own stencil
            self.assertIn(sub.node_idx, sub.stencil_nodes)

            # Support radius r_u must be strictly greater than integration radius r_s
            self.assertGreater(sub.radius_u, sub.radius_s)

            # Determinant must be positive and non-singular
            self.assertGreater(sub.det_a, 1e-12)

            # Inverse beta must satisfy A * beta = I_6 to high precision
            A_times_beta = sub.matrix_a @ sub.beta
            np.testing.assert_allclose(A_times_beta, np.eye(6), atol=1e-12)

            # Interpolation property on the 6 stencil nodes: N_k(x_m) . t_m = delta_km
            stencil_coords = coords[sub.stencil_nodes]
            stencil_vectors = vectors[sub.stencil_nodes]
            N_all = sub.eval_shape_functions(stencil_coords)  # shape (6, 2, 6)

            for m in range(6):
                # Projection on node m director vector: dot(N_k(x_m), t_m)
                t_m = stencil_vectors[m]
                projections = np.array([np.dot(N_all[m, :, k], t_m) for k in range(6)])
                expected = np.zeros(6)
                expected[m] = 1.0
                np.testing.assert_allclose(projections, expected, atol=1e-12)

            # Shape function curl and div must be length-6 vectors
            curl_N = sub.eval_shape_curl()
            div_N = sub.eval_shape_div()
            self.assertEqual(len(curl_N), 6)
            self.assertEqual(len(div_N), 6)

    def test_build_mlpg_subdomains_irregular_cloud(self):
        """Verifies robust construction under 20% spatial jitter."""
        coords, vectors, is_bnd = generate_cavity_mesh(Nx=7, Ny=7, jitter_frac=0.20, seed=123)
        subdomains = build_mlpg_subdomains_2d(
            coords=coords,
            vectors=vectors,
            is_boundary=is_bnd,
            alpha_s=0.8
        )

        self.assertEqual(len(subdomains), len(coords))
        for sub in subdomains:
            self.assertGreater(sub.det_a, 1e-14)
            self.assertGreater(sub.radius_u, sub.radius_s)

    def test_domain_coverage(self):
        """Verifies that the union of local subdomains covers the cavity domain without holes."""
        coords, vectors, is_bnd = generate_cavity_mesh(Nx=9, Ny=9, jitter_frac=0.0)
        subdomains = build_mlpg_subdomains_2d(
            coords=coords,
            vectors=vectors,
            is_boundary=is_bnd,
            alpha_s=0.85
        )

        ratio, counts = verify_mlpg_coverage_2d(subdomains, Lx=np.pi, Ly=np.pi, sample_density=20)
        # Ratio should be 1.0 (100% of domain points covered)
        self.assertGreaterEqual(ratio, 0.99)


if __name__ == "__main__":
    unittest.main()
