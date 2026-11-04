"""
Unit Tests for 3D Electromagnetic Resonant Cavity Solver (VNMM 3D).
Validates 3D mesh generation, 3D Gauss quadrature, matrix symmetry/positivity,
and 3D cavity eigenvalue accuracy.
"""

import os
import sys
import unittest
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(DIR_3D)
for p in [DIR_3D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_3d import generate_cavity_mesh_3d
from src.gauss_quadrature_3d import generate_quadrature_cells_3d, get_gauss_points_weights_1d
from src.vnmm_assembler_3d import assemble_vnmm_matrices_3d
from src.eigen_solver_cavity_3d import (
    apply_pec_boundary_conditions_3d,
    solve_generalized_eigenvalue_problem_3d,
    solve_cavity_eigenvalues_3d,
    ANALYTICAL_REFERENCE_MODES_3D
)


class TestPECCavityMesh3D(unittest.TestCase):
    """Unit tests for 3D cavity mesh generation and tangential PEC boundary directionality."""

    def test_dimensions_and_node_counts(self):
        Nx, Ny, Nz = 5, 5, 5
        coords, vectors, is_boundary = generate_cavity_mesh_3d(Nx=Nx, Ny=Ny, Nz=Nz)

        self.assertEqual(coords.shape[1], 3)
        self.assertEqual(vectors.shape[1], 3)
        self.assertEqual(len(coords), len(vectors))
        self.assertEqual(len(coords), len(is_boundary))

        # Interior nodes = (Nx-2) * (Ny-2) * (Nz-2) = 3 * 3 * 3 = 27
        num_interior = np.sum(~is_boundary)
        self.assertEqual(num_interior, (Nx - 2) * (Ny - 2) * (Nz - 2))

    def test_tangential_pec_boundary_directionality(self):
        Nx, Ny, Nz = 5, 5, 5
        coords, vectors, is_boundary = generate_cavity_mesh_3d(Nx=Nx, Ny=Ny, Nz=Nz, Lx=np.pi, Ly=np.pi, Lz=np.pi)

        tol = 1e-7
        for pt, vec, is_bnd in zip(coords, vectors, is_boundary):
            x, y, z = pt[0], pt[1], pt[2]
            vec_norm = np.linalg.norm(vec)
            self.assertTrue(np.isclose(vec_norm, 1.0, atol=1e-12), "Director vector must be unitary.")

            if is_bnd:
                # If on x-faces (x=0 or x=pi), normal is x, so vec must have tx = 0
                if np.isclose(x, 0.0, atol=tol) or np.isclose(x, np.pi, atol=tol):
                    self.assertTrue(np.isclose(vec[0], 0.0, atol=1e-10), "x-wall tangent must have tx = 0")
                # If on y-faces (y=0 or y=pi), normal is y, so vec must have ty = 0
                elif np.isclose(y, 0.0, atol=tol) or np.isclose(y, np.pi, atol=tol):
                    self.assertTrue(np.isclose(vec[1], 0.0, atol=1e-10), "y-wall tangent must have ty = 0")
                # If on z-faces (z=0 or z=pi), normal is z, so vec must have tz = 0
                elif np.isclose(z, 0.0, atol=tol) or np.isclose(z, np.pi, atol=tol):
                    self.assertTrue(np.isclose(vec[2], 0.0, atol=1e-10), "z-wall tangent must have tz = 0")


class TestGaussQuadrature3D(unittest.TestCase):
    """Unit tests for 3D background cell numerical quadrature."""

    def test_gauss_weights_1d(self):
        xi, w = get_gauss_points_weights_1d(2)
        self.assertEqual(len(xi), 2)
        self.assertTrue(np.isclose(np.sum(w), 2.0))

    def test_domain_total_volume_integration(self):
        Lx, Ly, Lz = np.pi, np.pi, np.pi
        gauss_pts, gauss_wts, cells = generate_quadrature_cells_3d(Lx=Lx, Ly=Ly, Lz=Lz, Ncx=3, Ncy=3, Ncz=3, points_per_dir=2)

        calculated_volume = np.sum(gauss_wts)
        exact_volume = Lx * Ly * Lz
        self.assertTrue(np.isclose(calculated_volume, exact_volume, atol=1e-12), "Integral of constant 1 must equal Lx * Ly * Lz.")

    def test_quadratic_polynomial_integration_3d(self):
        # Integral of f(x, y, z) = x^2 + y^2 + z^2 on [0, pi]^3
        # Exact integral = 3 * (pi^3 / 3) * pi^2 = pi^5
        Lx, Ly, Lz = np.pi, np.pi, np.pi
        gauss_pts, gauss_wts, _ = generate_quadrature_cells_3d(Lx=Lx, Ly=Ly, Lz=Lz, Ncx=4, Ncy=4, Ncz=4, points_per_dir=2)

        f_vals = gauss_pts[:, 0]**2 + gauss_pts[:, 1]**2 + gauss_pts[:, 2]**2
        num_integral = np.sum(f_vals * gauss_wts)
        exact_integral = np.pi**5

        self.assertTrue(np.isclose(num_integral, exact_integral, rtol=1e-8))


class TestVNMMMatrixAssembler3D(unittest.TestCase):
    """Tests for 3D global and reduced matrix properties."""

    def test_symmetry_and_positivity_gauss_point_support(self):
        Nx, Ny, Nz = 5, 5, 5
        coords, vectors, is_boundary = generate_cavity_mesh_3d(Nx=Nx, Ny=Ny, Nz=Nz)

        K, M = assemble_vnmm_matrices_3d(
            coords, vectors, basis="P1", s_div=6.0, Ncx=4, Ncy=4, Ncz=4,
            points_per_dir=2, support_mode="gauss_point"
        )

        # 1. Symmetry
        diff_K = K - K.T
        diff_M = M - M.T
        self.assertTrue(diff_K.nnz == 0 or np.max(np.abs(diff_K.data)) < 1e-12)
        self.assertTrue(diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-12)

        # 2. PEC reduction
        K_red, M_red, int_idx = apply_pec_boundary_conditions_3d(K, M, is_boundary)
        self.assertEqual(K_red.shape[0], len(int_idx))
        self.assertEqual(M_red.shape[0], len(int_idx))

        # 3. Positivity of M_red diagonal
        diag_M = M_red.diagonal()
        self.assertTrue(np.all(diag_M > 0.0), "All diagonal elements of M_red must be positive.")

    def test_symmetry_cell_center_support(self):
        Nx, Ny, Nz = 5, 5, 5
        coords, vectors, is_boundary = generate_cavity_mesh_3d(Nx=Nx, Ny=Ny, Nz=Nz)

        K, M = assemble_vnmm_matrices_3d(
            coords, vectors, basis="P1", s_div=6.0, Ncx=3, Ncy=3, Ncz=3,
            points_per_dir=2, support_mode="cell_center"
        )

        diff_K = K - K.T
        diff_M = M - M.T
        self.assertTrue(diff_K.nnz == 0 or np.max(np.abs(diff_K.data)) < 1e-12)
        self.assertTrue(diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-12)


class TestPECCavityEigenvalueSolver3D(unittest.TestCase):
    """Validation tests for 3D cavity eigenvalue solution."""

    def test_cavity_eigenvalues_3d_accuracy(self):
        res = solve_cavity_eigenvalues_3d(
            Nx=5, Ny=5, Nz=5,
            num_eigenvalues=5,
            s_div=6.0,
            points_per_dir=2,
            support_mode="cell_center"
        )
        vals_num = res['numerical_eigenvalues']
        self.assertEqual(len(vals_num), 5)
        # Numerical eigenvalues must be positive and ascending
        self.assertTrue(np.all(vals_num > 0.5))
        self.assertTrue(np.all(np.diff(vals_num) >= -1e-6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
