import os
import sys
import unittest
import numpy as np

# Path setup for tests
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.gauss_quadrature_2d import generate_quadrature_cells, get_gauss_points_weights_1d
from src.vnmm_assembler_2d import assemble_vnmm_matrices_2d
from src.eigen_solver_cavity_2d import (
    apply_pec_boundary_conditions, 
    solve_generalized_eigenvalue_problem, 
    solve_cavity_eigenvalues,
    ANALYTICAL_REFERENCE_MODES
)
from src.fem_edge_2d import solve_edge_fem_cavity_eigenvalues


class TestPECCavityMesh(unittest.TestCase):
    """Unit tests for cavity mesh generation and tangential PEC boundary directionality."""
    
    def test_dimensions_and_node_counts(self):
        Nx, Ny = 7, 7
        coords, vectors, is_boundary = generate_cavity_mesh(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        self.assertEqual(coords.shape, (Nx * Ny, 2))
        self.assertEqual(vectors.shape, (Nx * Ny, 2))
        self.assertEqual(is_boundary.shape, (Nx * Ny,))
        
        # Expected boundary node count in regular grid = 2*Nx + 2*Ny - 4
        expected_boundary_nodes = 2 * Nx + 2 * Ny - 4
        self.assertEqual(np.sum(is_boundary), expected_boundary_nodes)
        
    def test_tangential_pec_boundary_directionality(self):
        Nx, Ny = 9, 9
        coords, vectors, is_boundary = generate_cavity_mesh(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        tol = 1e-7
        for pt, vec, is_bnd in zip(coords, vectors, is_boundary):
            x, y = pt[0], pt[1]
            vec_norm = np.linalg.norm(vec)
            self.assertTrue(np.isclose(vec_norm, 1.0, atol=1e-12), "Director vector must be unitary.")
            
            # Bottom wall (y=0) or top wall (y=pi)
            if np.isclose(y, 0.0, atol=tol) or np.isclose(y, np.pi, atol=tol):
                self.assertTrue(is_bnd, "Node at y=0 or y=pi must be classified as boundary.")
                self.assertTrue(np.isclose(abs(vec[0]), 1.0, atol=1e-10))
                self.assertTrue(np.isclose(vec[1], 0.0, atol=1e-10))
                
            # Left wall (x=0) or right wall (x=pi)
            elif np.isclose(x, 0.0, atol=tol) or np.isclose(x, np.pi, atol=tol):
                self.assertTrue(is_bnd, "Node at x=0 or x=pi must be classified as boundary.")
                self.assertTrue(np.isclose(vec[0], 0.0, atol=1e-10))
                self.assertTrue(np.isclose(abs(vec[1]), 1.0, atol=1e-10))
            else:
                self.assertFalse(is_bnd, "Interior node must not be classified as boundary.")


class TestGaussQuadrature2D(unittest.TestCase):
    """Unit tests for background cell numerical quadrature."""
    
    def test_gauss_weights_1d(self):
        xi, w = get_gauss_points_weights_1d(2)
        self.assertEqual(len(xi), 2)
        self.assertTrue(np.isclose(np.sum(w), 2.0))
        
    def test_domain_total_area_integration(self):
        Lx, Ly = np.pi, np.pi
        gauss_pts, gauss_wts, cells = generate_quadrature_cells(Lx=Lx, Ly=Ly, Ncx=6, Ncy=6, points_per_dir=2)
        
        calculated_area = np.sum(gauss_wts)
        exact_area = Lx * Ly
        self.assertTrue(np.isclose(calculated_area, exact_area, atol=1e-12), "Integral of constant function 1 must equal Lx * Ly.")
        
    def test_quadratic_polynomial_integration(self):
        # Integral of f(x, y) = x^2 + y^2 on [0, pi]^2
        # Exact integral = 2 * pi^4 / 3
        Lx, Ly = np.pi, np.pi
        gauss_pts, gauss_wts, _ = generate_quadrature_cells(Lx=Lx, Ly=Ly, Ncx=8, Ncy=8, points_per_dir=2)
        
        f_vals = gauss_pts[:, 0]**2 + gauss_pts[:, 1]**2
        num_integral = np.sum(f_vals * gauss_wts)
        exact_integral = 2.0 * (np.pi**4) / 3.0
        
        self.assertTrue(np.isclose(num_integral, exact_integral, rtol=1e-10))


class TestVNMMMatrixAssembler(unittest.TestCase):
    """Tests for global and reduced sparse matrix properties."""
    
    def test_symmetry_and_positivity_gauss_point_support(self):
        Nx, Ny = 9, 9
        coords, vectors, is_boundary = generate_cavity_mesh(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        K, M = assemble_vnmm_matrices_2d(
            coords, vectors, basis="P1", s_div=6.0, Ncx=8, Ncy=8, 
            points_per_dir=2, support_mode="gauss_point"
        )
        
        # 1. Symmetry of global K and M
        diff_K = K - K.T
        diff_M = M - M.T
        self.assertTrue(diff_K.nnz == 0 or np.max(np.abs(diff_K.data)) < 1e-14)
        self.assertTrue(diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-14)
        
        # 2. PEC Reduction
        K_red, M_red, int_idx = apply_pec_boundary_conditions(K, M, is_boundary)
        self.assertEqual(K_red.shape[0], len(int_idx))
        self.assertEqual(M_red.shape[0], len(int_idx))
        
        # 3. Positivity of M_red diagonal
        diag_M = M_red.diagonal()
        self.assertTrue(np.all(diag_M > 0.0), "All diagonal elements of M_red must be strictly positive.")

    def test_symmetry_cell_center_support(self):
        Nx, Ny = 9, 9
        coords, vectors, is_boundary = generate_cavity_mesh(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        K, M = assemble_vnmm_matrices_2d(
            coords, vectors, basis="P1", s_div=6.0, Ncx=8, Ncy=8, 
            points_per_dir=2, support_mode="cell_center"
        )
        
        diff_K = K - K.T
        diff_M = M - M.T
        self.assertTrue(diff_K.nnz == 0 or np.max(np.abs(diff_K.data)) < 1e-14)
        self.assertTrue(diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-14)


class TestPECCavityEigenvalueSolver(unittest.TestCase):
    """Validation tests against Table 4-1 of Luilly Ortiz (UFMG, 2023)."""
    
    def test_cavity_eigenvalues_table_4_1_gauss_point(self):
        """
        Validates the first 10 numerical TEz eigenvalues in cavity [0, pi]^2
        using Gauss point support (EFG style).
        """
        result = solve_cavity_eigenvalues(
            Nx=21, 
            Ny=21, 
            Lx=np.pi, 
            Ly=np.pi, 
            Ncx=10,
            Ncy=10,
            basis="P1", 
            interior_type="alternating", 
            num_eigenvalues=10, 
            s_div=6.0,
            points_per_dir=2,
            support_mode="gauss_point"
        )
        
        vals_num = result['numerical_eigenvalues']
        vals_ref = result['analytical_eigenvalues']
        
        self.assertEqual(len(vals_num), 10, "Exactly 10 eigenvalues must be returned.")
        expected_ref = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0, 8.0, 9.0, 9.0])
        np.testing.assert_allclose(vals_ref, expected_ref, atol=1e-12)
        
        # Mean kc error < 1.5% and max kc error < 2.5%
        self.assertLess(result['mean_error_kc_pct'], 1.5)
        self.assertLess(result['max_error_kc_pct'], 2.5)

    def test_cavity_eigenvalues_table_4_1_cell_center(self):
        """
        Validates the first 10 numerical TEz eigenvalues with cell-center fixed support.
        """
        result = solve_cavity_eigenvalues(
            Nx=21, 
            Ny=21, 
            Lx=np.pi, 
            Ly=np.pi, 
            Ncx=20,
            Ncy=20,
            basis="P1", 
            interior_type="alternating", 
            num_eigenvalues=10, 
            s_div=6.0,
            points_per_dir=2,
            support_mode="cell_center"
        )
        
        self.assertLess(result['mean_error_kc_pct'], 2.5)
        self.assertLess(result['max_error_kc_pct'], 5.0)
        
    def test_absence_of_spurious_modes(self):
        """
        Verifies that no spurious/non-physical modes contaminate the operational spectrum.
        """
        result = solve_cavity_eigenvalues(
            Nx=17, 
            Ny=17, 
            Lx=np.pi, 
            Ly=np.pi, 
            basis="P1", 
            num_eigenvalues=8, 
            s_div=6.0
        )
        vals_num = result['numerical_eigenvalues']
        
        # No eigenvalue below physical fundamental resonance (TE10, lambda=1.0)
        self.assertTrue(np.all(vals_num >= 0.8), "No eigenvalues should lie below the fundamental physical mode.")
        
        # Strict monotonicity
        self.assertTrue(np.all(np.diff(vals_num) >= -1e-8), "Eigenvalue spectrum must be monotonically ascending.")


class TestFEMEdge2D(unittest.TestCase):
    """Unit tests for 2D triangular Nédélec edge finite element solver."""
    
    def test_edge_fem_cavity_solver(self):
        res = solve_edge_fem_cavity_eigenvalues(Nex=11, Ney=11, num_eigenvalues=10)
        
        self.assertEqual(len(res['numerical_eigenvalues']), 10)
        self.assertEqual(res['active_dofs'], 341)
        self.assertEqual(res['discarded_null_modes'], 100)
        
        # Mean kc error < 1.0%
        self.assertLess(res['mean_error_kc_pct'], 1.0)
        # Max error < 2.0%
        self.assertLess(res['max_error_kc_pct'], 2.0)

    def test_edge_fem_random_mesh_solver(self):
        res = solve_edge_fem_cavity_eigenvalues(
            Nex=12, Ney=12, num_eigenvalues=10, jitter_frac=0.25, seed=42
        )
        
        self.assertEqual(len(res['numerical_eigenvalues']), 10)
        # Under 25% jitter, mean kc error < 1.0%
        self.assertLess(res['mean_error_kc_pct'], 1.0)
        self.assertLess(res['max_error_kc_pct'], 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
