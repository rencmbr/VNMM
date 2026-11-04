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
from src.mlpg_subdomains import build_mlpg_subdomains_2d
from src.mlpg_assembler_2d import (
    get_disk_quadrature_points_2d,
    compute_subdomain_matrices_analytic,
    compute_subdomain_matrices_numeric,
    compute_subdomain_source_vector,
    assemble_mlpg_matrices_2d
)


class TestMLPGAssembler2D(unittest.TestCase):
    def setUp(self):
        self.coords, self.vectors, self.is_bnd = generate_cavity_mesh(Nx=7, Ny=7, jitter_frac=0.0)
        self.subdomains = build_mlpg_subdomains_2d(
            coords=self.coords,
            vectors=self.vectors,
            is_boundary=self.is_bnd,
            alpha_s=0.8
        )

    def test_disk_quadrature_area(self):
        """Verifies that disk quadrature reproduces exact area pi*r^2 and second moment pi*r^4/4."""
        center = np.array([1.2, 0.8])
        radius = 0.35
        exact_area = np.pi * (radius ** 2)
        exact_i2 = 0.25 * np.pi * (radius ** 4)

        pts, weights = get_disk_quadrature_points_2d(center, radius, n_r=6, n_theta=16)

        num_area = float(np.sum(weights))
        self.assertAlmostEqual(num_area, exact_area, places=10)

        # Second moment: integral (x - x0)^2 dA
        dx = pts[:, 0] - center[0]
        num_i2 = float(np.sum(weights * (dx ** 2)))
        self.assertAlmostEqual(num_i2, exact_i2, places=10)

    def test_analytic_vs_numeric_subdomain_matrices(self):
        """Verifies that closed-form analytic formulas match numerical polar Gauss quadrature."""
        # Pick interior subdomain
        sub = self.subdomains[len(self.subdomains) // 2]

        k_ana, m_ana = compute_subdomain_matrices_analytic(
            subdomain=sub,
            mu_r=1.0,
            epsilon_r=1.0,
            s_div=4.0
        )

        k_num, m_num = compute_subdomain_matrices_numeric(
            subdomain=sub,
            mu_r=1.0,
            epsilon_r=1.0,
            s_div=4.0,
            n_r=6,
            n_theta=16
        )

        # High precision agreement (quadrature of quadratic polynomials is exact with n_r >= 3)
        np.testing.assert_allclose(k_ana, k_num, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(m_ana, m_num, atol=1e-10, rtol=1e-8)

    def test_truncated_subdomain_analytic_vs_numeric(self):
        """Verifies that analytic stiffness scaling matches numerical integration on truncated boundary nodes."""
        bnd_indices = np.where(self.is_bnd)[0]
        sub_wall = None
        sub_corner_near = None
        for idx in bnd_indices:
            sub = self.subdomains[idx]
            if abs(sub.center[1] - 0.0) < 1e-12:
                if abs(sub.center[0] - np.pi / 2.0) < 0.5:
                    sub_wall = sub
                elif sub.center[0] < sub.radius_s:
                    sub_corner_near = sub

        self.assertIsNotNone(sub_wall)
        self.assertAlmostEqual(sub_wall.area_fraction, 0.5, places=7)
        k_ana, m_ana = compute_subdomain_matrices_analytic(sub_wall, s_div=4.0)
        k_num, m_num = compute_subdomain_matrices_numeric(sub_wall, s_div=4.0, n_r=32, n_theta=128)

        # Flat wall K and M must match numerical integration on truncated domain
        np.testing.assert_allclose(k_ana, k_num, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(m_ana, m_num, atol=1e-3, rtol=1e-2)

        if sub_corner_near is not None:
            # Case 3B: Boundary node near corner has active area strictly less than 0.5 and > 0.25
            self.assertTrue(0.25 < sub_corner_near.area_fraction < 0.5)
            k_ana_c, m_ana_c = compute_subdomain_matrices_analytic(sub_corner_near, s_div=4.0)
            k_num_c, m_num_c = compute_subdomain_matrices_numeric(sub_corner_near, s_div=4.0, n_r=32, n_theta=128)
            # Polar grid truncation has O(h_quad) cutoff across chord
            np.testing.assert_allclose(k_ana_c, k_num_c, atol=5e-2, rtol=1e-2)
            np.testing.assert_allclose(m_ana_c, m_num_c, atol=5e-2, rtol=1e-2)

    def test_global_assembly_dimensions_and_sparsity(self):
        """Verifies assembly of global CSR matrices K and M."""
        N = len(self.coords)
        K, M, F = assemble_mlpg_matrices_2d(
            subdomains=self.subdomains,
            mu_r=1.0,
            epsilon_r=1.0,
            s_div=4.0,
            integration_mode="analytic",
            apply_pec=False
        )

        self.assertEqual(K.shape, (N, N))
        self.assertEqual(M.shape, (N, N))
        self.assertIsNone(F)

        # Diagonal of mass matrix must be strictly positive (self-mass > 0)
        M_diag = M.diagonal()
        self.assertTrue(np.all(M_diag > 0.0), f"Mass diagonal has non-positive entries: {M_diag.min()}")

        # Check sparsity: each row has non-zeros exclusively inside its stencil
        for i, sub in enumerate(self.subdomains):
            row_k = K.getrow(i).toarray()[0]
            nonzeros = np.where(np.abs(row_k) > 1e-15)[0]
            for col in nonzeros:
                self.assertIn(col, sub.stencil_nodes)

    def test_assembly_with_pec(self):
        """Verifies identity rows for boundary PEC nodes."""
        K, M, _ = assemble_mlpg_matrices_2d(
            subdomains=self.subdomains,
            apply_pec=True
        )

        for i, sub in enumerate(self.subdomains):
            if sub.is_pec:
                # K(i, i) = 1, all other entries 0
                row_k = K.getrow(i).toarray()[0]
                expected_k = np.zeros(len(self.subdomains))
                expected_k[i] = 1.0
                np.testing.assert_allclose(row_k, expected_k, atol=1e-14)

                # M(i, :) = 0
                row_m = M.getrow(i).toarray()[0]
                np.testing.assert_allclose(row_m, np.zeros_like(row_m), atol=1e-14)

    def test_source_vector_assembly(self):
        """Verifies source vector assembly for constant and inhomogeneous sources."""
        # Constant source J_s = [1, 2]
        J_const = lambda pts: np.array([1.0, 2.0])

        K, M, F = assemble_mlpg_matrices_2d(
            subdomains=self.subdomains,
            source_func=J_const,
            apply_pec=False
        )

        self.assertIsNotNone(F)
        self.assertEqual(len(F), len(self.coords))

        # Check for first interior node against analytical integral of constant J_s
        sub = self.subdomains[len(self.subdomains) // 2]
        matches = np.where(sub.stencil_nodes == sub.node_idx)[0]
        i0 = matches[0]
        expected_F = (1.0 * sub.beta[0, i0] + 2.0 * sub.beta[1, i0]) * (np.pi * sub.radius_s ** 2)
        self.assertAlmostEqual(F[sub.node_idx], expected_F, places=8)

    def test_orthogonal_energy_stabilization(self):
        """Verifies orthogonal energy stabilization increases stiffness rank to 4 while preserving constant fields."""
        sub = self.subdomains[len(self.subdomains) // 2]
        
        # 1. Without stabilization: rank is at most 2
        K_unstab, _ = compute_subdomain_matrices_analytic(
            subdomain=sub, s_div=4.0, alpha_stab=0.0, matrix_format="6x6"
        )
        rank_unstab = np.linalg.matrix_rank(K_unstab)
        self.assertLessEqual(rank_unstab, 2)

        # 2. With stabilization: rank becomes 4 (full rank in 4 derivatives)
        K_stab, _ = compute_subdomain_matrices_analytic(
            subdomain=sub, s_div=4.0, alpha_stab=1.0, matrix_format="6x6"
        )
        rank_stab = np.linalg.matrix_rank(K_stab)
        self.assertEqual(rank_stab, 4)

        # 3. Patch Test consistency: constant field produces strictly zero stabilization force
        # Constant vector field u = [a0, a1]^T
        # In stencil DOFs: u_k = E . t_k = a0 * t_kx + a1 * t_ky
        for a0, a1 in [(1.0, 0.0), (0.0, 1.0), (3.5, -2.1)]:
            E_const = np.array([a0, a1])
            u_dofs = np.array([np.dot(E_const, self.vectors[node]) for node in sub.stencil_nodes])
            
            f_unstab = K_unstab @ u_dofs
            f_stab = K_stab @ u_dofs
            # Both forces must be zero because constant fields have zero curl, zero div, zero shear
            np.testing.assert_allclose(f_unstab, np.zeros(6), atol=1e-12)
            np.testing.assert_allclose(f_stab, np.zeros(6), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
