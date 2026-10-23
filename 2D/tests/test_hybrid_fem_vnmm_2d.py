import os
import sys
import unittest
import numpy as np
import scipy.linalg as la

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.fem_vnmm_hybrid_2d import (
    generate_hybrid_cavity_mesh,
    assemble_hybrid_fem_vnmm_matrices,
    solve_hybrid_fem_vnmm_eigenvalues
)


class TestHybridFemVnmm(unittest.TestCase):
    """
    Unit tests for hybrid coupled Edge FEM 2D + VNMM 2D (P1) solver.
    """
    
    def test_hybrid_mesh_generation(self):
        """Validates partitioned mesh generation and interface alignment."""
        Lx = np.pi
        Ly = np.pi
        Nex_fem = 6
        Ney = 10
        Nx_vnmm = 7
        Ny_vnmm = 11
        
        fem_data, vnmm_data = generate_hybrid_cavity_mesh(
            Lx=Lx, Ly=Ly, frac_fem=0.5, Nex_fem=Nex_fem, Ney=Ney,
            Nx_vnmm=Nx_vnmm, Ny_vnmm=Ny_vnmm
        )
        
        # 1. Check number of interface edges in FEM
        n_interface_fem = len(fem_data['interface_eids'])
        self.assertEqual(n_interface_fem, Ney)
        
        # 2. Check that VNMM interface nodes match this count
        n_interface_vnmm = np.sum(vnmm_data['is_interface'])
        self.assertEqual(n_interface_vnmm, Ney)
        
        # 3. Check that vectors at interface are vertical unit vectors t = [0, 1]
        idx_intf = np.where(vnmm_data['is_interface'])[0]
        for idx in idx_intf:
            vec = vnmm_data['vectors'][idx]
            self.assertAlmostEqual(vec[0], 0.0, places=6)
            self.assertAlmostEqual(vec[1], 1.0, places=6)
            self.assertAlmostEqual(vnmm_data['coords'][idx, 0], fem_data['x_int'], places=6)
            
    def test_hybrid_matrix_symmetry(self):
        """Validates strict symmetry of the coupled global matrices."""
        fem_data, vnmm_data = generate_hybrid_cavity_mesh(
            Nex_fem=4, Ney=6, Nx_vnmm=5, Ny_vnmm=7, frac_fem=0.5
        )
        
        K_glob, M_glob, info_dofs = assemble_hybrid_fem_vnmm_matrices(
            fem_data, vnmm_data, Ncx_vnmm=4, Ncy_vnmm=6, s_div_vnmm=6.0
        )
        
        K_dense = K_glob.toarray()
        M_dense = M_glob.toarray()
        
        sym_error_K = np.max(np.abs(K_dense - K_dense.T))
        sym_error_M = np.max(np.abs(M_dense - M_dense.T))
        
        self.assertLess(sym_error_K, 1e-12)
        self.assertLess(sym_error_M, 1e-12)
        
        # Check that M is strictly positive definite
        evals_M = la.eigvalsh(M_dense)
        self.assertTrue(np.all(evals_M > 0.0))
        
    def test_fundamental_mode_accuracy(self):
        """Validates accuracy of physical TE10, TE01, TE11 modes in the hybrid solver."""
        res = solve_hybrid_fem_vnmm_eigenvalues(
            Lx=np.pi, Ly=np.pi, frac_fem=0.5,
            Nex_fem=8, Ney=12, Nx_vnmm=9, Ny_vnmm=13,
            Ncx_vnmm=8, Ncy_vnmm=10, s_div_vnmm=6.0,
            num_eigenvalues=3
        )
        
        # Modes TE10, TE01, TE11
        errors_kc = res['rel_errors_kc_pct'][:3]
        for e in errors_kc:
            self.assertLess(e, 5.0, f"Cutoff wavenumber error ({e:.2f}%) exceeded 5.0% threshold.")
            
        self.assertLess(res['mean_error_kc_pct'], 4.0)

    def test_hybrid_random_meshes(self):
        """Validates operation and accuracy of hybrid coupling under random meshes and directions."""
        res = solve_hybrid_fem_vnmm_eigenvalues(
            Lx=np.pi, Ly=np.pi, frac_fem=0.5,
            Nex_fem=6, Ney=10, Nx_vnmm=7, Ny_vnmm=11,
            Ncx_vnmm=6, Ncy_vnmm=8, s_div_vnmm=6.0,
            points_per_dir=2,
            interior_type_vnmm="random",
            jitter_frac_fem=0.25,
            jitter_frac_vnmm=0.25,
            num_eigenvalues=3,
            seed=42
        )
        
        self.assertLess(res['mean_error_kc_pct'], 5.0)
        self.assertEqual(len(res['numerical_eigenvalues']), 3)


if __name__ == "__main__":
    unittest.main()
