import unittest
import numpy as np
from codigo.galerkin_patch_test import (
    run_patch_test_mode,
    generate_cavity_mesh
)


class TestGalerkinPatchTest(unittest.TestCase):
    def test_galerkin_patch_test_P1_regular(self):
        """Verifies that complete basis P1 passes the Galerkin Patch Test on a regular mesh."""
        coords, vectors, is_bnd = generate_cavity_mesh(
            Nx=7, Ny=7, Lx=np.pi, Ly=np.pi, interior_type="alternating", jitter_frac=0.0, seed=42
        )
        
        arbitrary_affine_field = {
            "name": "Arbitrary Affine Field",
            "func": lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y]),
            "rot": 0.4 - (-0.3),
            "div": 0.8 + 1.2,
            "harmonic": False
        }
        
        res_P1 = run_patch_test_mode(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            target_field=arbitrary_affine_field, basis="P1", s_div=6.0, Ncx=6, Ncy=6
        )
        
        self.assertLess(res_P1['error_dof_rel'], 1e-11, f"P1 failed on regular patch test: {res_P1['error_dof_rel']}")
        self.assertLess(res_P1['err_L2_E'], 1e-11)
        self.assertLess(res_P1['err_L2_rot'], 1e-10)
        self.assertLess(res_P1['err_L2_div'], 1e-10)

    def test_galerkin_patch_test_P1_irregular(self):
        """Verifies that complete basis P1 passes the Galerkin Patch Test on irregular cloud (25% jitter)."""
        coords, vectors, is_bnd = generate_cavity_mesh(
            Nx=7, Ny=7, Lx=np.pi, Ly=np.pi, interior_type="random", jitter_frac=0.25, seed=42
        )
        
        arbitrary_affine_field = {
            "name": "Arbitrary Affine Field",
            "func": lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y]),
            "rot": 0.4 - (-0.3),
            "div": 0.8 + 1.2,
            "harmonic": False
        }
        
        res_P1 = run_patch_test_mode(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            target_field=arbitrary_affine_field, basis="P1", s_div=6.0, Ncx=6, Ncy=6
        )
        
        self.assertLess(res_P1['error_dof_rel'], 1e-11, f"P1 failed on irregular patch test: {res_P1['error_dof_rel']}")
        self.assertLess(res_P1['err_L2_E'], 1e-11)
        self.assertLess(res_P1['err_L2_rot'], 1e-10)
        self.assertLess(res_P1['err_L2_div'], 1e-10)

    def test_galerkin_patch_test_L1_failure(self):
        """Verifies that incomplete basis L1 fails Patch Test for normal strain deformation."""
        coords, vectors, is_bnd = generate_cavity_mesh(
            Nx=7, Ny=7, Lx=np.pi, Ly=np.pi, interior_type="alternating", jitter_frac=0.0, seed=42
        )
        
        normal_strain_field = {
            "name": "Normal Strain",
            "func": lambda x, y: np.array([x, -y]),
            "rot": 0.0,
            "div": 0.0,
            "harmonic": True
        }
        
        res_L1 = run_patch_test_mode(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            target_field=normal_strain_field, basis="L1", s_div=6.0, Ncx=6, Ncy=6
        )
        
        self.assertGreater(res_L1['error_dof_rel'], 0.05, f"L1 should fail patch test for normal strain: {res_L1['error_dof_rel']}")


if __name__ == "__main__":
    unittest.main()
