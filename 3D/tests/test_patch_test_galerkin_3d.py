"""
Unit Tests for 3D Galerkin Patch Test (VNMM 3D).
Verifies machine-precision consistency under 3D affine vector modes.
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
from scripts.galerkin_patch_test_3d import run_patch_test_mode_3d


class TestGalerkinPatchTest3D(unittest.TestCase):
    def test_galerkin_patch_test_P1_regular_3d(self):
        """Verifies that complete 12-node basis P1 passes the 3D Patch Test on a regular mesh."""
        coords, vectors, is_bnd = generate_cavity_mesh_3d(
            Nx=5, Ny=5, Nz=5, Lx=np.pi, Ly=np.pi, Lz=np.pi,
            interior_type="alternating", jitter_frac=0.0, seed=42
        )

        arbitrary_affine_field_3d = {
            "name": "General Affine 3D",
            "func": lambda x, y, z: np.array([
                1.5 + 0.8 * x - 0.3 * y + 0.5 * z,
                -0.7 + 0.4 * x + 1.2 * y - 0.6 * z,
                 2.1 - 0.2 * x + 0.7 * y - 0.5 * z
            ]),
            "rot": [0.7 - (-0.6), 0.5 - (-0.2), 0.4 - (-0.3)],
            "div": 0.8 + 1.2 - 0.5
        }

        res = run_patch_test_mode_3d(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            target_field=arbitrary_affine_field_3d, s_div=6.0,
            Ncx=3, Ncy=3, Ncz=3, points_per_dir=2
        )

        self.assertLess(res['error_dof_rel'], 1e-10, f"3D P1 failed on regular patch test: {res['error_dof_rel']}")
        self.assertLess(res['err_L2_E_rel'], 1e-10)

    def test_galerkin_patch_test_P1_irregular_3d(self):
        """Verifies that complete 12-node basis P1 passes the 3D Patch Test on irregular cloud (15% jitter)."""
        coords, vectors, is_bnd = generate_cavity_mesh_3d(
            Nx=5, Ny=5, Nz=5, Lx=np.pi, Ly=np.pi, Lz=np.pi,
            interior_type="random", jitter_frac=0.15, seed=42
        )

        arbitrary_affine_field_3d = {
            "name": "General Affine 3D",
            "func": lambda x, y, z: np.array([
                1.5 + 0.8 * x - 0.3 * y + 0.5 * z,
                -0.7 + 0.4 * x + 1.2 * y - 0.6 * z,
                 2.1 - 0.2 * x + 0.7 * y - 0.5 * z
            ]),
            "rot": [0.7 - (-0.6), 0.5 - (-0.2), 0.4 - (-0.3)],
            "div": 0.8 + 1.2 - 0.5
        }

        res = run_patch_test_mode_3d(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            target_field=arbitrary_affine_field_3d, s_div=6.0,
            Ncx=3, Ncy=3, Ncz=3, points_per_dir=2
        )

        self.assertLess(res['error_dof_rel'], 1e-10, f"3D P1 failed on irregular patch test: {res['error_dof_rel']}")
        self.assertLess(res['err_L2_E_rel'], 1e-10)

    def test_galerkin_patch_test_L1_failure_3d(self):
        """Verifies that incomplete Nédélec L1 basis (6 nodes) fails the 3D Patch Test for normal strain."""
        coords, vectors, is_bnd = generate_cavity_mesh_3d(
            Nx=5, Ny=5, Nz=5, Lx=np.pi, Ly=np.pi, Lz=np.pi,
            interior_type="alternating", jitter_frac=0.0, seed=42
        )

        normal_strain_3d = {
            "name": "Normal Strain 3D",
            "func": lambda x, y, z: np.array([x, -y, 0.0]),
            "rot": [0.0, 0.0, 0.0],
            "div": 0.0
        }

        res_L1 = run_patch_test_mode_3d(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            target_field=normal_strain_3d, basis="L1", s_div=6.0,
            Ncx=3, Ncy=3, Ncz=3, points_per_dir=2
        )

        self.assertGreater(res_L1['error_dof_rel'], 0.05, f"L1 should fail patch test for normal strain: {res_L1['error_dof_rel']}")


if __name__ == "__main__":
    unittest.main()
