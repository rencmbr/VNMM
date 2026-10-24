import os
import sys
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
SCRIPTS_DIR = os.path.join(DIR_2D, "scripts")
for p in [DIR_2D, ROOT_DIR, SCRIPTS_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from scripts.mlpg_patch_test_2d import (
    AFFINE_MODES,
    run_mlpg_patch_test_mode
)


class TestMLPGPatchTest2D(unittest.TestCase):
    def test_mlpg_patch_test_regular_mesh(self):
        """Verifies machine-precision satisfaction of all 7 affine patch test modes on regular mesh."""
        coords, vectors, is_bnd = generate_cavity_mesh(
            Nx=9, Ny=9, interior_type="alternating", jitter_frac=0.0
        )

        for mode in AFFINE_MODES:
            res = run_mlpg_patch_test_mode(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                target_field=mode,
                s_div=6.0
            )
            rel_err = res["error_dof_rel"]
            self.assertLess(
                rel_err,
                1e-11,
                f"Patch test failed on regular mesh for mode '{mode['name']}': rel_err = {rel_err}"
            )

    def test_mlpg_patch_test_irregular_cloud(self):
        """Verifies machine-precision satisfaction of all 7 affine patch test modes on 25% jitter cloud."""
        coords, vectors, is_bnd = generate_cavity_mesh(
            Nx=9, Ny=9, interior_type="random", jitter_frac=0.25, seed=42
        )

        for mode in AFFINE_MODES:
            res = run_mlpg_patch_test_mode(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                target_field=mode,
                s_div=6.0
            )
            rel_err = res["error_dof_rel"]
            self.assertLess(
                rel_err,
                1e-11,
                f"Patch test failed on irregular cloud for mode '{mode['name']}': rel_err = {rel_err}"
            )


if __name__ == "__main__":
    unittest.main()
