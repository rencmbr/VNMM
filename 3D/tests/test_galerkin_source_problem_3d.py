"""
Unit Tests for 3D MMS Galerkin Source Problem (VNMM 3D).
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
from scripts.galerkin_source_problem_3d import assemble_and_solve_mms_3d


class TestGalerkinSourceProblem3D(unittest.TestCase):
    def test_mms_source_problem_3d_solvability(self):
        """Verifies solvability and error boundedness of 3D MMS source problem."""
        coords, vectors, is_bnd = generate_cavity_mesh_3d(Nx=5, Ny=5, Nz=5, jitter_frac=0.0)
        res = assemble_and_solve_mms_3d(
            coords=coords,
            vectors=vectors,
            is_boundary=is_bnd,
            s_div=6.0,
            kappa=1.0,
            Ncx=3,
            Ncy=3,
            Ncz=3,
            points_per_dir=2
        )

        self.assertIsNotNone(res['err_L2_E_rel'])
        self.assertIsNotNone(res['err_L2_curl_rel'])
        self.assertGreater(res['active_dofs'], 0)
        # Error must be finite and within expected numerical range for coarse 3D grid
        self.assertLess(res['err_L2_E_rel'], 2.0)


if __name__ == "__main__":
    unittest.main()

