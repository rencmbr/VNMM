"""
Unit Tests for Multimaterial Patch Test and First-Order Consistency
===================================================================
File: 2D/tests/test_multimaterial_patch_test_2d.py
"""

import unittest
import numpy as np
import sys
from pathlib import Path

DIR_TESTS = Path(__file__).resolve().parent
DIR_2D = DIR_TESTS.parent
ROOT_DIR = DIR_2D.parent

for p in [str(DIR_2D), str(ROOT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.multimaterial_mesh_2d import create_dielectric_cavity_cloud_2d
from src.multimaterial_assembler_2d import build_material_subsets
from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1


class TestMultimaterialPatchTest2D(unittest.TestCase):

    def test_patch_test_affine_machine_precision(self):
        """
        Ensures that for any affine linear vector field:
          E_patch(x, y) = [a0 + a1*x + a2*y, b0 + b1*x + b2*y]
        nodal reconstruction with P^1 reproduces both field and curl to machine precision.
        """
        coords, vectors, is_pec, mat_ids = create_dielectric_cavity_cloud_2d(
            Lx=2.0, Ly=2.0, x_int=1.0, Nx1=9, Nx2=9, Ny=17, jitter=0.15, seed=42
        )
        subsets = build_material_subsets(coords, vectors, mat_ids)

        a0, a1, a2 = 1.5, 0.8, -0.3
        b0, b1, b2 = -0.7, 0.4, 1.2
        curl_exact = b1 - a2  # 0.4 - (-0.3) = 0.7

        def exact_E(x, y):
            return np.array([a0 + a1 * x + a2 * y, b0 + b1 * x + b2 * y])

        u_nodes = np.zeros(len(coords))
        for i in range(len(coords)):
            u_nodes[i] = np.dot(exact_E(coords[i, 0], coords[i, 1]), vectors[i])

        eval_points = [
            (0.3, 0.4, 1),
            (0.7, 1.5, 1),
            (1.2, 0.8, 2),
            (1.8, 1.3, 2),
        ]

        for xp, yp, target_mat in eval_points:
            sub = subsets[target_mat]
            sel_loc, _, mat_A, _ = support_nodes_vnmm_2d_6_p1(
                P=np.array([xp, yp]),
                nodes_coords=sub['coords'],
                nodes_vectors=sub['vectors'],
                search_tree=sub['tree'],
                K=12,
                Tol_det=1e-5,
                adaptive=True
            )
            sel_glob = sub['indices'][sel_loc]
            beta = np.linalg.inv(mat_A)
            E_recon = beta[0:2, :] @ u_nodes[sel_glob]
            rot_recon = (beta[4, :] - beta[3, :]) @ u_nodes[sel_glob]

            E_ex = exact_E(xp, yp)
            np.testing.assert_allclose(E_recon, E_ex, atol=1e-13)
            self.assertAlmostEqual(rot_recon, curl_exact, delta=1e-12)


if __name__ == '__main__':
    unittest.main()
