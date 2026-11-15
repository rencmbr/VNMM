"""
Unit Tests for 2D Multimaterial Point Cloud Generators
======================================================
File: 2D/tests/test_multimaterial_mesh_2d.py
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

from src.multimaterial_mesh_2d import (
    create_dielectric_cavity_cloud_2d,
    create_h_ridge_waveguide_cloud_2d,
    create_circular_inclusion_guide_cloud_2d,
    create_quad_junction_cloud_2d,
    is_inside_h_guide,
    is_inside_circ_guide
)


class TestMultimaterialMesh2D(unittest.TestCase):

    def test_dielectric_cavity_cloud_structure(self):
        coords, vectors, is_pec, mat_ids = create_dielectric_cavity_cloud_2d(
            Lx=np.pi, Ly=np.pi, x_int=np.pi/2.0, Nx1=9, Nx2=9, Ny=17
        )
        self.assertEqual(len(coords), len(vectors))
        self.assertEqual(len(coords), len(is_pec))
        self.assertEqual(len(coords), len(mat_ids))

        # Check unit norm of all director vectors
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)

        # Check interface nodes (mat_id=0) located at x = pi/2
        int_nodes = coords[mat_ids == 0]
        self.assertGreater(len(int_nodes), 0)
        np.testing.assert_allclose(int_nodes[:, 0], np.pi/2.0, atol=1e-12)

        # Interface directors must be purely vertical (tangential)
        int_vecs = vectors[mat_ids == 0]
        np.testing.assert_allclose(int_vecs[:, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(int_vecs[:, 1]), 1.0, atol=1e-12)

    def test_h_ridge_waveguide_geometry(self):
        # Valid interior points
        self.assertTrue(is_inside_h_guide(0.0, 0.0))
        self.assertTrue(is_inside_h_guide(-1.5, 1.0))
        self.assertTrue(is_inside_h_guide(1.5, -1.0))

        # Points in exterior reentrant notches
        self.assertFalse(is_inside_h_guide(0.0, 1.0))
        self.assertFalse(is_inside_h_guide(0.0, -1.0))
        self.assertFalse(is_inside_h_guide(3.0, 0.0))

        coords, vectors, is_pec = create_h_ridge_waveguide_cloud_2d(nx_side=9, nx_mid=11, ny_side=17)
        self.assertGreater(len(coords), 100)
        for pt in coords:
            self.assertTrue(is_inside_h_guide(pt[0], pt[1]))

    def test_circular_inclusion_guide_geometry(self):
        r_hole = 1.0 / 3.0
        # Outside holes
        self.assertTrue(is_inside_circ_guide(0.0, 0.0, r_hole=r_hole))
        self.assertTrue(is_inside_circ_guide(-1.5, 0.5, r_hole=r_hole))

        # Inside PEC cylinder holes
        self.assertFalse(is_inside_circ_guide(-1.0, 0.0, r_hole=r_hole))
        self.assertFalse(is_inside_circ_guide(1.0, 0.0, r_hole=r_hole))

        coords, vectors, is_pec, mat_ids = create_circular_inclusion_guide_cloud_2d(
            nx_half=11, ny=11, n_circ=16, r_hole=r_hole
        )
        self.assertGreater(len(coords), 100)
        circ_pec = coords[is_pec & ((mat_ids == 1) | (mat_ids == 2))]
        self.assertGreater(len(circ_pec), 0)

    def test_quad_junction_cloud(self):
        coords, vectors, is_pec, tags = create_quad_junction_cloud_2d(Nq=8)
        self.assertEqual(len(coords), len(tags))

        # Origin point (0, 0) must be avoided
        dist_origin = np.hypot(coords[:, 0], coords[:, 1])
        self.assertTrue(np.all(dist_origin > 1e-6))

        # Must include all 4 semi-interfaces
        for tag in ['H_L', 'H_R', 'V_T', 'V_B']:
            self.assertIn(tag, tags)


if __name__ == '__main__':
    unittest.main()
