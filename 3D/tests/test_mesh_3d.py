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

from scripts.load_mesh_3d import load_mesh_3d


class TestMesh3D(unittest.TestCase):
    def setUp(self):
        self.tests_dir = os.path.dirname(os.path.abspath(__file__))
        self.root_3d = os.path.dirname(self.tests_dir)
        self.mesh_path = os.path.join(self.root_3d, "meshes", "mesh_3d.csv")

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(self.mesh_path), f"File {self.mesh_path} does not exist.")

    def test_node_count_and_columns(self):
        coords, vectors = load_mesh_3d(self.mesh_path)
        self.assertEqual(coords.shape, (12, 3), "Coordinates must have shape (12, 3)")
        self.assertEqual(vectors.shape, (12, 3), "Vectors must have shape (12, 3)")

    def test_unit_director_vectors(self):
        _, vectors = load_mesh_3d(self.mesh_path)
        norms = np.linalg.norm(vectors, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-2, err_msg="Director vectors must be unit norm")

    def test_placement_matrix_non_singular(self):
        """
        Verifies that the 12x12 placement matrix A for complete linear polynomial space P1 in 3D
        is non-singular and well-conditioned (cond < 100).
        """
        coords, vectors = load_mesh_3d(self.mesh_path)
        A = np.zeros((12, 12), dtype=float)
        for i in range(12):
            x, y, z = coords[i]
            tx, ty, tz = vectors[i]
            A[i, :] = [
                tx, ty, tz,
                x * tx, y * tx, z * tx,
                x * ty, y * ty, z * ty,
                x * tz, y * tz, z * tz,
            ]
        det = np.linalg.det(A)
        cond = np.linalg.cond(A)
        self.assertGreater(abs(det), 1e-4, f"Determinant of A is too small: {det}")
        self.assertLess(cond, 100.0, f"Condition number of A is too high: {cond}")


if __name__ == "__main__":
    unittest.main()
