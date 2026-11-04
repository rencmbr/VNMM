"""
Dense 3D Point Cloud Mesh Generator.
Generates volumetric point clouds on domain [-boundary_limit, boundary_limit]^3
with boundary nodes on the 6 bounding faces and interior nodes, saving to CSV (id, x, y, z, tx, ty, tz).
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.dirname(CURRENT_DIR)
DEFAULT_MESH_PATH_3D = os.path.join(DIR_3D, "meshes", "dense_mesh_3d.csv")


def generate_dense_mesh_3d(
    filename=DEFAULT_MESH_PATH_3D,
    num_boundary_nodes_per_face=16,
    num_interior_nodes=100,
    boundary_limit=1.0,
    seed=42,
    quiet=False
):
    """
    Generates a 3D mesh with boundary face nodes and interior nodes.
    """
    if seed is not None:
        np.random.seed(seed)

    coords = []
    vectors = []

    # 1. Boundary nodes on 6 faces:
    # Face -x (x = -boundary_limit): tangent in y-z
    # Face +x (x = +boundary_limit): tangent in y-z
    # Face -y (y = -boundary_limit): tangent in x-z
    # Face +y (y = +boundary_limit): tangent in x-z
    # Face -z (z = -boundary_limit): tangent in x-y
    # Face +z (z = +boundary_limit): tangent in x-y
    side_b = int(np.ceil(np.sqrt(num_boundary_nodes_per_face)))
    u = np.linspace(-boundary_limit * 0.9, boundary_limit * 0.9, side_b)
    v = np.linspace(-boundary_limit * 0.9, boundary_limit * 0.9, side_b)
    U, V = np.meshgrid(u, v)
    u_flat, v_flat = U.ravel(), V.ravel()

    for ui, vi in zip(u_flat, v_flat):
        # Face -x
        coords.append([-boundary_limit, ui, vi])
        vectors.append([0.0, 1.0, 0.0])
        # Face +x
        coords.append([boundary_limit, ui, vi])
        vectors.append([0.0, 0.0, 1.0])
        # Face -y
        coords.append([ui, -boundary_limit, vi])
        vectors.append([1.0, 0.0, 0.0])
        # Face +y
        coords.append([ui, boundary_limit, vi])
        vectors.append([0.0, 0.0, 1.0])
        # Face -z
        coords.append([ui, vi, -boundary_limit])
        vectors.append([1.0, 0.0, 0.0])
        # Face +z
        coords.append([ui, vi, boundary_limit])
        vectors.append([0.0, 1.0, 0.0])

    # 2. Interior nodes:
    int_pts = np.random.uniform(-boundary_limit * 0.8, boundary_limit * 0.8, size=(num_interior_nodes, 3))
    for pt in int_pts:
        coords.append(pt.tolist())
        # Random unit vector on S^2
        rnd_v = np.random.randn(3)
        rnd_v /= np.linalg.norm(rnd_v)
        vectors.append(rnd_v.tolist())

    coords = np.array(coords)
    vectors = np.array(vectors)

    df = pd.DataFrame({
        "id": np.arange(len(coords)),
        "x": np.round(coords[:, 0], 4),
        "y": np.round(coords[:, 1], 4),
        "z": np.round(coords[:, 2], 4),
        "tx": np.round(vectors[:, 0], 4),
        "ty": np.round(vectors[:, 1], 4),
        "tz": np.round(vectors[:, 2], 4)
    })

    if filename is not None:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        if not quiet:
            print(f"3D mesh saved to {filename} with {len(df)} nodes.")

    return coords, vectors


# Backward compatibility aliases
gerar_malha_densa_3d = generate_dense_mesh_3d
generate_dense_mesh = generate_dense_mesh_3d

