import pandas as pd


def load_mesh_3d(filepath):
    """
    Reads 3D nodal data file.
    Expects CSV with columns: id, x, y, z, tx, ty, tz
    where (tx, ty, tz) are unit director components.
    """
    df = pd.read_csv(filepath)
    nodes_coords = df[['x', 'y', 'z']].to_numpy()
    nodes_vectors = df[['tx', 'ty', 'tz']].to_numpy()
    return nodes_coords, nodes_vectors


# Backward compatibility aliases
load_mesh = load_mesh_3d
carregar_malha = load_mesh_3d
carregar_malha_3d = load_mesh_3d

