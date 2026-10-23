import pandas as pd


def load_mesh(filepath):
    """
    Reads nodal data file.
    Expects CSV with columns: id, x, y, tx, ty
    where (tx, ty) are unit director components.
    """
    df = pd.read_csv(filepath)
    nodes_coords = df[['x', 'y']].to_numpy()
    nodes_vectors = df[['tx', 'ty']].to_numpy()
    return nodes_coords, nodes_vectors


# Backward compatibility alias
carregar_malha = load_mesh
