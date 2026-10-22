from scipy.spatial import KDTree


def build_search_tree(nodes_coords):
    """
    Constructs spatial search structure (KDTree) from nodal coordinates.
    
    Parameters:
    - nodes_coords: Numpy array of global nodal coordinates.
    
    Returns:
    - tree: KDTree object ready for spatial nearest-neighbor queries.
    """
    return KDTree(nodes_coords)


# Backward compatibility alias
construir_arvore_busca = build_search_tree
