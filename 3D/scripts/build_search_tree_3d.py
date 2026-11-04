from scipy.spatial import KDTree


def build_search_tree_3d(nodes_coords):
    """
    Constructs a spatial search tree (KDTree) from 3D nodal coordinates.
    """
    return KDTree(nodes_coords)


# Backward compatibility aliases
build_search_tree = build_search_tree_3d
construir_arvore_busca_3d = build_search_tree_3d
construir_arvore_busca = build_search_tree_3d

