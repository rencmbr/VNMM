import numpy as np
import pandas as pd
import os
from scipy.spatial import KDTree

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
DEFAULT_MESH_PATH = os.path.join(DIR_2D, "meshes", "dense_mesh_2d.csv")


def generate_dense_mesh(
    filename=DEFAULT_MESH_PATH,
    num_boundary_nodes=40,
    num_interior_nodes=150,
    boundary_limit=10.0,
    min_boundary_distance=None,
    min_node_distance=None,
    seed=42,
    max_attempts=100000,
    quiet=False,
    # Compatibility aliases
    nome_arquivo=None,
    num_nos_fronteira=None,
    num_nos_interior=None,
    limite=None,
    dist_min_fronteira=None,
    dist_min_nos=None,
    max_tentativas=None,
    silencioso=None,
    n_boundary=None,
    n_interior=None,
    n_boundary_nodes=None,
    n_interior_nodes=None
):
    """
    Generates a mesh on domain [-boundary_limit, boundary_limit]^2 with boundary nodes
    and randomly distributed interior nodes.
    
    Uses KDTree spatial structure to enforce:
    1. Minimum distance from interior nodes to boundary nodes.
    2. Minimum Euclidean distance among nodes.
    """
    if nome_arquivo is not None:
        filename = nome_arquivo
    if num_nos_fronteira is not None:
        num_boundary_nodes = num_nos_fronteira
    elif n_boundary is not None:
        num_boundary_nodes = n_boundary
    elif n_boundary_nodes is not None:
        num_boundary_nodes = n_boundary_nodes

    if num_nos_interior is not None:
        num_interior_nodes = num_nos_interior
    elif n_interior is not None:
        num_interior_nodes = n_interior
    elif n_interior_nodes is not None:
        num_interior_nodes = n_interior_nodes
    if limite is not None:
        boundary_limit = limite
    if dist_min_fronteira is not None:
        min_boundary_distance = dist_min_fronteira
    if dist_min_nos is not None:
        min_node_distance = dist_min_nos
    if silencioso is not None:
        quiet = silencioso

    if seed is not None:
        np.random.seed(seed)
     
    if filename is not None and os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
     
    coords = []
    vectors = []
    
    # 1. Boundary nodes
    nodes_per_edge = num_boundary_nodes // 4
    step = (2 * boundary_limit) / nodes_per_edge
    
    if min_boundary_distance is None:
        min_boundary_distance = 0.5 * step
    if min_node_distance is None:
        min_node_distance = 0.45 * step
    
    # Bottom Edge: y = -boundary_limit, tangent +x
    for i in range(nodes_per_edge):
        x = -boundary_limit + (i + 0.5) * step
        y = -boundary_limit
        coords.append([x, y])
        vectors.append([1.0, 0.0])
        
    # Right Edge: x = +boundary_limit, tangent +y
    for i in range(nodes_per_edge):
        x = boundary_limit
        y = -boundary_limit + (i + 0.5) * step
        coords.append([x, y])
        vectors.append([0.0, 1.0])
        
    # Top Edge: y = +boundary_limit, tangent -x
    for i in range(nodes_per_edge):
        x = boundary_limit - (i + 0.5) * step
        y = boundary_limit
        coords.append([x, y])
        vectors.append([-1.0, 0.0])
        
    # Left Edge: x = -boundary_limit, tangent -y
    for i in range(nodes_per_edge):
        x = -boundary_limit
        y = boundary_limit - (i + 0.5) * step
        coords.append([x, y])
        vectors.append([0.0, -1.0])
        
    # 2. Interior nodes
    lim_int = boundary_limit - min_boundary_distance
    side_n = int(np.ceil(np.sqrt(num_interior_nodes)))
    dx = (2.0 * lim_int) / side_n
    
    xs = np.linspace(-lim_int + dx / 2.0, lim_int - dx / 2.0, side_n)
    ys = np.linspace(-lim_int + dx / 2.0, lim_int - dx / 2.0, side_n)
    X, Y = np.meshgrid(xs, ys)
    pts_base = np.column_stack([X.ravel(), Y.ravel()])
    
    jitter = np.random.uniform(-0.25 * dx, 0.25 * dx, size=pts_base.shape)
    pts_int = pts_base + jitter
    
    if len(pts_int) > num_interior_nodes:
        idx_sel = np.random.choice(len(pts_int), num_interior_nodes, replace=False)
        pts_int = pts_int[idx_sel]
        
    for pt in pts_int:
        coords.append(pt.tolist())
        theta = np.random.uniform(0, 2 * np.pi)
        vectors.append([np.cos(theta), np.sin(theta)])
        
    coords = np.array(coords)
    vectors = np.array(vectors)
    
    df = pd.DataFrame({
        'id': np.arange(len(coords)),
        'x': np.round(coords[:, 0], 4),
        'y': np.round(coords[:, 1], 4),
        'tx': np.round(vectors[:, 0], 4),
        'ty': np.round(vectors[:, 1], 4)
    })
    
    if filename is not None:
        df.to_csv(filename, index=False)
        if not quiet:
            print(f"Mesh successfully generated with {len(df)} nodes ({num_boundary_nodes} boundary + {len(pts_int)} interior).")
            print(f"Mean characteristic spacing: h ~ {step:.4f}")
            print(f"Saved to: {filename}")
            
    return coords, vectors


# Backward compatibility alias
gerar_malha_densa = generate_dense_mesh

if __name__ == "__main__":
    generate_dense_mesh()
