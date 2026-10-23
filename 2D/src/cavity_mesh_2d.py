import numpy as np


def generate_cavity_mesh(
    Nx=11, 
    Ny=11, 
    Lx=np.pi, 
    Ly=np.pi, 
    interior_type="alternating", 
    jitter_frac=0.0, 
    seed=42,
    tipo_interior=None,
    nx=None,
    ny=None,
    lx=None,
    ly=None
):
    if nx is not None:
        Nx = nx
    if ny is not None:
        Ny = ny
    if lx is not None:
        Lx = lx
    if ly is not None:
        Ly = ly
    """
    Generates a regular or quasi-regular nodal distribution for a rectangular cavity [0, Lx] x [0, Ly]
    with strict imposition of tangential directionality on PEC boundaries.
    
    Boundary Directionality Rules (PEC Walls):
    - Staggered distribution (centered on subintervals): nodes at midpoints (k + 0.5) * dx.
    - No nodes are placed on the four domain corners/vertices.
    - Horizontal edges (y=0 and y=Ly): unit vector t = [1, 0]^T (tangent to x-axis).
    - Vertical edges (x=0 and x=Lx): unit vector t = [0, 1]^T (tangent to y-axis).
    
    Parameters:
    - Nx: Number of nodal divisions along x-axis (minimum 3).
    - Ny: Number of nodal divisions along y-axis (minimum 3).
    - Lx: Cavity length along x direction (default: pi).
    - Ly: Cavity length along y direction (default: pi).
    - interior_type: Orientation pattern for internal nodes:
        * 'alternating' (or 'alternado'): Alternates directions at 45° and 135°.
        * 'diagonal': Fixed direction at 45° ([1/sqrt(2), 1/sqrt(2)]).
        * 'random' (or 'aleatorio'): Uniform angle theta in [0, 2*pi).
        * 'crossed' (or 'cruzado'): Cycles among [1,0], [0,1], [1,1]/sqrt(2), [-1,1]/sqrt(2).
    - jitter_frac: Maximum fractional random displacement for internal nodes (0.0 = regular).
    - seed: Seed for random reproducibility.
    
    Returns:
    - coords: Numpy array (N, 2) of Cartesian nodal coordinates (x, y).
    - vectors: Numpy array (N, 2) of associated unit direction vectors (tx, ty).
    - is_boundary: Boolean array (N,) True for nodes located on PEC boundaries.
    """
    if Nx < 3 or Ny < 3:
        raise ValueError("Nx and Ny must be greater than or equal to 3.")
        
    np.random.seed(seed)
    
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    coords_list = []
    vectors_list = []
    is_boundary_list = []
    
    # Normalize interior_type string
    if tipo_interior is not None:
        interior_type = tipo_interior
    type_str = interior_type.lower()
    if type_str in ["alternating", "alternado"]:
        mode = "alternating"
    elif type_str in ["diagonal"]:
        mode = "diagonal"
    elif type_str in ["random", "aleatorio"]:
        mode = "random"
    elif type_str in ["crossed", "cruzado"]:
        mode = "crossed"
    else:
        raise ValueError(f"Unknown interior_type: '{interior_type}'")
    
    # 1. Interior nodes
    for j in range(1, Ny - 1):
        for i in range(1, Nx - 1):
            x = i * dx
            y = j * dy
            
            if jitter_frac > 0.0:
                jx = np.random.uniform(-jitter_frac * dx, jitter_frac * dx)
                jy = np.random.uniform(-jitter_frac * dy, jitter_frac * dy)
                pos = np.array([x + jx, y + jy], dtype=float)
            else:
                pos = np.array([x, y], dtype=float)
                
            if mode == "diagonal":
                vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
            elif mode == "alternating":
                if (i + j) % 2 == 0:
                    vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                else:
                    vec = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
            elif mode == "crossed":
                idx_mod = (i + j) % 4
                if idx_mod == 0:
                    vec = np.array([1.0, 0.0], dtype=float)
                elif idx_mod == 1:
                    vec = np.array([0.0, 1.0], dtype=float)
                elif idx_mod == 2:
                    vec = np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
                else:
                    vec = np.array([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=float)
            elif mode == "random":
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)
                
            coords_list.append(pos)
            vectors_list.append(vec / np.linalg.norm(vec))
            is_boundary_list.append(False)
            
    # 2. Staggered boundary nodes (centered on subintervals)
    # No nodes on domain corners
    # Bottom boundary (y = 0): tangent [1, 0]
    for k in range(Nx - 1):
        pos = np.array([(k + 0.5) * dx, 0.0], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([1.0, 0.0], dtype=float))
        is_boundary_list.append(True)
        
    # Top boundary (y = Ly): tangent [1, 0]
    for k in range(Nx - 1):
        pos = np.array([(k + 0.5) * dx, Ly], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([1.0, 0.0], dtype=float))
        is_boundary_list.append(True)
        
    # Left boundary (x = 0): tangent [0, 1]
    for k in range(Ny - 1):
        pos = np.array([0.0, (k + 0.5) * dy], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([0.0, 1.0], dtype=float))
        is_boundary_list.append(True)
        
    # Right boundary (x = Lx): tangent [0, 1]
    for k in range(Ny - 1):
        pos = np.array([Lx, (k + 0.5) * dy], dtype=float)
        coords_list.append(pos)
        vectors_list.append(np.array([0.0, 1.0], dtype=float))
        is_boundary_list.append(True)
        
    return np.array(coords_list), np.array(vectors_list), np.array(is_boundary_list, dtype=bool)


# Backward compatibility alias
gerar_malha_cavidade = generate_cavity_mesh
