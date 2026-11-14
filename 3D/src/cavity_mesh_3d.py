"""
Cavity Mesh Generator in Three Dimensions (VNMM 3D).
Generates regular or quasi-regular 3D nodal point clouds in domain [0, Lx] x [0, Ly] x [0, Lz]
with strict enforcement of tangential directionality on PEC boundary walls.
"""

import numpy as np


def generate_cavity_mesh_3d(
    Nx=7,
    Ny=7,
    Nz=7,
    Lx=1.00,
    Ly=0.50,
    Lz=0.75,
    interior_type="alternating",
    jitter_frac=0.0,
    seed=42,
    # Backward compatibility aliases
    tipo_interior=None,
    nx=None,
    ny=None,
    nz=None,
    lx=None,
    ly=None,
    lz=None
):
    """
    Generates a nodal distribution for a 3D rectangular cavity [0, Lx] x [0, Ly] x [0, Lz]
    with strict imposition of tangential directionality on PEC boundaries (E x n = 0).
    
    Boundary Directionality Rules (PEC Walls):
    - Staggered distribution or grid nodes:
    - Normal along x (x=0 and x=Lx): unit vectors t lie in y-z plane (t_x = 0).
    - Normal along y (y=0 and y=Ly): unit vectors t lie in x-z plane (t_y = 0).
    - Normal along z (z=0 and z=Lz): unit vectors t lie in x-y plane (t_z = 0).
    
    Parameters:
    - Nx, Ny, Nz: Number of nodal divisions along x, y, and z axes (minimum 3).
    - Lx, Ly, Lz: Cavity lengths along x, y, and z directions (default: pi).
    - interior_type: Orientation pattern for internal nodes:
        * 'alternating' (or 'alternado'): Alternates 3D diagonal orientations.
        * 'cartesian' (or 'cartesiano'): Cycles through unit axes [1,0,0], [0,1,0], [0,0,1].
        * 'random' (or 'aleatorio'): Uniform random unit vectors on 2-sphere S^2.
    - jitter_frac: Maximum fractional random displacement for internal nodes (0.0 = regular).
    - seed: Seed for random reproducibility.
    
    Returns:
    - coords: Numpy array (N, 3) of Cartesian nodal coordinates (x, y, z).
    - vectors: Numpy array (N, 3) of associated unit direction vectors (tx, ty, tz).
    - is_boundary: Boolean array (N,) True for nodes located on PEC boundaries.
    """
    if nx is not None: Nx = nx
    if ny is not None: Ny = ny
    if nz is not None: Nz = nz
    if lx is not None: Lx = lx
    if ly is not None: Ly = ly
    if lz is not None: Lz = lz
    if tipo_interior is not None: interior_type = tipo_interior

    if Nx < 3 or Ny < 3 or Nz < 3:
        raise ValueError("Nx, Ny, and Nz must be greater than or equal to 3.")

    np.random.seed(seed)

    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dz = Lz / (Nz - 1)

    type_str = interior_type.lower()
    if type_str in ["alternating", "alternado"]:
        mode = "alternating"
    elif type_str in ["cartesian", "cartesiano"]:
        mode = "cartesian"
    elif type_str in ["random", "aleatorio"]:
        mode = "random"
    elif type_str in ["triad", "triade", "orthogonal_triad"]:
        mode = "triad"
    else:
        raise ValueError(f"Unknown interior_type: '{interior_type}'")

    coords_list = []
    vectors_list = []
    is_boundary_list = []

    if mode == "triad":
        # Complete 3-DoF orthogonal vector triad (Ex, Ey, Ez) at each 3D grid vertex
        # with exact tangential PEC boundary condition (n x E = 0) on all 6 faces and 12 edges
        for k in range(Nz):
            for j in range(Ny):
                for i in range(Nx):
                    x = i * dx
                    y = j * dy
                    z = k * dz
                    if jitter_frac > 0.0 and (0 < i < Nx - 1) and (0 < j < Ny - 1) and (0 < k < Nz - 1):
                        jx = np.random.uniform(-jitter_frac * dx, jitter_frac * dx)
                        jy = np.random.uniform(-jitter_frac * dy, jitter_frac * dy)
                        jz = np.random.uniform(-jitter_frac * dz, jitter_frac * dz)
                        pos = np.array([x + jx, y + jy, z + jz], dtype=float)
                    else:
                        pos = np.array([x, y, z], dtype=float)

                    # Ex degree of freedom: tangent to y-walls (j=0, Ny-1) and z-walls (k=0, Nz-1)
                    coords_list.append(pos)
                    vectors_list.append(np.array([1.0, 0.0, 0.0], dtype=float))
                    is_boundary_list.append(bool(j == 0 or j == Ny - 1 or k == 0 or k == Nz - 1))

                    # Ey degree of freedom: tangent to x-walls (i=0, Nx-1) and z-walls (k=0, Nz-1)
                    coords_list.append(pos)
                    vectors_list.append(np.array([0.0, 1.0, 0.0], dtype=float))
                    is_boundary_list.append(bool(i == 0 or i == Nx - 1 or k == 0 or k == Nz - 1))

                    # Ez degree of freedom: tangent to x-walls (i=0, Nx-1) and y-walls (j=0, Ny-1)
                    coords_list.append(pos)
                    vectors_list.append(np.array([0.0, 0.0, 1.0], dtype=float))
                    is_boundary_list.append(bool(i == 0 or i == Nx - 1 or j == 0 or j == Ny - 1))

        return (
            np.array(coords_list, dtype=float),
            np.array(vectors_list, dtype=float),
            np.array(is_boundary_list, dtype=bool)
        )

    # 1. Interior nodes (1 <= i < Nx-1, 1 <= j < Ny-1, 1 <= k < Nz-1)
    sq3 = 1.0 / np.sqrt(3.0)
    alt_vectors = [
        np.array([sq3, sq3, sq3]),
        np.array([-sq3, sq3, sq3]),
        np.array([sq3, -sq3, sq3]),
        np.array([sq3, sq3, -sq3]),
        np.array([-sq3, -sq3, sq3]),
        np.array([-sq3, sq3, -sq3]),
        np.array([sq3, -sq3, -sq3]),
        np.array([-sq3, -sq3, -sq3])
    ]

    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                x = i * dx
                y = j * dy
                z = k * dz

                if jitter_frac > 0.0:
                    jx = np.random.uniform(-jitter_frac * dx, jitter_frac * dx)
                    jy = np.random.uniform(-jitter_frac * dy, jitter_frac * dy)
                    jz = np.random.uniform(-jitter_frac * dz, jitter_frac * dz)
                    pos = np.array([x + jx, y + jy, z + jz], dtype=float)
                else:
                    pos = np.array([x, y, z], dtype=float)

                if mode == "alternating":
                    idx_vec = (i + 2 * j + 3 * k) % len(alt_vectors)
                    vec = alt_vectors[idx_vec].copy()
                elif mode == "cartesian":
                    idx_vec = (i + j + k) % 3
                    vec = np.zeros(3, dtype=float)
                    vec[idx_vec] = 1.0
                elif mode == "random":
                    v = np.random.randn(3)
                    vec = v / np.linalg.norm(v)

                coords_list.append(pos)
                vectors_list.append(vec / np.linalg.norm(vec))
                is_boundary_list.append(False)

    # 2. Boundary nodes (Staggered or centered on subintervals of the 6 faces)
    # Face 1 & 2: Bottom and Top z-faces (z = 0 and z = Lz) -> normal along z -> tangent in x-y plane
    for k_val in [0.0, Lz]:
        for j in range(Ny - 1):
            for i in range(Nx - 1):
                pos = np.array([(i + 0.5) * dx, (j + 0.5) * dy, k_val], dtype=float)
                # Alternate tangent between x and y
                if (i + j) % 2 == 0:
                    vec = np.array([1.0, 0.0, 0.0])
                else:
                    vec = np.array([0.0, 1.0, 0.0])
                coords_list.append(pos)
                vectors_list.append(vec)
                is_boundary_list.append(True)

    # Face 3 & 4: Left and Right x-faces (x = 0 and x = Lx) -> normal along x -> tangent in y-z plane
    for i_val in [0.0, Lx]:
        for k in range(Nz - 1):
            for j in range(Ny - 1):
                pos = np.array([i_val, (j + 0.5) * dy, (k + 0.5) * dz], dtype=float)
                if (j + k) % 2 == 0:
                    vec = np.array([0.0, 1.0, 0.0])
                else:
                    vec = np.array([0.0, 0.0, 1.0])
                coords_list.append(pos)
                vectors_list.append(vec)
                is_boundary_list.append(True)

    # Face 5 & 6: Front and Back y-faces (y = 0 and y = Ly) -> normal along y -> tangent in x-z plane
    for j_val in [0.0, Ly]:
        for k in range(Nz - 1):
            for i in range(Nx - 1):
                pos = np.array([(i + 0.5) * dx, j_val, (k + 0.5) * dz], dtype=float)
                if (i + k) % 2 == 0:
                    vec = np.array([1.0, 0.0, 0.0])
                else:
                    vec = np.array([0.0, 0.0, 1.0])
                coords_list.append(pos)
                vectors_list.append(vec)
                is_boundary_list.append(True)

    coords_arr = np.array(coords_list, dtype=float)
    vectors_arr = np.array(vectors_list, dtype=float)
    is_boundary_arr = np.array(is_boundary_list, dtype=bool)

    return coords_arr, vectors_arr, is_boundary_arr


# Backward compatibility aliases
gerar_malha_cavidade_3d = generate_cavity_mesh_3d
generate_cavity_mesh = generate_cavity_mesh_3d
gerar_malha_cavidade = generate_cavity_mesh_3d

