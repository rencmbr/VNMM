"""
Point Cloud Generators for 2D Multimaterial Domains (VNMM)
==========================================================
File: 2D/src/multimaterial_mesh_2d.py

Provides functions for creating and discretizing nodal point clouds in multimaterial geometries:
- create_dielectric_cavity_cloud_2d: bipartite rectangular cavity [0, Lx] x [0, Ly]
- create_h_ridge_waveguide_cloud_2d: H-shaped ridge waveguide with reentrant corners
- create_circular_inclusion_guide_cloud_2d: bimaterial waveguide with circular PEC cylinders
- create_quad_junction_cloud_2d: cavity with four-material cross junction
"""

import numpy as np


def create_dielectric_cavity_cloud_2d(Lx=np.pi, Ly=np.pi, x_int=np.pi/2.0, Nx1=11, Nx2=11, Ny=21, jitter=0.0, seed=None):
    """
    Generates point cloud for rectangular cavity with a vertical material interface at x = x_int.
    - Interior nodes with alternating +-45 degree directors.
    - Interface nodes on Gamma_int with purely vertical (tangential) directors.
    - Staggered midpoint PEC boundary nodes avoiding corner points.
    """
    if seed is not None:
        np.random.seed(seed)

    dx1 = x_int / (Nx1 - 1)
    dx2 = (Lx - x_int) / (Nx2 - 1)
    dy = Ly / (Ny - 1)
    h_min = min(dx1, dx2, dy)

    coords, vectors, is_pec, mat_ids = [], [], [], []

    # 1. Interior nodes for Region 1 (Material 1)
    for j in range(1, Ny - 1):
        y = j * dy
        for i in range(1, Nx1 - 1):
            x = i * dx1
            if jitter > 0.0:
                x += jitter * h_min * (np.random.rand() - 0.5)
                y_pert = y + jitter * h_min * (np.random.rand() - 0.5)
                x = np.clip(x, 0.1 * dx1, x_int - 0.1 * dx1)
                y_pert = np.clip(y_pert, 0.1 * dy, Ly - 0.1 * dy)
            else:
                y_pert = y
            ang = np.pi/4.0 if (i + j) % 2 == 0 else 3.0*np.pi/4.0
            coords.append([x, y_pert])
            vectors.append([np.cos(ang), np.sin(ang)])
            is_pec.append(False)
            mat_ids.append(1)

    # 2. Interior nodes for Region 2 (Material 2)
    for j in range(1, Ny - 1):
        y = j * dy
        for i in range(1, Nx2 - 1):
            x = x_int + i * dx2
            if jitter > 0.0:
                x += jitter * h_min * (np.random.rand() - 0.5)
                y_pert = y + jitter * h_min * (np.random.rand() - 0.5)
                x = np.clip(x, x_int + 0.1 * dx2, Lx - 0.1 * dx2)
                y_pert = np.clip(y_pert, 0.1 * dy, Ly - 0.1 * dy)
            else:
                y_pert = y
            ang = np.pi/4.0 if (i + j) % 2 == 0 else 3.0*np.pi/4.0
            coords.append([x, y_pert])
            vectors.append([np.cos(ang), np.sin(ang)])
            is_pec.append(False)
            mat_ids.append(2)

    # 3. Interface nodes on Gamma_int (x = x_int)
    for j in range(1, Ny - 1):
        y = j * dy
        coords.append([x_int, y])
        vectors.append([0.0, 1.0])  # Tangential to vertical interface
        is_pec.append(False)
        mat_ids.append(0)  # Shared interface tag

    # 4. Staggered PEC boundary nodes (avoiding corner singularities)
    for i in range(Nx1 - 1):
        coords.append([(i + 0.5) * dx1, 0.0]); vectors.append([1.0, 0.0]); is_pec.append(True); mat_ids.append(1)
        coords.append([(i + 0.5) * dx1, Ly]);  vectors.append([1.0, 0.0]); is_pec.append(True); mat_ids.append(1)

    for i in range(Nx2 - 1):
        coords.append([x_int + (i + 0.5) * dx2, 0.0]); vectors.append([1.0, 0.0]); is_pec.append(True); mat_ids.append(2)
        coords.append([x_int + (i + 0.5) * dx2, Ly]);  vectors.append([1.0, 0.0]); is_pec.append(True); mat_ids.append(2)

    for j in range(Ny - 1):
        coords.append([0.0, (j + 0.5) * dy]); vectors.append([0.0, 1.0]); is_pec.append(True); mat_ids.append(1)
        coords.append([Lx,  (j + 0.5) * dy]); vectors.append([0.0, 1.0]); is_pec.append(True); mat_ids.append(2)

    return (
        np.array(coords, dtype=float),
        np.array(vectors, dtype=float),
        np.array(is_pec, dtype=bool),
        np.array(mat_ids, dtype=int)
    )


def is_inside_h_guide(x, y, tol=1e-12):
    """Checks whether point (x, y) lies inside the H-shaped ridge waveguide."""
    if x < -2.5 - tol or x > 2.5 + tol or y < -1.5 - tol or y > 1.5 + tol:
        return False
    if -1.0 + tol < x < 1.0 - tol and y > 0.5 + tol:
        return False
    if -1.0 + tol < x < 1.0 - tol and y < -0.5 - tol:
        return False
    return True


def create_h_ridge_waveguide_cloud_2d(nx_side=11, nx_mid=15, ny_side=21):
    """
    Generates point cloud for the H-shaped ridge waveguide (Thesis Test 4.3.2).
    x in [-2.5, 2.5], y in [-1.5, 1.5], reentrant notch in [-1, 1] x (|y| > 0.5).
    """
    x_left = np.linspace(-2.5, -1.0, nx_side)
    x_mid = np.linspace(-1.0, 1.0, nx_mid)
    x_right = np.linspace(1.0, 2.5, nx_side)
    x_all = np.unique(np.concatenate([x_left, x_mid, x_right]))
    y_all = np.linspace(-1.5, 1.5, ny_side)

    coords, vectors, is_pec = [], [], []

    for i, x0 in enumerate(x_all):
        for j, y0 in enumerate(y_all):
            if not is_inside_h_guide(x0, y0):
                continue
            pec_flag = False
            t_pec = np.array([0.0, 0.0])

            if abs(x0 - (-2.5)) < 1e-6 or abs(x0 - 2.5) < 1e-6:
                pec_flag = True; t_pec = np.array([0.0, 1.0])
            elif (abs(y0 - 1.5) < 1e-6 or abs(y0 - (-1.5)) < 1e-6) and (x0 <= -1.0 + 1e-6 or x0 >= 1.0 - 1e-6):
                pec_flag = True; t_pec = np.array([1.0, 0.0])
            elif (abs(x0 - (-1.0)) < 1e-6 or abs(x0 - 1.0) < 1e-6) and abs(y0) >= 0.5 - 1e-6:
                pec_flag = True; t_pec = np.array([0.0, 1.0])
            elif (abs(y0 - 0.5) < 1e-6 or abs(y0 - (-0.5)) < 1e-6) and abs(x0) <= 1.0 + 1e-6:
                pec_flag = True; t_pec = np.array([1.0, 0.0])

            if pec_flag:
                t = t_pec / np.linalg.norm(t_pec)
            else:
                ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(ang), np.sin(ang)])

            coords.append([x0, y0])
            vectors.append(t)
            is_pec.append(pec_flag)

    return np.array(coords, dtype=float), np.array(vectors, dtype=float), np.array(is_pec, dtype=bool)


def is_inside_circ_guide(x, y, r_hole=1.0/3.0, tol=1e-12):
    """Checks whether point (x, y) lies inside the bimaterial waveguide with circular PEC cylinders."""
    if x < -2.0 - tol or x > 2.0 + tol or y < -1.0 - tol or y > 1.0 + tol:
        return False
    if (x + 1.0)**2 + y**2 < r_hole**2 - tol:
        return False
    if (x - 1.0)**2 + y**2 < r_hole**2 - tol:
        return False
    return True


def create_circular_inclusion_guide_cloud_2d(nx_half=17, ny=17, n_circ=20, r_hole=1.0/3.0):
    """
    Generates point cloud for bimaterial waveguide with circular PEC inclusions (Thesis Test 4.3.3).
    [-2, 2] x [-1, 1], interface at x = 0, PEC cylinders centered at (-1, 0) and (1, 0).
    """
    x_left = np.linspace(-2.0, 0.0, nx_half)
    x_right = np.linspace(0.0, 2.0, nx_half)
    x_all = np.unique(np.concatenate([x_left, x_right]))
    y_all = np.linspace(-1.0, 1.0, ny)

    coords, vectors, is_pec, mat_ids = [], [], [], []

    for i, x0 in enumerate(x_all):
        for j, y0 in enumerate(y_all):
            if not is_inside_circ_guide(x0, y0, r_hole=r_hole):
                continue
            d1 = np.hypot(x0 + 1.0, y0)
            d2 = np.hypot(x0 - 1.0, y0)
            if abs(d1 - r_hole) < 0.06 or abs(d2 - r_hole) < 0.06:
                continue

            pec_flag = False
            is_int = abs(x0) < 1e-6
            t = np.array([0.0, 0.0])

            if abs(x0 - (-2.0)) < 1e-6 or abs(x0 - 2.0) < 1e-6:
                pec_flag = True; t = np.array([0.0, 1.0])
            elif abs(y0 - (-1.0)) < 1e-6 or abs(y0 - 1.0) < 1e-6:
                pec_flag = True; t = np.array([1.0, 0.0])
            elif is_int:
                pec_flag = False; t = np.array([0.0, 1.0])
            else:
                ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(ang), np.sin(ang)])

            m_id = 0 if is_int else (1 if x0 < 0.0 else 2)
            coords.append([x0, y0])
            vectors.append(t)
            is_pec.append(pec_flag)
            mat_ids.append(m_id)

    # Nodes placed along the cylindrical PEC boundary rings
    thetas = np.linspace(0, 2 * np.pi, n_circ, endpoint=False)
    for xc_c, m_id in [(-1.0, 1), (1.0, 2)]:
        for th in thetas:
            xb = xc_c + r_hole * np.cos(th)
            yb = r_hole * np.sin(th)
            t_circ = np.array([-np.sin(th), np.cos(th)])
            coords.append([xb, yb])
            vectors.append(t_circ)
            is_pec.append(True)
            mat_ids.append(m_id)

    return (
        np.array(coords, dtype=float),
        np.array(vectors, dtype=float),
        np.array(is_pec, dtype=bool),
        np.array(mat_ids, dtype=int)
    )


def create_quad_junction_cloud_2d(Nq=12):
    """
    Generates point cloud for four-material corner junction cavity in [-1, 1] x [-1, 1].
    - Interior nodes with alternating +-45 degree directors in each quadrant.
    - Tangential nodes aligned along the 4 semi-interfaces (x=0, y=0).
    - Avoids placing nodes at the central singular vertex (0, 0) and the four PEC outer corners.
    """
    h = 1.0 / Nq
    coords, vectors, is_pec, tags = [], [], [], []

    # 1. Interior nodes for each quadrant
    for q_id, (sx, sy) in [(1, (-1, 1)), (2, (1, 1)), (3, (-1, -1)), (4, (1, -1))]:
        q_tag = f'Q{q_id}'
        for i in range(1, Nq):
            for j in range(1, Nq):
                ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                coords.append([sx * i * h, sy * j * h])
                vectors.append([np.cos(ang), np.sin(ang)])
                is_pec.append(False)
                tags.append(q_tag)

    # 2. Tangential nodes aligned along the 4 semi-interfaces
    for k in range(1, Nq):
        pos = k * h
        coords.append([-pos, 0.0]); vectors.append([1.0, 0.0]); is_pec.append(False); tags.append('H_L')
        coords.append([+pos, 0.0]); vectors.append([1.0, 0.0]); is_pec.append(False); tags.append('H_R')
        coords.append([0.0, +pos]); vectors.append([0.0, 1.0]); is_pec.append(False); tags.append('V_T')
        coords.append([0.0, -pos]); vectors.append([0.0, 1.0]); is_pec.append(False); tags.append('V_B')

    # 3. Staggered nodes on external PEC walls
    for k in range(Nq):
        pos = (k + 0.5) * h
        coords.append([-pos, 1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q1')
        coords.append([+pos, 1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q2')
        coords.append([-pos,-1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q3')
        coords.append([+pos,-1.0]); vectors.append([1.0, 0.0]); is_pec.append(True); tags.append('Q4')
        coords.append([-1.0,+pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q1')
        coords.append([-1.0,-pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q3')
        coords.append([ 1.0,+pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q2')
        coords.append([ 1.0,-pos]); vectors.append([0.0, 1.0]); is_pec.append(True); tags.append('Q4')

    return (
        np.array(coords, dtype=float),
        np.array(vectors, dtype=float),
        np.array(is_pec, dtype=bool),
        np.array(tags)
    )
