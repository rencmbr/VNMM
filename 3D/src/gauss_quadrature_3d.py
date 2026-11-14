"""
3D Tensor-Product Gauss-Legendre Quadrature Engine.
Maps 1D Gauss points and weights to 3D Cartesian background integration cells.
"""

import numpy as np


def get_gauss_points_weights_1d(num_points=2, n_pontos=None):
    """
    Returns 1D Gauss-Legendre quadrature points and weights in canonical interval [-1, 1].
    """
    if n_pontos is not None:
        num_points = n_pontos

    if num_points == 1:
        return np.array([0.0]), np.array([2.0])
    elif num_points == 2:
        val = 1.0 / np.sqrt(3.0)
        return np.array([-val, val]), np.array([1.0, 1.0])
    elif num_points == 3:
        val = np.sqrt(3.0 / 5.0)
        return np.array([-val, 0.0, val]), np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    elif num_points == 4:
        x1 = np.sqrt((3.0 - 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        x2 = np.sqrt((3.0 + 2.0 * np.sqrt(6.0 / 5.0)) / 7.0)
        w1 = (18.0 + np.sqrt(30.0)) / 36.0
        w2 = (18.0 - np.sqrt(30.0)) / 36.0
        return np.array([-x2, -x1, x1, x2]), np.array([w2, w1, w1, w2])
    else:
        xi, w = np.polynomial.legendre.leggauss(num_points)
        return xi, w


def generate_quadrature_cells_3d(
    Lx=1.00,
    Ly=0.50,
    Lz=0.75,
    Ncx=4,
    Ncy=4,
    Ncz=4,
    points_per_dir=2,
    pontos_por_dir=None,
    lx=None,
    ly=None,
    lz=None,
    ncx=None,
    ncy=None,
    ncz=None
):
    """
    Constructs a regular 3D brick cell grid covering [0, Lx] x [0, Ly] x [0, Lz]
    and maps 3D Gauss quadrature points and weights.
    
    Parameters:
    - Lx, Ly, Lz: Domain dimensions.
    - Ncx, Ncy, Ncz: Number of integration cells along each axis.
    - points_per_dir: Number of Gauss points per direction (e.g. 2 for 2x2x2 = 8 pts/cell).
    
    Returns:
    - gauss_points: Numpy array (M, 3) of Cartesian coordinates (xg, yg, zg).
    - gauss_weights: Numpy array (M,) of quadrature weights (w_g * det(J)).
    - cell_info: List of dictionaries with cell boundaries and metadata.
    """
    if lx is not None: Lx = lx
    if ly is not None: Ly = ly
    if lz is not None: Lz = lz
    if ncx is not None: Ncx = ncx
    if ncy is not None: Ncy = ncy
    if ncz is not None: Ncz = ncz
    if pontos_por_dir is not None: points_per_dir = pontos_por_dir

    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    z_edges = np.linspace(0.0, Lz, Ncz + 1)

    points_list = []
    weights_list = []
    cell_info = []

    for k in range(Ncz):
        z0, z1 = z_edges[k], z_edges[k + 1]
        dz = z1 - z0
        zc = 0.5 * (z0 + z1)

        for j in range(Ncy):
            y0, y1 = y_edges[j], y_edges[j + 1]
            dy = y1 - y0
            yc = 0.5 * (y0 + y1)

            for i in range(Ncx):
                x0, x1 = x_edges[i], x_edges[i + 1]
                dx = x1 - x0
                xc = 0.5 * (x0 + x1)

                det_J = 0.125 * dx * dy * dz

                cell_pts = []
                cell_wts = []

                for wk, zeta in zip(w_1d, xi_1d):
                    zg = zc + 0.5 * dz * zeta
                    for wj, eta in zip(w_1d, xi_1d):
                        yg = yc + 0.5 * dy * eta
                        for wi, xi in zip(w_1d, xi_1d):
                            xg = xc + 0.5 * dx * xi

                            wg = wi * wj * wk * det_J
                            pt = np.array([xg, yg, zg])

                            points_list.append(pt)
                            weights_list.append(wg)
                            cell_pts.append(pt)
                            cell_wts.append(wg)

                cell_info.append({
                    "bounds": ((x0, x1), (y0, y1), (z0, z1)),
                    "center": np.array([xc, yc, zc]),
                    "dims": np.array([dx, dy, dz]),
                    "det_J": det_J,
                    "points": np.array(cell_pts),
                    "weights": np.array(cell_wts)
                })

    return np.array(points_list), np.array(weights_list), cell_info


# Backward compatibility aliases
gerar_celulas_quadratura_3d = generate_quadrature_cells_3d
generate_quadrature_cells = generate_quadrature_cells_3d
obter_pontos_pesos_gauss_1d = get_gauss_points_weights_1d

