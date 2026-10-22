import numpy as np


def get_gauss_points_weights_1d(num_points=2, n_pontos=None):
    """
    Returns 1D Gauss-Legendre quadrature points and weights in the canonical interval [-1, 1].
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
        # Use numpy for arbitrary quadrature orders
        xi, w = np.polynomial.legendre.leggauss(num_points)
        return xi, w


def generate_quadrature_cells(
    Lx=np.pi, 
    Ly=np.pi, 
    Ncx=10, 
    Ncy=10, 
    points_per_dir=2,
    pontos_por_dir=None,
    lx=None,
    ly=None,
    ncx=None,
    ncy=None,
    num_cells_x=None,
    num_cells_y=None,
    points_per_direction=None
):
    if lx is not None:
        Lx = lx
    if ly is not None:
        Ly = ly
    if ncx is not None:
        Ncx = ncx
    elif num_cells_x is not None:
        Ncx = num_cells_x
    if ncy is not None:
        Ncy = ncy
    elif num_cells_y is not None:
        Ncy = num_cells_y
    if points_per_direction is not None:
        points_per_dir = points_per_direction
    elif pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    """
    Constructs a regular rectangular background cell grid covering [0, Lx] x [0, Ly]
    and maps 2D Gauss quadrature points and weights.
    
    Parameters:
    - Lx: Domain length in x.
    - Ly: Domain length in y.
    - Ncx: Number of integration cells along x.
    - Ncy: Number of integration cells along y.
    - points_per_dir: Number of Gauss points per direction per cell (e.g., 2 for 2x2 = 4 points/cell).
    
    Returns:
    - gauss_points: Numpy array (M, 2) of Cartesian coordinates (xg, yg) for all Gauss points.
    - gauss_weights: Numpy array (M,) of effective quadrature weights (w_g * det(J_e)).
    - cell_info: List of dictionaries with integration cell boundaries and metadata.
    """
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir

    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)
    
    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    
    points_list = []
    weights_list = []
    cell_info = []
    
    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        y_mid = 0.5 * (y0 + y1)
        
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            x_mid = 0.5 * (x0 + x1)
            
            # Jacobian of the canonical mapping [-1, 1]^2 -> [x0, x1] x [y0, y1]
            # dx_dxi = dx / 2, dy_deta = dy / 2 => det(J) = dx * dy / 4
            det_J = 0.25 * dx * dy
            
            cell_info.append({
                'cell_id': j * Ncx + i,
                'x_range': (x0, x1),
                'y_range': (y0, y1),
                'area': dx * dy
            })
            
            for wi, xi in zip(w_1d, xi_1d):
                xg = x_mid + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = y_mid + 0.5 * dy * eta
                    
                    effective_weight = wi * wj * det_J
                    
                    points_list.append([xg, yg])
                    weights_list.append(effective_weight)
                    
    return np.array(points_list, dtype=float), np.array(weights_list, dtype=float), cell_info
