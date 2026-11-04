"""
Analytical 3D Vector Fields and Differential Operators.
Provides exact fields, 3D curl, and 3D divergence for benchmarks and validation.
"""

import numpy as np


def vector_field_te101(coords):
    """
    Computes TE101 mode on [0, pi]^3:
        E(x, y, z) = [0, -sin(x) * sin(z), 0]^T
        curl(E)    = [sin(x) * cos(z), 0, -cos(x) * sin(z)]^T
        div(E)     = 0
    """
    coords = np.asarray(coords)
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]

    E = np.zeros_like(coords)
    E[:, 1] = -np.sin(x) * np.sin(z)

    curl_E = np.zeros_like(coords)
    curl_E[:, 0] = np.sin(x) * np.cos(z)
    curl_E[:, 2] = -np.cos(x) * np.sin(z)

    div_E = np.zeros(len(coords), dtype=float)

    return E, curl_E, div_E


def vector_field_affine_3d(coords, c=None, J=None):
    """
    Computes general 3D affine vector field:
        E(x, y, z) = c + J @ [x, y, z]^T
    where c is a 3-vector and J is a 3x3 matrix.
    
    Exact derivatives:
        curl(E)_x = J[2, 1] - J[1, 2]
        curl(E)_y = J[0, 2] - J[2, 0]
        curl(E)_z = J[1, 0] - J[0, 1]
        div(E)    = J[0, 0] + J[1, 1] + J[2, 2]
    """
    coords = np.asarray(coords)
    if c is None:
        c = np.array([1.5, -0.7, 2.1])
    if J is None:
        J = np.array([
            [0.8, -0.3,  0.5],
            [0.4,  1.2, -0.6],
            [-0.2, 0.7, -0.5]
        ])

    E = np.empty_like(coords)
    for i, pt in enumerate(coords):
        E[i] = c + J @ pt

    rot_const = np.array([
        J[2, 1] - J[1, 2],
        J[0, 2] - J[2, 0],
        J[1, 0] - J[0, 1]
    ])
    div_const = float(np.trace(J))

    curl_E = np.tile(rot_const, (len(coords), 1))
    div_E = np.full(len(coords), div_const)

    return E, curl_E, div_E

