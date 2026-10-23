import numpy as np


def vector_field_te11(P, A=1.0):
    """
    Computes electric field E(x, y) and curl(E) for TE11 mode (m=1, n=1)
    in a PEC cavity of dimensions Lx = 20, Ly = 20 shifted to interval [-10, 10] x [-10, 10].
    
    Equations of TE11 mode:
        E_x(x, y) =  A * cos(pi * (x + 10.0) / 20.0) * sin(pi * (y + 10.0) / 20.0)
        E_y(x, y) = -A * sin(pi * (x + 10.0) / 20.0) * cos(pi * (y + 10.0) / 20.0)
        
    Analytical curl (z-component):
        (curl E)_z = dE_y/dx - dE_x/dy
                   = -(A * pi / 10.0) * cos(pi * (x + 10.0) / 20.0) * cos(pi * (y + 10.0) / 20.0)
        
    Parameters:
    - P: 2D coordinates [x, y], tuple (x, y), or numpy array (2,) / (N, 2).
    - A: Field amplitude (default: 1.0).
         
    Returns:
    - E: Array containing vector [Ex, Ey] at P (shape (2,) or (N, 2)).
    - curl_E: Analytical curl z-component at P (float or shape (N,)).
    """
    P = np.asarray(P, dtype=float)
    
    if P.ndim == 1:
        x, y = P[0], P[1]
        u = np.pi * (x + 10.0) / 20.0
        v = np.pi * (y + 10.0) / 20.0
        
        Ex =  A * np.cos(u) * np.sin(v)
        Ey = -A * np.sin(u) * np.cos(v)
        curl_E = float(-(A * np.pi / 10.0) * np.cos(u) * np.cos(v))
        
        return np.array([Ex, Ey], dtype=float), curl_E
    elif P.ndim == 2:
        x = P[:, 0]
        y = P[:, 1]
        u = np.pi * (x + 10.0) / 20.0
        v = np.pi * (y + 10.0) / 20.0
        
        Ex =  A * np.cos(u) * np.sin(v)
        Ey = -A * np.sin(u) * np.cos(v)
        curl_E = -(A * np.pi / 10.0) * np.cos(u) * np.cos(v)
        
        return np.column_stack([Ex, Ey]), curl_E
    else:
        raise ValueError(f"Invalid input shape: {P.shape}. Expected (2,) or (N, 2).")


# Backward compatibility and convenience aliases
func_vet = vector_field_te11


def exact_field_te11(P_or_x, y=None, A=1.0):
    """Returns exact electric field vector for TE11 mode."""
    if y is not None:
        P = [P_or_x, y]
    else:
        P = P_or_x
    E, _ = vector_field_te11(P, A=A)
    return E


def exact_curl_te11(P_or_x, y=None, A=1.0):
    """Returns exact curl value for TE11 mode."""
    if y is not None:
        P = [P_or_x, y]
    else:
        P = P_or_x
    _, curl = vector_field_te11(P, A=A)
    return curl
