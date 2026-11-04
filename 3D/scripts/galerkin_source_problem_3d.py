"""
Manufactured Solution (MMS) Galerkin Source Problem for VNMM 3D.
Solves the inhomogeneous vector Helmholtz equation with volumetric divergence regularization:
    curl(curl E) - s_div * grad(div E) + kappa * E = f(x, y, z) on [0, pi]^3
with homogeneous essential Dirichlet boundary conditions on PEC walls.

Tests asymptotic convergence orders in L2, H(curl), and discrete DoF norms.
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_3D)
SRC_DIR = os.path.join(DIR_3D, "src")
for p in [DIR_3D, ROOT_DIR, CURRENT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_3d import generate_cavity_mesh_3d
from src.gauss_quadrature_3d import get_gauss_points_weights_1d
from scripts.support_nodes_vnmm_3d_12_p1 import support_nodes_vnmm_3d_12_p1


def exact_solution_mms_3d(x, y, z):
    """
    Exact divergence-free vector field E*(x, y, z) on [0, pi]^3 satisfying PEC boundary conditions:
        Ex =  cos(x) * sin(y) * sin(z)
        Ey =  sin(x) * cos(y) * sin(z)
        Ez = -2 * sin(x) * sin(y) * cos(z)
    div(E*) = 0 everywhere.
    curl(curl E*) = 3 * E*.
    """
    return np.array([
        np.cos(x) * np.sin(y) * np.sin(z),
        np.sin(x) * np.cos(y) * np.sin(z),
        -2.0 * np.sin(x) * np.sin(y) * np.cos(z)
    ])


def exact_curl_mms_3d(x, y, z):
    """
    curl(E*) for the manufactured 3D field:
        (curl E*)_x = -3 * sin(x) * cos(y) * cos(z)
        (curl E*)_y =  3 * cos(x) * sin(y) * cos(z)
        (curl E*)_z =  0
    """
    return np.array([
        -3.0 * np.sin(x) * np.cos(y) * np.cos(z),
         3.0 * np.cos(x) * np.sin(y) * np.cos(z),
         0.0
    ])


def exact_divergence_mms_3d(x, y, z):
    """div(E*) = 0 everywhere."""
    return 0.0


def source_mms_3d(x, y, z, kappa=1.0):
    """f(x, y, z) = (3 + kappa) * E*(x, y, z)."""
    return (3.0 + kappa) * exact_solution_mms_3d(x, y, z)


def assemble_and_solve_mms_3d(
    coords,
    vectors,
    is_boundary,
    s_div=6.0,
    kappa=1.0,
    Ncx=None,
    Ncy=None,
    Ncz=None,
    Lx=np.pi,
    Ly=np.pi,
    Lz=np.pi,
    points_per_dir=2,
    det_tolerance=1e-6
):
    """
    Assembles Galerkin system matrices and load vector for 3D MMS source problem:
        A = K_curl + s_div * K_div + kappa * M
        F_i = integral_Omega N_i . f(x) dOmega
    Enforces Dirichlet conditions and solves for interior degrees of freedom.
    """
    total_nodes = len(coords)
    tree = KDTree(coords)

    side_nodes = int(np.round(total_nodes ** (1.0 / 3.0)))
    if Ncx is None: Ncx = max(3, side_nodes - 1)
    if Ncy is None: Ncy = max(3, side_nodes - 1)
    if Ncz is None: Ncz = max(3, side_nodes - 1)

    h_char = max(Lx / max(side_nodes - 1, 1), Ly / max(side_nodes - 1, 1), Lz / max(side_nodes - 1, 1))

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    z_edges = np.linspace(0.0, Lz, Ncz + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)

    rows_A, cols_A, data_A = [], [], []
    F = np.zeros(total_nodes, dtype=float)
    gauss_info = []

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

                for wk, zeta in zip(w_1d, xi_1d):
                    zg = zc + 0.5 * dz * zeta
                    for wj, eta in zip(w_1d, xi_1d):
                        yg = yc + 0.5 * dy * eta
                        for wi, xi in zip(w_1d, xi_1d):
                            xg = xc + 0.5 * dx * xi
                            Pg = np.array([xg, yg, zg])
                            wg = wi * wj * wk * det_J

                            supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_3d_12_p1(
                                P=Pg,
                                nodes_coords=coords,
                                nodes_vectors=vectors,
                                search_tree=tree,
                                K=24,
                                Tol_det=det_tolerance,
                                adaptive=True,
                                step_K=6
                            )
                            beta = np.linalg.inv(A_mat)

                            Phi_g = beta[0:3, :]
                            curl_Phi = np.empty((3, 12), dtype=float)
                            curl_Phi[0, :] = beta[8, :] - beta[10, :]
                            curl_Phi[1, :] = beta[9, :] - beta[5, :]
                            curl_Phi[2, :] = beta[4, :] - beta[6, :]
                            div_Phi = beta[3, :] + beta[7, :] + beta[11, :]

                            gauss_info.append({
                                'Pg': Pg,
                                'wg': wg,
                                'nodes': supp_nodes,
                                'Phi_g': Phi_g,
                                'curl_Phi': curl_Phi,
                                'div_Phi': div_Phi
                            })

                            rot_mat = (curl_Phi.T @ curl_Phi) * wg
                            div_mat = np.outer(div_Phi, div_Phi) * (s_div * wg)
                            mass_mat = (Phi_g.T @ Phi_g) * (kappa * wg)
                            A_local = rot_mat + div_mat + mass_mat

                            f_val = source_mms_3d(xg, yg, zg, kappa=kappa)
                            f_local = (Phi_g.T @ f_val) * wg

                            for a in range(12):
                                Ia = supp_nodes[a]
                                F[Ia] += f_local[a]
                                for b in range(12):
                                    Ib = supp_nodes[b]
                                    val = A_local[a, b]
                                    if abs(val) > 1e-18:
                                        rows_A.append(Ia)
                                        cols_A.append(Ib)
                                        data_A.append(val)

    A = coo_matrix((data_A, (rows_A, cols_A)), shape=(total_nodes, total_nodes)).tocsr()
    A = 0.5 * (A + A.T)

    interior_idx = np.where(~is_boundary)[0]
    boundary_idx = np.where(is_boundary)[0]

    # Exact boundary condition e_b = 0
    e_exact = np.zeros(total_nodes, dtype=float)
    for idx in range(total_nodes):
        E_pt = exact_solution_mms_3d(coords[idx, 0], coords[idx, 1], coords[idx, 2])
        e_exact[idx] = np.dot(E_pt, vectors[idx])

    e_b = e_exact[boundary_idx]
    A_ii = A[interior_idx, :][:, interior_idx]
    A_ib = A[interior_idx, :][:, boundary_idx]
    F_i = F[interior_idx] - A_ib @ e_b

    e_i = spsolve(A_ii, F_i)

    e_num = np.zeros(total_nodes, dtype=float)
    e_num[boundary_idx] = e_b
    e_num[interior_idx] = e_i

    # Compute L2 and curl error norms
    sum_err_L2 = 0.0
    sum_exact_L2 = 0.0
    sum_err_curl = 0.0
    sum_exact_curl = 0.0

    for g in gauss_info:
        Pg = g['Pg']
        wg = g['wg']
        supp = g['nodes']
        Phi_g = g['Phi_g']
        curl_Phi = g['curl_Phi']

        E_num_pt = Phi_g @ e_num[supp]
        curl_num_pt = curl_Phi @ e_num[supp]

        E_ex_pt = exact_solution_mms_3d(Pg[0], Pg[1], Pg[2])
        curl_ex_pt = exact_curl_mms_3d(Pg[0], Pg[1], Pg[2])

        diff_E = E_num_pt - E_ex_pt
        diff_curl = curl_num_pt - curl_ex_pt

        sum_err_L2 += np.dot(diff_E, diff_E) * wg
        sum_exact_L2 += np.dot(E_ex_pt, E_ex_pt) * wg
        sum_err_curl += np.dot(diff_curl, diff_curl) * wg
        sum_exact_curl += np.dot(curl_ex_pt, curl_ex_pt) * wg

    err_L2_E_rel = float(np.sqrt(sum_err_L2) / np.sqrt(sum_exact_L2))
    err_L2_curl_rel = float(np.sqrt(sum_err_curl) / np.sqrt(sum_exact_curl))

    return {
        "err_L2_E_rel": err_L2_E_rel,
        "err_L2_curl_rel": err_L2_curl_rel,
        "h": h_char,
        "total_nodes": total_nodes,
        "active_dofs": len(interior_idx)
    }


if __name__ == "__main__":
    coords, vectors, is_bnd = generate_cavity_mesh_3d(Nx=5, Ny=5, Nz=5)
    res = assemble_and_solve_mms_3d(coords, vectors, is_bnd)
    print("MMS 3D L2 Error:", res["err_L2_E_rel"])

