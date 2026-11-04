"""
Galerkin Patch Test for Vector Nodal Meshless Method in Three Dimensions (VNMM 3D).
Verifies exact reproduction of 3D affine vector fields by solving the
Galerkin linear system K * e = F with essential Dirichlet boundary conditions.

Fields tested:
1. Pure translations: E = [1, 0, 0]^T, [0, 1, 0]^T, [0, 0, 1]^T
2. Rigid body rotation (Pure Curl)
3. Volumetric expansion (Pure Div)
4. General arbitrary 3D affine field E = c + J @ x (12 independent parameters)
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
from scripts.support_nodes_vnmm_3d_6_l1 import support_nodes_vnmm_3d_6_l1


def assemble_patch_test_system_3d(
    coords,
    vectors,
    target_field,
    basis="P1",
    s_div=6.0,
    Ncx=4,
    Ncy=4,
    Ncz=4,
    Lx=np.pi,
    Ly=np.pi,
    Lz=np.pi,
    points_per_dir=2,
    det_tolerance=1e-6
):
    """
    Assembles stiffness matrix K = K_curl + s_div * K_div and load vector F
    consistent with affine target field E_target.
    """
    total_nodes = len(coords)
    tree = KDTree(coords)

    is_P1 = basis.upper() in ["P1", "12_P1", "12NOS"]
    n_supp = 12 if is_P1 else 6

    rot_exact = np.asarray(target_field["rot"], dtype=float)
    div_exact = float(target_field["div"])

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    z_edges = np.linspace(0.0, Lz, Ncz + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)

    rows_K, cols_K, data_K = [], [], []
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

                            if is_P1:
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
                            else:
                                supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_3d_6_l1(
                                    P=Pg,
                                    nodes_coords=coords,
                                    nodes_vectors=vectors,
                                    search_tree=tree,
                                    K=14,
                                    Tol_det=det_tolerance,
                                    adaptive=True,
                                    step_K=4
                                )
                                beta = np.linalg.inv(A_mat)

                                Phi_g = beta[0:3, :]
                                curl_Phi = 2.0 * beta[3:6, :].copy()
                                div_Phi = np.zeros(6, dtype=float)

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
                            K_local = rot_mat + div_mat

                            # Load vector contribution
                            f_curl = curl_Phi.T @ rot_exact * wg
                            f_div = div_Phi * (s_div * div_exact * wg)
                            f_local = f_curl + f_div

                            for a in range(n_supp):
                                Ia = supp_nodes[a]
                                F[Ia] += f_local[a]
                                for b in range(n_supp):
                                    Ib = supp_nodes[b]
                                    val = K_local[a, b]
                                    if abs(val) > 1e-18:
                                        rows_K.append(Ia)
                                        cols_K.append(Ib)
                                        data_K.append(val)

    K = coo_matrix((data_K, (rows_K, cols_K)), shape=(total_nodes, total_nodes)).tocsr()
    K = 0.5 * (K + K.T)
    return K, F, gauss_info


def run_patch_test_mode_3d(
    coords,
    vectors,
    is_boundary,
    target_field,
    basis="P1",
    s_div=6.0,
    Ncx=4,
    Ncy=4,
    Ncz=4,
    Lx=np.pi,
    Ly=np.pi,
    Lz=np.pi,
    points_per_dir=2
):
    """
    Executes 3D patch test for a single target field and returns error metrics.
    """
    total_nodes = len(coords)
    func_exact = target_field["func"]

    # Exact DoF values at all nodes
    e_exact = np.zeros(total_nodes, dtype=float)
    for idx in range(total_nodes):
        E_pt = func_exact(coords[idx, 0], coords[idx, 1], coords[idx, 2])
        e_exact[idx] = np.dot(E_pt, vectors[idx])

    K, F, gauss_info = assemble_patch_test_system_3d(
        coords=coords,
        vectors=vectors,
        target_field=target_field,
        basis=basis,
        s_div=s_div,
        Ncx=Ncx,
        Ncy=Ncy,
        Ncz=Ncz,
        Lx=Lx,
        Ly=Ly,
        Lz=Lz,
        points_per_dir=points_per_dir
    )

    interior_idx = np.where(~is_boundary)[0]
    boundary_idx = np.where(is_boundary)[0]

    # Partition system: K_ii * e_i = F_i - K_ib * e_b
    K_ii = K[interior_idx, :][:, interior_idx]
    K_ib = K[interior_idx, :][:, boundary_idx]
    F_i = F[interior_idx]
    e_b = e_exact[boundary_idx]

    rhs_i = F_i - K_ib @ e_b

    # Solve interior DoFs
    e_i = spsolve(K_ii, rhs_i)

    e_num = np.zeros(total_nodes, dtype=float)
    e_num[boundary_idx] = e_b
    e_num[interior_idx] = e_i

    # Error metrics
    dof_diff = np.abs(e_num[interior_idx] - e_exact[interior_idx])
    norm_exact = np.linalg.norm(e_exact[interior_idx])
    error_dof_rel = float(np.linalg.norm(dof_diff) / (norm_exact if norm_exact > 1e-12 else 1.0))
    error_dof_max = float(np.max(dof_diff))

    # Evaluate continuous L2 error over quadrature points
    sum_err_L2_sq = 0.0
    sum_exact_L2_sq = 0.0

    for g in gauss_info:
        Pg = g['Pg']
        wg = g['wg']
        supp = g['nodes']
        Phi_g = g['Phi_g']

        E_approx = Phi_g @ e_num[supp]
        E_ex = func_exact(Pg[0], Pg[1], Pg[2])

        diff_E = E_approx - E_ex
        sum_err_L2_sq += np.dot(diff_E, diff_E) * wg
        sum_exact_L2_sq += np.dot(E_ex, E_ex) * wg

    err_L2_E = float(np.sqrt(sum_err_L2_sq))
    norm_L2_exact = float(np.sqrt(sum_exact_L2_sq))
    err_L2_E_rel = float(err_L2_E / (norm_L2_exact if norm_L2_exact > 1e-12 else 1.0))

    return {
        "error_dof_rel": error_dof_rel,
        "error_dof_max": error_dof_max,
        "err_L2_E": err_L2_E,
        "err_L2_E_rel": err_L2_E_rel,
        "num_interior": len(interior_idx),
        "num_boundary": len(boundary_idx)
    }


if __name__ == "__main__":
    coords, vectors, is_bnd = generate_cavity_mesh_3d(Nx=5, Ny=5, Nz=5)
    field = {
        "name": "General Affine 3D",
        "func": lambda x, y, z: np.array([1.5 + 0.8*x - 0.3*y + 0.5*z,
                                          -0.7 + 0.4*x + 1.2*y - 0.6*z,
                                           2.1 - 0.2*x + 0.7*y - 0.5*z]),
        "rot": [0.7 - (-0.6), 0.5 - (-0.2), 0.4 - (-0.3)],
        "div": 0.8 + 1.2 - 0.5
    }
    res = run_patch_test_mode_3d(coords, vectors, is_bnd, field)
    print("3D Patch Test Error:", res["error_dof_rel"])
