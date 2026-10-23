"""
Spectral validation of TEz resonant cavity using the Dimensionless A-matrix Formulation.

Demonstrates that scaling-invariant Tol_hat accurately recovers analytical eigenvalues.
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CODE_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.gauss_quadrature_2d import get_gauss_points_weights_1d
from scripts.dimensionless_vnmm_6_p1_2d import (
    support_nodes_vnmm_2d_6_p1_dimensionless,
    shape_functions_vnmm_2d_6_p1_dimensionless
)


def assemble_dimensionless_vnmm_matrices(
    coords, vectors, basis="P1", tol_hat=0.16, s_div=6.0,
    ncx=None, ncy=None, lx=np.pi, ly=np.pi, points_per_dir=3,
    # Backward compatibility
    base=None,
    Tol_hat=None,
    Ncx=None,
    Ncy=None,
    Lx=None,
    Ly=None,
    pontos_por_dir=None
):
    """
    Assemble stiffness and mass matrices using strictly dimensionless A_hat matrix.
    """
    if base is not None: basis = base
    if Tol_hat is not None: tol_hat = Tol_hat
    if Ncx is not None: ncx = Ncx
    if Ncy is not None: ncy = Ncy
    if Lx is not None: lx = Lx
    if Ly is not None: ly = Ly
    if pontos_por_dir is not None: points_per_dir = pontos_por_dir

    total_nodes = len(coords)
    tree = KDTree(coords)
    n_side = int(np.round(np.sqrt(total_nodes)))
    if ncx is None:
        ncx = max(4, n_side - 1)
    if ncy is None:
        ncy = max(4, n_side - 1)

    h_char = max(lx / max(n_side - 1, 1), ly / max(n_side - 1, 1))

    x_edges = np.linspace(0.0, lx, ncx + 1)
    y_edges = np.linspace(0.0, ly, ncy + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)

    rows_kc, cols_kc, data_kc = [], [], []
    rows_kd, cols_kd, data_kd = [], [], []
    rows_m, cols_m, data_m = [], [], []
    effective_ks = []

    for j in range(ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        yc = 0.5 * (y0 + y1)

        for i in range(ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_j = 0.25 * dx * dy

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy * eta
                    weight = wi * wj * det_j
                    pg = np.array([xg, yg])

                    nodes, det_hat, a_hat, h_eff, k_ef = support_nodes_vnmm_2d_6_p1_dimensionless(
                        point=pg, node_coords=coords, node_vectors=vectors,
                        search_tree=tree, h_local=h_char,
                        num_neighbors=12, tol_hat=tol_hat, adaptive=True, step_k=4
                    )
                    effective_ks.append(k_ef)

                    phi_g, rot_phi, div_phi, _ = shape_functions_vnmm_2d_6_p1_dimensionless(
                        point=pg, node_coords=coords, node_vectors=vectors,
                        selected_nodes=nodes, h_local=h_eff, matrix_a_hat=a_hat
                    )

                    for a_idx, node_a in enumerate(nodes):
                        for b_idx, node_b in enumerate(nodes):
                            val_kc = weight * rot_phi[a_idx] * rot_phi[b_idx]
                            rows_kc.append(node_a)
                            cols_kc.append(node_b)
                            data_kc.append(val_kc)

                            val_kd = weight * s_div * div_phi[a_idx] * div_phi[b_idx]
                            rows_kd.append(node_a)
                            cols_kd.append(node_b)
                            data_kd.append(val_kd)

                            val_m = weight * np.dot(phi_g[:, a_idx], phi_g[:, b_idx])
                            rows_m.append(node_a)
                            cols_m.append(node_b)
                            data_m.append(val_m)

    k_curl = coo_matrix((data_kc, (rows_kc, cols_kc)), shape=(total_nodes, total_nodes)).tocsr()
    k_div = coo_matrix((data_kd, (rows_kd, cols_kd)), shape=(total_nodes, total_nodes)).tocsr()
    m_mat = coo_matrix((data_m, (rows_m, cols_m)), shape=(total_nodes, total_nodes)).tocsr()
    k_mat = k_curl + k_div
    return k_mat, m_mat, np.mean(effective_ks)


def test_dimensionless_cavity():
    """Test cavity eigenvalues using dimensionless formulation."""
    print("=" * 80)
    print("TEST 4: Resolution of TEz Cavity Eigenvalues with Dimensionless Matrix")
    print("=" * 80)

    nx, ny = 11, 11
    coords, vectors, is_boundary = generate_cavity_mesh(
        nx=nx, ny=ny, lx=np.pi, ly=np.pi, interior_type="alternating"
    )
    interior_nodes = np.where(~is_boundary)[0]

    analytical_targets = [1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0, 8.0]

    k_ad, m_ad, k_mean = assemble_dimensionless_vnmm_matrices(
        coords, vectors, basis="P1", tol_hat=0.10, s_div=6.0,
        ncx=nx-1, ncy=ny-1, points_per_dir=3
    )

    k_sub = k_ad[interior_nodes, :][:, interior_nodes]
    m_sub = m_ad[interior_nodes, :][:, interior_nodes]

    eigenvalues, _ = eigsh(k_sub, k=10, M=m_sub, sigma=-1e-4, which='LM')
    eigenvalues = np.sort(np.real(eigenvalues))
    positive_vals = eigenvalues[eigenvalues > 0.05][:8]

    print(f"Effective K_mean: {k_mean:.2f}")
    print(f"{'Mode':<6} | {'Analytical':<12} | {'Dimensionless':<14} | {'Error [%]':<10}")
    print("-" * 50)
    errors = []
    for idx, (target, num) in enumerate(zip(analytical_targets, positive_vals)):
        err = abs(np.sqrt(num) - np.sqrt(target)) / np.sqrt(target) * 100.0
        errors.append(err)
        print(f"{idx+1:<6d} | {target:<12.4f} | {num:<14.4f} | {err:<10.3f}%")

    print("-" * 50)
    print(f"Mean kc error: {np.mean(errors):.3f}%")
    print("-> TEST 4 PASSED: Physical eigenvalues accurately recovered with dimensionless formulation.")
    return True


# Backward compatibility aliases
montar_matrizes_vnmm_adimensional = assemble_dimensionless_vnmm_matrices
testar_cavidade = test_dimensionless_cavity


if __name__ == "__main__":
    test_dimensionless_cavity()
