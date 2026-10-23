"""
Manufactured Solution (MMS) Galerkin Source Problem for VNMM 2D.
Solves the inhomogeneous vector Helmholtz equation with volumetric divergence regularization:
    curl(curl E) - s_div * grad(div E) + kappa * E = f(x, y) on [0, pi]^2
with essential Dirichlet boundary conditions on PEC walls.

Tests asymptotic convergence orders in L2, H(curl), and discrete DoF norms.
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, CURRENT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh import generate_cavity_mesh, gerar_malha_cavidade
from src.gauss_quadrature import get_gauss_points_weights_1d
from scripts.support_nodes_vnmm_2d_3_l1 import support_nodes_vnmm_2d_3_l1
from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1


def exact_solution_mms(x, y):
    """Exact vector field E*(x, y)."""
    return np.array([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)])


def exact_curl_mms(x, y):
    """(curl E*)_z = dEy/dx - dEx/dy = 2 * sin(x) * sin(y)."""
    return 2.0 * np.sin(x) * np.sin(y)


def exact_divergence_mms(x, y):
    """div(E*) = 0."""
    return 0.0


def source_mms(x, y, kappa=1.0):
    """f(x, y) = (2 + kappa) * E*(x, y)."""
    return (2.0 + kappa) * exact_solution_mms(x, y)


def assemble_and_solve_mms(
    coords,
    vectors,
    is_boundary,
    basis="P1",
    s_div=6.0,
    kappa=1.0,
    Ncx=None,
    Ncy=None,
    Lx=np.pi,
    Ly=np.pi,
    points_per_dir=3,
    det_tolerance=None,
    floor_tolerance=None,
    # Backward compatibility aliases
    base=None,
    pontos_por_dir=None,
    tolerancia_det=None,
    tol_piso=None
):
    """
    Assembles Galerkin system matrices and load vector for volumetric source MMS:
        A = K_curl + s_div * K_div + kappa * M
        F_i = integral_Omega N_i . f(x) dOmega
    Enforces essential Dirichlet conditions on boundary nodes and solves.
    Returns error metrics in L2, H(curl), and DoF norms.
    """
    if base is not None:
        basis = base
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    if tolerancia_det is not None:
        det_tolerance = tolerancia_det
    if tol_piso is not None:
        floor_tolerance = tol_piso

    total_nodes = len(coords)
    tree = KDTree(coords)
    is_P1 = basis.upper() in ["P1", "6_P1", "6NOS"]

    side_nodes = int(np.round(np.sqrt(total_nodes)))
    h_char = max(Lx / max(side_nodes - 1, 1), Ly / max(side_nodes - 1, 1))
    h_ref = np.pi / 20.0

    if Ncx is None:
        Ncx = max(side_nodes - 1, 6)
    if Ncy is None:
        Ncy = max(side_nodes - 1, 6)

    if det_tolerance is None:
        tol_base = 1e-4 * (h_char / h_ref)**4 if is_P1 else 1e-4 * (h_char / h_ref)
        det_tolerance = max(tol_base, floor_tolerance) if floor_tolerance is not None else tol_base

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)

    rows_A, cols_A, data_A = [], [], []
    F = np.zeros(total_nodes, dtype=float)
    gauss_info = []

    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        yc = 0.5 * (y0 + y1)
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_J = 0.25 * dx * dy

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy * eta
                    Pg = np.array([xg, yg])
                    wg = wi * wj * det_J

                    if is_P1:
                        supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_2d_6_p1(
                            P=Pg,
                            nodes_coords=coords,
                            nodes_vectors=vectors,
                            search_tree=tree,
                            K=12,
                            Tol_det=det_tolerance,
                            adaptive=True,
                            step_K=4
                        )
                        beta = np.linalg.inv(A_mat)
                        Phi_g = beta[0:2, :]
                        rot_Phi = beta[4, :] - beta[3, :]
                        div_Phi = beta[2, :] + beta[5, :]
                        n_supp = 6
                    else:
                        supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_2d_3_l1(
                            P=Pg,
                            nodes_coords=coords,
                            nodes_vectors=vectors,
                            search_tree=tree,
                            K=8,
                            Tol_det=det_tolerance,
                            adaptive=True,
                            step_K=2
                        )
                        beta = np.linalg.inv(A_mat)
                        Phi_g = beta[0:2, :]
                        rot_Phi = -2.0 * beta[2, :]
                        div_Phi = np.zeros(3, dtype=float)
                        n_supp = 3

                    f_val = source_mms(xg, yg, kappa=kappa)

                    rot_outer = np.outer(rot_Phi, rot_Phi) * wg
                    div_outer = np.outer(div_Phi, div_Phi) * (s_div * wg)
                    mass_local = (Phi_g.T @ Phi_g) * (kappa * wg)
                    a_local = rot_outer + div_outer + mass_local

                    f_local = (Phi_g.T @ f_val) * wg

                    gauss_info.append({
                        'Pg': Pg,
                        'wg': wg,
                        'nodes': supp_nodes,
                        'Phi_g': Phi_g,
                        'rot_Phi': rot_Phi,
                        'div_Phi': div_Phi
                    })

                    for a in range(n_supp):
                        Ia = supp_nodes[a]
                        F[Ia] += f_local[a]
                        for b in range(n_supp):
                            Ib = supp_nodes[b]
                            val = a_local[a, b]
                            if abs(val) > 1e-18:
                                rows_A.append(Ia)
                                cols_A.append(Ib)
                                data_A.append(val)

    A_mat = coo_matrix((data_A, (rows_A, cols_A)), shape=(total_nodes, total_nodes)).tocsr()
    A_mat = 0.5 * (A_mat + A_mat.T)

    e_exact = np.zeros(total_nodes, dtype=float)
    for k in range(total_nodes):
        E_pt = exact_solution_mms(coords[k, 0], coords[k, 1])
        e_exact[k] = np.dot(E_pt, vectors[k])

    idx_bnd = np.where(is_boundary)[0]
    idx_int = np.where(~is_boundary)[0]

    A_ii = A_mat[idx_int, :][:, idx_int]
    A_ib = A_mat[idx_int, :][:, idx_bnd]
    e_bnd = e_exact[idx_bnd]
    rhs_int = F[idx_int] - A_ib @ e_bnd

    e_sol_int = spsolve(A_ii, rhs_int)
    e_sol = np.zeros(total_nodes, dtype=float)
    e_sol[idx_bnd] = e_bnd
    e_sol[idx_int] = e_sol_int

    error_dof_inf = np.max(np.abs(e_sol_int - e_exact[idx_int]))
    norm_dof_inf = np.max(np.abs(e_exact[idx_int]))
    error_dof_rel = error_dof_inf / max(norm_dof_inf, 1e-14)

    err_L2_E_sq = 0.0
    norm_L2_E_sq = 0.0
    err_L2_rot_sq = 0.0
    norm_L2_rot_sq = 0.0
    err_L2_div_sq = 0.0

    for g in gauss_info:
        Pg = g['Pg']
        wg = g['wg']
        supp_nodes = g['nodes']
        Phi_g = g['Phi_g']
        rot_Phi = g['rot_Phi']
        div_Phi = g['div_Phi']

        e_supp = e_sol[supp_nodes]
        E_h = Phi_g @ e_supp
        rot_h = np.dot(rot_Phi, e_supp)
        div_h = np.dot(div_Phi, e_supp)

        E_ex = exact_solution_mms(Pg[0], Pg[1])
        rot_ex = exact_curl_mms(Pg[0], Pg[1])
        div_ex = exact_divergence_mms(Pg[0], Pg[1])

        err_L2_E_sq += np.sum((E_h - E_ex)**2) * wg
        norm_L2_E_sq += np.sum(E_ex**2) * wg

        err_L2_rot_sq += ((rot_h - rot_ex)**2) * wg
        norm_L2_rot_sq += (rot_ex**2) * wg

        err_L2_div_sq += ((div_h - div_ex)**2) * wg

    err_L2_E_rel = np.sqrt(err_L2_E_sq) / np.sqrt(norm_L2_E_sq)
    err_L2_rot_rel = np.sqrt(err_L2_rot_sq) / np.sqrt(norm_L2_rot_sq)
    res_div_L2 = np.sqrt(err_L2_div_sq)

    return {
        'h': h_char,
        'total_nodes': total_nodes,
        'internal_nodes': len(idx_int),
        'boundary_nodes': len(idx_bnd),
        'error_dof_rel': error_dof_rel,
        'err_L2_E_rel': err_L2_E_rel,
        'err_L2_curl_rel': err_L2_rot_rel,
        'div_res_L2': res_div_L2,
        # Backward compatibility aliases
        'N_total': total_nodes,
        'N_int': len(idx_int),
        'N_bnd': len(idx_bnd),
        'erro_dof_rel': error_dof_rel,
        'err_L2_rot_rel': err_L2_rot_rel,
        'res_div_L2': res_div_L2
    }


def run_mms_convergence_study():
    """
    Executes sequence of refinement on regular grid and computes
    asymptotic convergence orders O(h^p) for P1 and L1 bases.
    """
    print("=" * 78)
    print(" CONVERGENCE STUDY: MANUFACTURED SOLUTION (MMS) - VNMM 2D")
    print("=" * 78)
    print("Equation: curl(curl E) - s_div * grad(div E) + kappa * E = f(x) on [0, pi]^2")
    print("Exact: E* = [sin(x)cos(y), -cos(x)sin(y)]^T (solenoidal, curl=2*sin(x)sin(y))")
    print("Parameters: s_div = 6.0, kappa = 1.0, Quadrature: 3x3 Gauss per background cell")
    print("-" * 78)

    sizes = [9, 13, 17, 21, 25, 29]
    
    for basis in ["P1", "L1"]:
        print(f"\n>>> Basis: {basis} <<<")
        results = []
        for N in sizes:
            coords, vectors, is_boundary = generate_cavity_mesh(Nx=N, Ny=N, jitter_frac=0.0)
            res = assemble_and_solve_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_boundary,
                basis=basis,
                s_div=6.0,
                kappa=1.0,
                Ncx=N - 1,
                Ncy=N - 1,
                points_per_dir=3
            )
            results.append(res)
            print(f"N={N:2d}x{N:2d} (DoFs={res['total_nodes']:4d}, h={res['h']:.4f}): "
                  f"Err_L2(E) = {res['err_L2_E_rel']*100:6.3f}%, "
                  f"Err_H(curl) = {res['err_L2_curl_rel']*100:6.3f}%, "
                  f"Err_DoF = {res['error_dof_rel']*100:6.3f}%, "
                  f"Div_res = {res['div_res_L2']:.2e}")

        h_vals = np.array([r['h'] for r in results])
        e_L2 = np.array([r['err_L2_E_rel'] for r in results])
        e_curl = np.array([r['err_L2_curl_rel'] for r in results])
        e_dof = np.array([r['error_dof_rel'] for r in results])

        rate_L2, _ = np.polyfit(np.log(h_vals), np.log(e_L2), 1)
        rate_curl, _ = np.polyfit(np.log(h_vals), np.log(e_curl), 1)
        rate_dof, _ = np.polyfit(np.log(h_vals), np.log(e_dof), 1)

        print(f"\n--- Global Asymptotic Convergence Rates O(h^p) [{basis}] ---")
        print(f"Rate L2 (Field E):         p = {rate_L2:.2f}")
        print(f"Rate H(curl) (Curl):       p = {rate_curl:.2f}")
        print(f"Rate DoF (Discrete DoF):   p = {rate_dof:.2f}")

    print("\n" + "=" * 78)


# Backward compatibility aliases
solucao_exata_mms = exact_solution_mms
rotacional_exato_mms = exact_curl_mms
divergencia_exata_mms = exact_divergence_mms
fonte_mms = source_mms
montar_e_resolver_mms = assemble_and_solve_mms
executar_estudo_convergencia_mms = run_mms_convergence_study

if __name__ == "__main__":
    run_mms_convergence_study()
