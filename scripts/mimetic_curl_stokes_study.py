"""
Study and evaluation of the Mimetic Stokes formulation for VNMM 2D (without divergence penalty, s_div = 0).

Evaluates curl stiffness K_stokes assembled via boundary contour circulation
around background cells:
    (curl E)_e = (1 / Area_e) * oint_{dOmega_e} E . dl
"""

import os
import sys
import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cavity_mesh import generate_cavity_mesh
from src.gauss_quadrature import get_gauss_points_weights_1d
from src.eigen_solver_cavity import ANALYTICAL_REFERENCE_MODES
from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1


def assemble_vnmm_mimetic_stokes_matrices(
    coords, 
    vectors, 
    is_boundary, 
    lx=np.pi, 
    ly=np.pi, 
    ncx=10, 
    ncy=10, 
    edge_points=2, 
    mass_points=2, 
    mu_r=1.0, 
    epsilon_r=1.0, 
    det_tolerance=1e-4,
    # Backward compatibility
    Lx=None,
    Ly=None,
    Ncx=None,
    Ncy=None,
    pontos_aresta=None,
    pontos_massa=None,
    tolerancia_det=None
):
    """
    Assemble VNMM 2D matrices using Mimetic Stokes contour circulation:
    - K_stokes = sum_e (1 / Area_e) * (C_e^T C_e), with s_div = 0.
    - M: Standard mass matrix integrated over cell interior.
    """
    if Lx is not None: lx = Lx
    if Ly is not None: ly = Ly
    if Ncx is not None: ncx = Ncx
    if Ncy is not None: ncy = Ncy
    if pontos_aresta is not None: edge_points = pontos_aresta
    if pontos_massa is not None: mass_points = pontos_massa
    if tolerancia_det is not None: det_tolerance = tolerancia_det

    total_nodes = len(coords)
    tree = KDTree(coords)
    inv_mu = 1.0 / mu_r
    eps = epsilon_r
    
    x_edges = np.linspace(0.0, lx, ncx + 1)
    y_edges = np.linspace(0.0, ly, ncy + 1)
    
    xi_1d, w_1d = get_gauss_points_weights_1d(edge_points)
    xi_m, w_m = get_gauss_points_weights_1d(mass_points)
    
    k_stokes = np.zeros((total_nodes, total_nodes), dtype=float)
    rows_m, cols_m, data_m = [], [], []
    
    for j in range(ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        ym = 0.5 * (y0 + y1)
        
        for i in range(ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            xm = 0.5 * (x0 + x1)
            area = dx * dy
            
            # Boundary circulation vector C_e
            c_e = np.zeros(total_nodes, dtype=float)
            
            # 1. Bottom edge: y = y0, x from x0 to x1 (+Ex)
            for xi, w in zip(xi_1d, w_1d):
                xg = xm + 0.5 * dx * xi
                pg = np.array([xg, y0])
                nodes, _, a_mat, _ = support_nodes_vnmm_2d_6_p1(
                    point=pg, node_coords=coords, node_vectors=vectors,
                    search_tree=tree, num_neighbors=12, det_tolerance=det_tolerance, adaptive=True
                )
                beta = la.inv(a_mat)
                weight = 0.5 * dx * w
                for a in range(6):
                    c_e[nodes[a]] += weight * beta[0, a]
                    
            # 2. Right edge: x = x1, y from y0 to y1 (+Ey)
            for eta, w in zip(xi_1d, w_1d):
                yg = ym + 0.5 * dy * eta
                pg = np.array([x1, yg])
                nodes, _, a_mat, _ = support_nodes_vnmm_2d_6_p1(
                    point=pg, node_coords=coords, node_vectors=vectors,
                    search_tree=tree, num_neighbors=12, det_tolerance=det_tolerance, adaptive=True
                )
                beta = la.inv(a_mat)
                weight = 0.5 * dy * w
                for a in range(6):
                    c_e[nodes[a]] += weight * beta[1, a]
                    
            # 3. Top edge: y = y1, x from x1 to x0 (-Ex)
            for xi, w in zip(xi_1d, w_1d):
                xg = xm + 0.5 * dx * xi
                pg = np.array([xg, y1])
                nodes, _, a_mat, _ = support_nodes_vnmm_2d_6_p1(
                    point=pg, node_coords=coords, node_vectors=vectors,
                    search_tree=tree, num_neighbors=12, det_tolerance=det_tolerance, adaptive=True
                )
                beta = la.inv(a_mat)
                weight = -0.5 * dx * w
                for a in range(6):
                    c_e[nodes[a]] += weight * beta[0, a]
                    
            # 4. Left edge: x = x0, y from y1 to y0 (-Ey)
            for eta, w in zip(xi_1d, w_1d):
                yg = ym + 0.5 * dy * eta
                pg = np.array([x0, yg])
                nodes, _, a_mat, _ = support_nodes_vnmm_2d_6_p1(
                    point=pg, node_coords=coords, node_vectors=vectors,
                    search_tree=tree, num_neighbors=12, det_tolerance=det_tolerance, adaptive=True
                )
                beta = la.inv(a_mat)
                weight = -0.5 * dy * w
                for a in range(6):
                    c_e[nodes[a]] += weight * beta[1, a]
                    
            k_stokes += (inv_mu / area) * np.outer(c_e, c_e)
            
            # Mass matrix integration
            det_j = 0.25 * dx * dy
            for wi, xi in zip(w_m, xi_m):
                xg = xm + 0.5 * dx * xi
                for wj, eta in zip(w_m, xi_m):
                    yg = ym + 0.5 * dy * eta
                    pg = np.array([xg, yg])
                    
                    nodes, _, a_mat, _ = support_nodes_vnmm_2d_6_p1(
                        point=pg, node_coords=coords, node_vectors=vectors,
                        search_tree=tree, num_neighbors=12, det_tolerance=det_tolerance, adaptive=True
                    )
                    beta = la.inv(a_mat)
                    weight_m = wi * wj * det_j
                    
                    phi_x = beta[0, :]
                    phi_y = beta[1, :]
                    m_elem = eps * weight_m * (np.outer(phi_x, phi_x) + np.outer(phi_y, phi_y))
                    
                    for a in range(6):
                        na = nodes[a]
                        for b in range(6):
                            nb = nodes[b]
                            rows_m.append(na)
                            cols_m.append(nb)
                            data_m.append(m_elem[a, b])
                        
    m_glob = coo_matrix((data_m, (rows_m, cols_m)), shape=(total_nodes, total_nodes)).tocsr()
    k_glob = csr_matrix(k_stokes)
    
    k_glob = 0.5 * (k_glob + k_glob.T)
    m_glob = 0.5 * (m_glob + m_glob.T)
    
    return k_glob, m_glob


def test_mimetic_curl_stokes(
    nx=21, 
    ny=21, 
    ncx=10, 
    ncy=10, 
    lx=np.pi, 
    ly=np.pi, 
    edge_points=2, 
    mass_points=2, 
    num_eigenvalues=10, 
    zero_tol=0.05,
    # Backward compatibility
    Nx=None,
    Ny=None,
    Ncx=None,
    Ncy=None,
    Lx=None,
    Ly=None,
    pontos_aresta=None,
    pontos_massa=None,
    num_autovalores=None,
    tol_zero=None
):
    """Test cavity eigenvalues using the mimetic Stokes curl formulation."""
    if Nx is not None: nx = Nx
    if Ny is not None: ny = Ny
    if Ncx is not None: ncx = Ncx
    if Ncy is not None: ncy = Ncy
    if Lx is not None: lx = Lx
    if Ly is not None: ly = Ly
    if pontos_aresta is not None: edge_points = pontos_aresta
    if pontos_massa is not None: mass_points = pontos_massa
    if num_autovalores is not None: num_eigenvalues = num_autovalores
    if tol_zero is not None: zero_tol = tol_zero

    print(f"=== TESTING MIMETIC STOKES FORMULATION (Nx={nx}, Ny={ny}, Ncx={ncx}, Ncy={ncy}) ===")
    
    t0 = time.time()
    coords, vectors, is_boundary = generate_cavity_mesh(nx=nx, ny=ny, lx=lx, ly=ly)
    
    h_ref = np.pi / 20.0
    h_current = lx / (nx - 1)
    tol_det = 1e-4 * (h_current / h_ref)**4
    
    k_glob, m_glob = assemble_vnmm_mimetic_stokes_matrices(
        coords, vectors, is_boundary, lx=lx, ly=ly, ncx=ncx, ncy=ncy,
        edge_points=edge_points, mass_points=mass_points, det_tolerance=tol_det
    )
    
    interior_indices = np.where(~is_boundary)[0]
    k_red = k_glob[interior_indices, :][:, interior_indices].toarray()
    m_red = m_glob[interior_indices, :][:, interior_indices].toarray()
    
    assembly_time = time.time() - t0
    
    t0_eig = time.time()
    try:
        eigenvalues, _ = la.eigh(k_red, m_red)
    except Exception:
        eigenvalues, _ = la.eig(k_red, m_red)
        eigenvalues = np.real(eigenvalues)
    solve_time = time.time() - t0_eig
    
    null_mask = eigenvalues <= zero_tol
    positive_mask = eigenvalues > zero_tol
    
    num_null = int(np.sum(null_mask))
    physical_eigenvalues = np.sort(eigenvalues[positive_mask])[:num_eigenvalues]
    
    ref_vals = np.array([item[2] for item in ANALYTICAL_REFERENCE_MODES[:len(physical_eigenvalues)]])
    ref_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:len(physical_eigenvalues)]])
    
    kc_num = np.sqrt(np.maximum(physical_eigenvalues, 0.0))
    lambda_errors = np.abs(physical_eigenvalues - ref_vals) / ref_vals * 100.0
    kc_errors = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    result = {
        'nx': nx,
        'ny': ny,
        'ncx': ncx,
        'ncy': ncy,
        'num_dofs': len(interior_indices),
        'num_null_discarded': num_null,
        'assembly_time': assembly_time,
        'solve_time': solve_time,
        'total_time': assembly_time + solve_time,
        'all_eigenvalues': eigenvalues,
        'physical_eigenvalues': physical_eigenvalues,
        'analytical_eigenvalues': ref_vals,
        'numerical_kc': kc_num,
        'analytical_kc': ref_kc,
        'lambda_errors_pct': lambda_errors,
        'kc_errors_pct': kc_errors,
        'mean_lambda_error_pct': float(np.mean(lambda_errors)),
        'mean_kc_error_pct': float(np.mean(kc_errors)),
        'max_kc_error_pct': float(np.max(kc_errors)),
        # Backward compatibility aliases
        'Nx': nx,
        'Ny': ny,
        'Ncx': ncx,
        'Ncy': ncy,
        'N_dofs': len(interior_indices),
        'n_nulos_descartados': num_null,
        'tempo_montagem': assembly_time,
        'tempo_eig': solve_time,
        'tempo_total': assembly_time + solve_time,
        'autovalores_todos': eigenvalues,
        'autovalores_fisicos': physical_eigenvalues,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': lambda_errors,
        'erros_kc_pct': kc_errors,
        'erro_medio_lambda_pct': float(np.mean(lambda_errors)),
        'erro_medio_kc_pct': float(np.mean(kc_errors)),
        'erro_max_kc_pct': float(np.max(kc_errors))
    }
    
    print(f"  Active Unknowns: {len(interior_indices)}")
    print(f"  Discarded Null Gradient Eigenvalues: {num_null} / {len(interior_indices)}")
    print(f"  Assembly: {assembly_time:.3f}s | Solver: {solve_time:.3f}s")
    print(f"  Mean kc Err: {result['mean_kc_error_pct']:.2f}% | Max kc Err: {result['max_kc_error_pct']:.2f}%\n")
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    for k in range(min(10, len(physical_eigenvalues))):
        print(f"  Mode {mode_names[k]:>4s}: lambda={physical_eigenvalues[k]:7.4f} (ref={ref_vals[k]:5.2f}) | "
              f"kc={kc_num[k]:6.3f} (ref={ref_kc[k]:6.3f}) | kc Err={kc_errors[k]:5.2f}%")
              
    return result


def run_stokes_cell_sweep():
    """Parametric cell grid sweep for mimetic Stokes formulation."""
    print("\n==================================================================================")
    print("  PARAMETRIC CELL SWEEP: MIMETIC STOKES FORMULATION (s_div = 0)")
    print("==================================================================================")
    
    cell_grids = [
        (8, 8),
        (10, 10),
        (12, 12),
        (14, 14),
        (16, 16),
        (20, 20)
    ]
    
    results = []
    for (ncx, ncy) in cell_grids:
        res = test_mimetic_curl_stokes(nx=21, ny=21, ncx=ncx, ncy=ncy, num_eigenvalues=10)
        results.append(res)
        
    return results


def generate_stokes_plots_and_report(baseline_res, sweep_results, output_dir=REPORTS_DIR):
    """Generate plots and report for mimetic Stokes study."""
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    all_vals = baseline_res['all_eigenvalues']
    ax1.plot(all_vals, 'o-', color='#1f77b4', markersize=3.5, label=f"Full Spectrum ({len(all_vals)} eigenvalues)")
    ax1.axhline(0.0, color='r', linestyle='--', label=rf"{baseline_res['num_null_discarded']} Null Eigenvalues ($\lambda \approx 0$)")
    ax1.set_xlabel("Eigenvalue Index", fontsize=11)
    ax1.set_ylabel(r"Eigenvalue $\lambda = k_c^2$", fontsize=11)
    ax1.set_title("Global Spectrum: Mimetic Stokes Formulation", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    width = 0.38
    
    ax2.bar(indices - width/2, baseline_res['analytical_kc'], width, label="Analytical (Table 4-1)", color="#333333", alpha=0.7)
    ax2.bar(indices + width/2, baseline_res['numerical_kc'], width, 
            label=f"Mimetic Stokes ($s_{{div}}=0$, Mean Err: {baseline_res['mean_kc_error_pct']:.2f}%)", 
            color="#d62728", alpha=0.85)
    ax2.set_xticks(indices)
    ax2.set_xticklabels(mode_names, fontsize=9, fontweight="bold")
    ax2.set_ylabel(r"Cutoff Wavenumber $k_c$ [rad/m]", fontsize=11)
    ax2.set_title("First 10 Physical $TE_z$ Modes", fontsize=12, fontweight="bold")
    ax2.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "espectro_rotacional_mimetico_stokes.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Stokes plot saved to: {plot_path}")
    
    report_path = os.path.join(output_dir, "relatorio_rotacional_mimetico_stokes.md")
    
    content = []
    content.append("# Technical Report: Evaluation of Mimetic Stokes Formulation (s_div = 0)\n\n")
    content.append(f"- **Total Unknowns:** {baseline_res['num_dofs']}\n")
    content.append(f"- **Discarded Null Gradient Eigenvalues:** **{baseline_res['num_null_discarded']} exact zeros**\n")
    content.append(f"- **Mean $k_c$ Error:** **{baseline_res['mean_kc_error_pct']:.2f}%**\n")
    content.append(f"- **Max $k_c$ Error:** **{baseline_res['max_kc_error_pct']:.2f}%**\n\n")
    
    content.append("![Mimetic Stokes Spectrum](espectro_rotacional_mimetico_stokes.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"Report saved to: {report_path}")


def main():
    base_res = test_mimetic_curl_stokes(nx=21, ny=21, ncx=10, ncy=10)
    sweep_res = run_stokes_cell_sweep()
    generate_stokes_plots_and_report(base_res, sweep_res)
    print("\n>>> Mimetic Stokes curl study completed successfully!")


# Backward compatibility aliases
montar_matrizes_vnmm_stokes_mimetico = assemble_vnmm_mimetic_stokes_matrices
testar_rotacional_mimetico_stokes = test_mimetic_curl_stokes
varredura_celulas_stokes = run_stokes_cell_sweep
gerar_graficos_e_relatorio = generate_stokes_plots_and_report


if __name__ == "__main__":
    main()
