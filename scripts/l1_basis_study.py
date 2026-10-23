"""
Study of VNMM 2D using the L1 basis (3 terms, 3 support nodes) in the baseline cavity.

Solves the eigenvalue problem with identically solenoidal basis
L1 = <[1, 0]^T, [0, 1]^T, [y, -x]^T> (div = 0) and compares support
determined per Gauss point vs cell center.
"""

import os
import sys
import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.cavity_mesh import generate_cavity_mesh
from src.vnmm_assembler import assemble_vnmm_matrices_2d
from src.eigen_solver_cavity import apply_pec_boundary_conditions, ANALYTICAL_REFERENCE_MODES


def solve_l1_eigenvalues(
    nx=21, 
    ny=21, 
    lx=np.pi, 
    ly=np.pi, 
    ncx=10, 
    ncy=10, 
    points_per_dir=2, 
    support_mode="ponto_gauss",
    zero_tol=1e-2,
    num_eigenvalues=10,
    # Backward compatibility arguments:
    Nx=None,
    Ny=None,
    Lx=None,
    Ly=None,
    Ncx=None,
    Ncy=None,
    pontos_por_dir=None,
    modo_suporte=None,
    tol_zero=None,
    num_autovalores=None
):
    """
    Solve cavity eigenvalue problem using the L1 basis (3 terms, 3 support nodes).
    
    L1 = <[1, 0]^T, [0, 1]^T, [y, -x]^T> (identically solenoidal, div = 0).
    """
    if Nx is not None: nx = Nx
    if Ny is not None: ny = Ny
    if Lx is not None: lx = Lx
    if Ly is not None: ly = Ly
    if Ncx is not None: ncx = Ncx
    if Ncy is not None: ncy = Ncy
    if pontos_por_dir is not None: points_per_dir = pontos_por_dir
    if modo_suporte is not None: support_mode = modo_suporte
    if tol_zero is not None: zero_tol = tol_zero
    if num_autovalores is not None: num_eigenvalues = num_autovalores

    coords, vectors, is_boundary = generate_cavity_mesh(
        nx=nx, ny=ny, lx=lx, ly=ly, interior_type="alternating"
    )
    
    k_mat, m_mat = assemble_vnmm_matrices_2d(
        coords=coords, 
        vectors=vectors, 
        basis="L1", 
        s_div=0.0, 
        ncx=ncx, 
        ncy=ncy, 
        lx=lx, 
        ly=ly, 
        points_per_dir=points_per_dir, 
        support_mode=support_mode
    )
    
    k_red, m_red, _ = apply_pec_boundary_conditions(k_mat, m_mat, is_boundary)
    
    # Solve generalized eigenvalue problem
    try:
        eigenvalues, _ = la.eigh(k_red.toarray(), m_red.toarray())
    except Exception:
        eigenvalues, _ = la.eig(k_red.toarray(), m_red.toarray())
        eigenvalues = np.real(eigenvalues)
        
    positive_mask = eigenvalues > zero_tol
    positive_vals = np.sort(eigenvalues[positive_mask])
    num_null = np.sum(~positive_mask)
    
    numerical_eigenvalues = positive_vals[:num_eigenvalues]
    
    ref_vals = np.array([item[2] for item in ANALYTICAL_REFERENCE_MODES[:len(numerical_eigenvalues)]])
    ref_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:len(numerical_eigenvalues)]])
    
    kc_num = np.sqrt(np.maximum(numerical_eigenvalues, 0.0))
    lambda_errors = np.abs(numerical_eigenvalues - ref_vals) / ref_vals * 100.0
    kc_errors = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    mean_lambda_err = float(np.mean(lambda_errors)) if len(lambda_errors) > 0 else 0.0
    mean_kc_err = float(np.mean(kc_errors)) if len(kc_errors) > 0 else 0.0
    max_kc_err = float(np.max(kc_errors)) if len(kc_errors) > 0 else 0.0
    
    total_gauss_pts = ncx * ncy * points_per_dir * points_per_dir

    return {
        'nx': nx,
        'ny': ny,
        'nc': ncx,
        'p': points_per_dir,
        'num_cells': ncx * ncy,
        'total_gauss_points': total_gauss_pts,
        'support_mode': support_mode,
        'num_null_discarded': int(num_null),
        'positive_eigenvalues': numerical_eigenvalues,
        'analytical_eigenvalues': ref_vals,
        'numerical_kc': kc_num,
        'analytical_kc': ref_kc,
        'lambda_errors_pct': lambda_errors,
        'kc_errors_pct': kc_errors,
        'mean_lambda_error_pct': mean_lambda_err,
        'mean_kc_error_pct': mean_kc_err,
        'max_kc_error_pct': max_kc_err,
        # Backward compatibility aliases
        'Nx': nx,
        'Ny': ny,
        'Nc': ncx,
        'n_celulas': ncx * ncy,
        'total_pontos_gauss': total_gauss_pts,
        'modo_suporte': support_mode,
        'n_nulos_descartados': int(num_null),
        'autovalores_positivos': numerical_eigenvalues,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': lambda_errors,
        'erros_kc_pct': kc_errors,
        'erro_medio_lambda_pct': mean_lambda_err,
        'erro_medio_kc_pct': mean_kc_err,
        'erro_max_kc_pct': max_kc_err
    }


def run_l1_study():
    """Run parametric sweep of L1 basis with Gauss-point support vs cell-center support."""
    print("==================================================================================")
    print("  STUDY OF VNMM 2D: L1 BASIS (3 SUPPORT NODES) IN BASELINE CAVITY (21x21 = 441 NODES)")
    print("  Solenoidal Basis L1 = <[1,0]^T, [0,1]^T, [y,-x]^T> | Gauss-point support")
    print("==================================================================================\n")
    
    nc_list = [10, 15, 20, 30, 40]
    p_list = [2, 3, 4]
    
    results_gp = []
    results_cc = []
    
    print(">>> 1. L1 Basis with GAUSS POINT SUPPORT (EFG style)...")
    for nc in nc_list:
        for p in p_list:
            t0 = time.time()
            res = solve_l1_eigenvalues(
                nx=21, ny=21, 
                ncx=nc, ncy=nc, 
                points_per_dir=p, 
                support_mode="ponto_gauss"
            )
            elapsed = time.time() - t0
            res['elapsed_seconds'] = elapsed
            res['tempo_seg'] = elapsed
            results_gp.append(res)
            
            first_three = res['positive_eigenvalues'][:3]
            l_str = " ".join([f"{v:.4f}" for v in first_three])
            print(f"  [Gauss Point] Cells: {nc:2d}x{nc:2d} | Gauss: {p}x{p} | Total Pts: {res['total_gauss_points']:5d} | "
                  f"Zeros: {res['num_null_discarded']:3d} | λs: [{l_str}] | "
                  f"Mean kc Err: {res['mean_kc_error_pct']:5.2f}% | Time: {elapsed:.2f}s")
                  
    print("\n>>> 2. L1 Basis with CELL CENTER SUPPORT (Reference)...")
    for nc in nc_list:
        for p in p_list:
            t0 = time.time()
            res = solve_l1_eigenvalues(
                nx=21, ny=21, 
                ncx=nc, ncy=nc, 
                points_per_dir=p, 
                support_mode="centro_celula"
            )
            elapsed = time.time() - t0
            res['elapsed_seconds'] = elapsed
            res['tempo_seg'] = elapsed
            results_cc.append(res)
            
            first_three = res['positive_eigenvalues'][:3]
            l_str = " ".join([f"{v:.4f}" for v in first_three])
            print(f"  [Cell Center] Cells: {nc:2d}x{nc:2d} | Gauss: {p}x{p} | Total Pts: {res['total_gauss_points']:5d} | "
                  f"Zeros: {res['num_null_discarded']:3d} | λs: [{l_str}] | "
                  f"Mean kc Err: {res['mean_kc_error_pct']:5.2f}% | Time: {elapsed:.2f}s")
                  
    return results_gp, results_cc


def generate_l1_plots_and_report(results_gp, results_cc, output_dir=REPORTS_DIR):
    """Generate comparative plots and markdown report for L1 basis study."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_estudo_base_L1.md")
    
    # ----------------------------------------------------
    # Plot 1: Modal Comparison L1 vs Analytical (Table 4-1)
    # ----------------------------------------------------
    res_ref_gp = [r for r in results_gp if r['nc'] == 20 and r['p'] == 2][0]
    
    fig, ax = plt.subplots(figsize=(10, 5.5))
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    width = 0.35
    
    ax.bar(indices - width/2, res_ref_gp['analytical_kc'], width, label=r"Analytical $k_c$ (Table 4-1)", color="#1f77b4", alpha=0.85)
    ax.bar(indices + width/2, res_ref_gp['numerical_kc'], width, label=r"VNMM 2D $\mathcal{L}^1$ Basis (3 Nodes)", color="#e377c2", alpha=0.85)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Cutoff Wavenumber $k_c$ [rad/m]", fontsize=11)
    ax.set_title(r"Modal Spectrum with $\mathcal{L}^1$ Basis (3 Support Nodes)", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    plot_path = os.path.join(output_dir, "espectro_modos_base_L1.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"L1 basis plot saved to: {plot_path}")
    
    # ----------------------------------------------------
    # Markdown Report
    # ----------------------------------------------------
    content = []
    content.append("# Study of VNMM 2D with $\\mathcal{L}^1$ Basis (3 Support Nodes)\n\n")
    content.append("This report presents the spectral analysis of the VNMM 2D formulation using the incomplete polynomial vector basis "
                   "**$\\mathcal{L}^1 = \\langle [1, 0]^T, [0, 1]^T, [y, -x]^T \\rangle$** with **3 support nodes**, "
                   "evaluating support domain determination per Gauss point (EFG style) versus cell center.\n\n")
                    
    content.append("## 1. Mathematical Characteristics of the $\\mathcal{L}^1$ Basis\n\n")
    content.append("- **Identical Solenoidality:** Since $\\nabla \\cdot [1,0]^T = 0$, $\\nabla \\cdot [0,1]^T = 0$, and $\\nabla \\cdot [y,-x]^T = 0$, "
                   "the shape functions $\\vec{N}_i$ of the $\\mathcal{L}^1$ basis have identically zero divergence:\n")
    content.append("  $$\\nabla \\cdot \\vec{N}_i(x, y) \\equiv 0 \\implies K_{\\text{div}} \\equiv 0$$\n")
    content.append("- **Jacobian Incompleteness and Modal Aliasing:** The $\\mathcal{L}^1$ basis artificially imposes "
                   "$\\frac{\\partial E_x}{\\partial x} = 0$ and $\\frac{\\partial E_y}{\\partial y} = 0$, failing to represent normal derivatives.\n\n")
                    
    content.append("## 2. Results Table: Gauss Point Support (EFG Style)\n\n")
    content.append("| Cells ($N_c \\times N_c$) | Gauss ($p \\times p$) | Total Gauss Pts | Discarded Nulls | 1st $\\lambda$ | 2nd $\\lambda$ | 3rd $\\lambda$ | Mean $k_c$ Error (%) | Max $k_c$ Error (%) | Time (s) |\n")
    content.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in results_gp:
        l1 = r['positive_eigenvalues'][0] if len(r['positive_eigenvalues']) > 0 else 0.0
        l2 = r['positive_eigenvalues'][1] if len(r['positive_eigenvalues']) > 1 else 0.0
        l3 = r['positive_eigenvalues'][2] if len(r['positive_eigenvalues']) > 2 else 0.0
        content.append(f"| ${r['nc']} \\times {r['nc']}$ | ${r['p']} \\times {r['p']}$ | {r['total_gauss_points']} | {r['num_null_discarded']} | {l1:.4f} | {l2:.4f} | {l3:.4f} | **{r['mean_kc_error_pct']:5.2f}%** | {r['max_kc_error_pct']:5.2f}% | {r['elapsed_seconds']:.2f}s |\n")
        
    content.append("\n## 3. Results Table: Cell Center Support (Reference)\n\n")
    content.append("| Cells ($N_c \\times N_c$) | Gauss ($p \\times p$) | Total Gauss Pts | Discarded Nulls | 1st $\\lambda$ | 2nd $\\lambda$ | 3rd $\\lambda$ | Mean $k_c$ Error (%) | Max $k_c$ Error (%) | Time (s) |\n")
    content.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in results_cc:
        l1 = r['positive_eigenvalues'][0] if len(r['positive_eigenvalues']) > 0 else 0.0
        l2 = r['positive_eigenvalues'][1] if len(r['positive_eigenvalues']) > 1 else 0.0
        l3 = r['positive_eigenvalues'][2] if len(r['positive_eigenvalues']) > 2 else 0.0
        content.append(f"| ${r['nc']} \\times {r['nc']}$ | ${r['p']} \\times {r['p']}$ | {r['total_gauss_points']} | {r['num_null_discarded']} | {l1:.4f} | {l2:.4f} | {l3:.4f} | **{r['mean_kc_error_pct']:5.2f}%** | {r['max_kc_error_pct']:5.2f}% | {r['elapsed_seconds']:.2f}s |\n")
        
    content.append("\n## 4. Modal Comparison for the First 10 Modes ($20 \\times 20$, Gauss $2 \\times 2$)\n\n")
    content.append("| Mode ($TE_{nm}$) | $\\lambda_{\\text{analytical}}$ | $\\lambda_{\\text{VNMM } \\mathcal{L}^1}$ | $k_{c, \\text{analytical}}$ | $k_{c, \\text{VNMM } \\mathcal{L}^1}$ | $k_c$ Error (%) |\n")
    content.append("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i in range(10):
        m_name = mode_names[i]
        l_ref = res_ref_gp['analytical_eigenvalues'][i]
        l_num = res_ref_gp['positive_eigenvalues'][i]
        kc_r = res_ref_gp['analytical_kc'][i]
        kc_n = res_ref_gp['numerical_kc'][i]
        e_kc = res_ref_gp['kc_errors_pct'][i]
        content.append(f"| ${m_name}$ | {l_ref:6.2f} | {l_num:7.4f} | {kc_r:6.3f} | {kc_n:6.3f} | **{e_kc:5.2f}%** |\n")
        
    content.append("\n![L1 Base Spectrum](espectro_modos_base_L1.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"L1 basis report saved to: {report_path}")


def main():
    res_gp, res_cc = run_l1_study()
    generate_l1_plots_and_report(res_gp, res_cc)


# Backward compatibility aliases
resolver_autovalores_L1 = solve_l1_eigenvalues
executar_estudo_L1 = run_l1_study
gerar_graficos_e_relatorio_L1 = generate_l1_plots_and_report


if __name__ == "__main__":
    main()
