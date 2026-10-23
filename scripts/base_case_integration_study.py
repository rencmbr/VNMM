"""
Numerical integration study in the baseline cavity (Nx=21, Ny=21 = 441 nodes).

Evaluates the impact of:
- Number of background integration cells Ncx = Ncy
- Gauss quadrature order p (p x p points per cell)
- Gauss-point support (EFG style) vs cell-center support
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.eigen_solver_cavity import solve_cavity_eigenvalues, ANALYTICAL_REFERENCE_MODES


def run_baseline_integration_study(
    nx=21, 
    ny=21, 
    nc_list=[10, 15, 20, 30, 40], 
    p_list=[2, 3, 4, 5],
    s_div=6.0,
    basis="P1",
    Nx=None,
    Ny=None,
    lista_Nc=None,
    lista_p=None,
    base=None
):
    """
    Run 2D integration sweep in the baseline cavity:
    - Number of cells Ncx = Ncy in nc_list
    - Quadrature order p in p_list (p x p points per cell)
    - Comparison between support_mode='ponto_gauss' and 'centro_celula'
    """
    if Nx is not None: nx = Nx
    if Ny is not None: ny = Ny
    if lista_Nc is not None: nc_list = lista_Nc
    if lista_p is not None: p_list = lista_p
    if base is not None: basis = base

    print("==================================================================================")
    print(f"  NUMERICAL INTEGRATION STUDY IN BASELINE CAVITY: 2D PEC ({nx}x{ny} = {nx*ny} NODES)")
    print(f"  Basis: {basis} | s_div = {s_div:.1f} | Modes: Gauss Point (EFG) vs Cell Center")
    print("==================================================================================\n")
    
    results_gauss_point = []
    results_cell_center = []
    
    # 1. Sweep with support_mode = 'ponto_gauss'
    print(">>> 1. Running sweep with GAUSS POINT SUPPORT (EFG Style)...")
    for nc in nc_list:
        for p in p_list:
            t0 = time.time()
            res = solve_cavity_eigenvalues(
                nx=nx,
                ny=ny,
                lx=np.pi,
                ly=np.pi,
                ncx=nc,
                ncy=nc,
                basis=basis,
                interior_type="alternating",
                num_eigenvalues=10,
                s_div=s_div,
                points_per_dir=p,
                support_mode="ponto_gauss"
            )
            elapsed = time.time() - t0
            
            res['nc'] = nc
            res['p'] = p
            res['num_cells'] = nc * nc
            res['total_gauss_points'] = nc * nc * p * p
            res['elapsed_seconds'] = elapsed
            res['tempo_seg'] = elapsed
            res['support_mode'] = 'ponto_gauss'
            res['Nc'] = nc
            res['n_celulas'] = nc * nc
            results_gauss_point.append(res)
            
            print(f"  [Gauss Point] Cells: {nc:2d}x{nc:2d} ({nc*nc:4d} cells) | Gauss: {p}x{p} ({p*p:2d} pts/cell) | "
                  f"Total Pts: {nc*nc*p*p:5d} | Mean λ Err: {res['mean_lambda_error_pct']:5.2f}% | "
                  f"Mean kc Err: {res['mean_kc_error_pct']:5.2f}% | Max kc Err: {res['max_kc_error_pct']:5.2f}% | "
                  f"Time: {elapsed:.2f}s")
                  
    # 2. Sweep with support_mode = 'centro_celula'
    print("\n>>> 2. Running sweep with CELL CENTER SUPPORT (Reference)...")
    for nc in nc_list:
        for p in p_list:
            t0 = time.time()
            res = solve_cavity_eigenvalues(
                nx=nx,
                ny=ny,
                lx=np.pi,
                ly=np.pi,
                ncx=nc,
                ncy=nc,
                basis=basis,
                interior_type="alternating",
                num_eigenvalues=10,
                s_div=s_div,
                points_per_dir=p,
                support_mode="centro_celula"
            )
            elapsed = time.time() - t0
            
            res['nc'] = nc
            res['p'] = p
            res['num_cells'] = nc * nc
            res['total_gauss_points'] = nc * nc * p * p
            res['elapsed_seconds'] = elapsed
            res['tempo_seg'] = elapsed
            res['support_mode'] = 'centro_celula'
            res['Nc'] = nc
            res['n_celulas'] = nc * nc
            results_cell_center.append(res)
            
            print(f"  [Cell Center] Cells: {nc:2d}x{nc:2d} ({nc*nc:4d} cells) | Gauss: {p}x{p} ({p*p:2d} pts/cell) | "
                  f"Total Pts: {nc*nc*p*p:5d} | Mean λ Err: {res['mean_lambda_error_pct']:5.2f}% | "
                  f"Mean kc Err: {res['mean_kc_error_pct']:5.2f}% | Max kc Err: {res['max_kc_error_pct']:5.2f}% | "
                  f"Time: {elapsed:.2f}s")
                  
    return {
        'nx': nx,
        'ny': ny,
        'Nx': nx,
        'Ny': ny,
        'nc_list': nc_list,
        'lista_Nc': nc_list,
        'p_list': p_list,
        'lista_p': p_list,
        'gauss_point': results_gauss_point,
        'ponto_gauss': results_gauss_point,
        'cell_center': results_cell_center,
        'centro_celula': results_cell_center
    }


def generate_integration_study_plots(study_data, output_dir=REPORTS_DIR):
    """Generate comparative plots for integration sweep."""
    os.makedirs(output_dir, exist_ok=True)
    
    p_list = study_data['p_list']
    res_gp = study_data['gauss_point']
    res_cc = study_data['cell_center']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    markers = ['o-', 's--', '^-.', 'd:']
    
    for idx_p, p in enumerate(p_list):
        sub_gp = [r for r in res_gp if r['p'] == p]
        ncs_gp = [r['nc'] for r in sub_gp]
        errs_gp = [r['mean_kc_error_pct'] for r in sub_gp]
        ax1.plot(ncs_gp, errs_gp, markers[idx_p], color=colors[idx_p], linewidth=2.0, 
                 label=f"Gauss {p}x{p} ({p*p} pts/cell)")
                 
        sub_cc = [r for r in res_cc if r['p'] == p]
        ncs_cc = [r['nc'] for r in sub_cc]
        errs_cc = [r['mean_kc_error_pct'] for r in sub_cc]
        ax2.plot(ncs_cc, errs_cc, markers[idx_p], color=colors[idx_p], linewidth=2.0, 
                 label=f"Gauss {p}x{p} ({p*p} pts/cell)")
                 
    ax1.set_xlabel(r"Number of Cells per Direction $N_{cx} = N_{cy}$", fontsize=11)
    ax1.set_ylabel(r"Mean $k_c$ Wavenumber Error [%]", fontsize=11)
    ax1.set_title("Support per GAUSS POINT (EFG Style)", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)
    
    ax2.set_xlabel(r"Number of Cells per Direction $N_{cx} = N_{cy}$", fontsize=11)
    ax2.set_ylabel(r"Mean $k_c$ Wavenumber Error [%]", fontsize=11)
    ax2.set_title("Support per CELL CENTER (Previous)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)
    
    fig.suptitle("Impact of Integration Discretization in Baseline Cavity (21x21)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    fig1_path = os.path.join(output_dir, "estudo_integracao_erro_kc.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Integration error plot saved to: {fig1_path}")
    
    # Plot 2: Error vs Total Gauss Points
    fig, ax = plt.subplots(figsize=(8.5, 6))
    
    pts_gp = [r['total_gauss_points'] for r in res_gp]
    errs_gp = [r['mean_kc_error_pct'] for r in res_gp]
    
    pts_cc = [r['total_gauss_points'] for r in res_cc]
    errs_cc = [r['mean_kc_error_pct'] for r in res_cc]
    
    ax.scatter(pts_gp, errs_gp, color='#1f77b4', s=70, alpha=0.85, label='Gauss Point Support (EFG)', edgecolors='k')
    ax.scatter(pts_cc, errs_cc, color='#d62728', s=70, alpha=0.85, marker='s', label='Cell Center Support', edgecolors='k')
    
    idx_min_gp = np.argmin(errs_gp)
    ax.annotate(f"Minimum Gauss Point ({errs_gp[idx_min_gp]:.2f}%)\n{res_gp[idx_min_gp]['nc']}x{res_gp[idx_min_gp]['nc']} cells, p={res_gp[idx_min_gp]['p']}",
                xy=(pts_gp[idx_min_gp], errs_gp[idx_min_gp]),
                xytext=(pts_gp[idx_min_gp]*1.2, errs_gp[idx_min_gp]+0.5),
                arrowprops=dict(facecolor='blue', shrink=0.08, width=1.5, headwidth=6),
                fontsize=9, fontweight='bold', color='blue')
                
    ax.set_xscale('log')
    ax.set_xlabel("Total Number of Gauss Points in Domain (Log scale)", fontsize=11)
    ax.set_ylabel(r"Mean $k_c$ Wavenumber Error [%]", fontsize=11)
    ax.set_title(r"Trade-off: Spectral Accuracy vs Quadrature Resolution", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    fig2_path = os.path.join(output_dir, "estudo_integracao_tradeoff_pontos.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Trade-off plot saved to: {fig2_path}")


def generate_integration_markdown_report(study_data, output_dir=REPORTS_DIR):
    """Generate Markdown report for integration study."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_estudo_integracao_caso_base.md")
    
    res_gp = study_data['gauss_point']
    res_cc = study_data['cell_center']
    nx = study_data['nx']
    ny = study_data['ny']
    
    best_gp = min(res_gp, key=lambda x: x['mean_kc_error_pct'])
    best_cc = min(res_cc, key=lambda x: x['mean_kc_error_pct'])
    
    content = []
    content.append(f"# Report: Numerical Integration Study in Baseline Cavity ({nx}x{ny} = {nx*ny} Nodes)\n\n")
    content.append("This document evaluates numerical integration schemes for **VNMM 2D** in the 2D PEC resonant cavity $[0, \\pi]^2$.\n\n")
    
    content.append("## 1. Highlights and Best Configurations\n\n")
    content.append(f"- **Best Configuration [Gauss Point]:** Cells ${best_gp['nc']} \\times {best_gp['nc']}$, "
                   f"Quadrature ${best_gp['p']} \\times {best_gp['p']} "
                   f"$\\implies$ **Mean $k_c$ Error $= {best_gp['mean_kc_error_pct']:.2f}\\%$** (Max: {best_gp['max_kc_error_pct']:.2f}%, Time: {best_gp['elapsed_seconds']:.2f}s)\n")
    content.append(f"- **Best Configuration [Cell Center]:** Cells ${best_cc['nc']} \\times {best_cc['nc']}$, "
                   f"Quadrature ${best_cc['p']} \\times {best_cc['p']} "
                   f"$\\implies$ **Mean $k_c$ Error $= {best_cc['mean_kc_error_pct']:.2f}\\%$** (Max: {best_cc['max_kc_error_pct']:.2f}%, Time: {best_cc['elapsed_seconds']:.2f}s)\n\n")
                   
    content.append("![Integration Error](estudo_integracao_erro_kc.png)\n\n")
    content.append("![Gauss Points Trade-off](estudo_integracao_tradeoff_pontos.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"Integration study report saved to: {report_path}")


def main():
    study_data = run_baseline_integration_study()
    generate_integration_study_plots(study_data)
    generate_integration_markdown_report(study_data)
    print("\n>>> Integration study completed successfully!")


# Backward compatibility aliases
executar_estudo_integracao_caso_base = run_baseline_integration_study
gerar_graficos_estudo_integracao = generate_integration_study_plots
gerar_relatorio_markdown_estudo = generate_integration_markdown_report


if __name__ == "__main__":
    main()
