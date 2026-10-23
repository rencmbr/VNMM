"""
Convergence analysis of 2D cavity eigenvalues using VNMM.

Performs parametric convergence study of the first 10 cutoff wavenumbers (kc)
as a function of nodal spacing h_max for the PEC square cavity [0, pi]^2.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.eigen_solver_cavity import solve_cavity_eigenvalues, ANALYTICAL_REFERENCE_MODES


def run_cavity_convergence_analysis(nx_list=None, s_div=6.0, basis="P1", lista_Nx=None, base=None):
    """
    Execute parametric convergence analysis for the first 10 eigenvalues and cutoff wavenumbers (kc)
    as a function of nodal spacing h_max for the 2D PEC cavity [0, pi]^2.
    """
    if lista_Nx is not None:
        nx_list = lista_Nx
    if base is not None:
        basis = base
    if nx_list is None:
        nx_list = [9, 13, 17, 21, 25, 29]
        
    print("=================================================================")
    print("  CONVERGENCE ANALYSIS: 2D PEC CAVITY EIGENVALUES (VNMM)")
    print(f"  Basis: {basis} | Div-curl regularization s = {s_div:.1f}")
    print("=================================================================\n")
    
    results = []
    
    for nx in nx_list:
        res = solve_cavity_eigenvalues(
            nx=nx, 
            ny=nx, 
            lx=np.pi, 
            ly=np.pi, 
            basis=basis, 
            interior_type="alternating", 
            num_eigenvalues=10, 
            s_div=s_div
        )
        results.append(res)
        
        print(f"Mesh Nx={nx:2d} ({res['total_nodes']:4d} nodes, h={res['h_max']:.4f} m) | "
              f"Mean λ Error: {res['mean_lambda_error_pct']:5.2f}% | "
              f"Mean kc Error: {res['mean_kc_error_pct']:5.2f}% | "
              f"Max kc Error: {res['max_kc_error_pct']:5.2f}%")
              
    return results


def generate_cavity_convergence_plots(results, output_dir=REPORTS_DIR):
    """Generate and save spectral convergence and modal comparison plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    h_vals = [r['h_max'] for r in results]
    mean_lambda_errs = [r['mean_lambda_error_pct'] for r in results]
    mean_kc_errs = [r['mean_kc_error_pct'] for r in results]
    max_kc_errs = [r['max_kc_error_pct'] for r in results]
    
    # ----------------------------------------------------
    # Plot 1: Error Convergence vs h
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(h_vals, mean_lambda_errs, 'r-o', linewidth=2.2, label=r"Mean $\lambda$ Error (%)")
    ax.loglog(h_vals, mean_kc_errs, 'b-s', linewidth=2.2, label=r"Mean $k_c$ Error (%)")
    ax.loglog(h_vals, max_kc_errs, 'g--^', linewidth=1.8, label=r"Max $k_c$ Error (%)")
    
    ax.set_xlabel(r"Nodal Spacing $h = \pi / (N_x - 1)$ [m] (Log scale)", fontsize=11)
    ax.set_ylabel("Relative Percentage Error [%]", fontsize=11)
    ax.set_title(r"Convergence of $TE_z$ Modes in PEC Cavity", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    
    conv_plot_path = os.path.join(output_dir, "convergencia_autovalores_cavidade.png")
    fig.savefig(conv_plot_path, dpi=300)
    plt.close(fig)
    print(f"Convergence plot saved to: {conv_plot_path}")
    
    # ----------------------------------------------------
    # Plot 2: First 10 Modes Spectrum (Analytical vs VNMM)
    # ----------------------------------------------------
    fine_res = results[3] if len(results) > 3 else results[-1] # Nx=21
    
    mode_indices = np.arange(1, 11)
    kc_analytical = fine_res['analytical_kc']
    kc_vnmm = fine_res['numerical_kc']
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.35
    ax.bar(mode_indices - width/2, kc_analytical, width, label=r"Analytical $k_c = \sqrt{n^2 + m^2}$", color='#1f77b4', alpha=0.85)
    ax.bar(mode_indices + width/2, kc_vnmm, width, label=r"VNMM 2D $\mathcal{P}^1$ (Numerical)", color='#ff7f0e', alpha=0.85)
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    ax.set_xticks(mode_indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Cutoff Wavenumber $k_c$ [rad/m]", fontsize=11)
    ax.set_title(r"Modal Comparison: Analytical Spectrum vs VNMM 2D", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    
    spectrum_plot_path = os.path.join(output_dir, "espectro_modos_cavidade.png")
    fig.savefig(spectrum_plot_path, dpi=300)
    plt.close(fig)
    print(f"Spectrum plot saved to: {spectrum_plot_path}")


def generate_cavity_markdown_report(results, output_dir=REPORTS_DIR):
    """Generate technical Markdown report for cavity eigenvalue study."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_autovalores_cavidade_luilly.md")
    
    fine_res = results[3] if len(results) > 3 else results[-1] # Nx=21
    
    mode_names = [
        "TE_{10}", "TE_{01}", "TE_{11}", "TE_{20}", "TE_{02}", 
        "TE_{21}", "TE_{12}", "TE_{22}", "TE_{30}", "TE_{03}"
    ]
    
    content = []
    content.append("# Spectral Validation Report: 2D PEC Cavity\n\n")
    content.append("This report documents the implementation and validation of the 2D electromagnetic eigenvalue solver via the **Vector Nodal Meshless Method (VNMM)**, "
                   "benchmarked against analytical cutoff wavenumbers for transverse electric ($TE_z$) modes in a square cavity with perfect electric conductor (PEC) walls.\n\n")
    
    content.append("## 1. Physical Formulation\n\n")
    content.append("- **Domain:** Square cavity $\\Omega = [0, \\pi] \\times [0, \\pi]$ with $\\epsilon_r = 1.0, \\mu_r = 1.0$.\n")
    content.append("- **PEC Boundary Condition:** $\\hat{n} \\times \\vec{E} = \\mathbf{0} \\implies E_{tangential} = 0$ on $\\partial\\Omega$.\n")
    content.append("- **Weak Form with Div-Curl Regularization:**\n\n")
    content.append("$$\n\\int_{\\Omega} (\\nabla \\times \\vec{W}_t)_z \\cdot (\\nabla \\times \\vec{E})_z \\, d\\Omega + s \\int_{\\Omega} (\\nabla \\cdot \\vec{W}_t) (\\nabla \\cdot \\vec{E}) \\, d\\Omega = \\lambda \\int_{\\Omega} \\vec{W}_t \\cdot \\vec{E} \\, d\\Omega\n$$\n\n")
    
    content.append("## 2. Benchmark Modal Comparison\n\n")
    content.append(f"Results obtained with complete linear basis $\\mathcal{{P}}^1$ (6 support nodes) on $N_x = {fine_res['nx']}, N_y = {fine_res['ny']}$ ($N = {fine_res['total_nodes']}$ nodes):\n\n")
    
    content.append("| Mode ($TE_{nm}$) | $\\lambda_{analytical}$ | $\\lambda_{VNMM}$ | $\\lambda$ Error (%) | $k_{c, analytical}$ | $k_{c, VNMM}$ | $k_c$ Error (%) |\n")
    content.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i in range(10):
        m_name = mode_names[i]
        l_ref = fine_res['analytical_eigenvalues'][i]
        l_num = fine_res['numerical_eigenvalues'][i]
        e_l = fine_res['lambda_errors_pct'][i]
        kc_r = fine_res['analytical_kc'][i]
        kc_n = fine_res['numerical_kc'][i]
        e_kc = fine_res['kc_errors_pct'][i]
        content.append(f"| ${m_name}$ | {l_ref:6.2f} | {l_num:7.4f} | **{e_l:5.2f}%** | {kc_r:6.3f} | {kc_n:6.3f} | **{e_kc:5.2f}%** |\n")
        
    content.append("\n")
    content.append(f"- **Mean Relative $k_c$ Error:** **{fine_res['mean_kc_error_pct']:.2f}%**\n")
    content.append(f"- **Max Relative $k_c$ Error:** **{fine_res['max_kc_error_pct']:.2f}%**\n\n")
    
    content.append("![Cavity Mode Spectrum](espectro_modos_cavidade.png)\n\n")
    
    content.append("## 3. Convergence Analysis with Nodal Refinement\n\n")
    content.append("| $N_x \\times N_y$ | $N_{total}$ | $h_{max}$ [m] | Mean $\\lambda$ Error (%) | Mean $k_c$ Error (%) | Max $k_c$ Error (%) |\n")
    content.append("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in results:
        content.append(f"| ${r['nx']} \\times {r['ny']}$ | {r['total_nodes']} | {r['h_max']:.4f} | {r['mean_lambda_error_pct']:5.2f}% | **{r['mean_kc_error_pct']:5.2f}%** | {r['max_kc_error_pct']:5.2f}% |\n")
        
    content.append("\n![Eigenvalue Convergence](convergencia_autovalores_cavidade.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"\nTechnical report saved to: {report_path}")


def main():
    results = run_cavity_convergence_analysis()
    generate_cavity_convergence_plots(results)
    generate_cavity_markdown_report(results)
    print("\n=================================================================")
    print("VNMM CAVITY EIGENVALUE ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=================================================================")


# Backward compatibility and alternative aliases
executar_analise_convergencia_cavidade = run_cavity_convergence_analysis
run_convergence_analysis = run_cavity_convergence_analysis
gerar_graficos_convergencia_cavidade = generate_cavity_convergence_plots
gerar_relatorio_markdown_cavidade = generate_cavity_markdown_report


if __name__ == "__main__":
    main()
