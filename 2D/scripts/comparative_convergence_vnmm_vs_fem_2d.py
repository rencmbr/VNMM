"""
Comparative convergence analysis: VNMM 2D (P1 basis) vs Edge FEM (Nédélec).

Performs mesh refinement study for TEz cavity modes comparing VNMM 2D
and 1st-order triangular edge finite elements.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CODE_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.eigen_solver_cavity_2d import solve_cavity_eigenvalues, ANALYTICAL_REFERENCE_MODES
from src.fem_edge_2d import solve_edge_fem_cavity_eigenvalues


def run_comparative_convergence_analysis():
    """
    Run parametric mesh refinement comparing VNMM 2D (P1 basis) and Nédélec Edge FEM
    for the 2D PEC cavity [0, pi]^2.
    """
    print("==========================================================================================")
    print("  COMPARATIVE CONVERGENCE ANALYSIS: VNMM 2D (P1 BASIS) vs EDGE FEM (NÉDÉLEC)")
    print("  TEz Cavity Eigenvalue Problem [0, pi]^2")
    print("==========================================================================================\n")
    
    configurations = [
        {"level": "N1 (Very Coarse)", "nx": 9,  "nc": 5,  "nex": 4},
        {"level": "N2 (Coarse)",      "nx": 13, "nc": 8,  "nex": 7},
        {"level": "N3 (Medium-Coarse)","nx": 17, "nc": 10, "nex": 9},
        {"level": "N4 (Baseline)",    "nx": 21, "nc": 13, "nex": 11},
        {"level": "N5 (Medium-Dense)", "nx": 25, "nc": 15, "nex": 14},
        {"level": "N6 (Dense)",        "nx": 29, "nc": 17, "nex": 16},
        {"level": "N7 (Very Dense)",  "nx": 33, "nc": 20, "nex": 18},
    ]
    
    results_vnmm = []
    results_fem = []
    
    for cfg in configurations:
        level = cfg["level"]
        nx = cfg["nx"]
        nc = cfg["nc"]
        nex = cfg["nex"]
        
        # 1. VNMM 2D P1 Basis (with quartic tolerance scaling Tol_det(h) ~ h^4)
        h_current = np.pi / (nx - 1)
        h_ref = np.pi / 20.0 # h of baseline Nx=21
        tol_det_h = 1e-4 * (h_current / h_ref)**4
        
        t0 = time.time()
        res_vnmm = solve_cavity_eigenvalues(
            nx=nx, 
            ny=nx, 
            lx=np.pi, 
            ly=np.pi, 
            ncx=nc, 
            ncy=nc, 
            basis="P1", 
            interior_type="alternating", 
            num_eigenvalues=10, 
            s_div=6.0, 
            points_per_dir=3, 
            det_tolerance=tol_det_h,
            support_mode="ponto_gauss"
        )
        t_vnmm = time.time() - t0
        res_vnmm["level"] = level
        res_vnmm["total_time"] = t_vnmm
        res_vnmm["tempo_total"] = t_vnmm
        res_vnmm["nivel"] = level
        results_vnmm.append(res_vnmm)
        
        # 2. Triangular Edge FEM (Nédélec)
        t0 = time.time()
        res_fem = solve_edge_fem_cavity_eigenvalues(
            nex=nex, 
            ney=nex, 
            lx=np.pi, 
            ly=np.pi, 
            num_eigenvalues=10
        )
        t_fem = time.time() - t0
        res_fem["level"] = level
        res_fem["total_time"] = t_fem
        res_fem["tempo_total"] = t_fem
        res_fem["nivel"] = level
        results_fem.append(res_fem)
        
        print(f"[{level}]")
        print(f"  * VNMM 2D P1 : Nx={nx:2d} | DoFs={res_vnmm['active_dofs']:4d} | h={res_vnmm['h_max']:.4f}m | "
              f"Mean kc Err={res_vnmm['mean_kc_error_pct']:5.2f}% | Max kc Err={res_vnmm['max_kc_error_pct']:5.2f}% | Time={t_vnmm:.3f}s")
        print(f"  * FEM Edge   : Nex={nex:2d}| DoFs={res_fem['active_dofs']:4d} | h={res_fem['h_max']:.4f}m | "
              f"Mean kc Err={res_fem['mean_kc_error_pct']:5.2f}% | Max kc Err={res_fem['max_kc_error_pct']:5.2f}% | Time={t_fem:.3f}s\n")
              
    return results_vnmm, results_fem, configurations


def compute_convergence_rate(dofs, errors):
    """Compute asymptotic convergence slope in log-log scale."""
    log_dofs = np.log10(dofs)
    log_errors = np.log10(errors)
    poly = np.polyfit(log_dofs, log_errors, 1)
    return poly[0]


def generate_comparative_convergence_plots(res_vnmm, res_fem, output_dir=REPORTS_DIR):
    """Generate comparative convergence and CPU time trade-off plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    dofs_vnmm = np.array([r["active_dofs"] for r in res_vnmm])
    mean_kc_err_vnmm = np.array([r["mean_kc_error_pct"] for r in res_vnmm])
    max_kc_err_vnmm = np.array([r["max_kc_error_pct"] for r in res_vnmm])
    times_vnmm = np.array([r["total_time"] for r in res_vnmm])
    
    dofs_fem = np.array([r["active_dofs"] for r in res_fem])
    mean_kc_err_fem = np.array([r["mean_kc_error_pct"] for r in res_fem])
    max_kc_err_fem = np.array([r["max_kc_error_pct"] for r in res_fem])
    times_fem = np.array([r["total_time"] for r in res_fem])
    
    rate_vnmm = compute_convergence_rate(dofs_vnmm, mean_kc_err_vnmm)
    rate_fem = compute_convergence_rate(dofs_fem, mean_kc_err_fem)
    
    # ----------------------------------------------------
    # Plot 1: Error Convergence vs DoFs
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Subplot 1: Mean kc Error
    ax1.loglog(dofs_vnmm, mean_kc_err_vnmm, 'o-', color='#1f77b4', linewidth=2.2, markersize=7, 
               label=f"VNMM 2D $\\mathcal{{P}}^1$ (Slope = {rate_vnmm:.2f})")
    ax1.loglog(dofs_fem, mean_kc_err_fem, 's--', color='#2ca02c', linewidth=2.2, markersize=7, 
               label=f"FEM Edge Nédélec (Slope = {rate_fem:.2f})")
               
    ax1.set_xlabel("Active Degrees of Freedom (DoFs)", fontsize=11)
    ax1.set_ylabel(r"Mean Relative $k_c$ Error [%]", fontsize=11)
    ax1.set_title("Mean $k_c$ Error Convergence", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    
    # Subplot 2: Max kc Error
    ax2.loglog(dofs_vnmm, max_kc_err_vnmm, '^-', color='#1f77b4', linewidth=2.0, markersize=7, label=r"VNMM 2D $\mathcal{P}^1$ (Max)")
    ax2.loglog(dofs_fem, max_kc_err_fem, 'v--', color='#2ca02c', linewidth=2.0, markersize=7, label=r"FEM Edge Nédélec (Max)")
    
    ax2.set_xlabel("Active Degrees of Freedom (DoFs)", fontsize=11)
    ax2.set_ylabel(r"Max Relative $k_c$ Error [%]", fontsize=11)
    ax2.set_title("Max $k_c$ Error Convergence", fontsize=12, fontweight="bold")
    ax2.grid(True, which="both", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    fig.suptitle("Mesh Refinement Study: VNMM 2D $\\mathcal{P}^1$ vs Edge FEM", fontsize=13, fontweight="bold")
    fig.tight_layout()
    
    fig1_path = os.path.join(output_dir, "convergencia_comparativa_vnmm_vs_fem.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Convergence plot saved to: {fig1_path}")
    
    # ----------------------------------------------------
    # Plot 2: Error vs CPU Time Trade-off
    # ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    
    ax.loglog(times_vnmm, mean_kc_err_vnmm, 'o-', color='#1f77b4', linewidth=2.2, markersize=7, label=r"VNMM 2D $\mathcal{P}^1$")
    ax.loglog(times_fem, mean_kc_err_fem, 's--', color='#2ca02c', linewidth=2.2, markersize=7, label=r"FEM Edge Nédélec")
    
    for i, r in enumerate(res_vnmm):
        ax.annotate(r["level"].split()[0], (times_vnmm[i], mean_kc_err_vnmm[i]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=8, color='#1f77b4')
    for i, r in enumerate(res_fem):
        ax.annotate(r["level"].split()[0], (times_fem[i], mean_kc_err_fem[i]), textcoords="offset points", xytext=(0, -12), ha='center', fontsize=8, color='#2ca02c')
        
    ax.set_xlabel("Total Execution Time [seconds] (Log scale)", fontsize=11)
    ax.set_ylabel(r"Mean Relative $k_c$ Error [%]", fontsize=11)
    ax.set_title("Computational Efficiency: Spectral Error vs CPU Time", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    fig2_path = os.path.join(output_dir, "tradeoff_tempo_vnmm_vs_fem.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Trade-off plot saved to: {fig2_path}")


def generate_comparative_markdown_report(res_vnmm, res_fem, configurations, output_dir=REPORTS_DIR):
    """Generate technical comparative report in Markdown format."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_final_convergencia_vnmm_vs_fem.md")
    
    content = []
    content.append("# Final Report: Comparative Convergence Analysis (VNMM 2D $\\mathcal{P}^1$ vs Edge FEM)\n\n")
    content.append("This report presents the comparative investigation of 2D TEz electromagnetic cavity eigenvalues "
                   "in the PEC resonant cavity $[0, \\pi]^2$.\n\n")
                    
    content.append("## 1. Synthesis Table of Mesh Refinement\n\n")
    content.append("| Level | VNMM Mesh ($N_x \\times N_y$) | VNMM DoFs | Mean $k_c$ Err VNMM (%) | Max $k_c$ Err VNMM (%) | VNMM Time (s) | FEM Mesh ($N_{ex} \\times N_{ey}$) | FEM DoFs | Mean $k_c$ Err FEM (%) | Max $k_c$ Err FEM (%) | FEM Time (s) |\n")
    content.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i in range(len(configurations)):
        r_v = res_vnmm[i]
        r_f = res_fem[i]
        lbl = configurations[i]["level"]
        highlight = "**" if "Baseline" in lbl or "Caso Base" in lbl else ""
        content.append(
            f"| {highlight}{lbl}{highlight} | "
            f"${r_v['nx']} \\times {r_v['ny']}$ | {r_v['active_dofs']} | "
            f"{highlight}{r_v['mean_kc_error_pct']:5.2f}%{highlight} | {r_v['max_kc_error_pct']:5.2f}% | {r_v['total_time']:5.3f}s | "
            f"${r_f['nex']} \\times {r_f['ney']}$ | {r_f['active_dofs']} | "
            f"{highlight}{r_f['mean_kc_error_pct']:5.2f}%{highlight} | {r_f['max_kc_error_pct']:5.2f}% | {r_f['total_time']:5.3f}s |\n"
        )
        
    content.append("\n![Comparative Convergence](convergencia_comparativa_vnmm_vs_fem.png)\n\n")
    content.append("![CPU Time Trade-off](tradeoff_tempo_vnmm_vs_fem.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"Comparative report saved to: {report_path}")


def main():
    res_v, res_f, cfgs = run_comparative_convergence_analysis()
    generate_comparative_convergence_plots(res_v, res_f)
    generate_comparative_markdown_report(res_v, res_f, cfgs)
    print("\n>>> Comparative convergence study completed successfully!")


# Backward compatibility aliases
executar_analise_convergencia_comparativa = run_comparative_convergence_analysis
calcular_ordem_convergencia = compute_convergence_rate
gerar_graficos_convergencia_comparativa = generate_comparative_convergence_plots
gerar_relatorio_markdown_final = generate_comparative_markdown_report


if __name__ == "__main__":
    main()
