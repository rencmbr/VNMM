"""
Convergence comparison study: Hybrid FEM-VNMM coupling vs Pure VNMM vs Pure Edge FEM.

Analyzes convergence on the 2D PEC resonant cavity [0, pi]^2 for the first 10 TEz modes.
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
PAPER_DIR = os.path.join(ROOT_DIR, "paper", "HybridVNMM-EFEM")
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.eigen_solver_cavity_2d import solve_cavity_eigenvalues, ANALYTICAL_REFERENCE_MODES
from src.fem_edge_2d import solve_edge_fem_cavity_eigenvalues
from src.fem_vnmm_hybrid_2d import solve_hybrid_fem_vnmm_eigenvalues


def run_comparative_convergence_study():
    """Run comparative convergence study across pure VNMM, pure FEM, and hybrid formulation."""
    print("==========================================================================================")
    print("  CONVERGENCE STUDY: HYBRID COUPLING vs. PURE VNMM vs. PURE EDGE FEM")
    print("  PEC Resonant Cavity [0, pi]^2 (First 10 TEz Modes)")
    print("==========================================================================================\n")
    
    n_list = [9, 13, 17, 21, 25, 29]
    
    results_vnmm = []
    results_fem = []
    results_hybrid = []
    h_values = []
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    
    print(f"{'N':>3s} | {'h':>7s} | {'VNMM DoFs':>9s} {'VNMM Err':>10s} {'Time (s)':>9s} | {'FEM DoFs':>8s} {'FEM Err':>9s} {'Time (s)':>9s} | {'Hyb DoFs':>8s} {'Hyb Err':>9s} {'Time (s)':>9s}")
    print("-" * 110)
    
    for n in n_list:
        h = np.pi / (n - 1)
        h_values.append(h)
        
        # 1. Pure VNMM 2D (P1 basis, Gauss-point support)
        t0 = time.time()
        res_v = solve_cavity_eigenvalues(
            nx=n, ny=n, s_div=6.0, num_eigenvalues=10, support_mode="ponto_gauss"
        )
        t_v = time.time() - t0
        res_v['elapsed_time'] = t_v
        res_v['tempo'] = t_v
        results_vnmm.append(res_v)
        
        # 2. Pure Triangular Edge FEM (Nédélec 1st-order)
        t0 = time.time()
        res_f = solve_edge_fem_cavity_eigenvalues(
            nex=n-1, ney=n-1, num_eigenvalues=10
        )
        t_f = time.time() - t0
        res_f['elapsed_time'] = t_f
        res_f['tempo'] = t_f
        results_fem.append(res_f)
        
        # 3. Balanced Hybrid FEM-VNMM Coupling
        nex_f = (n - 1) // 2
        ney_f = n - 1
        nx_v = int(np.round(np.sqrt(3) * nex_f)) + 1
        ny_v = int(np.round(np.sqrt(3) * ney_f)) + 1
        
        t0 = time.time()
        res_h = solve_hybrid_fem_vnmm_eigenvalues(
            nex_fem=nex_f, ney=ney_f, nx_vnmm=nx_v, ny_vnmm=ny_v,
            ncx_vnmm=nx_v-1, ncy_vnmm=ny_v-1,
            s_div_vnmm=6.0, num_eigenvalues=10
        )
        t_h = time.time() - t0
        res_h['elapsed_time'] = t_h
        res_h['tempo'] = t_h
        results_hybrid.append(res_h)
        
        print(f"{n:3d} | {h:7.4f} | {res_v['active_dofs']:9d} {res_v['mean_kc_error_pct']:9.2f}% {t_v:8.3f}s | {res_f['active_dofs']:8d} {res_f['mean_kc_error_pct']:8.2f}% {t_f:8.3f}s | {res_h['dof_info']['num_global']:8d} {res_h['mean_kc_error_pct']:8.2f}% {t_h:8.3f}s")
        
    print("-" * 110 + "\n")
    
    return {
        'n_list': n_list,
        'N_list': n_list,
        'h_values': np.array(h_values),
        'h_valores': np.array(h_values),
        'vnmm': results_vnmm,
        'fem': results_fem,
        'hybrid': results_hybrid,
        'hibrido': results_hybrid
    }


def generate_comparative_plots(data, output_dir=REPORTS_DIR, paper_dir=PAPER_DIR):
    """Generate comparative error plots against characteristic spacing and DoFs."""
    os.makedirs(output_dir, exist_ok=True)
    if paper_dir is not None:
        os.makedirs(paper_dir, exist_ok=True)
    
    err_v = [r['mean_kc_error_pct'] for r in data['vnmm']]
    dofs_v = [r['active_dofs'] for r in data['vnmm']]
    
    err_f = [r['mean_kc_error_pct'] for r in data['fem']]
    dofs_f = [r['active_dofs'] for r in data['fem']]
    
    err_h = [r['mean_kc_error_pct'] for r in data['hybrid']]
    dofs_h = [r['dof_info']['num_global'] for r in data['hybrid']]
    
    domain_area = np.pi**2
    h_dof_v = np.sqrt(domain_area / np.array(dofs_v))
    h_dof_f = np.sqrt(domain_area / np.array(dofs_f))
    h_dof_h = np.sqrt(domain_area / np.array(dofs_h))
    
    # -------------------------------------------------------------
    # Publication Figure: 2-panel (DoFs and h_DoF) for Hybrid paper
    # -------------------------------------------------------------
    fig_pub, (ax_dofs, ax_h) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig_pub.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.92, wspace=0.22)
    
    ax_dofs.loglog(dofs_f, err_f, '^-', color='#2ca02c', markerfacecolor='#2ca02c',
                   linewidth=2.0, markersize=7.0, label=r"Pure Edge FEM (Nédélec, 1 DoF/edge)")
    ax_dofs.loglog(dofs_h, err_h, 's--', color='#ff7f0e', markerfacecolor='#ff7f0e',
                   linewidth=2.0, markersize=7.0, label=r"Balanced Hybrid ($\rho_{\mathrm{edge}}\approx\rho_{\mathrm{node}}$)")
    ax_dofs.loglog(dofs_v, err_v, 'o-', color='#1f77b4', markerfacecolor='#1f77b4',
                   linewidth=2.0, markersize=7.0, label=r"Pure VNMM 2D ($\mathcal{P}^1$, 1 DoF/node)")
    
    ax_dofs.set_xlabel(r"Number of Active Degrees of Freedom ($N_{\mathrm{DoF}}$)")
    ax_dofs.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_dofs.set_title(r"(a) Spectral Convergence vs. Problem Size ($N_{\mathrm{DoF}}$)")
    ax_dofs.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_dofs.legend(loc="upper right", framealpha=0.92, fontsize=9.2)
    
    p_fem = np.polyfit(np.log(h_dof_f), np.log(err_f), 1)[0]
    p_hybrid = np.polyfit(np.log(h_dof_h), np.log(err_h), 1)[0]
    p_vnmm = np.polyfit(np.log(h_dof_v), np.log(err_v), 1)[0]
    
    ax_h.loglog(h_dof_f, err_f, '^-', color='#2ca02c', markerfacecolor='#2ca02c',
                linewidth=2.0, markersize=7.0, label=f"Edge FEM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_fem:.2f}}})$")
    ax_h.loglog(h_dof_h, err_h, 's--', color='#ff7f0e', markerfacecolor='#ff7f0e',
                linewidth=2.0, markersize=7.0, label=f"Balanced Hybrid: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_hybrid:.2f}}})$")
    ax_h.loglog(h_dof_v, err_v, 'o-', color='#1f77b4', markerfacecolor='#1f77b4',
                linewidth=2.0, markersize=7.0, label=f"Pure VNMM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_vnmm:.2f}}})$")
    
    h_ref = np.linspace(min(h_dof_f.min(), h_dof_h.min(), h_dof_v.min()) * 0.9,
                        max(h_dof_f.max(), h_dof_h.max(), h_dof_v.max()) * 1.1, 100)
    ax_h.loglog(h_ref, 18.0 * (h_ref / h_ref[-1])**2, 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^2)$")
    ax_h.loglog(h_ref, 2.5 * (h_ref / h_ref[-1])**1, 'gray', linestyle='-.', alpha=0.55, label=r"Reference $\mathcal{O}(h^1)$")
    
    ax_h.set_xlabel(r"Characteristic DoF Spacing $h_{\mathrm{DoF}} = \sqrt{|\Omega|/N_{\mathrm{DoF}}}$ [m]")
    ax_h.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_h.set_title(r"(b) Convergence with Characteristic Spacing ($h_{\mathrm{DoF}} \to 0$)")
    ax_h.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_h.legend(loc="lower right", framealpha=0.92, fontsize=8.8)
    
    fig_pub.savefig(os.path.join(output_dir, "figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf"), bbox_inches='tight')
    fig_pub.savefig(os.path.join(output_dir, "figura_convergencia_hibrido_vs_vnmm_vs_fem.png"), bbox_inches='tight')
    if paper_dir is not None:
        fig_pub.savefig(os.path.join(paper_dir, "figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf"), bbox_inches='tight')
        fig_pub.savefig(os.path.join(paper_dir, "figura_convergencia_hibrido_vs_vnmm_vs_fem.png"), bbox_inches='tight')
    plt.close(fig_pub)
    print(f"Publication figure saved to: {os.path.join(paper_dir, 'figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf')}")
    
    # Plot 1: Error vs h_DoF
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.loglog(h_dof_v, err_v, 'o-', color='#1f77b4', linewidth=2.0, markersize=7, label=r"Pure VNMM 2D ($\mathcal{P}^1$)")
    ax.loglog(h_dof_h, err_h, 's--', color='#ff7f0e', linewidth=2.2, markersize=8, label="Balanced Hybrid Coupling (FEM + VNMM)")
    ax.loglog(h_dof_f, err_f, '^-.', color='#2ca02c', linewidth=2.0, markersize=7, label="Pure Edge FEM (Nédélec 1st Order)")
    
    ax.set_xlabel(r"Characteristic Spacing $h_{\mathrm{DoF}} = \sqrt{|\Omega|/N_{\mathrm{DoF}}}$ [m]", fontsize=11)
    ax.set_ylabel(r"Mean Relative $k_c$ Error (%)", fontsize=11)
    ax.set_title(r"Convergence with Characteristic Refinement ($h_{\mathrm{DoF}} \to 0$)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=10.5)
    fig.tight_layout()
    
    fig1_path = os.path.join(output_dir, "convergencia_hibrido_vs_vnmm_vs_fem.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Plot 1 saved to: {fig1_path}")
    
    # Plot 2: Error vs DoFs
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.loglog(dofs_v, err_v, 'o-', color='#1f77b4', linewidth=2.0, markersize=7, label="Pure VNMM 2D")
    ax.loglog(dofs_h, err_h, 's--', color='#ff7f0e', linewidth=2.2, markersize=8, label="Hybrid FEM + VNMM")
    ax.loglog(dofs_f, err_f, '^-.', color='#2ca02c', linewidth=2.0, markersize=7, label="Pure Edge FEM")
    
    ax.set_xlabel("Active Degrees of Freedom (DoFs)", fontsize=11)
    ax.set_ylabel(r"Mean Relative $k_c$ Error (%)", fontsize=11)
    ax.set_title("Spectral Efficiency: $k_c$ Error vs Number of Unknowns", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=10.5)
    fig.tight_layout()
    
    fig2_path = os.path.join(output_dir, "eficiencia_dofs_hibrido_vs_vnmm_vs_fem.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Plot 2 saved to: {fig2_path}")
    
    # Plot 3: Mode-by-mode comparison in baseline case (N = 21)
    idx_base = data['n_list'].index(21)
    res_v_base = data['vnmm'][idx_base]
    res_f_base = data['fem'][idx_base]
    res_h_base = data['hybrid'][idx_base]
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(indices - width, res_v_base['kc_errors_pct'], width, label=f"Pure VNMM (Mean: {res_v_base['mean_kc_error_pct']:.2f}%)", color='#1f77b4', alpha=0.85)
    ax.bar(indices, res_h_base['kc_errors_pct'], width, label=f"Hybrid FEM-VNMM (Mean: {res_h_base['mean_kc_error_pct']:.2f}%)", color='#ff7f0e', alpha=0.85)
    ax.bar(indices + width, res_f_base['kc_errors_pct'], width, label=f"Edge FEM (Mean: {res_f_base['mean_kc_error_pct']:.2f}%)", color='#2ca02c', alpha=0.85)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Relative $k_c$ Error (%)", fontsize=11)
    ax.set_title("Modal Error Comparison for First 10 TEz Modes (Baseline $N=21$)", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    fig3_path = os.path.join(output_dir, "comparacao_modos_hibrido_vs_puros.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Plot 3 saved to: {fig3_path}")


def generate_markdown_report(data, output_dir=REPORTS_DIR):
    """Generate Markdown technical report."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_convergencia_hibrido_vs_vnmm_vs_fem.md")
    
    n_list = data['n_list']
    h_vals = data['h_values']
    
    content = []
    content.append(r"# Comparative Convergence Report: Hybrid FEM-VNMM vs Pure VNMM vs Edge FEM" + "\n\n")
    content.append("**Problem:** Transverse Electric Modes ($TE_z$) in PEC Cavity $[0, \\pi]^2$\n\n")
    
    content.append("## 1. Convergence Table ($h \\to 0$)\n\n")
    content.append("| Level ($N$) | $h$ (m) | VNMM DoFs | Mean $k_c$ Err VNMM | FEM DoFs | Mean $k_c$ Err FEM | Hybrid DoFs | Mean $k_c$ Err Hybrid |\n")
    content.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i, n in enumerate(n_list):
        h = h_vals[i]
        rv = data['vnmm'][i]
        rf = data['fem'][i]
        rh = data['hybrid'][i]
        
        content.append(f"| **$N={n}$** | {h:6.4f} | {rv['active_dofs']} | **{rv['mean_kc_error_pct']:5.2f}%** | {rf['active_dofs']} | **{rf['mean_kc_error_pct']:5.2f}%** | {rh['dof_info']['num_global']} | **{rh['mean_kc_error_pct']:5.2f}%** |\n")
        
    content.append("\n![Convergence h](convergencia_hibrido_vs_vnmm_vs_fem.png)\n\n")
    content.append("![DoFs Efficiency](eficiencia_dofs_hibrido_vs_vnmm_vs_fem.png)\n\n")
    content.append("![Mode Comparison](comparacao_modos_hibrido_vs_puros.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"Final report saved to: {report_path}")


def main():
    data = run_comparative_convergence_study()
    generate_comparative_plots(data)
    generate_markdown_report(data)
    print("\n>>> Comparative convergence study completed successfully!")


# Backward compatibility aliases
executar_estudo_convergencia_comparativa = run_comparative_convergence_study
gerar_graficos_comparativos = generate_comparative_plots
gerar_relatorio_markdown = generate_markdown_report


if __name__ == "__main__":
    main()
