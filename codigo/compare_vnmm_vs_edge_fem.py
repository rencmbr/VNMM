import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "relatorios")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.eigen_solver_cavity import solve_cavity_eigenvalues, ANALYTICAL_REFERENCE_MODES
from src.fem_edge_2d import solve_edge_fem_cavity_eigenvalues


def run_comparison():
    print("==================================================================================")
    print("  NUMERICAL COMPARISON: VNMM 2D (P1 BASIS) vs 2D EDGE FINITE ELEMENTS (NÉDÉLEC)")
    print("  Problem: TEz Modes in 2D PEC Cavity [0, pi]^2 (Table 4-1 Luilly Ortiz)")
    print("==================================================================================\n")
    
    # 1. VNMM 2D P1 Basis (Base case: 21x21 nodes, 361 interior DoFs)
    print(">>> 1. Running VNMM 2D P1 Basis (Gauss point support, Nc=10, p=2, s_div=6.0)...")
    t0 = time.time()
    res_vnmm = solve_cavity_eigenvalues(
        Nx=21, 
        Ny=21, 
        Lx=np.pi, 
        Ly=np.pi, 
        Ncx=10, 
        Ncy=10, 
        basis="P1", 
        interior_type="alternating", 
        num_eigenvalues=10, 
        s_div=6.0, 
        points_per_dir=2, 
        support_mode="gauss_point"
    )
    t_vnmm = time.time() - t0
    res_vnmm['total_time'] = t_vnmm
    print(f"  VNMM 2D finished in {t_vnmm:.3f}s | DoFs: {res_vnmm['interior_nodes']} | Mean kc Error: {res_vnmm['mean_error_kc_pct']:.2f}%\n")
    
    # 2. Triangular Nédélec Edge FEM (11x11 cells => 341 interior DoFs)
    print(">>> 2. Running Nédélec 2D Edge FEM (Nex=11, Ney=11 => 341 interior active edges)...")
    t0 = time.time()
    res_fem = solve_edge_fem_cavity_eigenvalues(
        Nex=11, 
        Ney=11, 
        Lx=np.pi, 
        Ly=np.pi, 
        num_eigenvalues=10
    )
    t_fem = time.time() - t0
    res_fem['total_time'] = t_fem
    print(f"  Edge FEM finished in {t_fem:.3f}s | DoFs: {res_fem['active_dofs']} | Mean kc Error: {res_fem['mean_error_kc_pct']:.2f}%\n")
    
    return res_vnmm, res_fem


def generate_comparative_plots(res_vnmm, res_fem, output_dir=REPORTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    bar_width = 0.35
    
    # Plot 1: Percentage Error of kc per mode
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    ax.bar(indices - bar_width/2, res_vnmm['rel_errors_kc_pct'], bar_width, 
           label=f"VNMM 2D $\\mathcal{{P}}^1$ ({res_vnmm['interior_nodes']} DoFs, Mean Err: {res_vnmm['mean_error_kc_pct']:.2f}%)", 
           color="#1f77b4", alpha=0.85)
           
    ax.bar(indices + bar_width/2, res_fem['rel_errors_kc_pct'], bar_width, 
           label=f"Nédélec Edge FEM ({res_fem['active_dofs']} DoFs, Mean Err: {res_fem['mean_error_kc_pct']:.2f}%)", 
           color="#2ca02c", alpha=0.85)
           
    ax.set_xticks(indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Relative Error of $k_c$ [%]", fontsize=11)
    ax.set_title("Modal Accuracy Comparison: VNMM 2D $\\mathcal{P}^1$ vs Nédélec Edge FEM", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    fig1_path = os.path.join(output_dir, "comparacao_erro_modo_vnmm_vs_fem.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Error plot saved to: {fig1_path}")
    
    # Plot 2: Comparative Spectrum
    fig, ax = plt.subplots(figsize=(10, 5.5))
    width3 = 0.26
    
    ax.bar(indices - width3, res_vnmm['analytical_kc'], width3, label="Analytical (Table 4-1)", color="#333333", alpha=0.75)
    ax.bar(indices, res_vnmm['numerical_kc'], width3, label="VNMM 2D $\\mathcal{P}^1$", color="#1f77b4", alpha=0.85)
    ax.bar(indices + width3, res_fem['numerical_kc'], width3, label="Nédélec Edge FEM", color="#2ca02c", alpha=0.85)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Cutoff Wavenumber $k_c$ [rad/m]", fontsize=11)
    ax.set_title("Electromagnetic Spectrum for First 10 $TE_z$ Modes", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    fig2_path = os.path.join(output_dir, "comparacao_espectro_vnmm_vs_fem.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Spectrum plot saved to: {fig2_path}")


def generate_markdown_report(res_vnmm, res_fem, output_dir=REPORTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_comparacao_vnmm_vs_fem_aresta.md")
    
    mode_names = ["TE_{10}", "TE_{01}", "TE_{11}", "TE_{20}", "TE_{02}", "TE_{21}", "TE_{12}", "TE_{22}", "TE_{30}", "TE_{03}"]
    
    lines = []
    lines.append("# Comparative Report: VNMM 2D (Basis $\\mathcal{P}^1$) vs Nédélec Triangular Edge Elements\n\n")
    lines.append("This report presents a numerical comparison of 2D electromagnetic vector eigenvalue problem ($TE_z$) "
                 "on resonant PEC cavity $[0, \\pi]^2$ (Table 4-1 of Luilly Ortiz, UFMG, 2023) computed by:\n\n")
    lines.append("1. **Vector Nodal Meshless Method (VNMM 2D):** Complete linear basis $\\mathcal{P}^1$ (6 support nodes), "
                 "Gauss-point support (EFG style), div-curl regularization ($s_{\\text{div}} = 6.0$), $N_x = 21, N_y = 21$ ($361$ interior DoFs).\n")
    lines.append("2. **Nédélec Triangular Edge Finite Elements (FEM 2D):** Whitney 1-forms on triangles, "
                 "$N_{ex} = 11, N_{ey} = 11$ ($341$ interior active DoFs).\n\n")
                 
    lines.append("## 1. Overview Comparison Table\n\n")
    lines.append("| Feature | VNMM 2D (Basis $\\mathcal{P}^1$) | Nédélec Edge FEM (1st Order) |\n")
    lines.append("|:---|:---:|:---:|\n")
    lines.append(f"| **Discretization Type** | Vector nodal meshless (point cloud) | Conforming $H(\\text{{curl}})$ Finite Elements |\n")
    lines.append(f"| **Active DoFs** | **{res_vnmm['interior_nodes']} DoFs** | **{res_fem['active_dofs']} DoFs** |\n")
    lines.append(f"| **DoF Representation** | 1 scalar component per node | 1 tangential circulation per edge |\n")
    lines.append(f"| **Characteristic Spacing $h$** | $h = {res_vnmm['h_max']:.4f}\\text{{ m}}$ | $h = {res_fem['h_max']:.4f}\\text{{ m}}$ |\n")
    lines.append(f"| **Nullspace Treatment $\\nabla \\times (\\nabla \\phi)$** | Variational Divergence Penalty ($s_{{\\text{{div}}}} = 6.0$) | Discrete de Rham Complex (Exact Zeros) |\n")
    lines.append(f"| **Mean Relative Error of $k_c$** | **{res_vnmm['mean_error_kc_pct']:.2f}%** | **{res_fem['mean_error_kc_pct']:.2f}%** |\n")
    lines.append(f"| **Max Relative Error of $k_c$** | **{res_vnmm['max_error_kc_pct']:.2f}%** | **{res_fem['max_error_kc_pct']:.2f}%** |\n")
    lines.append(f"| **Total Runtime** | **{res_vnmm['total_time']:.3f}s** | **{res_fem['total_time']:.3f}s** |\n\n")
    
    lines.append("![Modal Error Comparison](comparacao_erro_modo_vnmm_vs_fem.png)\n\n")
    lines.append("![Spectrum Comparison](comparacao_espectro_vnmm_vs_fem.png)\n\n")
    
    lines.append("## 2. Modal Table: Eigenvalues and Errors per Mode (Table 4-1 Luilly Ortiz)\n\n")
    lines.append("| Mode ($TE_{nm}$) | $\\lambda_{\\text{exact}}$ | $k_{c, \\text{exact}}$ | $\\lambda_{\\text{VNMM}}$ | Error $k_c$ VNMM (%) | $\\lambda_{\\text{FEM}}$ | Error $k_c$ FEM (%) |\n")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i in range(10):
        m_name = mode_names[i]
        l_ref = res_vnmm['analytical_eigenvalues'][i]
        kc_ref = res_vnmm['analytical_kc'][i]
        
        l_vnmm = res_vnmm['numerical_eigenvalues'][i]
        e_kc_vnmm = res_vnmm['rel_errors_kc_pct'][i]
        
        l_fem = res_fem['numerical_eigenvalues'][i]
        e_kc_fem = res_fem['rel_errors_kc_pct'][i]
        
        lines.append(f"| ${m_name}$ | {l_ref:6.2f} | {kc_ref:6.3f} | {l_vnmm:7.4f} | **{e_kc_vnmm:5.2f}%** | {l_fem:7.4f} | **{e_kc_fem:5.2f}%** |\n")
        
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"Comparative report saved to: {report_path}")


def main():
    res_vnmm, res_fem = run_comparison()
    generate_comparative_plots(res_vnmm, res_fem)
    generate_markdown_report(res_vnmm, res_fem)
    print("\n>>> VNMM 2D vs Edge FEM comparison completed successfully!")


# Backward compatibility aliases
executar_comparacao = run_comparison
gerar_graficos_comparativos = generate_comparative_plots
gerar_relatorio_markdown = generate_markdown_report

if __name__ == "__main__":
    main()
