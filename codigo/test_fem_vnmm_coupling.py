"""
Test and demonstration of the coupled hybrid solver: 2D Edge FEM + 2D VNMM (P1 basis).

Problem: TEz modes in PEC cavity [0, pi]^2.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "relatorios")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fem_vnmm_hybrid_2d import solve_hybrid_fem_vnmm_eigenvalues, generate_hybrid_cavity_mesh
from src.eigen_solver_cavity import solve_cavity_eigenvalues
from src.fem_edge_2d import solve_edge_fem_cavity_eigenvalues


def run_hybrid_coupling_test():
    """Run test solving eigenvalues with the hybrid Edge-FEM + VNMM formulation."""
    print("==================================================================================")
    print("  COUPLED HYBRID SOLVER TEST: 2D EDGE FEM + 2D VNMM (P1 BASIS)")
    print("  Problem: TEz Modes in PEC Cavity [0, pi]^2")
    print("==================================================================================\n")
    
    t0 = time.time()
    res_hybrid = solve_hybrid_fem_vnmm_eigenvalues(
        lx=np.pi, 
        ly=np.pi, 
        fem_fraction=0.5, 
        nex_fem=8, 
        ney=12, 
        nx_vnmm=9, 
        ny_vnmm=13,
        ncx_vnmm=8, 
        ncy_vnmm=10,
        s_div_vnmm=6.0,
        num_eigenvalues=10
    )
    t_hybrid = time.time() - t0
    res_hybrid['total_time'] = t_hybrid
    res_hybrid['tempo_total'] = t_hybrid
    
    info = res_hybrid['dof_info']
    print(f"FEM Subdomain: x in [0, pi/2] | VNMM Subdomain: x in [pi/2, pi]")
    print(f"Total Global DoFs: {info['num_global']}")
    print(f"  * Interior FEM Edges: {info['num_fem_internal']}")
    print(f"  * Interface Gamma_int Edges: {info['num_interface_gamma']} (delta_y = {info['delta_y_gamma']:.4f}m)")
    print(f"  * Interior VNMM Nodes: {info['num_vnmm_internal']}")
    print(f"  * Discarded Nulls: {res_hybrid['num_null_discarded']}")
    print(f"Elapsed Time: {t_hybrid:.3f}s\n")
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    print("Modal Table for First 10 TEz Modes:")
    print("----------------------------------------------------------------------------------")
    print("  Mode   | lambda_ref | kc_ref | lambda_hybrid | kc_hybrid | kc Error (%)")
    print("----------------------------------------------------------------------------------")
    for i in range(10):
        m_name = mode_names[i]
        l_ref = res_hybrid['analytical_eigenvalues'][i]
        kc_r = res_hybrid['analytical_kc'][i]
        l_num = res_hybrid['numerical_eigenvalues'][i]
        kc_n = res_hybrid['numerical_kc'][i]
        e_kc = res_hybrid['kc_errors_pct'][i]
        print(f"  {m_name:>6s} | {l_ref:10.2f} | {kc_r:6.3f} | {l_num:14.4f} | {kc_n:10.3f} | {e_kc:8.2f}%")
    print("----------------------------------------------------------------------------------")
    print(f"Mean Relative kc Error: {res_hybrid['mean_kc_error_pct']:.2f}%")
    print(f"Max Relative kc Error: {res_hybrid['max_kc_error_pct']:.2f}%\n")
    
    return res_hybrid


def generate_hybrid_plots(res_hybrid, output_dir=REPORTS_DIR):
    """Generate modal spectrum and hybrid mesh partition plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    width = 0.38
    
    # Plot 1: Modal Spectrum
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(indices - width/2, res_hybrid['analytical_kc'], width, label="Analytical (Table 4-1)", color="#333333", alpha=0.7)
    ax.bar(indices + width/2, res_hybrid['numerical_kc'], width, 
           label=f"Hybrid FEM-VNMM (Mean Err: {res_hybrid['mean_kc_error_pct']:.2f}%)", 
           color="#ff7f0e", alpha=0.85)
           
    ax.set_xticks(indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Cutoff Wavenumber $k_c$ [rad/m]", fontsize=11)
    ax.set_title("Electromagnetic Spectrum: Hybrid Edge FEM + 2D VNMM Solver", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    fig1_path = os.path.join(output_dir, "espectro_hibrido_fem_vnmm.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Hybrid spectrum plot saved to: {fig1_path}")
    
    # Plot 2: Hybrid Mesh Illustration
    fem_data, vnmm_data = generate_hybrid_cavity_mesh(
        nex_fem=8, ney=12, nx_vnmm=9, ny_vnmm=13, fem_fraction=0.5
    )
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    nodes_f = fem_data['nodes']
    for elem in fem_data['elements']:
        pts = nodes_f[elem]
        tri = np.vstack([pts, pts[0]])
        ax.plot(tri[:, 0], tri[:, 1], 'b-', linewidth=0.7, alpha=0.6)
        
    nodes_v = vnmm_data['coords']
    is_intf = vnmm_data['is_interface']
    is_pec = vnmm_data['is_pec']
    is_int = ~is_intf & ~is_pec
    
    ax.scatter(nodes_v[is_int, 0], nodes_v[is_int, 1], color='red', s=30, label="Interior VNMM Nodes", zorder=3)
    ax.scatter(nodes_v[is_pec, 0], nodes_v[is_pec, 1], color='black', s=30, label="PEC VNMM Nodes", zorder=3)
    ax.scatter(nodes_v[is_intf, 0], nodes_v[is_intf, 1], color='green', s=60, marker='s', label=r"Interface $\Gamma_{\mathrm{int}}$ Nodes", zorder=4)
    
    ax.axvline(fem_data['interface_x'], color='green', linestyle='--', linewidth=2.0, label=r"Interface $\Gamma_{\mathrm{int}}$ ($x = \pi/2$)")
    
    ax.set_xlim(-0.1, np.pi + 0.1)
    ax.set_ylim(-0.1, np.pi + 0.1)
    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title("Hybrid Domain Partition: FEM Subdomain (Left) + VNMM 2D (Right)", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    
    fig2_path = os.path.join(output_dir, "malha_hibrida_fem_vnmm.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Hybrid mesh plot saved to: {fig2_path}")


def generate_hybrid_markdown_report(res_hybrid, output_dir=REPORTS_DIR):
    """Generate technical report in Markdown format."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_acoplamento_hibrido_fem_vnmm.md")
    
    info = res_hybrid['dof_info']
    mode_names = ["TE_{10}", "TE_{01}", "TE_{11}", "TE_{20}", "TE_{02}", "TE_{21}", "TE_{12}", "TE_{22}", "TE_{30}", "TE_{03}"]
    
    content = []
    content.append(r"# Technical Report: Coupled Hybrid 2D Edge FEM + 2D VNMM ($\mathcal{P}^1$)" + "\n\n")
    content.append("This report documents the numerical results of the coupled hybrid solver **Triangular Edge FEM + 2D VNMM** for the PEC cavity $[0, \\pi]^2$.\n\n")
    
    content.append(f"- **Total Active Master Unknowns:** **{info['num_global']} DoFs**\n")
    content.append(f"  - FEM Interior Edges: {info['num_fem_internal']}\n")
    content.append(f"  - Interface $\\Gamma_{{\\text{{int}}}}$ Edges: {info['num_interface_gamma']}\n")
    content.append(f"  - VNMM Interior Nodes: {info['num_vnmm_internal']}\n")
    content.append(f"- **Total Time:** **{res_hybrid['total_time']:.3f}s**\n\n")
    
    content.append("![Hybrid Mesh](malha_hibrida_fem_vnmm.png)\n\n")
    content.append("![Hybrid Spectrum](espectro_hibrido_fem_vnmm.png)\n\n")
    
    content.append("## 1. Modal Results (First 10 TEz Modes)\n\n")
    content.append("| Mode ($TE_{nm}$) | $\\lambda_{\\text{analytical}}$ | $k_{c, \\text{analytical}}$ | $\\lambda_{\\text{hybrid}}$ | $k_{c, \\text{hybrid}}$ | $k_c$ Error (%) |\n")
    content.append("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    for i in range(10):
        m_name = mode_names[i]
        l_ref = res_hybrid['analytical_eigenvalues'][i]
        kc_r = res_hybrid['analytical_kc'][i]
        l_n = res_hybrid['numerical_eigenvalues'][i]
        kc_n = res_hybrid['numerical_kc'][i]
        e_kc = res_hybrid['kc_errors_pct'][i]
        content.append(f"| ${m_name}$ | {l_ref:6.2f} | {kc_r:6.3f} | {l_n:7.4f} | {kc_n:6.3f} | **{e_kc:5.2f}%** |\n")
        
    content.append("\n")
    content.append(f"- **Mean $k_c$ Error in Hybrid Solver:** **{res_hybrid['mean_kc_error_pct']:.2f}%**\n")
    content.append(f"- **Max $k_c$ Error in Hybrid Solver:** **{res_hybrid['max_kc_error_pct']:.2f}%**\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"Hybrid report saved to: {report_path}")


def main():
    res_hybrid = run_hybrid_coupling_test()
    generate_hybrid_plots(res_hybrid)
    generate_hybrid_markdown_report(res_hybrid)
    print("\n>>> Coupled FEM-VNMM hybrid demonstration completed successfully!")


# Backward compatibility aliases
executar_teste_hibrido = run_hybrid_coupling_test
gerar_graficos_hibrido = generate_hybrid_plots
gerar_relatorio_hibrido = generate_hybrid_markdown_report


if __name__ == "__main__":
    main()
