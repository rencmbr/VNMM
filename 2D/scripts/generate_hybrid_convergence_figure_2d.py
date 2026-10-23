"""
Publication figure generator for the Hybrid FEM-VNMM paper.
Produces vector PDF and 300 DPI PNG versions of:
- figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf / .png (Section 6.3)

Target destination: paper/HybridVNMM-EFEM/ and reports/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Global Publication Styling (IEEE / Elsevier standard)
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 11.0,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.titlesize": 12.0,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.45,
    "lines.markersize": 6.5,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

COLOR_NAVY = "#0f4c81"
COLOR_ORANGE = "#d35400"
COLOR_GREEN = "#27ae60"

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(ROOT_DIR, "reports")
PAPER_DIR = os.path.join(ROOT_DIR, "paper", "HybridVNMM-EFEM")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def generate_figure_convergence_hybrid_vnmm_fem():
    """Generate comparative convergence figure for Hybrid FEM-VNMM vs Pure VNMM vs Edge FEM."""
    print(">>> Generating: figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf / .png...")
    
    # 1. Edge FEM (Nédélec 1st order): 1 DoF per internal edge (~3 DoFs / vertex)
    #    6 mesh levels: N in [9, 13, 17, 21, 25, 29]
    dofs_fem = np.array([176, 408, 736, 1160, 1680, 2296])
    h_fem = np.sqrt(np.pi**2 / dofs_fem)
    err_fem = np.array([0.810, 0.367, 0.208, 0.133, 0.093, 0.068])
    
    # 2. Balanced Hybrid (FEM-VNMM): Equivalence rho_edge,FEM approx rho_node,VNMM
    #    Balanced subdomains: FEM edges ~ VNMM nodes (~50% DoFs each)
    dofs_hybrid = np.array([170, 390, 727, 1134, 1672, 2243])
    h_hybrid = np.sqrt(np.pi**2 / dofs_hybrid)
    err_hybrid = np.array([3.144, 0.888, 0.636, 0.359, 0.173, 0.101])
    
    # 3. Pure VNMM 2D (P1): Edge-equivalent nodal density (Nv ~ sqrt(3)*(N-1)+1)
    dofs_vnmm = np.array([169, 400, 729, 1156, 1681, 2209])
    h_vnmm = np.sqrt(np.pi**2 / dofs_vnmm)
    err_vnmm = np.array([10.233, 3.667, 1.645, 0.839, 0.493, 0.259])
    
    fig, (ax_dofs, ax_h) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.92, wspace=0.22)
    
    # --- Subplot (a): Convergence vs Degrees of Freedom (DoFs) ---
    ax_dofs.loglog(dofs_fem, err_fem, '^-', color=COLOR_GREEN, markerfacecolor=COLOR_GREEN,
                   linewidth=2.0, markersize=7.0, label=r"Pure Edge FEM (Nédélec, 1 DoF/edge)")
    ax_dofs.loglog(dofs_hybrid, err_hybrid, 's--', color=COLOR_ORANGE, markerfacecolor=COLOR_ORANGE,
                   linewidth=2.0, markersize=7.0, label=r"Balanced Hybrid ($\rho_{\mathrm{edge}}\approx\rho_{\mathrm{node}}$)")
    ax_dofs.loglog(dofs_vnmm, err_vnmm, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                   linewidth=2.0, markersize=7.0, label=r"Pure VNMM 2D ($\mathcal{P}^1$, 1 DoF/node)")
    
    ax_dofs.set_xlabel(r"Number of Active Degrees of Freedom ($N_{\mathrm{DoF}}$)")
    ax_dofs.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_dofs.set_title(r"(a) Spectral Convergence vs. Problem Size ($N_{\mathrm{DoF}}$)")
    ax_dofs.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_dofs.legend(loc="upper right", framealpha=0.92, fontsize=9.2)
    
    # --- Subplot (b): Convergence vs Characteristic DoF Spacing h_DoF ---
    p_fem = np.polyfit(np.log(h_fem), np.log(err_fem), 1)[0]
    p_hybrid = np.polyfit(np.log(h_hybrid), np.log(err_hybrid), 1)[0]
    p_vnmm = np.polyfit(np.log(h_vnmm), np.log(err_vnmm), 1)[0]
    
    ax_h.loglog(h_fem, err_fem, '^-', color=COLOR_GREEN, markerfacecolor=COLOR_GREEN,
                linewidth=2.0, markersize=7.0, label=f"Edge FEM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_fem:.2f}}})$")
    ax_h.loglog(h_hybrid, err_hybrid, 's--', color=COLOR_ORANGE, markerfacecolor=COLOR_ORANGE,
                linewidth=2.0, markersize=7.0, label=f"Balanced Hybrid: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_hybrid:.2f}}})$")
    ax_h.loglog(h_vnmm, err_vnmm, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                linewidth=2.0, markersize=7.0, label=f"Pure VNMM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_vnmm:.2f}}})$")
    
    # Reference lines
    h_ref = np.linspace(0.060, 0.25, 100)
    ax_h.loglog(h_ref, 18.0 * (h_ref / h_ref[-1])**2, 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^2)$")
    ax_h.loglog(h_ref, 2.5 * (h_ref / h_ref[-1])**1, 'gray', linestyle='-.', alpha=0.55, label=r"Reference $\mathcal{O}(h^1)$")
    
    ax_h.set_xlabel(r"Characteristic DoF Spacing $h_{\mathrm{DoF}} = \sqrt{|\Omega|/N_{\mathrm{DoF}}}$ [m]")
    ax_h.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_h.set_title(r"(b) Convergence with Characteristic Spacing ($h_{\mathrm{DoF}} \to 0$)")
    ax_h.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_h.legend(loc="lower right", framealpha=0.92, fontsize=8.8)
    
    # Save to reports/
    pdf_rep = os.path.join(OUTPUT_DIR, "figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf")
    png_rep = os.path.join(OUTPUT_DIR, "figura_convergencia_hibrido_vs_vnmm_vs_fem.png")
    fig.savefig(pdf_rep, bbox_inches='tight')
    fig.savefig(png_rep, bbox_inches='tight')
    
    # Save to paper/HybridVNMM-EFEM/
    pdf_paper = os.path.join(PAPER_DIR, "figura_convergencia_hibrido_vs_vnmm_vs_fem.pdf")
    png_paper = os.path.join(PAPER_DIR, "figura_convergencia_hibrido_vs_vnmm_vs_fem.png")
    fig.savefig(pdf_paper, bbox_inches='tight')
    fig.savefig(png_paper, bbox_inches='tight')
    
    plt.close(fig)
    print(f"  Saved report: {pdf_rep}")
    print(f"  Saved report: {png_rep}")
    print(f"  Saved paper:  {pdf_paper}")
    print(f"  Saved paper:  {png_paper}")


def main():
    print("=================================================================")
    print("  GENERATING PUBLICATION FIGURE FOR HYBRID FEM-VNMM PAPER")
    print(f"  Target: {PAPER_DIR}")
    print("=================================================================\n")
    generate_figure_convergence_hybrid_vnmm_fem()
    print("\nFigure generated successfully!")


if __name__ == "__main__":
    main()
