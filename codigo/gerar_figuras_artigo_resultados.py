#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate publication-quality figures for the VNMM 2D paper.
All figures are produced in vector PDF format (with 300 DPI PNG previews)
with vertical 2x1 layout for optimal readability and compatibility with caption font size.

Target Figures:
- Figure 5: figura_convergencia_interpolacao_L1_vs_P1.pdf (Section 5.2)
- Figure 6: figura_robustez_perturbacao_estocastica.pdf (Section 5.3)
- Figure 7: figura_espectro_cavidade_com_sem_regularizacao.pdf (Section 5.4)
- Figure 8: figura_convergencia_vnmm_vs_fem.pdf (Section 5.5)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import matplotlib.transforms as mtransforms

# -----------------------------------------------------------------------------
# Global Publication Styling (IEEE Trans. / Elsevier standard)
# Sized for 1-column / centered vertical layout (figsize ~ 6.2 x 6.6-7.2 in)
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

# Color palette: curated, high-contrast, publication-grade
COLOR_NAVY = "#0f4c81"       # Deep Navy (Primary / P1)
COLOR_CRIMSON = "#c0392b"    # Crimson / Red (L1 / Spurious / Without Floor)
COLOR_TEAL = "#16a085"       # Teal / Emerald (Curl / Regularized)
COLOR_ORANGE = "#d35400"     # Amber / Orange
COLOR_GREEN = "#27ae60"      # Green (FEM)
COLOR_PURPLE = "#8e44ad"     # Violet / Purple
COLOR_GRAY = "#555555"       # Neutral gray for reference lines

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "relatorios")
ARTIGO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artigo")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIGO_DIR, exist_ok=True)


# =============================================================================
# FIGURE 5: Interpolation Convergence L1 vs P1 (Section 5.2)
# =============================================================================
def generate_figure_1():
    print(">>> Generating Figure 5: L1 vs P1 Interpolation Convergence...")
    
    h_vals = np.array([0.5236, 0.3491, 0.2244, 0.1496, 0.0982, 0.0654])
    N_nodes = np.array([84, 186, 416, 884, 1928, 4192])
    
    # L1 (3-node incomplete formulation)
    E_rms_L1 = np.array([1.9136e-1, 1.1462e-1, 7.4377e-2, 5.0783e-2, 3.5623e-2, 2.4643e-2])
    rot_rms_L1 = np.array([8.6074e-1, 8.3762e-1, 8.3360e-1, 8.4263e-1, 8.8781e-1, 8.6317e-1])
    
    # P1 (6-node complete formulation)
    E_rms_P1 = np.array([1.2430e-1, 3.9749e-2, 1.4656e-2, 6.3940e-3, 2.8591e-3, 1.2416e-3])
    rot_rms_P1 = np.array([2.5542e-1, 1.6616e-1, 1.0686e-1, 7.2390e-2, 4.8940e-2, 3.3319e-2])
    
    # Fitted slopes
    p_E_L1 = np.polyfit(np.log(h_vals), np.log(E_rms_L1), 1)[0]
    p_E_P1 = np.polyfit(np.log(h_vals), np.log(E_rms_P1), 1)[0]
    p_rot_L1 = np.polyfit(np.log(h_vals), np.log(rot_rms_L1), 1)[0]
    p_rot_P1 = np.polyfit(np.log(h_vals), np.log(rot_rms_P1), 1)[0]
    
    fig, (ax_E, ax_rot) = plt.subplots(2, 1, figsize=(6.2, 6.6))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.08, top=0.96, hspace=0.28)
    
    # --- Subplot (a): Electric Field RMS Error ---
    ax_E.loglog(h_vals, E_rms_L1, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                markeredgewidth=1.8, label=f"Incomplete Base $\\mathcal{{L}}^1$ (3 nodes): $\\mathcal{{O}}(h^{{{p_E_L1:.2f}}})$")
    ax_E.loglog(h_vals, E_rms_P1, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                label=f"Complete Base $\\mathcal{{P}}^1$ (6 nodes): $\\mathcal{{O}}(h^{{{p_E_P1:.2f}}})$")
    
    # Reference lines
    h_ref = np.linspace(h_vals.min() * 0.85, h_vals.max() * 1.15, 60)
    c_h2 = E_rms_P1[2] / (h_vals[2]**2)
    c_h1 = E_rms_L1[1] / (h_vals[1]**1)
    ax_E.loglog(h_ref, c_h2 * (h_ref**2), 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^2)$ slope")
    ax_E.loglog(h_ref, c_h1 * (h_ref**1), 'gray', linestyle='-.', alpha=0.45, label=r"Reference $\mathcal{O}(h^1)$ slope")
    
    ax_E.set_ylim(3.0e-4, 1.8)
    ax_E.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{nom}}$ [m]")
    ax_E.set_ylabel(r"Electric Field RMS Error $\|\vec{E} - \vec{E}^h\|_{\mathrm{RMS}}$")
    ax_E.set_title(r"(a)", loc='center', fontweight='bold', fontsize=11.5)
    ax_E.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_E.legend(loc="upper left", framealpha=0.92, fontsize=8.8)
    
    # --- Subplot (b): Curl RMS Error ---
    ax_rot.loglog(h_vals, rot_rms_L1, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                  markeredgewidth=1.8, label=r"Incomplete Base $\mathcal{L}^1$ (3 nodes): $\mathcal{O}(1)$ Stagnation")
    ax_rot.loglog(h_vals, rot_rms_P1, 'd-', color=COLOR_TEAL, markerfacecolor=COLOR_TEAL,
                  label=f"Complete Base $\\mathcal{{P}}^1$ (6 nodes): $\\mathcal{{O}}(h^{{{p_rot_P1:.2f}}})$")
    
    c_curl1 = rot_rms_P1[2] / (h_vals[2]**1)
    ax_rot.loglog(h_ref, c_curl1 * (h_ref**1), 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^1)$ slope")
    
    ax_rot.set_ylim(8.0e-4, 2.5)
    ax_rot.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{nom}}$ [m]")
    ax_rot.set_ylabel(r"Curl RMS Error $\|(\nabla \times \vec{E})_z - (\nabla \times \vec{E}^h)_z\|_{\mathrm{RMS}}$")
    ax_rot.set_title(r"(b)", loc='center', fontweight='bold', fontsize=11.5)
    ax_rot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_rot.legend(loc="lower right", framealpha=0.92, fontsize=8.8)
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_convergencia_interpolacao_L1_vs_P1.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_convergencia_interpolacao_L1_vs_P1.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_convergencia_interpolacao_L1_vs_P1.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_convergencia_interpolacao_L1_vs_P1.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# FIGURE 7: Cavity Discrete Spectrum With/Without Regularization (Section 5.4)
# =============================================================================
def generate_figure_2():
    print(">>> Generating Figure 7: Cavity Spectrum Regularization (Section 5.4)...")
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    
    # Analytical eigenvalues and cutoffs
    lambda_exact = np.array([1.00, 1.00, 2.00, 4.00, 4.00, 5.00, 5.00, 8.00, 9.00, 9.00])
    kc_exact = np.sqrt(lambda_exact)
    
    # Computed eigenvalues without regularization (s_div = 0.0)
    lambda_s0 = np.array([0.9127, 1.0368, 1.0795, 1.7091, 2.0485, 3.6800, 4.3059, 5.0050, 5.0925, 6.2524])
    is_spurious_s0 = [False, False, True, True, False, True, False, False, False, True]
    
    # Computed eigenvalues with div-curl regularization (s_div = 6.0)
    lambda_s6 = np.array([0.9788, 0.9940, 1.9706, 3.8566, 4.0798, 5.0120, 5.0666, 7.8063, 9.3389, 9.3929])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 7.2))
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.08, top=0.96, hspace=0.34)
    
    width = 0.38
    
    # --- Subplot (a): Unregularized Spectrum (s_div = 0.0) ---
    bars_exact1 = ax1.bar(indices - width/2, lambda_exact, width, color="#7f8c8d", alpha=0.55,
                          edgecolor="#2c3e50", linewidth=1.0, label="Analytical Reference $\\lambda_{\\mathrm{exact}}$")
    
    for i, idx in enumerate(indices):
        if is_spurious_s0[i]:
            ax1.bar(idx + width/2, lambda_s0[i], width, color="#fadbd8", edgecolor=COLOR_CRIMSON,
                    hatch="//", linewidth=1.4)
        else:
            ax1.bar(idx + width/2, lambda_s0[i], width, color="#34495e", alpha=0.85,
                    edgecolor="#2c3e50", linewidth=1.0)
    
    ax1.set_xticks(indices)
    labels1 = ax1.set_xticklabels([f"Mode {i}" for i in indices], fontsize=9.2)
    # Shift 'Mode 10' tick label slightly to the right to avoid collision with 'Mode 9'
    offset_mode10 = mtransforms.ScaledTranslation(8.5 / 72.0, 0, fig.dpi_scale_trans)
    labels1[-1].set_transform(labels1[-1].get_transform() + offset_mode10)
    ax1.set_ylabel(r"Discrete Eigenvalue $\lambda = k_c^2$")
    ax1.set_title(r"(a)", loc='center', fontweight='bold', fontsize=11.5)
    ax1.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax1.set_ylim(0, 11.8)
    
    legend_elements1 = [
        Patch(facecolor="#7f8c8d", alpha=0.55, edgecolor="#2c3e50", label="Analytical Reference"),
        Patch(facecolor="#34495e", alpha=0.85, edgecolor="#2c3e50", label="Physical Mode"),
        Patch(facecolor="#fadbd8", edgecolor=COLOR_CRIMSON, hatch="//", label="Spurious Mode")
    ]
    ax1.legend(handles=legend_elements1, loc="upper left", framealpha=0.92, fontsize=8.6)
    
    ax1.annotate("Gradient nullspace leakage:\n" r"$\nabla_h \times \nabla_h \phi \neq \mathbf{0}$",
                 xy=(3.15, 1.3), xytext=(1.5, 6.2),
                 arrowprops=dict(arrowstyle="->", color=COLOR_CRIMSON, lw=1.3),
                 fontsize=8.5, fontweight="bold", color=COLOR_CRIMSON,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="#fde8e8", edgecolor=COLOR_CRIMSON, alpha=0.95))
    
    # --- Subplot (b): Div-Curl Regularized Spectrum (s_div = 6.0) ---
    bars_exact2 = ax2.bar(indices - width/2, lambda_exact, width, color="#7f8c8d", alpha=0.55,
                          edgecolor="#2c3e50", linewidth=1.0, label="Analytical Reference $\\lambda_{\\mathrm{exact}}$")
    bars_s6 = ax2.bar(indices + width/2, lambda_s6, width, color=COLOR_TEAL, alpha=0.88,
                      edgecolor="#0e6251", linewidth=1.0, label=r"VNMM 2D $\mathcal{P}^1$ ($s_{\mathrm{div}} = 6.0$)")
    
    ax2.set_xticks(indices)
    ax2.set_xticklabels(mode_names, fontsize=9.2, fontweight="bold")
    ax2.set_ylabel(r"Discrete Eigenvalue $\lambda = k_c^2$")
    ax2.set_title(r"(b)", loc='center', fontweight='bold', fontsize=11.5)
    ax2.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax2.set_ylim(0, 11.8)
    
    for i, idx in enumerate(indices):
        err_pct = abs(np.sqrt(lambda_s6[i]) - kc_exact[i]) / kc_exact[i] * 100.0
        ax2.text(idx + width/2, lambda_s6[i] + 0.15, f"{err_pct:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, color="#0e6251", fontweight="bold")
        
    ax2.annotate("Spurious-Free Bandwidth:\nAll gradient modes $\\lambda > 50$\nMean $k_c$ error = 1.09%",
                 xy=(4.1, 4.1), xytext=(1.5, 5.8),
                 arrowprops=dict(arrowstyle="->", color=COLOR_TEAL, lw=1.3),
                 fontsize=8.3, fontweight="bold", color="#0e6251",
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="#e6f7f2", edgecolor=COLOR_TEAL, alpha=0.95))
    
    ax2.legend(loc="upper left", framealpha=0.92, fontsize=8.6)
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_espectro_cavidade_com_sem_regularizacao.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_espectro_cavidade_com_sem_regularizacao.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_espectro_cavidade_com_sem_regularizacao.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_espectro_cavidade_com_sem_regularizacao.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# FIGURE 8: Convergence Pure VNMM vs Edge FEM (Section 5.5)
# =============================================================================
def generate_figure_3():
    print(">>> Generating Figure 8: Convergence Pure VNMM vs Edge FEM (Section 5.5)...")
    
    # 1. Edge FEM (Nédélec 1st order): 1 DoF per internal edge
    dofs_fem = np.array([176, 408, 736, 1160, 1680, 2296])
    h_fem = np.sqrt(np.pi**2 / dofs_fem)
    err_fem = np.array([0.810, 0.367, 0.208, 0.133, 0.093, 0.068])
    
    # 2. Pure VNMM 2D (P1): Edge-equivalent nodal density
    dofs_vnmm = np.array([169, 400, 729, 1156, 1681, 2209])
    h_vnmm = np.sqrt(np.pi**2 / dofs_vnmm)
    err_vnmm = np.array([9.484, 3.454, 1.534, 0.737, 0.463, 0.285])
    
    fig, (ax_dofs, ax_h) = plt.subplots(2, 1, figsize=(6.2, 6.6))
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.08, top=0.96, hspace=0.28)
    
    # --- Subplot (a): Convergence vs Degrees of Freedom (DoFs) ---
    ax_dofs.loglog(dofs_fem, err_fem, '^-', color=COLOR_GREEN, markerfacecolor=COLOR_GREEN,
                   linewidth=2.0, markersize=7.0, label=r"Pure Edge FEM (Nédélec, 1 DoF/edge)")
    ax_dofs.loglog(dofs_vnmm, err_vnmm, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                   linewidth=2.0, markersize=7.0, label=r"Pure VNMM 2D ($\mathcal{P}^1$, 1 DoF/node)")
    
    ax_dofs.set_ylim(0.035, 25.0)
    ax_dofs.set_xlabel(r"Number of Active Degrees of Freedom ($N_{\mathrm{DoF}}$)")
    ax_dofs.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_dofs.set_title(r"(a)", loc='center', fontweight='bold', fontsize=11.5)
    ax_dofs.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_dofs.legend(loc="upper right", framealpha=0.92, fontsize=8.8)
    
    # --- Subplot (b): Convergence vs Characteristic DoF Spacing h_DoF ---
    p_fem = np.polyfit(np.log(h_fem), np.log(err_fem), 1)[0]
    p_vnmm = np.polyfit(np.log(h_vnmm), np.log(err_vnmm), 1)[0]
    
    ax_h.loglog(h_fem, err_fem, '^-', color=COLOR_GREEN, markerfacecolor=COLOR_GREEN,
                linewidth=2.0, markersize=7.0, label=f"Edge FEM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_fem:.2f}}})$")
    ax_h.loglog(h_vnmm, err_vnmm, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                linewidth=2.0, markersize=7.0, label=f"Pure VNMM: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_vnmm:.2f}}})$")
    
    # Reference lines
    h_ref = np.linspace(0.060, 0.25, 100)
    ax_h.loglog(h_ref, 18.0 * (h_ref / h_ref[-1])**2, 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^2)$")
    ax_h.loglog(h_ref, 2.5 * (h_ref / h_ref[-1])**1, 'gray', linestyle='-.', alpha=0.55, label=r"Reference $\mathcal{O}(h^1)$")
    
    ax_h.set_ylim(0.010, 25.0)
    ax_h.set_xlabel(r"Characteristic DoF Spacing $h_{\mathrm{DoF}} = \sqrt{|\Omega|/N_{\mathrm{DoF}}}$ [m]")
    ax_h.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
    ax_h.set_title(r"(b)", loc='center', fontweight='bold', fontsize=11.5)
    ax_h.grid(True, which="both", linestyle="--", alpha=0.45)
    ax_h.legend(loc="lower right", framealpha=0.92, fontsize=8.0, ncol=2)
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_convergencia_vnmm_vs_fem.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_convergencia_vnmm_vs_fem.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_convergencia_vnmm_vs_fem.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_convergencia_vnmm_vs_fem.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# FIGURE 6: Robustness Under Stochastic Perturbations & Dimensionless Threshold (Section 5.3)
# =============================================================================
def generate_figure_4():
    print(">>> Generating Figure 6: Stochastic Robustness & Dimensionless Threshold (Section 5.3)...")
    
    h_vals = np.array([0.5236, 0.3491, 0.2244, 0.1496, 0.0982, 0.0654])
    
    # Overly permissive threshold: Tol_hat <= 0.065
    E_rms_sem = np.array([1.5734e-1, 6.8018e-2, 2.1490e-2, 8.8302e-3, 3.9952e-3, 1.9496e-3])
    rot_rms_sem = np.array([3.9767e-1, 2.4035e-1, 1.6794e-1, 1.1930e-1, 8.6573e-2, 6.0053e-2])
    
    # Stabilized threshold: Tol_hat = 0.82
    E_rms_com = np.array([1.2430e-1, 3.9749e-2, 1.4656e-2, 6.3940e-3, 2.8591e-3, 1.2100e-3])
    rot_rms_com = np.array([2.5542e-1, 1.6616e-1, 1.0686e-1, 7.2390e-2, 4.8940e-2, 3.2000e-2])
    
    # Dimensionless threshold sweep on fine mesh (N = 4192, h = 0.0654 m) across [0.05, 6.00]
    tol_hat_sweep = np.array([0.05, 0.10, 0.25, 0.50, 0.80, 1.00, 1.50, 2.00, 3.00, 4.50, 6.00])
    E_rms_sweep = np.array([2.26e-3, 1.66e-3, 1.21e-3, 1.15e-3, 1.18e-3, 1.23e-3, 1.38e-3, 1.50e-3, 1.76e-3, 2.00e-3, 2.19e-3])
    rot_rms_sweep = np.array([6.98e-2, 5.31e-2, 3.56e-2, 3.19e-2, 3.18e-2, 3.27e-2, 3.32e-2, 3.30e-2, 3.36e-2, 3.25e-2, 3.44e-2])
    K_avg_sweep = np.array([6.3, 6.6, 7.6, 9.1, 10.3, 10.7, 11.2, 11.4, 11.8, 12.8, 13.9])
    
    fig, (ax_conv, ax_tol) = plt.subplots(2, 1, figsize=(6.2, 7.2))
    fig.subplots_adjust(left=0.14, right=0.86, bottom=0.08, top=0.96, hspace=0.30)
    
    # --- Subplot (a): Convergence under Stochastic Perturbations ---
    ax_conv.loglog(h_vals, rot_rms_sem, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                   markeredgewidth=1.8, label=r"Curl: Permissive ($\widehat{\mathrm{Tol}} = 0.05$)")
    ax_conv.loglog(h_vals, rot_rms_com, 'd-', color=COLOR_TEAL, markerfacecolor=COLOR_TEAL,
                   label=r"Curl: Stabilized ($\widehat{\mathrm{Tol}} = 0.80$)")
    
    ax_conv.loglog(h_vals, E_rms_sem, '^--', color="#7f8c8d", markerfacecolor='white',
                   markeredgewidth=1.8, label=r"$\vec{E}$: Permissive ($\widehat{\mathrm{Tol}} = 0.05$)")
    ax_conv.loglog(h_vals, E_rms_com, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                   label=r"$\vec{E}$: Stabilized ($\widehat{\mathrm{Tol}} = 0.80$)")
    
    # Reference lines
    h_ref = np.linspace(h_vals.min() * 0.85, h_vals.max() * 1.15, 60)
    c_h2 = E_rms_com[2] / (h_vals[2]**2)
    c_h1 = rot_rms_com[2] / (h_vals[2]**1)
    ax_conv.loglog(h_ref, c_h2 * (h_ref**2), 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^2)$ slope")
    ax_conv.loglog(h_ref, c_h1 * (h_ref**1), 'gray', linestyle='-.', alpha=0.55, label=r"Reference $\mathcal{O}(h^1)$ slope")
    
    ax_conv.set_ylim(1.0e-4, 1.2)
    ax_conv.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{avg}}$ [m]")
    ax_conv.set_ylabel("RMS Interpolation Error")
    ax_conv.set_title(r"(a)", loc='center', fontweight='bold', fontsize=11.5)
    ax_conv.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_conv.legend(loc="lower right", fontsize=7.6, framealpha=0.93, ncol=2, borderpad=0.25, columnspacing=0.8, handletextpad=0.4)
    
    # --- Subplot (b): Stabilizing Role of the Dimensionless Threshold ---
    ax_tol.set_xscale('log')
    ax_tol.set_yscale('log')
    
    l1 = ax_tol.plot(tol_hat_sweep, rot_rms_sweep, 'd-', color=COLOR_CRIMSON, linewidth=2.0,
                     markerfacecolor=COLOR_CRIMSON, label="Curl RMS")
    l2 = ax_tol.plot(tol_hat_sweep, E_rms_sweep, 'o-', color=COLOR_NAVY, linewidth=2.0,
                     markerfacecolor=COLOR_NAVY, label=r"Field $\vec{E}$ RMS")
    
    ax_tol.set_xlabel(r"Dimensionless Threshold $\widehat{\mathrm{Tol}}$")
    ax_tol.set_ylabel("RMS Interpolation Error", color="#2c3e50")
    ax_tol.tick_params(axis='y', labelcolor="#2c3e50")
    ax_tol.set_xlim(0.038, 7.5)
    ax_tol.set_ylim(1.0e-3, 1.0e-1)
    
    # Format x-ticks with clean numbers
    ax_tol.set_xticks([0.05, 0.1, 0.5, 1.0, 3.0, 6.0])
    ax_tol.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    
    # Twin axis for neighborhood size K_avg
    ax_k = ax_tol.twinx()
    l3 = ax_k.plot(tol_hat_sweep, K_avg_sweep, 's--', color=COLOR_ORANGE, linewidth=2.0,
                   markerfacecolor='white', markeredgewidth=1.8, label=r"Neighbors $K_{\mathrm{avg}}$ (right)")
    ax_k.set_ylabel(r"Candidate Neighbors Queried $K_{\mathrm{avg}}$", color=COLOR_ORANGE)
    ax_k.tick_params(axis='y', labelcolor=COLOR_ORANGE)
    ax_k.set_ylim(5, 18)
    
    # Optimal stabilization window band and threshold indicator
    ax_tol.axvspan(0.80, 1.50, color="#27ae60", alpha=0.10)
    l4 = [ax_tol.axvline(0.80, color="#27ae60", linestyle="--", linewidth=1.6, alpha=0.85,
                         label=r"Stabilizing $\widehat{\mathrm{Tol}} = 0.80$")]
    
    # Combined legend for ax_tol and ax_k positioned inside on the left, between curl and neighbors
    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax_tol.legend(lines, labels, loc="center left", bbox_to_anchor=(0.02, 0.48), fontsize=7.6, framealpha=0.93, borderpad=0.25, handletextpad=0.4)
    
    ax_tol.set_title(r"(b)", loc='center', fontweight='bold', fontsize=11.5)
    ax_tol.grid(True, which="both", linestyle="--", alpha=0.5)
    
    pdf_path = os.path.join(OUTPUT_DIR, "figura_robustez_perturbacao_estocastica.pdf")
    png_path = os.path.join(OUTPUT_DIR, "figura_robustez_perturbacao_estocastica.png")
    fig.savefig(pdf_path, bbox_inches='tight')
    fig.savefig(png_path, bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_robustez_perturbacao_estocastica.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(ARTIGO_DIR, "figura_robustez_perturbacao_estocastica.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {pdf_path}")
    print(f"  Saved: {png_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("=================================================================")
    print("  GENERATING PUBLICATION-QUALITY FIGURES FOR VNMM 2D PAPER")
    print("  Output format: PDF (Vector) + PNG (300 DPI Preview)")
    print(f"  Destination directory: {OUTPUT_DIR}")
    print("=================================================================\n")
    
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    generate_figure_4()
    
    print("\nAll 4 publication figures generated successfully!")
