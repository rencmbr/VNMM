#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate publication-quality figure for Spectral Validation on Irregular Point Clouds
comparing Regular Grid vs. Irregular Cloud across 6 refinement levels (N in {9, 13, 17, 21, 25, 29}),
strictly mirroring the variables, metrics, numerical conditions, and styling of Figure 8 (VNMM vs. Edge FEM).
Vertical 2x1 layout for optimal readability and compatibility with caption font size.

Subplot (a): Mean relative spectral error in k_c (%) vs. Active Degrees of Freedom (N_DoF)
Subplot (b): Convergence rate vs. Characteristic DoF Spacing h_DoF = sqrt(|Omega| / N_DoF) [m]
"""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
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

# Curated publication color palette matching Figure 8
COLOR_NAVY = "#0f4c81"       # Deep Navy (Regular Grid / P1 reference in Figure 8)
COLOR_CRIMSON = "#c0392b"    # Crimson / Red (Irregular Cloud with Jitter & Random Directors)
COLOR_GRAY = "#555555"       # Neutral gray for reference lines

# 6 refinement levels: N in {9, 13, 17, 21, 25, 29}, N_v in {15, 22, 29, 36, 43, 49}
# Active interior DoFs: N_DoF = (N_v - 2)^2
dofs = np.array([169, 400, 729, 1156, 1681, 2209])

# Characteristic DoF spacing: h_DoF = sqrt(|Omega| / N_DoF) = pi / sqrt(N_DoF)
h_dof = np.sqrt(np.pi**2 / dofs)

# Mean relative error in k_c (%) across the 6 refinement levels
err_reg = np.array([10.233, 3.667, 1.645, 0.839, 0.493, 0.259])
err_aleat = np.array([24.577, 11.775, 5.858, 3.108, 1.967, 1.632])

# Fitted asymptotic convergence slopes over all 6 refinement levels
p_reg = np.polyfit(np.log(h_dof), np.log(err_reg), 1)[0]
p_aleat = np.polyfit(np.log(h_dof), np.log(err_aleat), 1)[0]

fig, (ax_dofs, ax_h) = plt.subplots(2, 1, figsize=(6.2, 6.6))
fig.subplots_adjust(left=0.14, right=0.96, bottom=0.08, top=0.96, hspace=0.28)

# -----------------------------------------------------------------------------
# Subplot (a): Convergence vs Problem Size (N_DoF)
# -----------------------------------------------------------------------------
ax_dofs.loglog(dofs, err_reg, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
               linewidth=2.0, markersize=7.0,
               label=r"Pure VNMM: Regular Grid")
ax_dofs.loglog(dofs, err_aleat, 's-', color=COLOR_CRIMSON, markerfacecolor=COLOR_CRIMSON,
               linewidth=2.0, markersize=7.0,
               label=r"Pure VNMM: Irregular Cloud (25% Jitter)")

ax_dofs.set_xlabel(r"Number of Active Degrees of Freedom ($N_{\mathrm{DoF}}$)")
ax_dofs.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
ax_dofs.set_title(r"(a)", loc='center', fontweight='bold', fontsize=11.5)
ax_dofs.grid(True, which="both", linestyle="--", alpha=0.45)
ax_dofs.legend(loc="upper right", framealpha=0.92, fontsize=8.8)
ax_dofs.set_ylim(0.12, 50.0)

# -----------------------------------------------------------------------------
# Subplot (b): Convergence vs Characteristic DoF Spacing h_DoF
# -----------------------------------------------------------------------------
ax_h.loglog(h_dof, err_reg, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
            linewidth=2.0, markersize=7.0,
            label=f"Regular Grid: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_reg:.2f}}})$")
ax_h.loglog(h_dof, err_aleat, 's-', color=COLOR_CRIMSON, markerfacecolor=COLOR_CRIMSON,
            linewidth=2.0, markersize=7.0,
            label=f"Irregular Cloud: $\\mathcal{{O}}(h_{{\\mathrm{{DoF}}}}^{{{p_aleat:.2f}}})$")

# Reference lines O(h^2) and O(h^1)
h_ref = np.linspace(h_dof.min() * 0.90, h_dof.max() * 1.10, 100)
c_ref2 = err_aleat[2] / (h_dof[2]**2)
c_ref1 = err_aleat[0] / (h_dof[0]**1)
ax_h.loglog(h_ref, c_ref2 * (h_ref**2), 'k:', alpha=0.55, label=r"Reference $\mathcal{O}(h^2)$")
ax_h.loglog(h_ref, c_ref1 * (h_ref**1), 'gray', linestyle='-.', alpha=0.55, label=r"Reference $\mathcal{O}(h^1)$")

ax_h.set_xlabel(r"Characteristic DoF Spacing $h_{\mathrm{DoF}} = \sqrt{|\Omega|/N_{\mathrm{DoF}}}$ [m]")
ax_h.set_ylabel(r"Mean Relative Error in $k_c$ (%)")
ax_h.set_title(r"(b)", loc='center', fontweight='bold', fontsize=11.5)
ax_h.grid(True, which="both", linestyle="--", alpha=0.45)
ax_h.legend(loc="lower right", framealpha=0.92, fontsize=8.0, ncol=2)
ax_h.set_ylim(0.08, 45.0)

# Save figure in artigo and relatorios
dir_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
caminho_pdf_artigo = os.path.join(dir_raiz, "artigo", "figura_convergencia_cavidade_aleatoria.pdf")
caminho_png_artigo = os.path.join(dir_raiz, "artigo", "figura_convergencia_cavidade_aleatoria.png")
caminho_pdf_rel = os.path.join(dir_raiz, "relatorios", "figura_convergencia_cavidade_aleatoria.pdf")
caminho_png_rel = os.path.join(dir_raiz, "relatorios", "figura_convergencia_cavidade_aleatoria.png")

fig.savefig(caminho_pdf_artigo, bbox_inches='tight')
fig.savefig(caminho_png_artigo, bbox_inches='tight')
fig.savefig(caminho_pdf_rel, bbox_inches='tight')
fig.savefig(caminho_png_rel, bbox_inches='tight')
plt.close(fig)

print("Successfully generated Figure 9 with unified conditions:")
print(f"  {caminho_pdf_artigo}")
print(f"  {caminho_png_artigo}")
