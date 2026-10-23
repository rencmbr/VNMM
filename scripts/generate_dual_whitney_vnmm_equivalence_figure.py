#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate publication-quality Figure 1: figura_equivalencia_dual_whitney_vnmm
Illustrating the local dual constructive equivalence between:
  - Whitney Edge Finite Elements (lowest-order Nedelec 1-forms on triangles)
  - Vector Nodal Meshless Method 2D (VNMM 2D) with the solenoidal L1 basis.

Optimized for Elsevier preprint format with font sizes matching caption text (~10 pt).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Directory configuration
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
PAPER_DIR = os.path.join(ROOT_DIR, "paper")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def generate_dual_whitney_vnmm_equivalence_figure():
    """
    Generates high-resolution publication-quality Figure 1.
    Design criteria:
      - Compact 3-row layout (6.6 x 5.6 in) with native 1:1 scaling in LaTeX.
      - Fonts (10.0 - 10.8 pt) strictly harmonized with the document caption font size (~10 pt).
      - Rows:
          Row 1: (a) FEM Whitney Edge Element | (b) VNMM 2D Support Stencil
          Row 2: (c) Conceptual Asymptotic Dual Collapse Limit (Lk -> 0)
          Row 3: (d) FEM Whitney 1-Form | (e) VNMM 2D Basis (with colorbar)
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'axes.labelsize': 10.2,
        'axes.titlesize': 10.5,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # Geometry of a general scalene, non-degenerate reference triangle
    v1 = np.array([0.15, 0.15])
    v2 = np.array([1.25, 0.25])
    v3 = np.array([0.55, 1.15])
    vertices = np.array([v1, v2, v3])

    edges = [(v1, v2), (v2, v3), (v3, v1)]
    edge_lengths = [np.linalg.norm(eb - ea) for ea, eb in edges]
    tangents = [(eb - ea) / np.linalg.norm(eb - ea) for ea, eb in edges]
    midpoints = [0.5 * (ea + eb) for ea, eb in edges]

    det2 = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v3[0] - v1[0]) * (v2[1] - v1[1])
    area = 0.5 * det2

    grad_lambda = np.zeros((3, 2))
    grad_lambda[0] = [(v2[1] - v3[1]) / det2, (v3[0] - v2[0]) / det2]
    grad_lambda[1] = [(v3[1] - v1[1]) / det2, (v1[0] - v3[0]) / det2]
    grad_lambda[2] = [(v1[1] - v2[1]) / det2, (v2[0] - v1[0]) / det2]

    def bary_coords(x, y):
        l1 = ((v2[1] - v3[1]) * (x - v3[0]) + (v3[0] - v2[0]) * (y - v3[1])) / det2
        l2 = ((v3[1] - v1[1]) * (x - v1[0]) + (v1[0] - v3[0]) * (y - v1[1])) / det2
        l3 = 1.0 - l1 - l2
        return l1, l2, l3

    def whitney_w1(x, y):
        l1, l2, l3 = bary_coords(x, y)
        wx = l1 * grad_lambda[1, 0] - l2 * grad_lambda[0, 0]
        wy = l1 * grad_lambda[1, 1] - l2 * grad_lambda[0, 1]
        return wx, wy

    # VNMM L1 moment matrix with nodes placed at edge midpoints
    matrix_a = np.zeros((3, 3))
    for i in range(3):
        matrix_a[i, 0] = tangents[i][0]
        matrix_a[i, 1] = tangents[i][1]
        matrix_a[i, 2] = midpoints[i][1] * tangents[i][0] - midpoints[i][0] * tangents[i][1]
    beta = np.linalg.inv(matrix_a)

    def vnmm_n1(x, y):
        nx_val = beta[0, 0] + beta[2, 0] * y
        ny_val = beta[1, 0] - beta[2, 0] * x
        return nx_val, ny_val

    # Vector fields evaluation
    nx, ny = 45, 45
    x_grid = np.linspace(v1[0] - 0.02, v2[0] + 0.02, nx)
    y_grid = np.linspace(v1[1] - 0.02, v3[1] + 0.02, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    tri_path = Path(vertices)
    pts = np.vstack((X.ravel(), Y.ravel())).T
    inside_mask = tri_path.contains_points(pts).reshape(X.shape)

    Wx, Wy = np.zeros_like(X), np.zeros_like(Y)
    Nx, Ny = np.zeros_like(X), np.zeros_like(Y)

    for iy in range(ny):
        for ix in range(nx):
            if inside_mask[iy, ix]:
                wx, wy = whitney_w1(X[iy, ix], Y[iy, ix])
                nx_val, ny_val = vnmm_n1(X[iy, ix], Y[iy, ix])
                Wx[iy, ix] = wx
                Wy[iy, ix] = wy
                Nx[iy, ix] = nx_val / edge_lengths[0]
                Ny[iy, ix] = ny_val / edge_lengths[0]
            else:
                Wx[iy, ix] = np.nan
                Wy[iy, ix] = np.nan
                Nx[iy, ix] = np.nan
                Ny[iy, ix] = np.nan

    w_mag = np.sqrt(Wx**2 + Wy**2)
    n_mag = np.sqrt(Nx**2 + Ny**2)
    norm_mag = Normalize(vmin=0.0, vmax=np.nanmax(w_mag))

    # Visual styling colors
    c_tri = "#2c3e50"
    c_edge1 = "#c0392b"
    c_edge2 = "#2980b9"
    c_edge3 = "#27ae60"
    c_vnmm = "#8e44ad"
    c_bg = "#ffffff"

    # Canvas dimensions: 6.6 x 5.6 inches (matches 1 column / 1 textwidth perfectly)
    fig = plt.figure(figsize=(6.6, 5.6))

    # Grid specifications
    gs_row1 = fig.add_gridspec(1, 2, left=0.08, right=0.92, top=0.97, bottom=0.66, wspace=0.22)
    gs_row2 = fig.add_gridspec(1, 1, left=0.05, right=0.95, top=0.62, bottom=0.48)
    gs_row3 = fig.add_gridspec(1, 2, left=0.08, right=0.92, top=0.44, bottom=0.08, wspace=0.24)

    # =========================================================================
    # ROW 1, COL 1: PANEL (a) - FEM Whitney Edge Element
    # =========================================================================
    ax_a = fig.add_subplot(gs_row1[0, 0])
    ax_a.set_facecolor(c_bg)

    tri_patch_a = patches.Polygon(vertices, closed=True, facecolor="#ebf5fb", edgecolor=c_tri, linewidth=1.9, zorder=2)
    ax_a.add_patch(tri_patch_a)

    for v in vertices:
        ax_a.plot(v[0], v[1], 'o', color=c_tri, markersize=5.5, zorder=6)
    ax_a.text(v1[0] - 0.09, v1[1] - 0.08, r"$V_1$", fontsize=10.5, fontweight="bold", color=c_tri)
    ax_a.text(v2[0] + 0.04, v2[1] - 0.06, r"$V_2$", fontsize=10.5, fontweight="bold", color=c_tri)
    ax_a.text(v3[0] - 0.02, v3[1] + 0.08, r"$V_3$", fontsize=10.5, fontweight="bold", color=c_tri)

    edge_colors = [c_edge1, c_edge2, c_edge3]
    edge_labels = [r"$e_1$", r"$e_2$", r"$e_3$"]
    for i, (ea, eb) in enumerate(edges):
        ax_a.annotate("", xy=eb, xytext=ea,
                      arrowprops=dict(arrowstyle="-|>", color=edge_colors[i], lw=2.4, mutation_scale=14),
                      zorder=4)
        mid = midpoints[i]
        offset = [np.array([0.0, -0.15]), np.array([0.12, 0.03]), np.array([-0.13, 0.03])][i]
        ax_a.text(mid[0] + offset[0], mid[1] + offset[1], edge_labels[i],
                  fontsize=11.0, fontweight="bold", color=edge_colors[i], ha="center")

    centroid = np.mean(vertices, axis=0)
    ax_a.text(centroid[0], centroid[1], r"$\Omega_e$",
              fontsize=11.2, fontweight="bold", color="#34495e", ha="center", va="center",
              bbox=dict(boxstyle="circle,pad=0.25", facecolor="white", edgecolor="#bdc3c7", lw=1.2, alpha=0.95))

    ax_a.set_xlim(-0.18, 1.50)
    ax_a.set_ylim(-0.16, 1.35)
    ax_a.set_aspect('equal')
    ax_a.set_title(r"$\mathbf{(a)}$", pad=6, fontsize=11.0, fontweight="bold")
    ax_a.axis("off")

    # =========================================================================
    # ROW 1, COL 2: PANEL (b) - VNMM 2D Support Stencil
    # =========================================================================
    ax_b = fig.add_subplot(gs_row1[0, 1])
    ax_b.set_facecolor(c_bg)

    tri_patch_b = patches.Polygon(vertices, closed=True, facecolor="#f8f9f9", edgecolor="#7f8c8d",
                                  linestyle="--", linewidth=1.6, zorder=2)
    ax_b.add_patch(tri_patch_b)

    point_eval = np.array([0.62, 0.52])
    circle_eval = patches.Circle(point_eval, 0.44, facecolor="#af7ac5", alpha=0.14, edgecolor="#8e44ad",
                                 linestyle=":", linewidth=1.6, zorder=3)
    ax_b.add_patch(circle_eval)
    ax_b.plot(point_eval[0], point_eval[1], 'X', color="#8e44ad", markersize=7.0, zorder=6)
    ax_b.text(point_eval[0] + 0.05, point_eval[1] - 0.03, r"$\mathbf{P}$",
              fontsize=10.5, fontweight="bold", color="#6c3483")

    node_labels = [r"$(P_1, \vec{t}_1)$", r"$(P_2, \vec{t}_2)$", r"$(P_3, \vec{t}_3)$"]
    vec_len = 0.28
    off_t_list = [np.array([0.0, -0.19]), np.array([0.22, 0.05]), np.array([-0.22, 0.05])]
    ha_list = ["center", "left", "right"]
    for i in range(3):
        p_i = midpoints[i]
        t_i = tangents[i]
        ax_b.plot(p_i[0], p_i[1], 'o', color=edge_colors[i], markersize=6.5, zorder=5)
        ax_b.annotate("", xy=p_i + vec_len * t_i, xytext=p_i,
                      arrowprops=dict(arrowstyle="-|>", color=edge_colors[i], lw=2.5, mutation_scale=16),
                      zorder=4)
        ax_b.text(p_i[0] + off_t_list[i][0], p_i[1] + off_t_list[i][1], node_labels[i],
                  fontsize=9.8, fontweight="bold", color=edge_colors[i], ha=ha_list[i], va="center")

    ax_b.text(0.62, 1.02, r"Support $\Omega_x$",
              fontsize=9.8, color="#6c3483", fontstyle="italic", ha="center")

    ax_b.set_xlim(-0.25, 1.55)
    ax_b.set_ylim(-0.22, 1.38)
    ax_b.set_aspect('equal')
    ax_b.set_title(r"$\mathbf{(b)}$", pad=6, fontsize=11.0, fontweight="bold")
    ax_b.axis("off")

    # =========================================================================
    # ROW 2: PANEL (c) - Horizontal Asymptotic Dual Collapse Limit (Lk -> 0)
    # =========================================================================
    ax_c = fig.add_subplot(gs_row2[0, 0])
    ax_c.set_facecolor(c_bg)

    # 3 horizontal stages
    x1, x2, x3 = 0.15, 0.50, 0.85
    y0 = 0.48
    y_top = 0.85
    y_bot = 0.12

    # Top stage titles
    ax_c.text(x1, y_top, r"Continuous Edge", fontsize=9.8, fontweight="bold", color=c_edge1, ha="center")
    ax_c.text(x2, y_top, r"Contracted Edge", fontsize=9.8, fontweight="bold", color="#d35400", ha="center")
    ax_c.text(x3, y_top, r"Vector Node", fontsize=9.8, fontweight="bold", color=c_vnmm, ha="center")

    # Stage 1: Continuous Edge
    ax_c.plot([x1 - 0.08, x1 + 0.08], [y0, y0], lw=3.2, color=c_edge1, solid_capstyle='round', zorder=3)
    ax_c.annotate("", xy=(x1 + 0.08, y0), xytext=(x1 - 0.08, y0),
                  arrowprops=dict(arrowstyle="-|>", color=c_edge1, lw=2.2, mutation_scale=13), zorder=4)
    ax_c.plot(x1, y0, 'o', color="#2c3e50", markersize=4.5, zorder=5)
    ax_c.text(x1, y_bot, r"$e_k$ (length $L_k$)", fontsize=9.2, color="#2c3e50", ha="center")

    # Transition 1 -> 2
    ax_c.annotate("", xy=(0.38, y0), xytext=(0.27, y0),
                  arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6, mutation_scale=12), zorder=4)
    ax_c.text(0.325, y_bot, r"$L_k \to 0$", fontsize=9.2, color="#444444", fontstyle="italic", ha="center")

    # Stage 2: Contracted Edge
    ax_c.plot([x2 - 0.04, x2 + 0.04], [y0, y0], lw=2.8, color="#e67e22", solid_capstyle='round', zorder=3)
    ax_c.annotate("", xy=(x2 + 0.04, y0), xytext=(x2 - 0.04, y0),
                  arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=2.0, mutation_scale=11), zorder=4)
    ax_c.plot(x2, y0, 'o', color="#2c3e50", markersize=4.5, zorder=5)
    ax_c.text(x2, y_bot, r"$L_k \ll 1$", fontsize=9.2, color="#d35400", ha="center")

    # Transition 2 -> 3
    ax_c.annotate("", xy=(0.73, y0), xytext=(0.62, y0),
                  arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.6, mutation_scale=12), zorder=4)
    ax_c.text(0.675, y_bot, r"pointwise limit", fontsize=8.8, color="#444444", fontstyle="italic", ha="center")

    # Stage 3: Vector Node
    ax_c.plot(x3 - 0.03, y0, 'o', color=c_vnmm, markersize=6.5, zorder=5)
    ax_c.annotate("", xy=(x3 + 0.05, y0), xytext=(x3 - 0.03, y0),
                  arrowprops=dict(arrowstyle="-|>", color=c_vnmm, lw=2.4, mutation_scale=14), zorder=4)
    ax_c.text(x3 - 0.04, y_bot, r"$P_k$", fontsize=10.0, fontweight="bold", color=c_vnmm, ha="center")
    ax_c.text(x3 + 0.05, y_bot, r"$\vec{t}_k$", fontsize=10.0, fontweight="bold", color=c_vnmm, ha="center")

    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_title(r"$\mathbf{(c)}$", pad=6, fontsize=11.0, fontweight="bold")
    ax_c.axis("off")

    # =========================================================================
    # ROW 3, COL 1: PANEL (d) - FEM Whitney 1-Form
    # =========================================================================
    ax_d = fig.add_subplot(gs_row3[0, 0])
    ax_d.set_facecolor(c_bg)

    cf_d = ax_d.contourf(X, Y, w_mag, levels=25, cmap="YlGnBu", norm=norm_mag, alpha=0.88)
    ax_d.add_patch(patches.Polygon(vertices, closed=True, fill=False, edgecolor=c_tri, lw=1.8, zorder=5))

    step = 3
    ax_d.quiver(X[::step, ::step], Y[::step, ::step],
                Wx[::step, ::step], Wy[::step, ::step],
                color="#1b4f72", scale=16.0, width=0.0075, headwidth=4.0, headlength=5.0, zorder=4)

    ax_d.plot([v1[0], v2[0]], [v1[1], v2[1]], color=c_edge1, lw=3.8, zorder=6)
    ax_d.text(midpoints[0][0], midpoints[0][1] - 0.12, r"Active Edge $e_1$",
              fontsize=10.2, fontweight="bold", color=c_edge1, ha="center")

    ax_d.set_xlim(-0.02, 1.40)
    ax_d.set_ylim(-0.06, 1.30)
    ax_d.set_aspect('equal')
    ax_d.set_title(r"$\mathbf{(d)}$", pad=6, fontsize=11.0, fontweight="bold")
    ax_d.set_xlabel(r"$x$ [m]", fontsize=10.2)
    ax_d.set_ylabel(r"$y$ [m]", fontsize=10.2)
    ax_d.tick_params(labelsize=9.5)

    # =========================================================================
    # ROW 3, COL 2: PANEL (e) - VNMM 2D Basis Function
    # =========================================================================
    ax_e = fig.add_subplot(gs_row3[0, 1])
    ax_e.set_facecolor(c_bg)

    cf_e = ax_e.contourf(X, Y, n_mag, levels=25, cmap="YlGnBu", norm=norm_mag, alpha=0.88)
    ax_e.add_patch(patches.Polygon(vertices, closed=True, fill=False, edgecolor=c_tri, lw=1.8, zorder=5))

    ax_e.quiver(X[::step, ::step], Y[::step, ::step],
                Nx[::step, ::step], Ny[::step, ::step],
                color="#1b4f72", scale=16.0, width=0.0075, headwidth=4.0, headlength=5.0, zorder=4)

    ax_e.plot(midpoints[0][0], midpoints[0][1], 'o', color=c_edge1, markersize=7.0, zorder=7)
    ax_e.annotate("", xy=midpoints[0] + 0.20 * tangents[0], xytext=midpoints[0],
                  arrowprops=dict(arrowstyle="-|>", color=c_edge1, lw=2.6, mutation_scale=15), zorder=8)
    ax_e.text(midpoints[0][0], midpoints[0][1] - 0.12, r"Active Node $P_1$",
              fontsize=10.2, fontweight="bold", color=c_edge1, ha="center")

    ax_e.set_xlim(-0.02, 1.40)
    ax_e.set_ylim(-0.06, 1.30)
    ax_e.set_aspect('equal')
    ax_e.set_title(r"$\mathbf{(e)}$", pad=6, fontsize=11.0, fontweight="bold")
    ax_e.set_xlabel(r"$x$ [m]", fontsize=10.2)
    ax_e.tick_params(labelsize=9.5)

    # Colorbar
    divider = make_axes_locatable(ax_e)
    cax = divider.append_axes("right", size="5.5%", pad=0.08)
    cbar = fig.colorbar(cf_e, cax=cax)
    cbar.set_label(r"Magnitude $\|\vec{E}\|$", fontsize=10.0)
    cbar.ax.tick_params(labelsize=9.0)

    # Output file paths
    rel_png_path = os.path.join(REPORTS_DIR, "figura_equivalencia_dual_whitney_vnmm.png")
    rel_pdf_path = os.path.join(REPORTS_DIR, "figura_equivalencia_dual_whitney_vnmm.pdf")
    art_png_path = os.path.join(PAPER_DIR, "figura_equivalencia_dual_whitney_vnmm.png")
    art_pdf_path = os.path.join(PAPER_DIR, "figura_equivalencia_dual_whitney_vnmm.pdf")

    fig.savefig(rel_png_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(rel_pdf_path, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(art_png_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
    fig.savefig(art_pdf_path, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)

    print(f"[SUCCESS] Saved to reports: {rel_pdf_path}")
    print(f"[SUCCESS] Saved to article: {art_pdf_path}")


# Backward compatibility alias
criar_figura_equivalencia_dual = generate_dual_whitney_vnmm_equivalence_figure

if __name__ == "__main__":
    generate_dual_whitney_vnmm_equivalence_figure()
