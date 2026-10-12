#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate publication-quality Figure 2: figura_singularidades_matriz_A
Illustrating the geometric and directional singularity configurations of the
VNMM 2D local placement matrix A in R^{3x3} for the solenoidal L1 basis.

Optimized for Elsevier preprint format:
  - Native 1:1 scale with textwidth (figsize=6.6 x 4.4 inches).
  - Fonts (10.0-10.5 pt) harmonized with document caption text (~10 pt).
  - Clean subfigure titles: (a), (b), (c), (d), (e), (f).
  - All redundant text removed from inside subplots; detailed explanations in caption.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Arc

# Directories
DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
DIRETORIO_RELATORIOS = os.path.join(DIRETORIO_RAIZ, "relatorios")
DIRETORIO_ARTIGO = os.path.join(DIRETORIO_RAIZ, "artigo")
os.makedirs(DIRETORIO_RELATORIOS, exist_ok=True)
os.makedirs(DIRETORIO_ARTIGO, exist_ok=True)


def gerar_figura_singularidades():
    """Generates Figure 2 in high resolution."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    fig = plt.figure(figsize=(6.6, 4.4), facecolor='white')
    gs = fig.add_gridspec(2, 3, left=0.03, right=0.97, top=0.92, bottom=0.04, wspace=0.18, hspace=0.25)

    c_tri = "#34495e"
    c_red = "#c0392b"
    c_blue = "#2980b9"
    c_green = "#27ae60"
    c_purple = "#8e44ad"
    c_orange = "#d35400"
    c_gold = "#b7950b"
    c_bg = "#fdfefe"

    def style_ax(ax, title_str):
        ax.set_facecolor(c_bg)
        ax.set_aspect('equal')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title_str, pad=5, fontsize=10.8, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#bdc3c7')
            spine.set_linewidth(1.0)

    # =========================================================================
    # PANEL (a): Global Vector Parallelism (t1 || t2 || t3)
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a, "(a)")

    # Shift triangle slightly left so t2 is completely within the frame
    p1_a = np.array([0.15, 0.30])
    p2_a = np.array([0.70, 0.35])
    p3_a = np.array([0.42, 0.78])
    poly_a = patches.Polygon([p1_a, p2_a, p3_a], closed=True, facecolor="#eaf2f8", edgecolor="#aed6f1",
                             linestyle="--", lw=1.5, zorder=1)
    ax_a.add_patch(poly_a)

    theta_a = np.deg2rad(15.0)
    t_par = np.array([np.cos(theta_a), np.sin(theta_a)])
    nodes_a = [p1_a, p2_a, p3_a]
    colors = [c_red, c_blue, c_green]
    lbl_off_a = [(-0.09, -0.08), (0.07, -0.08), (-0.08, 0.07)]
    arr_off_a = [(0.01, -0.11), (0.05, 0.04), (0.05, 0.04)]

    for i in range(3):
        ax_a.plot(nodes_a[i][0], nodes_a[i][1], 'o', color=c_tri, markersize=5.5, zorder=5)
        ax_a.text(nodes_a[i][0] + lbl_off_a[i][0], nodes_a[i][1] + lbl_off_a[i][1], rf"$P_{i+1}$",
                  fontsize=10.0, fontweight="bold", color=colors[i], ha="center", va="center")

        p_start = nodes_a[i]
        p_end = nodes_a[i] + 0.24 * t_par
        ax_a.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color="#95a5a6", linestyle=":", lw=1.2, alpha=0.65, zorder=2)

        end = nodes_a[i] + 0.17 * t_par
        arrow = FancyArrowPatch(nodes_a[i], end, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=colors[i], lw=2.2, zorder=4)
        ax_a.add_patch(arrow)
        ax_a.text(end[0] + arr_off_a[i][0], end[1] + arr_off_a[i][1], rf"$\vec{{t}}_{i+1}$",
                  fontsize=10.0, color=colors[i], ha="center", va="center")

    # =========================================================================
    # PANEL (b): Concurrency of Lines of Action (Focal Point Pc)
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b, "(b)")

    Pc = np.array([0.50, 0.52])
    p1_b = np.array([0.20, 0.26])
    p2_b = np.array([0.80, 0.26])
    p3_b = np.array([0.50, 0.88])
    nodes_b = [p1_b, p2_b, p3_b]
    lbl_off_b = [(-0.09, -0.08), (0.09, -0.08), (0.09, 0.03)]
    arr_off_b = [(-0.06, 0.06), (0.06, 0.06), (-0.08, 0.0)]

    tb = [(Pc - pt) / np.linalg.norm(Pc - pt) for pt in nodes_b]
    for i in range(3):
        ax_b.plot(nodes_b[i][0], nodes_b[i][1], 'o', color=c_tri, markersize=5.5, zorder=5)
        ax_b.text(nodes_b[i][0] + lbl_off_b[i][0], nodes_b[i][1] + lbl_off_b[i][1], rf"$P_{i+1}$",
                  fontsize=10.0, fontweight="bold", color=colors[i], ha="center", va="center")

        p_start = nodes_b[i]
        p_end = nodes_b[i] + 0.48 * tb[i]
        ax_b.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color="#e74c3c", linestyle="--", lw=1.2, alpha=0.6, zorder=2)

        end = nodes_b[i] + 0.18 * tb[i]
        arrow = FancyArrowPatch(nodes_b[i], end, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=colors[i], lw=2.2, zorder=4)
        ax_b.add_patch(arrow)
        ha_b = "right" if i == 2 else "center"
        ax_b.text(end[0] + arr_off_b[i][0], end[1] + arr_off_b[i][1], rf"$\vec{{t}}_{i+1}$",
                  fontsize=10.0, color=colors[i], ha=ha_b, va="center")

    ax_b.plot(Pc[0], Pc[1], 'X', color=c_red, markersize=8.5, markeredgecolor='black', markeredgewidth=1.0, zorder=8)
    ax_b.text(Pc[0] + 0.05, Pc[1] + 0.03, r"$P_c$", fontsize=10.5, fontweight="bold", color=c_red, ha="left", va="bottom")

    # =========================================================================
    # PANEL (c): Shared Line of Action (P1, P2 in L)
    # =========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c, "(c)")

    theta_c = np.deg2rad(30.0)
    dir_L = np.array([np.cos(theta_c), np.sin(theta_c)])
    p1_c = np.array([0.18, 0.30])
    p2_c = p1_c + 0.46 * dir_L
    p3_c = np.array([0.46, 0.82])

    p_start = p1_c - 0.03 * dir_L
    p_end = p1_c + 0.72 * dir_L
    ax_c.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color="#e74c3c", linestyle="-.", lw=1.6, alpha=0.75, zorder=2)
    ax_c.text(p_end[0] + 0.04, p_end[1] + 0.02, r"$\mathcal{L}$", fontsize=11.5, fontweight="bold", color=c_red, ha="left", va="center")

    # P1 label placed clearly below the line with ample distance (red)
    ax_c.plot(p1_c[0], p1_c[1], 'o', color=c_tri, markersize=5.5, zorder=5)
    ax_c.text(p1_c[0] - 0.03, p1_c[1] - 0.10, r"$P_1$", fontsize=10.0, fontweight="bold", color=c_red, ha="center", va="center")

    # P2 label placed below line with clear offset (blue)
    ax_c.plot(p2_c[0], p2_c[1], 'o', color=c_tri, markersize=5.5, zorder=5)
    ax_c.text(p2_c[0] + 0.08, p2_c[1] - 0.09, r"$P_2$", fontsize=10.0, fontweight="bold", color=c_blue, ha="center", va="center")

    # P3 label placed to the right with clear offset (green)
    ax_c.plot(p3_c[0], p3_c[1], 'o', color=c_tri, markersize=5.5, zorder=5)
    ax_c.text(p3_c[0] + 0.08, p3_c[1] + 0.03, r"$P_3$", fontsize=10.0, fontweight="bold", color=c_green, ha="left", va="center")

    # t1 arrow and label (above line)
    end1 = p1_c + 0.17 * dir_L
    arrow1 = FancyArrowPatch(p1_c, end1, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=c_red, lw=2.2, zorder=4)
    ax_c.add_patch(arrow1)
    ax_c.text(p1_c[0] + 0.08 * dir_L[0] - 0.04, p1_c[1] + 0.08 * dir_L[1] + 0.09, r"$\vec{t}_1$",
              fontsize=10.0, color=c_red, ha="center", va="center")

    # t2 arrow and label (above line, before end)
    end2 = p2_c + 0.17 * dir_L
    arrow2 = FancyArrowPatch(p2_c, end2, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=c_blue, lw=2.2, zorder=4)
    ax_c.add_patch(arrow2)
    ax_c.text(p2_c[0] + 0.08 * dir_L[0] - 0.04, p2_c[1] + 0.08 * dir_L[1] + 0.09, r"$\vec{t}_2$",
              fontsize=10.0, color=c_blue, ha="center", va="center")

    # t3 arrow and label
    t3_c = np.array([-0.7, -0.6]) / np.linalg.norm([-0.7, -0.6])
    end3 = p3_c + 0.17 * t3_c
    arrow3 = FancyArrowPatch(p3_c, end3, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=c_green, lw=2.2, zorder=4)
    ax_c.add_patch(arrow3)
    ax_c.text(end3[0] - 0.08, end3[1] + 0.01, r"$\vec{t}_3$", fontsize=10.0, color=c_green, ha="right", va="center")

    # =========================================================================
    # PANEL (d): Nodal Coincidence / Spatial Collapse (P1 ~ P2)
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 0])
    style_ax(ax_d, "(d)")

    p1_d = np.array([0.34, 0.43])
    p2_d = np.array([0.41, 0.48])
    p3_d = np.array([0.76, 0.80])

    # Ellipse enclosing P1 and P2 with clean boundary
    halo = patches.Ellipse((0.375, 0.455), 0.32, 0.22, angle=25, facecolor="#fadbd8", edgecolor="#e74c3c", linestyle="--", lw=1.4, zorder=2)
    ax_d.add_patch(halo)

    # P1 dot and label placed clearly outside the ellipse to the bottom-left (red)
    ax_d.plot(p1_d[0], p1_d[1], 'o', color=c_red, markersize=5.5, zorder=5)
    ax_d.text(0.14, 0.33, r"$P_1$", fontsize=10.0, fontweight="bold", color=c_red, ha="center", va="center")

    # P2 dot and label placed clearly outside the ellipse to the top-left (blue)
    ax_d.plot(p2_d[0], p2_d[1], 'o', color=c_blue, markersize=5.5, zorder=5)
    ax_d.text(0.26, 0.64, r"$P_2$", fontsize=10.0, fontweight="bold", color=c_blue, ha="center", va="center")

    # P3 dot and label (green)
    ax_d.plot(p3_d[0], p3_d[1], 'o', color=c_green, markersize=5.5, zorder=5)
    ax_d.text(p3_d[0] + 0.08, p3_d[1] - 0.04, r"$P_3$", fontsize=10.0, fontweight="bold", color=c_green, ha="left", va="center")

    # t1 arrow and label
    t1_d = np.array([0.90, -0.30]) / np.linalg.norm([0.90, -0.30])
    end1 = p1_d + 0.18 * t1_d
    arrow1 = FancyArrowPatch(p1_d, end1, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=c_red, lw=2.2, zorder=4)
    ax_d.add_patch(arrow1)
    ax_d.text(end1[0] + 0.07, end1[1] - 0.01, r"$\vec{t}_1$", fontsize=10.0, color=c_red, ha="left", va="center")

    # t2 arrow and label
    t2_d = np.array([0.65, 0.75]) / np.linalg.norm([0.65, 0.75])
    end2 = p2_d + 0.18 * t2_d
    arrow2 = FancyArrowPatch(p2_d, end2, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=c_blue, lw=2.2, zorder=4)
    ax_d.add_patch(arrow2)
    ax_d.text(end2[0] + 0.06, end2[1] + 0.06, r"$\vec{t}_2$", fontsize=10.0, color=c_blue, ha="left", va="center")

    # t3 arrow and label
    t3_d = np.array([-0.3, 0.95]) / np.linalg.norm([-0.3, 0.95])
    end3 = p3_d + 0.13 * t3_d
    arrow3 = FancyArrowPatch(p3_d, end3, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=c_green, lw=2.2, zorder=4)
    ax_d.add_patch(arrow3)
    ax_d.text(end3[0] - 0.07, end3[1] - 0.02, r"$\vec{t}_3$", fontsize=10.0, color=c_green, ha="right", va="center")

    # =========================================================================
    # PANEL (e): Decoupling: Spatial Geometry vs. Vector Alignment
    # =========================================================================
    ax_e = fig.add_subplot(gs[1, 1])
    style_ax(ax_e, "(e)")

    ax_e.axhline(0.50, color="#bdc3c7", linestyle=":", lw=1.2, zorder=1)

    # Sub-case E1 (Top): Spatially Non-collinear, Parallel Vectors
    p1_e1 = np.array([0.20, 0.66])
    p2_e1 = np.array([0.80, 0.66])
    p3_e1 = np.array([0.50, 0.90])
    poly_e1 = patches.Polygon([p1_e1, p2_e1, p3_e1], closed=True, facecolor="#fadbd8", edgecolor="#e74c3c", lw=1.2, zorder=2)
    ax_e.add_patch(poly_e1)

    t_e1 = np.array([1.0, 0.0])
    for pt in [p1_e1, p2_e1, p3_e1]:
        ax_e.plot(pt[0], pt[1], 'o', color=c_tri, markersize=5.0, zorder=4)
        arrow = FancyArrowPatch(pt, pt + 0.15 * t_e1, arrowstyle="-|>,head_length=5.0,head_width=3.5", color=c_red, lw=1.8, zorder=5)
        ax_e.add_patch(arrow)
    ax_e.text(0.50, 0.56, r"$\det(\mathbf{A}) = 0$", fontsize=10.2, fontweight="bold", color=c_red, ha="center", va="center")

    # Sub-case E2 (Bottom): Collinear Nodes, Non-parallel Vectors
    p1_e2 = np.array([0.20, 0.25])
    p2_e2 = np.array([0.50, 0.25])
    p3_e2 = np.array([0.80, 0.25])
    ax_e.plot([p1_e2[0], p3_e2[0]], [p1_e2[1], p3_e2[1]], color="#d35400", lw=2.0, zorder=2)

    dirs_e2 = [np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    for i, pt in enumerate([p1_e2, p2_e2, p3_e2]):
        ax_e.plot(pt[0], pt[1], 's', color=c_tri, markersize=5.5, zorder=4)
        arrow = FancyArrowPatch(pt, pt + 0.15 * dirs_e2[i], arrowstyle="-|>,head_length=5.0,head_width=3.5", color=c_purple, lw=1.8, zorder=5)
        ax_e.add_patch(arrow)
    ax_e.text(0.50, 0.10, r"$\det(\mathbf{A}) \neq 0$", fontsize=10.2, fontweight="bold", color=c_gold, ha="center", va="center")

    # =========================================================================
    # PANEL (f): Admissible and Stable Support (Circulation)
    # =========================================================================
    ax_f = fig.add_subplot(gs[1, 2])
    style_ax(ax_f, "(f)")

    P_eval = np.array([0.50, 0.53])
    v1_f = np.array([0.18, 0.28])
    v2_f = np.array([0.84, 0.28])
    v3_f = np.array([0.50, 0.88])

    poly_macro = patches.Polygon([v1_f, v2_f, v3_f], closed=True, facecolor="#eafaf1", edgecolor="#a9dfbf", lw=1.5, zorder=1)
    ax_f.add_patch(poly_macro)

    m1_f = 0.5 * (v1_f + v2_f)
    m2_f = 0.5 * (v2_f + v3_f)
    m3_f = 0.5 * (v3_f + v1_f)
    nodes_f = [m1_f, m2_f, m3_f]

    dirs_f = [(v2_f - v1_f) / np.linalg.norm(v2_f - v1_f),
              (v3_f - v2_f) / np.linalg.norm(v3_f - v2_f),
              (v1_f - v3_f) / np.linalg.norm(v1_f - v3_f)]

    # Dashed extensions
    for i in range(3):
        p_start = nodes_f[i] - 0.12 * dirs_f[i]
        p_end = nodes_f[i] + 0.22 * dirs_f[i]
        ax_f.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], color="#27ae60", linestyle="--", lw=1.2, alpha=0.55, zorder=2)
        ax_f.plot(nodes_f[i][0], nodes_f[i][1], 'o', color=c_tri, markersize=5.5, zorder=5)

    # Label P1, P2, P3 (red, blue, green)
    ax_f.text(m1_f[0], m1_f[1] - 0.10, r"$P_1$", fontsize=10.0, fontweight="bold", color=c_red, ha="center", va="center")
    ax_f.text(m2_f[0] + 0.09, m2_f[1] - 0.01, r"$P_2$", fontsize=10.0, fontweight="bold", color=c_blue, ha="left", va="center")
    ax_f.text(m3_f[0] - 0.09, m3_f[1] + 0.03, r"$P_3$", fontsize=10.0, fontweight="bold", color=c_green, ha="right", va="center")

    # Arrows and labels for t1, t2, t3
    # t1: arrow along bottom, label below arrow outside triangle
    end_f1 = m1_f + 0.18 * dirs_f[0]
    arrow_f1 = FancyArrowPatch(m1_f, end_f1, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=colors[0], lw=2.2, zorder=4)
    ax_f.add_patch(arrow_f1)
    ax_f.text(end_f1[0] + 0.01, end_f1[1] - 0.09, r"$\vec{t}_1$", fontsize=10.0, color=colors[0], ha="center", va="center")

    # t2: arrow along right edge, label to the right
    end_f2 = m2_f + 0.18 * dirs_f[1]
    arrow_f2 = FancyArrowPatch(m2_f, end_f2, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=colors[1], lw=2.2, zorder=4)
    ax_f.add_patch(arrow_f2)
    ax_f.text(end_f2[0] + 0.07, end_f2[1] + 0.04, r"$\vec{t}_2$", fontsize=10.0, color=colors[1], ha="left", va="center")

    # t3: arrow along left edge, label to the left
    end_f3 = m3_f + 0.18 * dirs_f[2]
    arrow_f3 = FancyArrowPatch(m3_f, end_f3, arrowstyle="-|>,head_length=5.5,head_width=3.8", color=colors[2], lw=2.2, zorder=4)
    ax_f.add_patch(arrow_f3)
    ax_f.text(end_f3[0] - 0.08, end_f3[1] + 0.02, r"$\vec{t}_3$", fontsize=10.0, color=colors[2], ha="right", va="center")

    # Center point P and arc
    ax_f.plot(P_eval[0], P_eval[1], '*', color=c_orange, markersize=10, markeredgecolor='black', markeredgewidth=1.0, zorder=8)
    ax_f.text(P_eval[0] + 0.08, P_eval[1] - 0.05, r"$P$", fontsize=10.5, fontweight="bold", color=c_orange, ha="left", va="center")

    # Circulation arc: radius 0.10 (diameter 0.20)
    arc = Arc(P_eval, 0.20, 0.20, angle=0, theta1=20, theta2=280, color=c_orange, linestyle=":", lw=1.8, zorder=6)
    ax_f.add_patch(arc)
    ax_f.annotate("", xy=(P_eval[0] + 0.03, P_eval[1] + 0.10), xytext=(P_eval[0] - 0.03, P_eval[1] + 0.10),
                  arrowprops=dict(arrowstyle="-|>", color=c_orange, lw=2.0, mutation_scale=13), zorder=7)

    # Save to both relatorios and artigo
    caminho_rel_pdf = os.path.join(DIRETORIO_RELATORIOS, "figura_singularidades_matriz_A.pdf")
    caminho_rel_png = os.path.join(DIRETORIO_RELATORIOS, "figura_singularidades_matriz_A.png")
    caminho_art_pdf = os.path.join(DIRETORIO_ARTIGO, "figura_singularidades_matriz_A.pdf")
    caminho_art_png = os.path.join(DIRETORIO_ARTIGO, "figura_singularidades_matriz_A.png")

    fig.savefig(caminho_rel_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(caminho_rel_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(caminho_art_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(caminho_art_png, format="png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close()

    print(f"[SUCCESS] Saved to relatorios:\n  - {caminho_rel_pdf}\n  - {caminho_rel_png}")
    print(f"[SUCCESS] Saved to artigo:\n  - {caminho_art_pdf}\n  - {caminho_art_png}")


if __name__ == "__main__":
    gerar_figura_singularidades()
