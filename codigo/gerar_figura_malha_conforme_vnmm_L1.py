#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate the publication-quality figure: figura_malha_conforme_vnmm_L1
Illustrating the conformal triangulation with VNMM L1 support nodes located at edge midpoints.
All text, labels, titles, and annotations are strictly in English.
Optimized for compact display matching caption font size.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_ATUAL)
DIRETORIO_RELATORIOS = os.path.join(DIRETORIO_RAIZ, "relatorios")
DIRETORIO_ARTIGO = os.path.join(DIRETORIO_RAIZ, "artigo")
os.makedirs(DIRETORIO_RELATORIOS, exist_ok=True)
os.makedirs(DIRETORIO_ARTIGO, exist_ok=True)

# Publication styling (IEEE / Elsevier standard)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 10.8,
    "axes.titlesize": 11.2,
    "xtick.labelsize": 9.8,
    "ytick.labelsize": 9.8,
    "legend.fontsize": 9.0,
    "figure.titlesize": 11.5,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.55,
    "grid.alpha": 0.45,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})


def gerar_figura_malha_conforme_vnmm_L1():
    print(">>> Generating figura_malha_conforme_vnmm_L1 (strictly in English)...")
    
    # 1. Mesh generation: 8x8 conformal triangulation with 10% controlled nodal jitter
    np.random.seed(42)
    Lx = np.pi
    Ly = np.pi
    Nex, Ney = 8, 8
    dx = Lx / Nex
    dy = Ly / Ney
    jitter = 0.10
    
    x_lin = np.linspace(0.0, Lx, Nex + 1)
    y_lin = np.linspace(0.0, Ly, Ney + 1)
    
    vertices = []
    grid_v = np.zeros((Ney + 1, Nex + 1), dtype=int)
    vid = 0
    for j in range(Ney + 1):
        for i in range(Nex + 1):
            x, y = x_lin[i], y_lin[j]
            if jitter > 0.0 and 0 < i < Nex and 0 < j < Ney:
                x += np.random.uniform(-jitter * dx, jitter * dx)
                y += np.random.uniform(-jitter * dy, jitter * dy)
            vertices.append([x, y])
            grid_v[j, i] = vid
            vid += 1
    vertices = np.array(vertices, dtype=float)
    
    triangulos = []
    for j in range(Ney):
        for i in range(Nex):
            v_bl = grid_v[j, i]
            v_br = grid_v[j, i + 1]
            v_tl = grid_v[j + 1, i]
            v_tr = grid_v[j + 1, i + 1]
            triangulos.append([v_bl, v_br, v_tr])
            triangulos.append([v_bl, v_tr, v_tl])
    triangulos = np.array(triangulos, dtype=int)
    
    # Unique edges and midpoints
    edge_dict = {}
    for tri in triangulos:
        loc = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
        for v1, v2 in loc:
            pair = (min(v1, v2), max(v1, v2))
            if pair not in edge_dict:
                edge_dict[pair] = len(edge_dict)
                
    vnmm_coords = []
    for v1, v2 in edge_dict.keys():
        p1, p2 = vertices[v1], vertices[v2]
        vnmm_coords.append(0.5 * (p1 + p2))
    vnmm_coords = np.array(vnmm_coords)
    
    # 2. Plotting with compact canvas size (4.8 x 4.15 in)
    fig, ax = plt.subplots(figsize=(4.8, 4.15))
    
    # Plot triangular mesh edges
    for tri in triangulos:
        pts = vertices[tri]
        pts_closed = np.vstack([pts, pts[0]])
        ax.plot(pts_closed[:, 0], pts_closed[:, 1], color='#7f8c8d', lw=0.55, alpha=0.55)
        
    # Select a well-proportioned representative support triangle in the center of the domain
    centroids = np.mean(vertices[triangulos], axis=1)
    center_domain = np.array([Lx / 2.0, Ly / 2.0])
    t_sel = int(np.argmin(np.linalg.norm(centroids - center_domain, axis=1)))
    
    pts_sel = vertices[triangulos[t_sel]]
    pts_sel_closed = np.vstack([pts_sel, pts_sel[0]])
    tri_centroid = np.mean(pts_sel, axis=0)
    
    ax.fill(pts_sel[:, 0], pts_sel[:, 1], color='#3498db', alpha=0.25, label=r"Local Support Simplex $T$")
    ax.plot(pts_sel_closed[:, 0], pts_sel_closed[:, 1], color='#1f77b4', lw=2.1, zorder=4)
    
    # Plot global VNMM midpoint nodes
    ax.plot(vnmm_coords[:, 0], vnmm_coords[:, 1], 'o', color='#95a5a6', markersize=2.6, alpha=0.6,
            label=r"Global VNMM Nodes (Midpoints)", zorder=3)
            
    # Highlight the 3 local support nodes of triangle T with directional tangent vectors
    loc_edges = [(pts_sel[0], pts_sel[1]), (pts_sel[1], pts_sel[2]), (pts_sel[2], pts_sel[0])]
    for k, (va, vb) in enumerate(loc_edges):
        mid = 0.5 * (va + vb)
        diff = vb - va
        Lk = np.linalg.norm(diff)
        tk = diff / Lk
        
        # Vector pointing away from the triangle centroid for label placement
        outward = mid - tri_centroid
        outward = outward / np.linalg.norm(outward)
        
        # Red node marker
        ax.plot(mid[0], mid[1], 'ro', markersize=7.0, markeredgecolor='#800000', markeredgewidth=1.1, zorder=6)
        
        # Directional arrow along edge
        ax.quiver(mid[0], mid[1], tk[0], tk[1], color='#c0392b', scale=10.5, width=0.009,
                  headwidth=4.5, headlength=5.5, zorder=7)
                  
        # Specific adjustments for perfectly clear orientation without overlap with the triangle
        if k == 0:  # Bottom edge
            label_pos = mid + np.array([0.0, -0.22])
        elif k == 1:  # Right/vertical edge
            label_pos = mid + np.array([0.36, 0.08])
        elif k == 2:  # Slanted edge
            label_pos = mid + np.array([-0.32, 0.18])
            
        ax.annotate(rf"$(P_{k+1}, \vec{{t}}_{k+1})$", xy=mid, xytext=label_pos,
                    fontsize=10.0, fontweight='bold', color='#8b0000', zorder=8,
                    ha='center', va='center',
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor="#c0392b", alpha=0.95, lw=0.85))
                    
    # Evaluation point P inside the triangle
    P_eval = tri_centroid + np.array([0.01, -0.04])
    ax.plot(P_eval[0], P_eval[1], 'm*', markersize=12.5, markeredgecolor='#4a148c', markeredgewidth=1.1,
            zorder=9, label=r"Collocation Point $\mathbf{P}$")
    ax.annotate(r"$\mathbf{P}$ (Interpolation)", xy=P_eval, xytext=(P_eval[0] + 0.48, P_eval[1] - 0.12),
                arrowprops=dict(arrowstyle="->", color='#6a1b9a', lw=1.1),
                fontsize=9.8, fontweight='bold', color='#4a148c', zorder=10,
                ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.20", facecolor="#f3e5f5", edgecolor='#6a1b9a', alpha=0.95, lw=0.85))
                
    ax.set_aspect('equal')
    ax.set_xlabel(r"Cartesian Coordinate $x$ [m]", fontsize=10.8)
    ax.set_ylabel(r"Cartesian Coordinate $y$ [m]", fontsize=10.8)
    ax.tick_params(labelsize=9.8)
    ax.grid(True, linestyle='--', alpha=0.45)
    
    leg = ax.legend(loc='upper right', framealpha=0.92, edgecolor='#cccccc', fontsize=8.8)
    if len(leg.legend_handles) > 2:
        leg.legend_handles[2]._sizes = [60]
    
    fig.tight_layout(pad=0.6)
    
    # Save to both relatorios and artigo
    caminho_rel_pdf = os.path.join(DIRETORIO_RELATORIOS, "figura_malha_conforme_vnmm_L1.pdf")
    caminho_rel_png = os.path.join(DIRETORIO_RELATORIOS, "figura_malha_conforme_vnmm_L1.png")
    caminho_art_pdf = os.path.join(DIRETORIO_ARTIGO, "figura_malha_conforme_vnmm_L1.pdf")
    caminho_art_png = os.path.join(DIRETORIO_ARTIGO, "figura_malha_conforme_vnmm_L1.png")
    
    fig.savefig(caminho_rel_pdf, bbox_inches='tight')
    fig.savefig(caminho_rel_png, bbox_inches='tight')
    fig.savefig(caminho_art_pdf, bbox_inches='tight')
    fig.savefig(caminho_art_png, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {caminho_rel_pdf} and {caminho_art_pdf}")


if __name__ == "__main__":
    gerar_figura_malha_conforme_vnmm_L1()
