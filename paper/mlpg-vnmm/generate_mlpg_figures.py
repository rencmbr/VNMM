#!/usr/bin/env python3
"""
generate_mlpg_figures.py
========================
Generates publication-quality vector figures in PDF format for the IJNME article:
"A Truly Cell-Free Vector Nodal Meshless Method via Meshless Local Petrov-Galerkin (VNMM-MLPG) 
for 2D Electromagnetic Problems"

Figures produced (saved as PDF):
1. fig_mlpg_subdomains_schematic.pdf (Figure 1: Schematic of MLPG Subdomains)
2. fig_mlpg_mms_convergence.pdf      (Figure 2: MMS Asymptotic Spatial Convergence)
3. fig_mlpg_alpha_s_sensitivity.pdf  (Figure 3: Parametric Sensitivity of Subdomain Radius)
4. fig_mlpg_cavity_jitter.pdf        (Figure 4: Cavity Cutoff Error under Jitter)
5. fig_mlpg_cavity_stabilization.pdf (Figure 5: Orthogonal Energy Stabilization Convergence)
6. fig_mlpg_benchmark_comparison.pdf (Figure 6: Computational Benchmark VNMM-MLPG vs Gauss)
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DIR_2D = os.path.join(ROOT_DIR, "2D")
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d
from scripts.mlpg_source_problem_2d import run_mlpg_mms_convergence_study

# Style setup for Wiley IJNME standard
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


def save_figure(fig, base_name):
    out_path = os.path.join(CURRENT_DIR, f"{base_name}.pdf")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig1_schematic():
    print("Generating Figure 1: Schematic of MLPG Subdomains...")
    fig, ax = plt.subplots(figsize=(7.2, 7.8), dpi=300)

    Nx, Ny = 9, 9
    Lx, Ly = np.pi, np.pi
    coords, vectors, is_bnd = generate_cavity_mesh(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, jitter_frac=0.25, interior_type='random', seed=42
    )
    subs = build_mlpg_subdomains_2d(coords, vectors, is_boundary=is_bnd, alpha_s=1.1)

    domain_rect = patches.Rectangle((0, 0), Lx, Ly, linewidth=2.0, edgecolor='#2c3e50', facecolor='#fafbfc', zorder=1)
    ax.add_patch(domain_rect)

    # Interior nodes with random directors (matching node color, smaller arrow heads)
    ax.quiver(coords[~is_bnd, 0], coords[~is_bnd, 1], vectors[~is_bnd, 0], vectors[~is_bnd, 1],
              angles='xy', scale_units='xy', scale=7.0, color='#2c3e50', alpha=0.85,
              width=0.0055, headwidth=3.0, headlength=3.5, headaxislength=3.0, zorder=2)
    p_int = ax.scatter(coords[~is_bnd, 0], coords[~is_bnd, 1], c='#2c3e50', s=22, zorder=3,
                       label=r'Interior node $\mathbf{x}_k$ (random orientation)')

    # Boundary nodes with strictly tangential directors (smaller arrow heads, smaller squares)
    ax.quiver(coords[is_bnd, 0], coords[is_bnd, 1], vectors[is_bnd, 0], vectors[is_bnd, 1],
              angles='xy', scale_units='xy', scale=7.0, color='#c0392b', alpha=0.90,
              width=0.0060, headwidth=3.0, headlength=3.5, headaxislength=3.0, zorder=2)
    p_bnd = ax.scatter(coords[is_bnd, 0], coords[is_bnd, 1], c='#c0392b', s=16, marker='s', zorder=3,
                       label=r'Boundary node (tangential $\mathbf{t}_k$)')

    def draw_stencil(idx, color):
        sub = subs[idx]
        c = sub.center
        st_idx = sub.stencil_nodes
        st_pts = coords[st_idx]
        # Support stencil represented strictly by node marks and dotted lines (no shaded patch)
        for pt in st_pts:
            ax.plot([c[0], pt[0]], [c[1], pt[1]], color=color, linestyle=':', linewidth=1.2, alpha=0.7, zorder=4)
        ax.scatter(st_pts[:, 0], st_pts[:, 1], edgecolors=color, facecolor='none', s=85, linewidth=1.5, zorder=5)

    # Case 1: Pure interior node 24
    draw_stencil(24, '#1f77b4')
    sub1 = subs[24]
    circ1 = patches.Circle(sub1.center, sub1.radius_s, facecolor='#3498db', edgecolor='#1f77b4',
                           alpha=0.38, linewidth=1.6, zorder=4)
    ax.add_patch(circ1)
    ax.scatter(sub1.center[0], sub1.center[1], c='#1f77b4', s=95, marker='*', zorder=6)

    # Case 2: Interior node intersecting boundary (node 3) - strictly truncated inside Omega
    draw_stencil(3, '#d35400')
    sub2 = subs[3]
    c2 = sub2.center
    r2 = sub2.radius_s
    alpha2 = np.arcsin(c2[1] / r2)
    thetas2 = np.linspace(-alpha2, np.pi + alpha2, 200)
    arc2_x = c2[0] + r2 * np.cos(thetas2)
    arc2_y = c2[1] + r2 * np.sin(thetas2)
    poly2_pts = np.column_stack([arc2_x, arc2_y])
    poly2 = patches.Polygon(poly2_pts, closed=True, facecolor='#f39c12', edgecolor='#d35400', alpha=0.42, linewidth=1.6, zorder=4)
    ax.add_patch(poly2)
    ax.scatter(c2[0], c2[1], c='#d35400', s=95, marker='*', zorder=6)
    dx2 = np.sqrt(r2**2 - c2[1]**2)
    ax.plot([c2[0] - dx2, c2[0] + dx2], [0, 0], color='#b9770e', linewidth=4.0, zorder=7)

    # Case 3: Boundary node B (node 68 on x=0) - strictly truncated inside Omega
    draw_stencil(68, '#8e44ad')
    sub3 = subs[68]
    c3 = sub3.center
    r3 = sub3.radius_s
    thetas3 = np.linspace(-np.pi/2, np.pi/2, 200)
    arc3_x = c3[0] + r3 * np.cos(thetas3)
    arc3_y = c3[1] + r3 * np.sin(thetas3)
    poly3_pts = np.column_stack([arc3_x, arc3_y])
    poly3 = patches.Polygon(poly3_pts, closed=True, facecolor='#9b59b6', edgecolor='#7d3c98', alpha=0.42, linewidth=1.6, zorder=4)
    ax.add_patch(poly3)
    ax.scatter(c3[0], c3[1], c='#7d3c98', s=65, marker='s', zorder=6)
    ax.plot([0, 0], [c3[1] - r3, c3[1] + r3], color='#6c3483', linewidth=4.0, zorder=7)

    # Domain labels
    ax.text(Lx * 0.78, Ly * 0.90, r'Domain $\Omega$', fontsize=11, fontweight='bold', color='#2c3e50', alpha=0.65)
    ax.text(Lx * 0.5, Ly + 0.07, r'Boundary $\partial\Omega$', fontsize=10.5, color='#2c3e50', ha='center')

    # Annotations with capital B for Case 3
    ax.annotate(r'$\mathbf{Case\ 1:\ Fully\ Interior}$' + '\n' + r'Subdomain $\Omega_s^{I_1} \subset \Omega$' + '\n' + r'Stencil $\Omega_u^{I_1}$ (6 nodes)',
                xy=(sub1.center[0] + 0.12, sub1.center[1] + 0.12),
                xytext=(sub1.center[0] + 0.50, sub1.center[1] + 0.55),
                arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2),
                fontsize=8.0, color='#1f77b4',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#ebf5fb', edgecolor='#1f77b4', alpha=0.92))

    ax.annotate(r'$\mathbf{Case\ 2:\ Truncated\ Interior}$' + '\n' + r'Subdomain $\Omega_s^{I_2} \cap \Omega$' + '\n' + r'Stencil $\Omega_u^{I_2}$ (6 nodes)',
                xy=(c2[0] + 0.25, c2[1] + 0.12),
                xytext=(c2[0] + 0.42, c2[1] + 0.12),
                arrowprops=dict(arrowstyle='->', color='#d35400', lw=1.2),
                fontsize=7.8, linespacing=1.25, color='#d35400',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='#fef5e7', edgecolor='#d35400', alpha=0.92))

    ax.annotate(r'$\mathbf{Case\ 3:\ Truncated\ Boundary}$' + '\n' + r'Subdomain $\Omega_s^B \cap \Omega$' + '\n' + r'Stencil $\Omega_u^B$ (6 nodes)',
                xy=(c3[0] + 0.12, c3[1] + 0.22),
                xytext=(c3[0] + 0.08, c3[1] + 0.72),
                arrowprops=dict(arrowstyle='->', color='#8e44ad', lw=1.2),
                fontsize=8.0, color='#8e44ad',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#f5eef8', edgecolor='#8e44ad', alpha=0.92))

    margin = 0.25
    ax.set_xlim(-margin, Lx + margin)
    ax.set_ylim(-margin, Ly + margin)
    ax.set_aspect('equal')
    ax.axis('off')

    custom_handles = [
        p_int,
        p_bnd,
        patches.Patch(facecolor='#3498db', edgecolor='#1f77b4', alpha=0.5, label=r'Subdomain $\Omega_s^{I_1} \subset \Omega$ (internal)'),
        patches.Patch(facecolor='#f39c12', edgecolor='#d35400', alpha=0.5, label=r'Subdomain $\Omega_s^{I_2} \cap \Omega$ (truncated)'),
        patches.Patch(facecolor='#9b59b6', edgecolor='#7d3c98', alpha=0.5, label=r'Subdomain $\Omega_s^B \cap \Omega$ (truncated)'),
        plt.Line2D([0], [0], color='#b9770e', linewidth=3, label=r'Boundary segment $\Gamma_{sI}, \Gamma_{sB} = \Omega_s \cap \partial\Omega$'),
        plt.Line2D([0], [0], color='#2c3e50', linestyle=':', marker='o', markerfacecolor='none', markeredgecolor='#2c3e50',
                   linewidth=1.2, markersize=7, label=r'6-node trial stencil $\Omega_u$ (connections and nodes)')
    ]

    ax.legend(handles=custom_handles, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=2, frameon=True, framealpha=0.95, edgecolor='#bdc3c7', fontsize=8.0)

    save_figure(fig, "fig_mlpg_subdomains_schematic")


def plot_fig2_mms_convergence():
    print("Generating Figure 2: MMS Asymptotic Refinement Convergence...")
    json_path = os.path.join(DIR_2D, "scripts", "mms_eabe_6meshes.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        results = data['results']
        h_vals = np.array([r['h'] for r in results])
        err_L2_dir = np.array([r['err_L2_A'] for r in results]) / 100.0
        err_curl_dir = np.array([r['err_curl_A'] for r in results]) / 100.0
        err_L2_neu = np.array([r['err_L2_B'] for r in results]) / 100.0
        err_curl_neu = np.array([r['err_curl_B'] for r in results]) / 100.0
        rate_L2_dir = data['rates']['rate_L2_A']
        rate_curl_dir = data['rates']['rate_curl_A']
        rate_L2_neu = data['rates']['rate_L2_B']
        rate_curl_neu = data['rates']['rate_curl_B']
    else:
        grids = [15, 22, 29, 36, 43, 49]
        res_dir = run_mlpg_mms_convergence_study(grid_sizes=grids, alpha_s=0.8, s_div=6.0, boundary_condition="dirichlet")
        res_neu = run_mlpg_mms_convergence_study(grid_sizes=grids, alpha_s=0.8, s_div=6.0, boundary_condition="mixed_neumann")
        h_vals = res_dir['h_values']
        err_L2_dir = res_dir['errors_L2']
        err_curl_dir = res_dir['errors_curl']
        err_L2_neu = res_neu['errors_L2']
        err_curl_neu = res_neu['errors_curl']
        rate_L2_dir = res_dir['rate_L2']
        rate_curl_dir = res_dir['rate_curl']
        rate_L2_neu = res_neu['rate_L2']
        rate_curl_neu = res_neu['rate_curl']

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    ax.loglog(h_vals, err_L2_dir, 'o-b', linewidth=1.8, markersize=6,
              label=f'Dirichlet $L^2(\\mathbf{{E}})$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_L2_dir:.2f}}})$)')
    ax.loglog(h_vals, err_curl_dir, 's-r', linewidth=1.8, markersize=6,
              label=f'Dirichlet $|H(\\mathbf{{curl}})|$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_curl_dir:.2f}}})$)')

    ax.loglog(h_vals, err_L2_neu, 'o--c', linewidth=1.8, markersize=6, fillstyle='none',
              label=f'Mixed Neumann $L^2(\\mathbf{{E}})$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_L2_neu:.2f}}})$)')
    ax.loglog(h_vals, err_curl_neu, 's--m', linewidth=1.8, markersize=6, fillstyle='none',
              label=f'Mixed Neumann $|H(\\mathbf{{curl}})|$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_curl_neu:.2f}}})$)')

    h_ref = np.array([h_vals[1], h_vals[-1]])
    slope2 = err_L2_dir[1] * (h_ref / h_ref[0]) ** 2.0
    slope1 = err_curl_dir[1] * (h_ref / h_ref[0]) ** 1.0
    ax.loglog(h_ref, slope2, 'k:', linewidth=1.2, label=r'Theoretical $\mathcal{O}(h_{\mathrm{nom}}^2)$ slope')
    ax.loglog(h_ref, slope1, 'gray', linestyle=':', linewidth=1.2, label=r'Theoretical $\mathcal{O}(h_{\mathrm{nom}}^1)$ slope')

    ax.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]')
    ax.set_ylabel('Relative Error Norm')
    ax.legend(loc='lower right', framealpha=0.92, fontsize=8.5)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    save_figure(fig, "fig_mlpg_mms_convergence")


def plot_fig3_alpha_s_sensitivity():
    print("Generating Figure 3: Parametric Sensitivity of Subdomain Radius alpha_s...")
    json_path = os.path.join(DIR_2D, "scripts", "alpha_s_parametric_data.json")
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found, skipping alpha_s sensitivity figure.")
        return

    with open(json_path, "r") as f:
        data_by_jitter = json.load(f)

    alpha_vals = [r['alpha_s'] for r in data_by_jitter['jitter_0']]
    alphas = np.array(alpha_vals)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)

    curve_styles = [
        ("jitter_0", '#1f77b4', '-', 'o', 'Jitter 0% (Regular Lattice)'),
        ("jitter_25", '#2ca02c', '--', 's', 'Jitter 25% (Perturbed)')
    ]

    # (a) Domain Integration Coverage
    ax1.axvspan(0.70, 1.20, color='#2ecc71', alpha=0.18, label=r'Operational Window $[0.7, 1.2]$')
    ax1.axvline(0.7071, color='#e74c3c', linestyle='--', linewidth=1.4, label=r'Regular Limit $\alpha_s = \sqrt{2}/2 \approx 0.7071$')
    ax1.axhline(100.0, color='#7f8c8d', linestyle=':', linewidth=1.0)

    for k, col, ls, marker, lbl in curve_styles:
        covs = np.array([r['coverage'] for r in data_by_jitter[k]])
        ax1.plot(alphas, covs, marker=marker, linestyle=ls, color=col, linewidth=2.0, markersize=6, label=lbl)

    ax1.set_xlabel(r'Subdomain Radius Scaling Factor $\alpha_s$', fontsize=10.5)
    ax1.set_ylabel(r'Domain Integration Coverage (%)', fontsize=10.5)
    ax1.set_title('(a)', fontsize=11.5)
    ax1.set_xlim(0.25, 2.05)
    ax1.set_ylim(10, 105)
    ax1.tick_params(labelsize=9.5)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.92, fontsize=9.0)

    # (b) MMS Source Problem Convergence
    ax2.axvspan(0.70, 1.20, color='#2ecc71', alpha=0.18, label=r'Operational Window $[0.7, 1.2]$')
    ax2.axvline(0.7071, color='#e74c3c', linestyle='--', linewidth=1.4, label=r'Regular Limit $\alpha_s = \sqrt{2}/2$')

    for k, col, ls, marker, lbl in curve_styles:
        mms_err = np.array([r['mms_l2'] for r in data_by_jitter[k]])
        ax2.plot(alphas, mms_err, marker=marker, linestyle=ls, color=col, linewidth=2.0, markersize=6, label=lbl)
        val_08 = mms_err[alphas == 0.80][0]
        ax2.plot([0.80], [val_08], '*', color=col, markersize=11)

    ax2.plot([], [], '*', color='#2c3e50', markersize=11, label=r'Adopted Standard $\alpha_s = 0.80$')
    ax2.set_xlabel(r'Subdomain Radius Scaling Factor $\alpha_s$', fontsize=10.5)
    ax2.set_ylabel(r'MMS Relative $L^2(\vec{E})$ Error (%)', fontsize=10.5)
    ax2.set_title('(b)', fontsize=11.5)
    ax2.set_xlim(0.25, 2.05)
    ax2.tick_params(labelsize=9.5)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left', framealpha=0.92, fontsize=9.0)

    fig.tight_layout()
    save_figure(fig, "fig_mlpg_alpha_s_sensitivity")


def plot_fig4_cavity_jitter():
    print("Generating Figure 4: MLPG Cavity Resonances Convergence under Jitter...")
    json_path = os.path.join(DIR_2D, "scripts", "jitter_benchmark_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"Missing {json_path}")

    h_vals = np.array(data['h'])
    rates = data['rates']

    fig, ax1 = plt.subplots(figsize=(6.2, 4.6), dpi=300)

    mlpg_0 = [d['mean_err_kc'] for d in data['mlpg']['0%']]
    mlpg_25 = [d['mean_err_kc'] for d in data['mlpg']['25%']]

    p_m0 = rates['mlpg']['0%']['rate_p']
    p_m25 = rates['mlpg']['25%']['rate_p']

    ax1.loglog(h_vals, mlpg_0, 'o-', color='#1f77b4', linewidth=2.0, markersize=6,
               label=f'0% Jitter (Regular, $p={p_m0:.2f}$)')
    ax1.loglog(h_vals, mlpg_25, 's-', color='#2ca02c', linewidth=2.0, markersize=6,
               label=f'25% Jitter ($p={p_m25:.2f}$)')

    ax1.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]')
    ax1.set_ylabel(r'Mean Cutoff Error $\overline{\varepsilon}_{k_c}$ [\%]')
    ax1.grid(True, which="both", linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.92)

    fig.tight_layout()
    save_figure(fig, "fig_mlpg_cavity_jitter")


def plot_fig5_cavity_stabilization():
    print("Generating Figure 5: Orthogonal Energy Stabilization Convergence...")
    grids = [15, 22, 29, 36, 43, 49]
    h = np.array([np.pi / (N - 1) for N in grids])

    # Empirical cavity cutoff errors computed with dimensionless scaling alpha_stab = c_stab * (h / L)^2 (L = pi)
    data = {
        0.0: {
            0: [3.87, 3.55, 2.58, 2.28, 2.03, 2.32],
            20: [6.69, 3.31, 1.84, 2.11, 1.59, 1.92],
            40: [11.14, 4.41, 2.06, 2.24, 1.34, 1.59],
            70: [17.67, 7.11, 3.34, 2.87, 1.39, 1.29],
            100: [23.81, 9.73, 5.38, 3.94, 1.67, 1.17]
        },
        0.25: {
            0: [12.67, 5.70, 5.61, 5.15, 4.27, 3.95],
            20: [4.44, 3.73, 4.37, 4.02, 3.46, 3.37],
            40: [5.02, 4.39, 3.93, 3.17, 2.68, 2.80],
            70: [11.88, 6.82, 3.77, 2.36, 1.77, 1.97],
            100: [18.61, 10.12, 4.78, 2.17, 1.38, 1.45]
        }
    }

    colors = {
        0: '#7f7f7f',
        20: '#1f77b4',
        40: '#2ca02c',
        70: '#ff7f0e',
        100: '#d62728'
    }
    markers = {
        0: 'x',
        20: 's',
        40: '^',
        70: 'o',
        100: 'd'
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=300)

    def get_rate(h_vals, err_vals):
        log_h = np.log(h_vals)
        log_e = np.log(err_vals)
        p, _ = np.polyfit(log_h, log_e, 1)
        return p

    # Panel (a): 0% Jitter (Regular)
    for c_stab in [0, 20, 40, 70, 100]:
        errs = data[0.0][c_stab]
        p = get_rate(h, errs)
        lbl = rf'$c_{{\mathrm{{stab}}}}={c_stab}$ ($p={p:.2f}$)' if c_stab > 0 else f'Unstabilized ($c=0$, $p={p:.2f}$)'
        ls = '--' if c_stab == 0 else '-'
        lw = 1.5 if c_stab == 0 else 2.0
        ax1.loglog(h, errs, marker=markers[c_stab], linestyle=ls, color=colors[c_stab],
                   linewidth=lw, markersize=6, label=lbl)

    ax1.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]', fontsize=10.5)
    ax1.set_ylabel(r'Mean Cutoff Error $\overline{\varepsilon}_{k_c}$ [\%]', fontsize=10.5)
    ax1.set_title(r'(a) Regular Lattice ($0\%$ Jitter)', fontsize=11.5)
    ax1.grid(True, which="both", linestyle=':', alpha=0.5)

    # Panel (b): 25% Jitter (Perturbed)
    for c_stab in [0, 20, 40, 70, 100]:
        errs = data[0.25][c_stab]
        p = get_rate(h, errs)
        lbl = rf'$c_{{\mathrm{{stab}}}}={c_stab}$ ($p={p:.2f}$)' if c_stab > 0 else f'Unstabilized ($c=0$, $p={p:.2f}$)'
        ls = '--' if c_stab == 0 else '-'
        lw = 1.5 if c_stab == 0 else 2.0
        ax2.loglog(h, errs, marker=markers[c_stab], linestyle=ls, color=colors[c_stab],
                   linewidth=lw, markersize=6, label=lbl)

    ax2.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]', fontsize=10.5)
    ax2.set_ylabel(r'Mean Cutoff Error $\overline{\varepsilon}_{k_c}$ [\%]', fontsize=10.5)
    ax2.set_title(r'(b) Perturbed Point Cloud ($25\%$ Jitter)', fontsize=11.5)
    ax2.grid(True, which="both", linestyle=':', alpha=0.5)

    xlim = (min(h) * 0.85, max(h) * 1.15)
    ylim = (0.8, 30.0)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    ax1.legend(loc='lower right', framealpha=0.92, fontsize=8.0)
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=8.0)

    fig.tight_layout()
    save_figure(fig, "fig_mlpg_cavity_stabilization")


def plot_fig6_benchmark():
    print("Generating Figure 6: Computational Benchmark Convergence...")
    grids = [15, 22, 29, 36, 43, 49]
    h_vals = np.array([np.pi / (N - 1) for N in grids])

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)

    gauss_0 = [25.25, 13.46, 6.18, 4.22, 2.46, 1.53]
    gauss_25 = [24.67, 11.49, 5.94, 3.10, 1.96, 1.63]

    mlpg_0 = [17.67, 7.11, 3.34, 2.87, 1.39, 1.29]
    mlpg_25 = [11.88, 6.82, 3.77, 2.36, 1.77, 1.97]

    p_g0, _ = np.polyfit(np.log(h_vals), np.log(gauss_0), 1)
    p_g25, _ = np.polyfit(np.log(h_vals), np.log(gauss_25), 1)
    p_m0, _ = np.polyfit(np.log(h_vals), np.log(mlpg_0), 1)
    p_m25, _ = np.polyfit(np.log(h_vals), np.log(mlpg_25), 1)

    ax.loglog(h_vals, gauss_0, 's--', color='#1f77b4', markerfacecolor='none', markeredgewidth=1.4,
              linewidth=1.6, label=f'Gauss 0% ($p={p_g0:.2f}$)')
    ax.loglog(h_vals, mlpg_0, 'o-', color='#1f77b4', linewidth=2.0,
              label=f'MLPG 0% ($p={p_m0:.2f}$)')

    ax.loglog(h_vals, gauss_25, 's--', color='#2ca02c', markerfacecolor='none', markeredgewidth=1.4,
              linewidth=1.6, label=f'Gauss 25% ($p={p_g25:.2f}$)')
    ax.loglog(h_vals, mlpg_25, 'o-', color='#2ca02c', linewidth=2.0,
              label=f'MLPG 25% ($p={p_m25:.2f}$)')

    ax.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]')
    ax.set_ylabel(r'Mean Cutoff Error $\overline{\varepsilon}_{k_c}$ [\%]')
    ax.grid(True, which="both", linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.92, fontsize=8.5, ncol=2)

    fig.tight_layout()
    save_figure(fig, "fig_mlpg_benchmark_comparison")


def main():
    print("=" * 65)
    print("Generating Figures (PDF) for IJNME Paper in paper/mlpg-vnmm/")
    print("=" * 65)
    plot_fig1_schematic()
    plot_fig2_mms_convergence()
    plot_fig3_alpha_s_sensitivity()
    plot_fig4_cavity_jitter()
    plot_fig5_cavity_stabilization()
    plot_fig6_benchmark()
    print("=" * 65)
    print("All 6 manuscript figures generated successfully as PDF.")
    print("=" * 65)


if __name__ == "__main__":
    main()
