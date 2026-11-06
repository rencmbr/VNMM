#!/usr/bin/env python3
"""
generate_mlpg_figures.py
========================
Generates publication-quality figures for the EABE article:
"A Truly Cell-Free Vector Nodal Meshless Method via Meshless Local Petrov-Galerkin (VNMM-MLPG) 
for 2D Electromagnetic Problems"

Figures produced (saved as PNG):
1. fig_mlpg_subdomains_schematic.png
2. fig_mlpg_patch_test.png
3. fig_mlpg_mms_convergence.png
4. fig_mlpg_cavity_modes_and_purification.png
5. fig_mlpg_benchmark_comparison.png
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
DIR_2D = os.path.join(ROOT_DIR, "2D")
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d
from src.mlpg_assembler_2d import assemble_mlpg_matrices_2d
from src.eigen_solver_cavity_2d import apply_pec_boundary_conditions, ANALYTICAL_REFERENCE_MODES
from scripts.mlpg_patch_test_2d import AFFINE_MODES, run_mlpg_patch_test_mode
from scripts.mlpg_source_problem_2d import run_mlpg_mms_convergence_study
from scripts.mlpg_cavity_eigenvalues_2d import (
    solve_mlpg_cavity_eigenvalues,
    run_mlpg_s_div_sensitivity_study,
    run_mlpg_vs_gauss_benchmark
)

# Style setup for IEEE / Elsevier standard
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
    'savefig.bbox': 'tight'
})


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
                fontsize=7.3, linespacing=1.1, color='#d35400',
                bbox=dict(boxstyle='round,pad=0.18', facecolor='#fef5e7', edgecolor='#d35400', alpha=0.92))

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

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_subdomains_schematic.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig2_patch_test():
    print("Generating Figure 2: Patch Test Consistency (Unsymmetric vs. Symmetric)...")
    states_labels = [
        "Rigid X", "Rigid Y", "Dilatation",
        "Pure Shear", "Affine Solenoidal", "Irrotational", "General Affine"
    ]
    err_reg_unsym = []
    err_reg_sym = []
    err_jit_unsym = []
    err_jit_sym = []

    coords_reg, vecs_reg, bnd_reg = generate_cavity_mesh(Nx=11, Ny=11, jitter_frac=0.0)
    subs_reg = build_mlpg_subdomains_2d(coords_reg, vecs_reg, is_boundary=bnd_reg, alpha_s=0.8)

    coords_jit, vecs_jit, bnd_jit = generate_cavity_mesh(Nx=11, Ny=11, jitter_frac=0.25, seed=42)
    subs_jit = build_mlpg_subdomains_2d(coords_jit, vecs_jit, is_boundary=bnd_jit, alpha_s=0.8)

    for mode in AFFINE_MODES:
        res_reg_u = run_mlpg_patch_test_mode(coords_reg, vecs_reg, bnd_reg, mode, subdomains=subs_reg, assembly_mode="row_petrov_galerkin")
        res_reg_s = run_mlpg_patch_test_mode(coords_reg, vecs_reg, bnd_reg, mode, subdomains=subs_reg, assembly_mode="subdomain_galerkin")
        res_jit_u = run_mlpg_patch_test_mode(coords_jit, vecs_jit, bnd_jit, mode, subdomains=subs_jit, assembly_mode="row_petrov_galerkin")
        res_jit_s = run_mlpg_patch_test_mode(coords_jit, vecs_jit, bnd_jit, mode, subdomains=subs_jit, assembly_mode="subdomain_galerkin")

        err_reg_unsym.append(max(res_reg_u['error_dof_rel'], 1e-16))
        err_reg_sym.append(max(res_reg_s['error_dof_rel'], 1e-16))
        err_jit_unsym.append(max(res_jit_u['error_dof_rel'], 1e-16))
        err_jit_sym.append(max(res_jit_s['error_dof_rel'], 1e-16))

    x = np.arange(len(states_labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

    # Panel (a): Regular Mesh
    ax1.bar(x - width/2, err_reg_unsym, width, label='Unsymmetric (Row Petrov-Gal.)', color='steelblue')
    ax1.bar(x + width/2, err_reg_sym, width, label='Symmetric (Subdomain Galerkin)', color='mediumseagreen')
    ax1.axhline(1e-11, color='red', linestyle='--', linewidth=1.2, label='Patch Test Tol. ($10^{-11}$)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Relative Error $\\|\\mathbf{e}_{\\mathrm{int}} - \\mathbf{e}^*\\|_\\infty / \\|\\mathbf{e}^*\\|_\\infty$')
    ax1.set_title('(a) Regular Mesh ($11 \\times 11$)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(states_labels, rotation=25, ha='right')
    ax1.set_ylim(1e-16, 1e-8)
    ax1.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    # Panel (b): Irregular Cloud (25% Jitter)
    ax2.bar(x - width/2, err_jit_unsym, width, label='Unsymmetric (Row Petrov-Gal.)', color='darkorange')
    ax2.bar(x + width/2, err_jit_sym, width, label='Symmetric (Subdomain Galerkin)', color='mediumpurple')
    ax2.axhline(1e-11, color='red', linestyle='--', linewidth=1.2, label='Patch Test Tol. ($10^{-11}$)')
    ax2.set_yscale('log')
    ax2.set_title('(b) Irregular Cloud (25% Jitter)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(states_labels, rotation=25, ha='right')
    ax2.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)

    fig.tight_layout()
    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_patch_test.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig3_mms_convergence():
    print("Generating Figure 3: MMS Asymptotic Refinement Convergence on 6 EABE Meshes...")
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

    # Pure Dirichlet curves
    ax.loglog(h_vals, err_L2_dir, 'o-b', linewidth=1.8, markersize=6,
              label=f'Dirichlet $L^2(\\mathbf{{E}})$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_L2_dir:.2f}}})$)')
    ax.loglog(h_vals, err_curl_dir, 's-r', linewidth=1.8, markersize=6,
              label=f'Dirichlet $|H(\\mathbf{{curl}})|$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_curl_dir:.2f}}})$)')

    # Mixed Dirichlet-Neumann curves
    ax.loglog(h_vals, err_L2_neu, 'o--c', linewidth=1.8, markersize=6, fillstyle='none',
              label=f'Mixed Neumann $L^2(\\mathbf{{E}})$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_L2_neu:.2f}}})$)')
    ax.loglog(h_vals, err_curl_neu, 's--m', linewidth=1.8, markersize=6, fillstyle='none',
              label=f'Mixed Neumann $|H(\\mathbf{{curl}})|$ (Rate: $\\mathcal{{O}}(h_{{\\mathrm{{nom}}}}^{{{rate_curl_neu:.2f}}})$)')

    # Theoretical asymptotic slopes
    h_ref = np.array([h_vals[1], h_vals[-1]])
    slope2 = err_L2_dir[1] * (h_ref / h_ref[0]) ** 2.0
    slope1 = err_curl_dir[1] * (h_ref / h_ref[0]) ** 1.0
    ax.loglog(h_ref, slope2, 'k:', linewidth=1.2, label=r'Theoretical $\mathcal{O}(h_{\mathrm{nom}}^2)$ slope')
    ax.loglog(h_ref, slope1, 'gray', linestyle=':', linewidth=1.2, label=r'Theoretical $\mathcal{O}(h_{\mathrm{nom}}^1)$ slope')

    ax.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]')
    ax.set_ylabel('Relative Error Norm')
    ax.legend(loc='lower right', framealpha=0.92, fontsize=8.5)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_mms_convergence.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig4_cavity_modes():
    print("Generating Figure 4: Cavity Resonances and Spurious Mode Shifting...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel (a): First 10 physical modes vs Reference (Ortiz 2023)
    res_cav = solve_mlpg_cavity_eigenvalues(Nx=21, Ny=21, s_div=6.0, alpha_s=0.80, num_eigenvalues=10)
    indices = np.arange(1, 11)
    labels = res_cav['mode_labels']
    kc_num = res_cav['numerical_kc']
    kc_ref = res_cav['analytical_kc']

    ax1.plot(indices, kc_ref, 'sk', markersize=8, fillstyle='none', linewidth=1.5, label='Analytical Reference (Ortiz, 2023)')
    ax1.plot(indices, kc_num, 'o-b', markersize=6, linewidth=1.8, label=r'VNMM-MLPG ($21 \times 21$, $s_{\mathrm{div}}=6.0$)')

    for i in range(10):
        ax1.text(indices[i], kc_num[i] + 0.08, labels[i], ha='center', fontsize=7.5, color='darkblue')

    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel(r'Cutoff Wavenumber $k_c = \sqrt{\lambda}$ [m$^{-1}$]')
    ax1.set_title('(a) First 10 Cavity TE Wavenumbers')
    ax1.set_xticks(indices)
    ax1.legend(loc='lower right', framealpha=0.92)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel (b): s_div sensitivity study (spurious mode shifting)
    sens = run_mlpg_s_div_sensitivity_study(Nx=21, Ny=21, s_div_list=[0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0], verbose=False)
    s_vals = [item['s_div'] for item in sens]
    spur_counts = [item['spur_in_10'] for item in sens]
    mean_errs = [item['mean_err_kc'] for item in sens]

    color = 'tab:red'
    ax2.set_xlabel(r'Divergence Regularization Parameter $s_{\mathrm{div}}$')
    ax2.set_ylabel(r'Spurious Modes in Operational Band ($\leq 10$)', color=color)
    line1 = ax2.plot(s_vals, spur_counts, 's-', color=color, linewidth=1.8, label='Spurious Count')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-0.5, 11)

    ax2_twin = ax2.twinx()
    color2 = 'tab:blue'
    ax2_twin.set_ylabel('Mean Cutoff Error $\\overline{\\varepsilon}_{k_c}$ (%)', color=color2)
    line2 = ax2_twin.plot(s_vals, mean_errs, 'o--', color=color2, linewidth=1.8, label='Mean $k_c$ Error')
    ax2_twin.tick_params(axis='y', labelcolor=color2)
    ax2_twin.set_ylim(0, 35)

    ax2.set_title('(b) Spurious Mode Shifting vs. $s_{\\mathrm{div}}$')
    ax2.grid(True, linestyle=':', alpha=0.5)

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_cavity_modes_and_purification.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig_cavity_jitter():
    print("Generating Figure: MLPG Cavity Resonances Convergence under Jitter...")
    json_path = os.path.join(DIR_2D, "scripts", "jitter_benchmark_results.json")
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"Missing {json_path}")

    grids = data['grids']
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
    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_cavity_jitter.png")
    fig.savefig(out_path)
    # Also save with legacy name for backward compatibility
    fig.savefig(os.path.join(CURRENT_DIR, "fig_mlpg_cavity_jitter_convergence.png"))
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig_cavity_stabilization():
    print("Generating Figure: Orthogonal Energy Stabilization Convergence (Regular vs. Jitter)...")
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=300)

    # (a) Regular Grid (0% Jitter) vs c_stab
    ax1.loglog(h, data[0.0][0], 'o-', color='#1f77b4', linewidth=1.6, markersize=5.5,
               label=r'Unstabilized ($c_{\mathrm{stab}}=0, p=0.53$)')
    ax1.loglog(h, data[0.0][20], '^-', color='#ff7f0e', linewidth=1.6, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=20$ ($p=1.06$)')
    ax1.loglog(h, data[0.0][40], 'v-', color='#2ca02c', linewidth=1.6, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=40$ ($p=1.64$)')
    ax1.loglog(h, data[0.0][70], 'd-', color='#9467bd', linewidth=1.6, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=70$ ($p=2.15$)')
    ax1.loglog(h, data[0.0][100], 's-', color='#d62728', linewidth=1.8, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=100$ ($p=2.39$)')

    # Reference O(h^2) slope on ax1
    h_ref = np.array([h[2], h[5]])
    err_ref = data[0.0][100][2] * (h_ref / h_ref[0])**2
    ax1.loglog(h_ref, err_ref, 'k--', linewidth=1.3, label=r'$\mathcal{O}(h_{\mathrm{nom}}^2)$ reference slope')

    ax1.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]')
    ax1.set_ylabel(r'Mean Cutoff Error $\overline{\varepsilon}_{k_c}$ [%]')
    ax1.set_title('(a)', fontsize=11)
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.92, fontsize=8.0)

    # (b) Perturbed Point Cloud (25% Jitter) vs c_stab
    ax2.loglog(h, data[0.25][0], 'o-', color='#1f77b4', linewidth=1.6, markersize=5.5,
               label=r'Unstabilized ($c_{\mathrm{stab}}=0, p=0.84$)')
    ax2.loglog(h, data[0.25][20], '^-', color='#ff7f0e', linewidth=1.6, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=20$ ($p=0.18$)')
    ax2.loglog(h, data[0.25][40], 'v-', color='#2ca02c', linewidth=1.6, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=40$ ($p=0.54$)')
    ax2.loglog(h, data[0.25][70], 'd-', color='#9467bd', linewidth=1.6, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=70$ ($p=1.62$)')
    ax2.loglog(h, data[0.25][100], 's-', color='#d62728', linewidth=1.8, markersize=5.5,
               label=r'$c_{\mathrm{stab}}=100$ ($p=2.30$)')

    # Reference O(h^2) slope on ax2
    h_ref2 = np.array([h[2], h[5]])
    err_ref2 = data[0.25][100][2] * (h_ref2 / h_ref2[0])**2
    ax2.loglog(h_ref2, err_ref2, 'k--', linewidth=1.3, label=r'$\mathcal{O}(h_{\mathrm{nom}}^2)$ reference slope')

    ax2.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]')
    ax2.set_ylabel(r'Mean Cutoff Error $\overline{\varepsilon}_{k_c}$ [%]')
    ax2.set_title('(b)', fontsize=11)
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)

    # Identical scales for both panels
    xlim = (0.055, 0.26)
    ylim = (0.8, 32.0)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)

    ax1.legend(loc='lower right', framealpha=0.92, fontsize=8.0)
    ax2.legend(loc='lower right', framealpha=0.92, fontsize=8.0)

    fig.tight_layout()
    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_cavity_stabilization.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig5_benchmark():
    print("Generating Figure 6: Computational Benchmark Convergence (Gauss vs. Stabilized MLPG c_stab=70)...")
    grids = [15, 22, 29, 36, 43, 49]
    h_vals = np.array([np.pi / (N - 1) for N in grids])

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)

    # Cutoff errors for Gauss vs. Stabilized MLPG (c_stab = 70)
    gauss_0 = [25.25, 13.46, 6.18, 4.22, 2.46, 1.53]
    gauss_25 = [24.67, 11.49, 5.94, 3.10, 1.96, 1.63]

    mlpg_0 = [17.67, 7.11, 3.34, 2.87, 1.39, 1.29]
    mlpg_25 = [11.88, 6.82, 3.77, 2.36, 1.77, 1.97]

    p_g0 = 2.25
    p_g25 = 2.29

    p_m0 = 2.15
    p_m25 = 1.62

    # 0% Jitter (Blue)
    ax.loglog(h_vals, gauss_0, 's--', color='#1f77b4', markerfacecolor='none', markeredgewidth=1.4,
              linewidth=1.6, label=f'Gauss 0% ($p={p_g0:.2f}$)')
    ax.loglog(h_vals, mlpg_0, 'o-', color='#1f77b4', linewidth=2.0,
              label=f'MLPG 0% ($p={p_m0:.2f}$)')

    # 25% Jitter (Green)
    ax.loglog(h_vals, gauss_25, 's--', color='#2ca02c', markerfacecolor='none', markeredgewidth=1.4,
              linewidth=1.6, label=f'Gauss 25% ($p={p_g25:.2f}$)')
    ax.loglog(h_vals, mlpg_25, 'o-', color='#2ca02c', linewidth=2.0,
              label=f'MLPG 25% ($p={p_m25:.2f}$)')

    ax.set_xlabel(r'Nominal Nodal Spacing $h_{\mathrm{nom}}$ [m]')
    ax.set_ylabel(r'Mean Cutoff Error $\overline{\varepsilon}_{k_c}$ [\%]')
    ax.grid(True, which="both", linestyle=':', alpha=0.5)
    ax.legend(loc='upper left', framealpha=0.92, fontsize=8.5, ncol=2)

    fig.tight_layout()
    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_benchmark_comparison.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig_alpha_s_sensitivity():
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
    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_alpha_s_sensitivity.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("=" * 65)
    print("Generating Figures for EABE Paper in paper/mlpg-vnmm/")
    print("=" * 65)
    plot_fig1_schematic()
    plot_fig2_patch_test()
    plot_fig3_mms_convergence()
    plot_fig_alpha_s_sensitivity()
    plot_fig4_cavity_modes()
    plot_fig_cavity_jitter()
    plot_fig_cavity_stabilization()
    plot_fig5_benchmark()
    print("=" * 65)
    print("All figures generated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
