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
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    Nx, Ny = 9, 9
    Lx, Ly = np.pi, np.pi
    coords, vectors, is_bnd = generate_cavity_mesh(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, jitter_frac=0.08, seed=42)
    subdomains = build_mlpg_subdomains_2d(coords, vectors, is_boundary=is_bnd, alpha_s=0.85)

    # Draw domain boundary
    rect = patches.Rectangle((0, 0), Lx, Ly, linewidth=1.5, edgecolor='black', facecolor='whitesmoke', zorder=1)
    ax.add_patch(rect)

    # Plot all nodes and director vectors
    ax.quiver(coords[:, 0], coords[:, 1], vectors[:, 0], vectors[:, 1],
              angles='xy', scale_units='xy', scale=7, color='gray', alpha=0.6, width=0.003, zorder=2)
    ax.scatter(coords[~is_bnd, 0], coords[~is_bnd, 1], c='navy', s=25, label='Interior Nodes ($c_k$ free)', zorder=3)
    ax.scatter(coords[is_bnd, 0], coords[is_bnd, 1], c='crimson', s=35, marker='s', label='PEC Boundary Nodes ($e_k = 0$)', zorder=3)

    # Highlight an interior node I
    idx_int = 20  # near center
    sub_int = subdomains[idx_int]
    c_int = sub_int.center
    r_int = sub_int.radius_s

    # Subdomain disk Omega_s^I
    circle_int = patches.Circle(c_int, r_int, edgecolor='blue', facecolor='cyan', alpha=0.35, linewidth=1.8,
                                linestyle='--', zorder=4, label='Local Subdomain $\\Omega_s^I$ (radius $r_s$)')
    ax.add_patch(circle_int)
    ax.scatter(c_int[0], c_int[1], c='blue', s=70, marker='*', zorder=6, label='Central Node $\\mathbf{x}_I$')

    # Draw stencil connections for Omega_u^I
    st_coords = coords[sub_int.stencil_nodes]
    for pt in st_coords:
        ax.plot([c_int[0], pt[0]], [c_int[1], pt[1]], 'b:', alpha=0.7, linewidth=1.2, zorder=4)
    ax.scatter(st_coords[:, 0], st_coords[:, 1], edgecolors='blue', facecolor='none', s=80, linewidth=1.5, zorder=5,
               label='Trial Support Stencil $\\Omega_u^I$ (6 nodes)')

    # Highlight a boundary node B
    bnd_indices = np.where(is_bnd)[0]
    idx_bnd = bnd_indices[4]  # on bottom wall
    sub_b = subdomains[idx_bnd]
    c_b = sub_b.center
    r_b = sub_b.radius_s

    circle_b = patches.Circle(c_b, r_b, edgecolor='red', facecolor='coral', alpha=0.4, linewidth=1.8,
                              linestyle='--', zorder=4, label='Boundary Subdomain $\\Omega_s^B$ (intersects $\\partial\\Omega$)')
    ax.add_patch(circle_b)
    ax.scatter(c_b[0], c_b[1], c='red', s=70, marker='s', zorder=6)

    ax.set_xlim(-0.25, Lx + 0.25)
    ax.set_ylim(-0.25, Ly + 0.25)
    ax.set_aspect('equal')
    ax.set_xlabel('$x$ [rad]')
    ax.set_ylabel('$y$ [rad]')
    ax.set_title('VNMM-MLPG 2D Discretization: Local Subdomains $\\Omega_s$ and Stencils $\\Omega_u$')
    ax.legend(loc='upper right', framealpha=0.92, fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.4)

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_subdomains_schematic.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig2_patch_test():
    print("Generating Figure 2: Patch Test Consistency across 7 Affine States...")
    states_labels = [
        "Rigid X", "Rigid Y", "Dilatation",
        "Pure Shear", "Affine Solenoidal", "Irrotational", "General Affine"
    ]
    err_regular = []
    err_jitter = []

    coords_reg, vecs_reg, bnd_reg = generate_cavity_mesh(Nx=11, Ny=11, jitter_frac=0.0)
    subs_reg = build_mlpg_subdomains_2d(coords_reg, vecs_reg, is_boundary=bnd_reg, alpha_s=0.8)

    coords_jit, vecs_jit, bnd_jit = generate_cavity_mesh(Nx=11, Ny=11, jitter_frac=0.25, seed=42)
    subs_jit = build_mlpg_subdomains_2d(coords_jit, vecs_jit, is_boundary=bnd_jit, alpha_s=0.8)

    for mode in AFFINE_MODES:
        res_reg = run_mlpg_patch_test_mode(coords_reg, vecs_reg, bnd_reg, mode, subdomains=subs_reg)
        res_jit = run_mlpg_patch_test_mode(coords_jit, vecs_jit, bnd_jit, mode, subdomains=subs_jit)
        err_regular.append(max(res_reg['error_dof_rel'], 1e-16))
        err_jitter.append(max(res_jit['error_dof_rel'], 1e-16))

    x = np.arange(len(states_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    rects1 = ax.bar(x - width/2, err_regular, width, label='Regular Grid ($N=11 \\times 11$)', color='steelblue')
    rects2 = ax.bar(x + width/2, err_jitter, width, label='Irregular Cloud (25% Jitter)', color='darkorange')

    ax.axhline(1e-11, color='red', linestyle='--', linewidth=1.2, label='Patch Test Tolerance ($10^{-11}$)')
    ax.set_yscale('log')
    ax.set_ylabel('Relative Discrete Error $\\|\\mathbf{e}_{\\mathrm{int}} - \\mathbf{e}^*\\|_\\infty / \\|\\mathbf{e}^*\\|_\\infty$')
    ax.set_title('Galerkin Patch Test Satisfaction: Machine-Precision Linearity')
    ax.set_xticks(x)
    ax.set_xticklabels(states_labels, rotation=25, ha='right')
    ax.set_ylim(1e-17, 1e-8)
    ax.legend(loc='upper right', framealpha=0.92)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_patch_test.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig3_mms_convergence():
    print("Generating Figure 3: MMS Asymptotic Refinement Convergence...")
    grids = [9, 13, 17, 21, 25]
    res = run_mlpg_mms_convergence_study(grid_sizes=grids, alpha_s=0.8, s_div=6.0)

    h_vals = res['h_values']
    err_L2 = res['errors_L2']
    err_curl = res['errors_curl']
    err_dof = res['errors_dof']

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    ax.loglog(h_vals, err_L2, 'o-b', linewidth=1.8, markersize=6,
              label=f'$L^2(\\mathbf{{E}})$ error (Rate: $\\mathcal{{O}}(h^{{{res["rate_L2"]:.2f}}})$)')
    ax.loglog(h_vals, err_curl, 's-r', linewidth=1.8, markersize=6,
              label=f'$H(\\mathbf{{curl}})$ error (Rate: $\\mathcal{{O}}(h^{{{res["rate_curl"]:.2f}}})$)')
    ax.loglog(h_vals, err_dof, '^--k', linewidth=1.4, markersize=6,
              label=f'Discrete DoF error (Rate: $\\mathcal{{O}}(h^{{{res["rate_dof"]:.2f}}})$)')

    # Add theoretical asymptotic slopes
    h_ref = np.array([h_vals[1], h_vals[-1]])
    slope2 = err_L2[1] * (h_ref / h_ref[0]) ** 2.0
    slope1 = err_curl[1] * (h_ref / h_ref[0]) ** 1.0
    ax.loglog(h_ref, slope2, 'b:', linewidth=1.2, label='Theoretical $\\mathcal{O}(h^2)$ slope')
    ax.loglog(h_ref, slope1, 'r:', linewidth=1.2, label='Theoretical $\\mathcal{O}(h^1)$ slope')

    ax.set_xlabel('Characteristic Nodal Spacing $h = \\pi / (N - 1)$ [rad]')
    ax.set_ylabel('Relative Error Norm')
    ax.set_title('MMS Asymptotic Convergence: VNMM-MLPG Optimal $\\mathcal{P}^1$ Rates')
    ax.legend(loc='lower right', framealpha=0.92)
    ax.grid(True, which='both', linestyle=':', alpha=0.5)

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_mms_convergence.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig4_cavity_modes():
    print("Generating Figure 4: Cavity Resonances and Spectral Purification...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

    # Panel (a): First 10 physical modes vs Reference (Ortiz 2023)
    res_cav = solve_mlpg_cavity_eigenvalues(Nx=21, Ny=21, s_div=15.0, alpha_s=0.9, num_eigenvalues=10)
    indices = np.arange(1, 11)
    labels = res_cav['mode_labels']
    kc_num = res_cav['numerical_kc']
    kc_ref = res_cav['analytical_kc']

    ax1.plot(indices, kc_ref, 'sk', markersize=8, fillstyle='none', linewidth=1.5, label='Analytical Reference (Ortiz, 2023)')
    ax1.plot(indices, kc_num, 'o-b', markersize=6, linewidth=1.8, label='VNMM-MLPG ($21 \\times 21$, $s_{\\mathrm{div}}=15$)')

    for i in range(10):
        ax1.text(indices[i], kc_num[i] + 0.08, labels[i], ha='center', fontsize=7.5, color='darkblue')

    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Cutoff Wavenumber $k_c = \\sqrt{\\lambda}$ [rad/m]')
    ax1.set_title('(a) First 10 Cavity TE Wavenumbers')
    ax1.set_xticks(indices)
    ax1.legend(loc='lower right', framealpha=0.92)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Panel (b): s_div sensitivity study (purification)
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

    ax2.set_title('(b) Spectral Purification vs. $s_{\\mathrm{div}}$')
    ax2.grid(True, linestyle=':', alpha=0.5)

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_cavity_modes_and_purification.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_fig5_benchmark():
    print("Generating Figure 5: Computational Benchmark (VNMM-MLPG vs VNMM-Gauss)...")
    grids = [11, 15, 21, 27]
    times_mlpg = []
    times_gauss = []
    nnz_mlpg = []
    nnz_gauss = []

    for N in grids:
        b = run_mlpg_vs_gauss_benchmark(Nx=N, Ny=N, s_div=6.0, verbose=False)
        times_mlpg.append(b['time_mlpg_sec'])
        times_gauss.append(b['time_gauss_sec'])
        nnz_mlpg.append(b['nnz_K_mlpg'])
        nnz_gauss.append(b['nnz_K_gauss'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))

    x = np.arange(len(grids))
    width = 0.35

    # Assembly Time
    ax1.bar(x - width/2, times_gauss, width, label='VNMM-Gauss ($3 \\times 3$ cells)', color='silver', edgecolor='black')
    ax1.bar(x + width/2, times_mlpg, width, label='VNMM-MLPG (Cell-Free)', color='forestgreen', edgecolor='black')
    ax1.set_xlabel('Grid Dimension ($N \\times N$)')
    ax1.set_ylabel('Assembly Time [s]')
    ax1.set_title('(a) Assembly Execution Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{N}x{N}' for N in grids])
    ax1.legend(loc='upper left', framealpha=0.92)
    ax1.grid(True, linestyle=':', alpha=0.5)

    # Matrix Sparsity (Non-zeros)
    ax2.bar(x - width/2, nnz_gauss, width, label='VNMM-Gauss', color='silver', edgecolor='black')
    ax2.bar(x + width/2, nnz_mlpg, width, label='VNMM-MLPG (Compact Stencils)', color='royalblue', edgecolor='black')
    ax2.set_xlabel('Grid Dimension ($N \\times N$)')
    ax2.set_ylabel('Matrix Non-Zeros (NNZ in $\\mathbf{K}$)')
    ax2.set_title('(b) Global Matrix Sparsity')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{N}x{N}' for N in grids])
    ax2.legend(loc='upper left', framealpha=0.92)
    ax2.grid(True, linestyle=':', alpha=0.5)

    out_path = os.path.join(CURRENT_DIR, "fig_mlpg_benchmark_comparison.png")
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
    plot_fig4_cavity_modes()
    plot_fig5_benchmark()
    print("=" * 65)
    print("All 5 figures generated successfully.")
    print("=" * 65)


if __name__ == "__main__":
    main()
