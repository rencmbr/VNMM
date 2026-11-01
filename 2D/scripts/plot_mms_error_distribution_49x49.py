#!/usr/bin/env python3
"""
plot_mms_error_distribution_49x49.py
====================================
Analyzes the spatial distribution of errors for the densest MMS mesh (49x49)
in the 2D Vector Nodal Meshless Method via MLPG (VNMM-MLPG 2D).

Computes and maps:
1. Vector field error norm ||E_h - E*|| across the domain [0, pi]^2.
2. Curl error |curl(E_h) - curl(E*)| across the domain.
3. Discrete DoF error |e_i - e_i*| at each node.
4. Divergence residue |div(E_h)| across the domain.
5. 1D profiles along centerlines and boundary walls.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
SRC_DIR = os.path.join(DIR_2D, "src")
for p in [DIR_2D, ROOT_DIR, CURRENT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d
from scripts.mlpg_source_problem_2d import (
    assemble_and_solve_mlpg_mms,
    exact_solution_mms,
    exact_curl_mms
)

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

N = 49
Lx, Ly = np.pi, np.pi
print("=" * 75)
print(f"COMPUTING MMS ERROR SPATIAL DISTRIBUTION ON DENSEST MESH ({N}x{N})")
print("=" * 75)

coords, vectors, is_bnd = generate_cavity_mesh(Nx=N, Ny=N, Lx=Lx, Ly=Ly)
subs = build_mlpg_subdomains_2d(coords, vectors, is_bnd, alpha_s=0.80)

# 1. Solve Case A (Pure Dirichlet)
print("Solving Case A (Pure Dirichlet)...")
resA = assemble_and_solve_mlpg_mms(coords, vectors, is_bnd, subdomains=subs, boundary_condition='dirichlet')

# 2. Solve Case B (Mixed Neumann / PMC)
print("Solving Case B (Mixed Dirichlet-Neumann / PMC)...")
resB = assemble_and_solve_mlpg_mms(coords, vectors, is_bnd, subdomains=subs, boundary_condition='mixed_neumann')

def compute_node_errors(res):
    e_sol = res['e_sol']
    e_exact = res['e_exact']
    num_nodes = len(coords)

    err_field = np.zeros(num_nodes)
    err_curl = np.zeros(num_nodes)
    err_div = np.zeros(num_nodes)
    diff_Ex = np.zeros(num_nodes)
    diff_Ey = np.zeros(num_nodes)
    err_dof = np.abs(e_sol - e_exact)

    for i, sub in enumerate(subs):
        e_st = e_sol[sub.stencil_nodes]
        N_val = sub.eval_shape_functions(np.array([sub.center])) # (1, 2, 6)
        Eh = N_val[0] @ e_st # (2,)
        E_ex = exact_solution_mms(coords[i, 0], coords[i, 1])

        dE = Eh - E_ex
        diff_Ex[i] = dE[0]
        diff_Ey[i] = dE[1]
        err_field[i] = np.linalg.norm(dE)

        ch = float(np.dot(sub.eval_shape_curl(), e_st))
        c_ex = exact_curl_mms(coords[i, 0], coords[i, 1])
        err_curl[i] = abs(ch - c_ex)

        dh = float(np.dot(sub.eval_shape_div(), e_st))
        err_div[i] = abs(dh)

    return {
        'err_field': err_field,
        'err_curl': err_curl,
        'err_div': err_div,
        'err_dof': err_dof,
        'diff_Ex': diff_Ex,
        'diff_Ey': diff_Ey
    }

nodeA = compute_node_errors(resA)
nodeB = compute_node_errors(resB)

# High-resolution 2D evaluation grid (200 x 200)
grid_x, grid_y = np.meshgrid(np.linspace(0, np.pi, 200), np.linspace(0, np.pi, 200))

def interpolate_to_grid(values):
    grid_val = griddata(coords, values, (grid_x, grid_y), method='cubic')
    # Fill any boundary NaNs with linear
    nan_mask = np.isnan(grid_val)
    if np.any(nan_mask):
        grid_lin = griddata(coords, values, (grid_x, grid_y), method='nearest')
        grid_val[nan_mask] = grid_lin[nan_mask]
    return grid_val

gridA_field = interpolate_to_grid(nodeA['err_field'])
gridA_curl = interpolate_to_grid(nodeA['err_curl'])
gridA_dof = interpolate_to_grid(nodeA['err_dof'])

gridB_field = interpolate_to_grid(nodeB['err_field'])
gridB_curl = interpolate_to_grid(nodeB['err_curl'])
gridB_dof = interpolate_to_grid(nodeB['err_dof'])

# Summary Statistics
print("\n" + "=" * 75)
print("ERROR CONCENTRATION SUMMARY (49x49 Mesh):")
print("=" * 75)
for name, f in [("Case A (Pure Dirichlet)", nodeA), ("Case B (Mixed Neumann)", nodeB)]:
    print(f"\n--- {name} ---")
    print(f"Field Error ||Eh - E*||:  Mean = {np.mean(f['err_field']):.4e}, Max = {np.max(f['err_field']):.4e}")
    max_f_idx = np.argmax(f['err_field'])
    print(f"  Max located at (x, y) = ({coords[max_f_idx, 0]:.3f}, {coords[max_f_idx, 1]:.3f}) [Interior vortex region]")
    
    print(f"Curl Error |curl(Eh) - curl(E*)|: Mean = {np.mean(f['err_curl']):.4e}, Max = {np.max(f['err_curl']):.4e}")
    max_c_idx = np.argmax(f['err_curl'])
    print(f"  Max located at (x, y) = ({coords[max_c_idx, 0]:.3f}, {coords[max_c_idx, 1]:.3f}) [Truncated boundary]")

    print(f"DoF Error |e_i - e_i*|:    Mean = {np.mean(f['err_dof']):.4e}, Max = {np.max(f['err_dof']):.4e}")
    print(f"Divergence Residue:        Mean = {np.mean(f['err_div']):.4e}, Max = {np.max(f['err_div']):.4e}")

# ----------------- 2D Heatmap Figure -----------------
fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.0))

# Shared color limits for direct comparison
vmax_f = max(np.max(gridA_field), np.max(gridB_field))
vmax_c = max(np.max(gridA_curl), np.max(gridB_curl))
vmax_d = max(np.max(gridA_dof), np.max(gridB_dof))

# Row 1: Case A (Pure Dirichlet)
# (a1) Field Error
cf1 = axes[0, 0].contourf(grid_x, grid_y, gridA_field, levels=40, cmap='viridis', vmin=0, vmax=vmax_f)
cb1 = fig.colorbar(cf1, ax=axes[0, 0])
cb1.set_label(r'$\|\mathbf{E}^h - \mathbf{E}^*\|$')
axes[0, 0].set_title(r'(a1) Case A: Field Error $\|\mathbf{E}^h - \mathbf{E}^*\|$')
axes[0, 0].set_ylabel('$y$ [m]')
axes[0, 0].set_aspect('equal')

# (a2) Curl Error
cf2 = axes[0, 1].contourf(grid_x, grid_y, gridA_curl, levels=40, cmap='magma', vmin=0, vmax=vmax_c)
cb2 = fig.colorbar(cf2, ax=axes[0, 1])
cb2.set_label(r'$|(\nabla\times\mathbf{E}^h)_z - (\nabla\times\mathbf{E}^*)_z|$')
axes[0, 1].set_title(r'(a2) Case A: Curl Error $|(\nabla\times\mathbf{E}^h)_z - (\nabla\times\mathbf{E}^*)_z|$')
axes[0, 1].set_aspect('equal')

# (a3) DoF Error and Quiver
cf3 = axes[0, 2].contourf(grid_x, grid_y, gridA_dof, levels=40, cmap='plasma', vmin=0, vmax=vmax_d)
cb3 = fig.colorbar(cf3, ax=axes[0, 2])
cb3.set_label(r'DoF Error $|e_i - e_i^*|$')
int_mask = ~is_bnd
step_q = 60
axes[0, 2].quiver(coords[int_mask, 0][::step_q], coords[int_mask, 1][::step_q],
                 nodeA['diff_Ex'][int_mask][::step_q], nodeA['diff_Ey'][int_mask][::step_q],
                 color='cyan', scale=0.15, width=0.0035, alpha=0.9, label=r'$\mathbf{E}^h - \mathbf{E}^*$')
axes[0, 2].set_title(r'(a3) Case A: DoF Error and Error Quiver')
axes[0, 2].legend(loc='upper right', framealpha=0.85, fontsize=8)
axes[0, 2].set_aspect('equal')

# Row 2: Case B (Mixed Neumann / PMC)
# (b1) Field Error
cf4 = axes[1, 0].contourf(grid_x, grid_y, gridB_field, levels=40, cmap='viridis', vmin=0, vmax=vmax_f)
cb4 = fig.colorbar(cf4, ax=axes[1, 0])
cb4.set_label(r'$\|\mathbf{E}^h - \mathbf{E}^*\|$')
axes[1, 0].set_title(r'(b1) Case B (PMC): Field Error $\|\mathbf{E}^h - \mathbf{E}^*\|$')
axes[1, 0].set_xlabel('$x$ [m]')
axes[1, 0].set_ylabel('$y$ [m]')
axes[1, 0].set_aspect('equal')

# (b2) Curl Error
cf5 = axes[1, 1].contourf(grid_x, grid_y, gridB_curl, levels=40, cmap='magma', vmin=0, vmax=vmax_c)
cb5 = fig.colorbar(cf5, ax=axes[1, 1])
cb5.set_label(r'$|(\nabla\times\mathbf{E}^h)_z - (\nabla\times\mathbf{E}^*)_z|$')
axes[1, 1].set_title(r'(b2) Case B (PMC): Curl Error $|(\nabla\times\mathbf{E}^h)_z - (\nabla\times\mathbf{E}^*)_z|$')
axes[1, 1].set_xlabel('$x$ [m]')
axes[1, 1].set_aspect('equal')

# (b3) DoF Error and Quiver
cf6 = axes[1, 2].contourf(grid_x, grid_y, gridB_dof, levels=40, cmap='plasma', vmin=0, vmax=vmax_d)
cb6 = fig.colorbar(cf6, ax=axes[1, 2])
cb6.set_label(r'DoF Error $|e_i - e_i^*|$')
axes[1, 2].quiver(coords[int_mask, 0][::step_q], coords[int_mask, 1][::step_q],
                 nodeB['diff_Ex'][int_mask][::step_q], nodeB['diff_Ey'][int_mask][::step_q],
                 color='cyan', scale=0.15, width=0.0035, alpha=0.9, label=r'$\mathbf{E}^h - \mathbf{E}^*$')
axes[1, 2].set_title(r'(b3) Case B (PMC): DoF Error and Error Quiver')
axes[1, 2].legend(loc='upper right', framealpha=0.85, fontsize=8)
axes[1, 2].set_xlabel('$x$ [m]')
axes[1, 2].set_aspect('equal')

# Mark PMC walls on Case B
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1.5, label='PMC ($y=0, \\pi$)')
axes[1, 0].axhline(np.pi, color='red', linestyle='--', linewidth=1.5)
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1, 1].axhline(np.pi, color='red', linestyle='--', linewidth=1.5)
axes[1, 2].axhline(0, color='red', linestyle='--', linewidth=1.5)
axes[1, 2].axhline(np.pi, color='red', linestyle='--', linewidth=1.5)

fig.tight_layout()
out_paper = os.path.join(ROOT_DIR, "paper", "mlpg-vnmm", "fig_mms_error_distribution_49x49.png")
fig.savefig(out_paper)
print(f"\nSaved publication plot: {out_paper}")

# ----------------- 1D Centerline & Boundary Profiles -----------------
fig_prof, (ax_p1, ax_p2) = plt.subplots(1, 2, figsize=(12.0, 4.4))

# 1. Profile along horizontal centerline y = pi/2
y_mid = np.pi / 2.0
dy = np.pi / (N - 1)
mask_center = np.abs(coords[:, 1] - y_mid) < 0.5 * dy
x_cent = coords[mask_center, 0]
sort_cent = np.argsort(x_cent)
x_cent = x_cent[sort_cent]

err_fA_cent = nodeA['err_field'][mask_center][sort_cent]
err_fB_cent = nodeB['err_field'][mask_center][sort_cent]
err_dA_cent = nodeA['err_dof'][mask_center][sort_cent]
err_dB_cent = nodeB['err_dof'][mask_center][sort_cent]

ax_p1.plot(x_cent, err_fA_cent, 'b-o', markersize=3.5, linewidth=1.5, label=r'Case A: Field Error $\|\mathbf{E}^h - \mathbf{E}^*\|$')
ax_p1.plot(x_cent, err_fB_cent, 'c--s', markersize=3.5, linewidth=1.5, label=r'Case B: Field Error $\|\mathbf{E}^h - \mathbf{E}^*\|$')
ax_p1.plot(x_cent, err_dA_cent, 'm-.^', markersize=3.0, linewidth=1.2, label=r'Case A: DoF Error $|e_i - e_i^*|$')
ax_p1.plot(x_cent, err_dB_cent, 'k:v', markersize=3.0, linewidth=1.2, label=r'Case B: DoF Error $|e_i - e_i^*|$')
ax_p1.set_xlabel('Coordinate $x$ [m] along Centerline $y = \\pi / 2$')
ax_p1.set_ylabel('Absolute Error')
ax_p1.set_title(r'(a) Error Distribution along Horizontal Centerline ($y = \pi/2$)')
ax_p1.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
ax_p1.grid(True, linestyle=':', alpha=0.5)

# 2. Curl error: Boundary y = 0 vs Centerline y = pi/2
mask_bnd0 = np.abs(coords[:, 1] - 0.0) < 0.5 * dy
x_bnd0 = coords[mask_bnd0, 0]
sort_b0 = np.argsort(x_bnd0)
x_b0 = x_bnd0[sort_b0]

curl_A_b0 = nodeA['err_curl'][mask_bnd0][sort_b0]
curl_B_b0 = nodeB['err_curl'][mask_bnd0][sort_b0]
curl_A_cent = nodeA['err_curl'][mask_center][sort_cent]
curl_B_cent = nodeB['err_curl'][mask_center][sort_cent]

ax_p2.plot(x_b0, curl_A_b0, 'r-o', markersize=3.5, linewidth=1.5, label='Case A: Boundary Wall $y = 0$ (Dirichlet)')
ax_p2.plot(x_b0, curl_B_b0, 'darkorange', linestyle='--', marker='s', markersize=3.5, linewidth=1.5, label='Case B: Boundary Wall $y = 0$ (PMC Wall)')
ax_p2.plot(x_cent, curl_A_cent, 'b-.', linewidth=1.5, label='Case A: Interior Centerline $y = \\pi/2$')
ax_p2.plot(x_cent, curl_B_cent, 'g:', linewidth=1.5, label='Case B: Interior Centerline $y = \\pi/2$')
ax_p2.set_xlabel('Coordinate $x$ [m]')
ax_p2.set_ylabel('Absolute Curl Error')
ax_p2.set_title(r'(b) Curl Error: Boundary Wall ($y=0$) vs. Interior ($y=\pi/2$)')
ax_p2.legend(loc='upper right', framealpha=0.92, fontsize=8.5)
ax_p2.grid(True, linestyle=':', alpha=0.5)

fig_prof.tight_layout()
out_prof = os.path.join(ROOT_DIR, "paper", "mlpg-vnmm", "fig_mms_error_profiles_49x49.png")
fig_prof.savefig(out_prof)
print(f"Saved profile plot: {out_prof}")
print("Done!")
