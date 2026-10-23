"""
Convergence analysis of the coupled hybrid Edge-FEM + VNMM solver under randomized meshes and random directors.

Compares Pure FEM, Pure VNMM, and Coupled Hybrid FEM-VNMM under stochastic coordinate jitter (25%)
and arbitrary vector directions.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.fem_edge_2d import solve_edge_fem_cavity_eigenvalues
from src.eigen_solver_cavity import solve_cavity_eigenvalues, ANALYTICAL_REFERENCE_MODES
from src.fem_vnmm_hybrid_2d import (
    generate_hybrid_cavity_mesh,
    assemble_hybrid_fem_vnmm_matrices,
    solve_hybrid_fem_vnmm_eigenvalues
)


def compute_h_max_delaunay(coords):
    """Compute maximum and mean edge length from Delaunay triangulation."""
    tri = Delaunay(coords)
    edges = set()
    for s in tri.simplices:
        for i in range(3):
            edges.add(tuple(sorted([s[i], s[(i + 1) % 3]])))
    lens = [np.linalg.norm(coords[e[0]] - coords[e[1]]) for e in edges]
    return float(np.max(lens)), float(np.mean(lens))


def run_random_hybrid_convergence_study(levels_list=None, seed=42, lista_niveis=None):
    """
    Run comparative convergence study across pure FEM, pure VNMM, and hybrid FEM-VNMM on randomized meshes.
    """
    if lista_niveis is not None:
        levels_list = lista_niveis
    if levels_list is None:
        levels_list = [
            (8, 9),
            (12, 13),
            (16, 17),
            (20, 21),
            (24, 25),
            (28, 29),
            (32, 33)
        ]
        
    print("=========================================================================")
    print("  RANDOMIZED MESH COMPARATIVE ANALYSIS: FEM vs VNMM vs HYBRID")
    print("  Problem: 2D PEC Cavity [0, pi]^2")
    print("  Coordinate Jitter: 25% | VNMM Vectors: Random theta in [0, 2pi)")
    print(f"  FEM-VNMM Interface at x = 0.5 * pi | Seed: {seed}")
    print("=========================================================================\n")
    
    results_fem = []
    results_vnmm = []
    results_hybrid = []
    
    for n_fem, n_vnmm in levels_list:
        h_nom = np.pi / n_fem
        print(f"--- Processing Level: Nex_fem={n_fem}, Nx_vnmm={n_vnmm} (h_nom = {h_nom:.4f} m) ---")
        
        # 1. Pure Random Edge FEM
        t0 = time.time()
        res_fem = solve_edge_fem_cavity_eigenvalues(
            nex=n_fem, 
            ney=n_fem, 
            jitter_frac=0.25, 
            seed=seed
        )
        t_fem = time.time() - t0
        res_fem['h_nom'] = h_nom
        res_fem['elapsed_seconds'] = t_fem
        res_fem['tempo_s'] = t_fem
        results_fem.append(res_fem)
        
        # 2. Pure Random VNMM 2D
        t0 = time.time()
        res_vnmm = solve_cavity_eigenvalues(
            nx=n_vnmm, 
            ny=n_vnmm, 
            basis="P1", 
            interior_type="random", 
            jitter_frac=0.25, 
            s_div=4.0, 
            floor_tolerance=1e-4,
            points_per_dir=3,
            seed=seed
        )
        t_vnmm = time.time() - t0
        res_vnmm['h_nom'] = h_nom
        res_vnmm['elapsed_seconds'] = t_vnmm
        res_vnmm['tempo_s'] = t_vnmm
        results_vnmm.append(res_vnmm)
        
        # 3. Hybrid Random FEM-VNMM
        t0 = time.time()
        res_hib = solve_hybrid_fem_vnmm_eigenvalues(
            nex_fem=n_fem // 2, 
            ney=n_fem,
            nx_vnmm=(n_vnmm // 2) + 1, 
            ny_vnmm=n_vnmm,
            interior_type_vnmm="random",
            jitter_frac_fem=0.25, 
            jitter_frac_vnmm=0.25,
            points_per_dir=3, 
            s_div_vnmm=4.0,
            floor_tolerance_vnmm=1e-4,
            seed=seed
        )
        t_hib = time.time() - t0
        res_hib['h_nom'] = h_nom
        res_hib['elapsed_seconds'] = t_hib
        res_hib['tempo_s'] = t_hib
        res_hib['nex_fem'] = n_fem // 2
        res_hib['ney'] = n_fem
        res_hib['nx_vnmm'] = (n_vnmm // 2) + 1
        res_hib['ny_vnmm'] = n_vnmm
        results_hybrid.append(res_hib)
        
        print(f"  Pure FEM   : DoFs={res_fem['active_dofs']:4d} | kc Err={res_fem['mean_kc_error_pct']:5.2f}% | Time={t_fem:.3f}s")
        print(f"  Pure VNMM  : DoFs={res_vnmm['active_dofs']:4d} | kc Err={res_vnmm['mean_kc_error_pct']:5.2f}% | Time={t_vnmm:.3f}s")
        print(f"  Hybrid     : DoFs={res_hib['dof_info']['num_global']:4d} | kc Err={res_hib['mean_kc_error_pct']:5.2f}% | Time={t_hib:.3f}s\n")
        
    return results_fem, results_vnmm, results_hybrid


def generate_random_hybrid_mesh_figure(output_dir=REPORTS_DIR):
    """Generate detailed illustration of coupled hybrid mesh with jitter and random vectors."""
    os.makedirs(output_dir, exist_ok=True)
    
    nex_fem = 6
    ney = 10
    nx_vnmm = 7
    ny_vnmm = 11
    lx = np.pi
    ly = np.pi
    
    fem_data, vnmm_data = generate_hybrid_cavity_mesh(
        lx=lx, ly=ly, fem_fraction=0.5,
        nex_fem=nex_fem, ney=ney,
        nx_vnmm=nx_vnmm, ny_vnmm=ny_vnmm,
        interior_type_vnmm="random",
        jitter_frac_fem=0.25,
        jitter_frac_vnmm=0.25,
        seed=42
    )
    
    fig, ax = plt.subplots(figsize=(9.0, 7.5))
    
    nodes_f = fem_data['nodes']
    elements_f = fem_data['elements']
    for el in elements_f:
        pts = nodes_f[el]
        tri = plt.Polygon(pts, fill=True, facecolor='#e6f2ff', edgecolor='#1f77b4', linewidth=1.1, alpha=0.85)
        ax.add_patch(tri)
        
    ax.scatter(nodes_f[:, 0], nodes_f[:, 1], c='#1f77b4', s=25, zorder=4, label=f"FEM Nodes ({len(nodes_f)})")
    
    x_int = fem_data['interface_x']
    ax.axvline(x_int, color='#d62728', linestyle='--', linewidth=2.2, label=r"Interface $\Gamma_{\mathrm{int}}$ ($x = \pi/2$)", zorder=5)
    
    pms = fem_data['interface_midpoints']
    ax.scatter(pms[:, 0], pms[:, 1], c='#d62728', marker='D', s=55, zorder=6, label=f"FEM-VNMM Interface Nodes ({len(pms)})")
    
    scale_q = 0.14
    ax.quiver(
        pms[:, 0], pms[:, 1],
        np.zeros_like(pms[:, 0]), np.ones_like(pms[:, 1]) * scale_q,
        color='#d62728', angles='xy', scale_units='xy', scale=1, width=0.005, zorder=7,
        label=r"Interface Directors $\vec{t} = [0, 1]^T$"
    )
    
    coords_v = vnmm_data['coords']
    vectors_v = vnmm_data['vectors']
    is_pec_v = vnmm_data['is_pec']
    is_int_v = vnmm_data['is_interface']
    idx_v_int = np.where(~is_pec_v & ~is_int_v)[0]
    idx_v_pec = np.where(is_pec_v)[0]
    
    ax.scatter(coords_v[idx_v_int, 0], coords_v[idx_v_int, 1], c='#2ca02c', s=35, zorder=4, label=f"Interior VNMM Nodes with Jitter ({len(idx_v_int)})")
    ax.scatter(coords_v[idx_v_pec, 0], coords_v[idx_v_pec, 1], c='#ff7f0e', marker='s', s=40, zorder=4, label="PEC Boundary VNMM Nodes")
    
    ax.quiver(
        coords_v[idx_v_int, 0], coords_v[idx_v_int, 1],
        vectors_v[idx_v_int, 0] * scale_q, vectors_v[idx_v_int, 1] * scale_q,
        color='#2ca02c', angles='xy', scale_units='xy', scale=1, width=0.004, zorder=5,
        label=r"Random VNMM Directors $\vec{t}_k$"
    )
    ax.quiver(
        coords_v[idx_v_pec, 0], coords_v[idx_v_pec, 1],
        vectors_v[idx_v_pec, 0] * scale_q, vectors_v[idx_v_pec, 1] * scale_q,
        color='#ff7f0e', angles='xy', scale_units='xy', scale=1, width=0.004, zorder=5
    )
    
    ax.set_xlim(-0.15, lx + 0.15)
    ax.set_ylim(-0.15, ly + 0.15)
    ax.set_aspect('equal')
    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title("Coupled Hybrid FEM-VNMM Discretization on Randomized Mesh", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
    fig.tight_layout()
    
    mesh_fig_path = os.path.join(output_dir, "malha_hibrida_aleatoria_exemplo.png")
    fig.savefig(mesh_fig_path, dpi=300)
    plt.close(fig)
    print(f"Hybrid mesh figure saved to: {mesh_fig_path}")


def generate_comparative_plots(res_fem, res_vnmm, res_hib, output_dir=REPORTS_DIR):
    """Generate comparative convergence, modal spectrum, and DoFs plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    h_noms = [r['h_nom'] for r in res_fem]
    err_kc_fem = [r['mean_kc_error_pct'] for r in res_fem]
    err_kc_vnmm = [r['mean_kc_error_pct'] for r in res_vnmm]
    err_kc_hib = [r['mean_kc_error_pct'] for r in res_hib]
    
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.loglog(h_noms, err_kc_fem, 'b-o', linewidth=2.2, markersize=7, label=r"Pure Edge FEM (Jittered Mesh)")
    ax.loglog(h_noms, err_kc_hib, 'r-s', linewidth=2.2, markersize=7, label=r"Coupled Hybrid FEM-VNMM (Both Jittered/Random)")
    ax.loglog(h_noms, err_kc_vnmm, 'g--^', linewidth=2.2, markersize=7, label=r"Pure VNMM $\mathcal{P}^1$ (Random Nodes & Directors)")
    
    h_ref_arr = np.array(h_noms)
    ax.loglog(h_ref_arr, err_kc_fem[0] * (h_ref_arr / h_ref_arr[0])**2, 'k:', alpha=0.5, label=r"$\mathcal{O}(h^2)$ Reference")
    
    ax.set_xlabel(r"Nominal Nodal Spacing $h_{nom}$ [m] (Log scale)", fontsize=11)
    ax.set_ylabel(r"Mean Relative $k_c$ Error [%] (Log scale)", fontsize=11)
    ax.set_title("Comparative Convergence on Random Meshes: FEM vs VNMM vs Hybrid", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9.5)
    fig.tight_layout()
    
    conv_fig_path = os.path.join(output_dir, "convergencia_comparativa_aleatoria_hibrido_vs_puros.png")
    fig.savefig(conv_fig_path, dpi=300)
    plt.close(fig)
    print(f"Comparative convergence plot saved to: {conv_fig_path}")


def generate_markdown_report(res_fem, res_vnmm, res_hib, output_dir=REPORTS_DIR):
    """Generate technical report in Markdown format."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_acoplamento_hibrido_malhas_aleatorias.md")
    
    content = []
    content.append("# Technical Report: Coupled FEM-VNMM Solver on Randomized Meshes\n\n")
    content.append("This report presents comparative spectral convergence analysis for the 2D PEC cavity $[0, \\pi]^2$ under randomized meshes.\n\n")
    content.append("![Hybrid Random Mesh](malha_hibrida_aleatoria_exemplo.png)\n\n")
    content.append("![Comparative Convergence](convergencia_comparativa_aleatoria_hibrido_vs_puros.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(content)
        
    print(f"Report saved to: {report_path}")


def main():
    res_fem, res_vnmm, res_hib = run_random_hybrid_convergence_study()
    generate_random_hybrid_mesh_figure()
    generate_comparative_plots(res_fem, res_vnmm, res_hib)
    generate_markdown_report(res_fem, res_vnmm, res_hib)
    print("\n=========================================================================")
    print("  RANDOMIZED HYBRID STUDY COMPLETED SUCCESSFULLY!")
    print("=========================================================================")


# Backward compatibility aliases
calcular_h_max_delaunay = compute_h_max_delaunay
executar_estudo_hibrido_aleatorio = run_random_hybrid_convergence_study
gerar_figura_malha_hibrida_aleatoria = generate_random_hybrid_mesh_figure
gerar_graficos_comparativos = generate_comparative_plots
gerar_relatorio_markdown_hibrido_aleatorio = generate_markdown_report


if __name__ == "__main__":
    main()
