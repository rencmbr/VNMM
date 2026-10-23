"""
Convergence analysis of 2D cavity eigenvalues on random/jittered nodal distributions with random directors.

Evaluates the robustness of VNMM 2D (P1 basis) when subjected to coordinate jitter (25% dx)
and random azimuthal angles (theta in [0, 2pi)).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CODE_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.eigen_solver_cavity_2d import solve_cavity_eigenvalues, ANALYTICAL_REFERENCE_MODES
from src.cavity_mesh_2d import generate_cavity_mesh


def compute_spacing_metrics_h(coords):
    """Compute geometric nodal spacing metrics from Delaunay triangulation."""
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            e = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
            edges.add(e)
            
    edge_lens = np.array([np.linalg.norm(coords[e[0]] - coords[e[1]]) for e in edges])
    
    return {
        'h_max_geom': float(np.max(edge_lens)),
        'h_mean_geom': float(np.mean(edge_lens)),
        'h_min_geom': float(np.min(edge_lens)),
        'h_med_geom': float(np.mean(edge_lens))
    }


def run_random_cavity_convergence_study(
    nx_list=None, 
    jitter_frac=0.25, 
    interior_type="random", 
    s_div=4.0, 
    basis="P1", 
    seed=42,
    # Backward compatibility
    lista_Nx=None,
    tipo_interior=None,
    base=None
):
    """
    Run convergence analysis on random/jittered nodal distributions with random directors.
    """
    if lista_Nx is not None: nx_list = lista_Nx
    if tipo_interior is not None: interior_type = tipo_interior
    if base is not None: basis = base
    if nx_list is None:
        nx_list = [9, 13, 17, 21, 25, 29, 33, 37]
        
    print("=========================================================================")
    print("  VNMM 2D CONVERGENCE (P1 BASIS) WITH RANDOM MESH AND DIRECTORS")
    print(f"  PEC 2D Cavity Benchmark")
    print(f"  Coordinate Jitter: {jitter_frac*100:.1f}% dx | Interior Vectors: {interior_type}")
    print(f"  Div-curl regularization: s = {s_div:.1f} | Seed: {seed}")
    print("=========================================================================\n")
    
    results_random = []
    results_regular = []
    
    for nx in nx_list:
        # 1. Random mesh case
        res_rand = solve_cavity_eigenvalues(
            nx=nx, 
            ny=nx, 
            lx=np.pi, 
            ly=np.pi, 
            basis=basis, 
            interior_type=interior_type, 
            jitter_frac=jitter_frac,
            num_eigenvalues=10, 
            s_div=s_div,
            seed=seed
        )
        
        metrics_h = compute_spacing_metrics_h(res_rand['coords'])
        res_rand['h_nom'] = np.pi / (nx - 1)
        res_rand.update(metrics_h)
        results_random.append(res_rand)
        
        # 2. Regular mesh reference case
        res_reg = solve_cavity_eigenvalues(
            nx=nx, 
            ny=nx, 
            lx=np.pi, 
            ly=np.pi, 
            basis=basis, 
            interior_type="alternating", 
            jitter_frac=0.0,
            num_eigenvalues=10, 
            s_div=s_div,
            seed=seed
        )
        res_reg['h_nom'] = np.pi / (nx - 1)
        res_reg['h_max_geom'] = res_reg['h_nom']
        results_regular.append(res_reg)
        
        print(f"Nx={nx:2d} ({res_rand['total_nodes']:4d} nodes) | "
              f"h_nom={res_rand['h_nom']:.4f} m, h_max_geom={res_rand['h_max_geom']:.4f} m | "
              f"Mean λ Err: {res_rand['mean_lambda_error_pct']:5.2f}% | "
              f"Mean kc Err: {res_rand['mean_kc_error_pct']:5.2f}% | "
              f"Max kc Err: {res_rand['max_kc_error_pct']:5.2f}%")
              
    return results_random, results_regular


def generate_random_convergence_plots(
    res_rand, 
    res_reg, 
    output_dir=REPORTS_DIR
):
    """Generate comparative plots for randomized mesh study."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot random mesh with quiver
    example_nx = 13
    coords_ex, vectors_ex, is_b_ex = generate_cavity_mesh(
        nx=example_nx, 
        ny=example_nx, 
        lx=np.pi, 
        ly=np.pi, 
        interior_type="random", 
        jitter_frac=0.25, 
        seed=42
    )
    
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    idx_int = np.where(~is_b_ex)[0]
    idx_bnd = np.where(is_b_ex)[0]
    
    ax.scatter(coords_ex[idx_int, 0], coords_ex[idx_int, 1], c='#1f77b4', s=35, label=f"Internal Nodes with Jitter ({len(idx_int)})", zorder=3)
    ax.scatter(coords_ex[idx_bnd, 0], coords_ex[idx_bnd, 1], c='#d62728', s=45, marker='s', label=f"Boundary PEC Nodes ({len(idx_bnd)})", zorder=3)
    
    quiver_scale = 0.18
    ax.quiver(
        coords_ex[idx_int, 0], coords_ex[idx_int, 1],
        vectors_ex[idx_int, 0] * quiver_scale, vectors_ex[idx_int, 1] * quiver_scale,
        color='#2ca02c', angles='xy', scale_units='xy', scale=1, width=0.004,
        label=r"Random Directors $\vec{t}_k$ (Interior)"
    )
    ax.quiver(
        coords_ex[idx_bnd, 0], coords_ex[idx_bnd, 1],
        vectors_ex[idx_bnd, 0] * quiver_scale, vectors_ex[idx_bnd, 1] * quiver_scale,
        color='#9467bd', angles='xy', scale_units='xy', scale=1, width=0.005,
        label=r"PEC Tangent Directors $\vec{t}_k$ (Boundary)"
    )
    
    ax.set_xlim(-0.15, np.pi + 0.15)
    ax.set_ylim(-0.15, np.pi + 0.15)
    ax.set_aspect('equal')
    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)
    ax.set_title(r"Irregular Nodal Distribution with Jitter and Random Directors ($N_x=13$)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    
    fig_mesh_path = os.path.join(output_dir, "malha_cavidade_aleatoria_exemplo.png")
    fig.savefig(fig_mesh_path, dpi=300)
    plt.close(fig)
    print(f"Random mesh figure saved to: {fig_mesh_path}")
    
    # 2. Error Convergence vs h_max
    h_max_geom = [r['h_max_geom'] for r in res_rand]
    mean_lambda_err = [r['mean_lambda_error_pct'] for r in res_rand]
    mean_kc_err = [r['mean_kc_error_pct'] for r in res_rand]
    max_kc_err = [r['max_kc_error_pct'] for r in res_rand]
    
    p_fit_kc = np.polyfit(np.log(h_max_geom[2:]), np.log(mean_kc_err[2:]), 1)[0]
    p_fit_lam = np.polyfit(np.log(h_max_geom[2:]), np.log(mean_lambda_err[2:]), 1)[0]
    
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.loglog(h_max_geom, mean_lambda_err, 'r-o', linewidth=2.2, markersize=7, label=f"Mean $\\lambda$ Error (%) [Slope ~ {p_fit_lam:.2f}]")
    ax.loglog(h_max_geom, mean_kc_err, 'b-s', linewidth=2.2, markersize=7, label=f"Mean $k_c$ Error (%) [Slope ~ {p_fit_kc:.2f}]")
    ax.loglog(h_max_geom, max_kc_err, 'g--^', linewidth=1.8, markersize=7, label="Max $k_c$ Error (%)")
    
    h_ref_line = np.array(h_max_geom)
    ax.loglog(h_ref_line, mean_kc_err[0] * (h_ref_line / h_ref_line[0])**2, 'k:', alpha=0.6, label=r"Theoretical Reference $\mathcal{O}(h^2)$")
    ax.loglog(h_ref_line, mean_kc_err[0] * (h_ref_line / h_ref_line[0]), 'gray', linestyle='-.', alpha=0.6, label=r"Theoretical Reference $\mathcal{O}(h)$")
    
    ax.set_xlabel(r"Maximum Element Spacing $h_{max}^{geom}$ [m] (Log scale)", fontsize=11)
    ax.set_ylabel("Relative Percentage Error [%] (Log scale)", fontsize=11)
    ax.set_title(r"Spectral Convergence VNMM 2D ($\mathcal{P}^1$ Basis) on Random Meshes", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    
    fig_conv_path = os.path.join(output_dir, "convergencia_cavidade_malha_aleatoria.png")
    fig.savefig(fig_conv_path, dpi=300)
    plt.close(fig)
    print(f"Convergence plot saved to: {fig_conv_path}")
    
    # 3. Modal Comparison
    fine_res = res_rand[-2] # Nx=33
    mode_indices = np.arange(1, 11)
    kc_analytical = fine_res['analytical_kc']
    kc_vnmm = fine_res['numerical_kc']
    
    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    width = 0.35
    ax.bar(mode_indices - width/2, kc_analytical, width, label=r"Analytical $k_c$", color='#1f77b4', alpha=0.88)
    ax.bar(mode_indices + width/2, kc_vnmm, width, label=f"VNMM 2D $\\mathcal{{P}}^1$ Random Mesh ($N={fine_res['total_nodes']}$)", color='#ff7f0e', alpha=0.88)
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    ax.set_xticks(mode_indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Cutoff Wavenumber $k_c$ [rad/m]", fontsize=11)
    ax.set_title(r"Modal Spectrum: Analytical vs VNMM 2D with Random Coordinates and Directors", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    
    fig_spec_path = os.path.join(output_dir, "espectro_modos_cavidade_malha_aleatoria.png")
    fig.savefig(fig_spec_path, dpi=300)
    plt.close(fig)
    print(f"Spectrum plot saved to: {fig_spec_path}")
    
    # 4. Comparative Convergence: Random vs Regular
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    kc_err_rand = [r['mean_kc_error_pct'] for r in res_rand]
    kc_err_reg = [r['mean_kc_error_pct'] for r in res_reg]
    h_nom_list = [r['h_nom'] for r in res_rand]
    
    ax.loglog(h_nom_list, kc_err_rand, 's-', color='#e377c2', linewidth=2.2, markersize=7, label=r"VNMM $\mathcal{P}^1$ (Random Mesh & Directors)")
    ax.loglog(h_nom_list, kc_err_reg, 'o--', color='#17becf', linewidth=2.2, markersize=7, label=r"VNMM $\mathcal{P}^1$ (Regular Mesh & Alternating Directors)")
    
    ax.set_xlabel(r"Nominal Nodal Spacing $h_{nom} = \pi / (N_x - 1)$ [m] (Log scale)", fontsize=11)
    ax.set_ylabel("Mean Relative $k_c$ Error [%] (Log scale)", fontsize=11)
    ax.set_title("VNMM 2D Robustness: Random Mesh vs Regular Mesh", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    
    fig_comp_path = os.path.join(output_dir, "comparativo_convergencia_aleatoria_vs_regular.png")
    fig.savefig(fig_comp_path, dpi=300)
    plt.close(fig)
    print(f"Comparative plot saved to: {fig_comp_path}")
    
    return p_fit_kc, p_fit_lam


def generate_random_markdown_report(
    res_rand, 
    res_reg, 
    p_fit_kc, 
    p_fit_lam, 
    output_dir=REPORTS_DIR
):
    """Generate technical report in Markdown format for randomized mesh study."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_convergencia_cavidade_malha_aleatoria.md")
    
    fine_res = res_rand[-2] # Nx=33
    mode_names = [
        "TE_{10}", "TE_{01}", "TE_{11}", "TE_{20}", "TE_{02}", 
        "TE_{21}", "TE_{12}", "TE_{22}", "TE_{30}", "TE_{03}"
    ]
    
    lines = []
    lines.append("# Spectral Convergence Report: 2D PEC Cavity with Random Mesh and Directors\n\n")
    lines.append("**Vector Nodal Meshless Method (VNMM 2D) - Complete Linear Basis $\\mathcal{P}^1$**\n\n")
    lines.append("This report documents the numerical convergence of the 2D electromagnetic cavity solver "
                 "under spatial coordinate perturbation (jitter = 25% dx) and random internal vector orientations.\n\n")
    
    lines.append("![Example Random Mesh](malha_cavidade_aleatoria_exemplo.png)\n\n")
    lines.append("![Convergence Curve](convergencia_cavidade_malha_aleatoria.png)\n\n")
    lines.append("![Modal Spectrum](espectro_modos_cavidade_malha_aleatoria.png)\n\n")
    lines.append("![Random vs Regular Comparison](comparativo_convergencia_aleatoria_vs_regular.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"Report saved to: {report_path}")


def main():
    res_rand, res_reg = run_random_cavity_convergence_study()
    p_kc, p_lam = generate_random_convergence_plots(res_rand, res_reg)
    generate_random_markdown_report(res_rand, res_reg, p_kc, p_lam)
    print("\n=========================================================================")
    print("  RANDOM MESH CONVERGENCE STUDY COMPLETED SUCCESSFULLY!")
    print("=========================================================================")


# Backward compatibility aliases
calcular_metricas_espacamento_h = compute_spacing_metrics_h
executar_estudo_convergencia_aleatoria = run_random_cavity_convergence_study
gerar_graficos_convergencia_aleatoria = generate_random_convergence_plots
gerar_relatorio_markdown = generate_random_markdown_report


if __name__ == "__main__":
    main()
