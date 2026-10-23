"""
Parametric analysis of tolerance and mesh density for VNMM 2D L1 basis.

Evaluates:
1. Interpolation error sensitivity to |det(A)| threshold.
2. Error convergence rate vs nodal spacing h (log-log scale).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
DEFAULT_REPORT_PATH = os.path.join(REPORTS_DIR, "relatorio_analise_parametrica.md")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from scripts.generate_dense_mesh import generate_dense_mesh
from scripts.build_search_tree import build_search_tree
from scripts.evaluate_point_grid import evaluate_point_grid


def run_tolerance_analysis(
    coords, 
    vectors, 
    search_tree, 
    eval_points, 
    tolerance_list=None, 
    init_neighborhood_size=8,
    # Backward compatibility
    arvore=None,
    pontos_avaliacao=None,
    lista_tolerancias=None,
    tamanho_vizinhanca_ini=None
):
    """
    Run parametric analysis varying the determinant threshold |det(A)|.
    """
    if arvore is not None: search_tree = arvore
    if pontos_avaliacao is not None: eval_points = pontos_avaliacao
    if lista_tolerancias is not None: tolerance_list = lista_tolerancias
    if tamanho_vizinhanca_ini is not None: init_neighborhood_size = tamanho_vizinhanca_ini

    if tolerance_list is None:
        tolerance_list = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5]
        
    print(f"\n=======================================================")
    print(f"STUDY 1: Parametric Analysis of Tolerance |det(A)|")
    print(f"=======================================================")
    print(f"Tolerance range: {tolerance_list}\n")
    
    results = []
    
    for tol in tolerance_list:
        res = evaluate_point_grid(
            coords=coords,
            vectors=vectors,
            search_tree=search_tree,
            eval_points=eval_points,
            tolerance=tol,
            neighborhood_size=init_neighborhood_size,
            adaptive=True
        )
        res['tolerance'] = tol
        res['tolerancia'] = tol
        results.append(res)
        
        print(f"Tol = {tol:6.3f} | Success: {res['success_rate']:5.1f}% | "
              f"|det(A)| mean: {res['mean_det']:6.4f} | "
              f"E Err (mean/RMS): {res['vector_err_mean']:.4e} / {res['vector_err_rms']:.4e} | "
              f"Rot Err (mean/RMS): {res['curl_err_mean']:.4e} / {res['curl_err_rms']:.4e}")
              
    return results


def run_density_analysis(
    mesh_configs=None, 
    eval_points=None, 
    ref_tolerance=1.0, 
    ref_h=2.0,
    init_neighborhood_size=8,
    # Backward compatibility
    lista_configs_malha=None,
    pontos_avaliacao=None,
    tolerancia_ref=None,
    h_ref=None,
    tamanho_vizinhanca_ini=None
):
    """
    Run parametric analysis varying nodal mesh density.
    Determinant threshold scales linearly: Tol_det(h) = Tol_ref * (h / h_ref).
    """
    if lista_configs_malha is not None: mesh_configs = lista_configs_malha
    if pontos_avaliacao is not None: eval_points = pontos_avaliacao
    if tolerancia_ref is not None: ref_tolerance = tolerancia_ref
    if h_ref is not None: ref_h = h_ref
    if tamanho_vizinhanca_ini is not None: init_neighborhood_size = tamanho_vizinhanca_ini

    if mesh_configs is None:
        mesh_configs = [
            (24, 60, "Coarse (N=84)"),
            (36, 150, "Medium-Coarse (N=186)"),
            (56, 360, "Medium (N=416)"),
            (84, 800, "Medium-Dense (N=884)"),
            (128, 1800, "Dense (N=1928)"),
            (192, 4000, "Very Dense (N=4192)"),
            (368, 8040, "Ultra Dense (N=8408)")
        ]
        
    print(f"\n=======================================================")
    print(f"STUDY 2: Parametric Analysis of Mesh Density")
    print(f"=======================================================")
    print(f"Density range: {mesh_configs[0][0]+mesh_configs[0][1]} to {mesh_configs[-1][0]+mesh_configs[-1][1]} nodes")
    print(f"Reference tolerance: {ref_tolerance:.4f} (for h_ref={ref_h:.2f})")
    print(f"Initial neighborhood K: {init_neighborhood_size}\n")
    
    results = []
    
    for n_front, n_int, label in mesh_configs:
        coords, vectors = generate_dense_mesh(
            filename=None,
            num_boundary_nodes=n_front,
            num_interior_nodes=n_int,
            limit=10.0,
            seed=42,
            silent=True
        )
        
        search_tree = build_search_tree(coords)
        n_total = len(coords)
        h_mean = 80.0 / n_front
        tol_h = ref_tolerance * (h_mean / ref_h)
        
        res = evaluate_point_grid(
            coords=coords,
            vectors=vectors,
            search_tree=search_tree,
            eval_points=eval_points,
            tolerance=tol_h,
            neighborhood_size=init_neighborhood_size,
            adaptive=True
        )
        
        res['n_front'] = n_front
        res['n_int'] = n_int
        res['n_total'] = n_total
        res['h_mean'] = h_mean
        res['h_medio'] = h_mean
        res['tol_h'] = tol_h
        res['label'] = label
        results.append(res)
        
        print(f"N_total = {n_total:5d} (Front={n_front:3d}, Int={n_int:4d}, h={h_mean:6.4f}) | "
              f"Tol(h)={tol_h:6.4f} | |det(A)| (mean/min): {res['mean_det']:.4f} / {res['min_det']:.4f} | "
              f"K neighbors (mean/max): {res['mean_k']:.1f} / {res['max_k']:2d} | "
              f"E Err (mean/RMS): {res['vector_err_mean']:.4e} / {res['vector_err_rms']:.4e} | "
              f"Rot Err (mean/RMS): {res['curl_err_mean']:.4e} / {res['curl_err_rms']:.4e}")
              
    return results


def generate_parametric_plots(res_tolerance, res_density, output_dir=REPORTS_DIR):
    """Generate high-resolution parametric evaluation plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Tolerance Variation
    tols = [r['tolerance'] for r in res_tolerance]
    e_v_med = [r['vector_err_mean'] for r in res_tolerance]
    e_v_rms = [r['vector_err_rms'] for r in res_tolerance]
    e_v_max = [r['vector_err_max'] for r in res_tolerance]
    
    e_r_med = [r['curl_err_mean'] for r in res_tolerance]
    e_r_rms = [r['curl_err_rms'] for r in res_tolerance]
    e_r_max = [r['curl_err_max'] for r in res_tolerance]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    ax1.plot(tols, e_v_med, 'o-', color='#1f77b4', linewidth=2, label='Mean Error')
    ax1.plot(tols, e_v_rms, 's--', color='#2ca02c', linewidth=2, label='RMS Error')
    ax1.plot(tols, e_v_max, '^:', color='#d62728', linewidth=2, label='Max Error')
    ax1.set_xlabel(r'Determinant Tolerance ($Tol_{det}$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Field Error $\| \vec{E}^h - \vec{E} \|$', fontsize=11, fontweight='bold')
    ax1.set_title('Vector Field Error vs. Tolerance', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(fontsize=10)
    
    ax2.plot(tols, e_r_med, 'o-', color='#1f77b4', linewidth=2, label='Mean Error')
    ax2.plot(tols, e_r_rms, 's--', color='#2ca02c', linewidth=2, label='RMS Error')
    ax2.plot(tols, e_r_max, '^:', color='#d62728', linewidth=2, label='Max Error')
    ax2.set_xlabel(r'Determinant Tolerance ($Tol_{det}$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Curl Error $|\nabla \times \vec{E}^h - \nabla \times \vec{E}|$', fontsize=11, fontweight='bold')
    ax2.set_title('Curl Error vs. Tolerance', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(fontsize=10)
    
    fig.tight_layout()
    plot_tol_path = os.path.join(output_dir, 'analise_tolerancia.png')
    plt.savefig(plot_tol_path)
    plt.close(fig)
    print(f"Tolerance analysis plot saved to: {plot_tol_path}")
    
    # Plot 2: Mesh Density Variation (Log-Log)
    h_means = [r['h_mean'] for r in res_density]
    e_v_med_d = [r['vector_err_mean'] for r in res_density]
    e_v_rms_d = [r['vector_err_rms'] for r in res_density]
    e_v_max_d = [r['vector_err_max'] for r in res_density]
    
    e_r_med_d = [r['curl_err_mean'] for r in res_density]
    e_r_rms_d = [r['curl_err_rms'] for r in res_density]
    e_r_max_d = [r['curl_err_max'] for r in res_density]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    ax1.loglog(h_means, e_v_med_d, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='Mean Error')
    ax1.loglog(h_means, e_v_rms_d, 's--', color='#2ca02c', linewidth=2, markersize=7, label='RMS Error')
    ax1.loglog(h_means, e_v_max_d, '^:', color='#d62728', linewidth=2, markersize=7, label='Max Error')
    ax1.set_xlabel(r'Mean Nodal Spacing ($h$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel(r'Field Error $\| \vec{E}^h - \vec{E} \|$', fontsize=11, fontweight='bold')
    ax1.set_title(r'Convergence of $\vec{E}$ vs. $h$ (Log-Log scale)', fontsize=12, fontweight='bold')
    ax1.grid(True, which="both", linestyle='--', alpha=0.5)
    ax1.legend(fontsize=10)
    
    ax2.loglog(h_means, e_r_med_d, 'o-', color='#1f77b4', linewidth=2, markersize=7, label='Mean Error')
    ax2.loglog(h_means, e_r_rms_d, 's--', color='#2ca02c', linewidth=2, markersize=7, label='RMS Error')
    ax2.loglog(h_means, e_r_max_d, '^:', color='#d62728', linewidth=2, markersize=7, label='Max Error')
    ax2.set_xlabel(r'Mean Nodal Spacing ($h$)', fontsize=11, fontweight='bold')
    ax2.set_ylabel(r'Curl Error $|\nabla \times \vec{E}^h - \nabla \times \vec{E}|$', fontsize=11, fontweight='bold')
    ax2.set_title(r'Convergence of $\nabla \times \vec{E}$ vs. $h$ (Log-Log scale)', fontsize=12, fontweight='bold')
    ax2.grid(True, which="both", linestyle='--', alpha=0.5)
    ax2.legend(fontsize=10)
    
    fig.tight_layout()
    plot_dens_path = os.path.join(output_dir, 'analise_densidade.png')
    plt.savefig(plot_dens_path)
    plt.close(fig)
    print(f"Density analysis plot saved to: {plot_dens_path}")


def generate_parametric_markdown_report(res_tolerance, res_density, report_path=DEFAULT_REPORT_PATH):
    """Generate Markdown technical report."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    lines = []
    lines.append("# Technical Report: Parametric Analysis of VNMM 2D (L1 Basis)\n\n")
    lines.append("![Tolerance Analysis](analise_tolerancia.png)\n\n")
    lines.append("![Density Analysis](analise_densidade.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Report saved to: {report_path}")


def main():
    coords, vectors = generate_dense_mesh(
        filename=None,
        num_boundary_nodes=56,
        num_interior_nodes=360,
        limit=10.0,
        seed=42,
        silent=True
    )
    search_tree = build_search_tree(coords)
    
    xs = np.linspace(-7.0, 7.0, 15)
    ys = np.linspace(-7.0, 7.0, 15)
    xx, yy = np.meshgrid(xs, ys)
    eval_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    res_tol = run_tolerance_analysis(coords, vectors, search_tree, eval_points)
    res_dens = run_density_analysis(eval_points=eval_points)
    generate_parametric_plots(res_tol, res_dens)
    generate_parametric_markdown_report(res_tol, res_dens)


# Backward compatibility aliases
executar_analise_tolerancia = run_tolerance_analysis
executar_analise_densidade = run_density_analysis
gerar_graficos_relatorio = generate_parametric_plots
gerar_relatorio_markdown = generate_parametric_markdown_report


if __name__ == "__main__":
    main()
