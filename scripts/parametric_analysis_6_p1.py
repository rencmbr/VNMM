"""
Parametric analysis of tolerance and mesh density for VNMM 2D complete linear basis P1 (6 nodes).

Evaluates:
1. Interpolation error sensitivity to |det(A)| threshold.
2. Error convergence rate vs nodal spacing h with quartic law Tol_det(h) ~ h^4.
3. Comparative convergence with and without tolerance floor (Tol_floor).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
DEFAULT_REPORT_PATH = os.path.join(REPORTS_DIR, "relatorio_analise_parametrica_P1.md")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from scripts.generate_dense_mesh import generate_dense_mesh
from scripts.build_search_tree import build_search_tree
from scripts.evaluate_point_grid_6_p1 import evaluate_point_grid_6_p1
from scripts.evaluate_point_grid import evaluate_point_grid


def run_tolerance_analysis_6_p1(
    coords, 
    vectors, 
    search_tree, 
    eval_points, 
    tolerance_list=None, 
    init_neighborhood_size=12,
    # Backward compatibility
    arvore=None,
    pontos_avaliacao=None,
    lista_tolerancias=None,
    tamanho_vizinhanca_ini=None
):
    """
    Run parametric analysis varying the determinant threshold |det(A)| for complete P1 basis.
    """
    if arvore is not None: search_tree = arvore
    if pontos_avaliacao is not None: eval_points = pontos_avaliacao
    if lista_tolerancias is not None: tolerance_list = lista_tolerancias
    if tamanho_vizinhanca_ini is not None: init_neighborhood_size = tamanho_vizinhanca_ini

    if tolerance_list is None:
        tolerance_list = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 4e-1, 6e-1, 8e-1, 1.0]
        
    print(f"\n=======================================================")
    print(f"STUDY 1 (P1): Parametric Analysis of Tolerance |det(A)|")
    print(f"=======================================================")
    print(f"Tolerance range: {tolerance_list}\n")
    
    results = []
    
    for tol in tolerance_list:
        res = evaluate_point_grid_6_p1(
            coords=coords,
            vectors=vectors,
            search_tree=search_tree,
            eval_points=eval_points,
            tolerance=tol,
            neighborhood_size=init_neighborhood_size,
            adaptive=True,
            step_k=4
        )
        res['tolerance'] = tol
        res['tolerancia'] = tol
        results.append(res)
        
        print(f"Tol = {tol:8.1e} | Success: {res['success_rate']:5.1f}% | "
              f"|det(A)| mean: {res['mean_det']:8.2e} | "
              f"K mean: {res['mean_k']:4.1f} | "
              f"E Err (mean/RMS): {res['vector_err_mean']:.4e} / {res['vector_err_rms']:.4e} | "
              f"Rot Err (mean/RMS): {res['curl_err_mean']:.4e} / {res['curl_err_rms']:.4e}")
              
    return results


def run_density_analysis_6_p1(
    mesh_configs=None, 
    eval_points=None, 
    ref_tolerance=1.0, 
    ref_h=2.0,
    floor_tolerance=0.0,
    init_neighborhood_size=12,
    evaluate_l1=False,
    # Backward compatibility
    lista_configs_malha=None,
    pontos_avaliacao=None,
    tolerancia_ref=None,
    h_ref=None,
    tol_piso=None,
    tamanho_vizinhanca_ini=None,
    avaliar_L1=None
):
    """
    Run parametric mesh density analysis for P1 basis (6 nodes).
    Tolerance follows quartic law: Tol_det(h) = max(Tol_ref * (h / h_ref)^4, floor_tolerance).
    """
    if lista_configs_malha is not None: mesh_configs = lista_configs_malha
    if pontos_avaliacao is not None: eval_points = pontos_avaliacao
    if tolerancia_ref is not None: ref_tolerance = tolerancia_ref
    if h_ref is not None: ref_h = h_ref
    if tol_piso is not None: floor_tolerance = tol_piso
    if tamanho_vizinhanca_ini is not None: init_neighborhood_size = tamanho_vizinhanca_ini
    if avaliar_L1 is not None: evaluate_l1 = avaliar_L1

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
        
    floor_str = f"with Floor Tol_floor={floor_tolerance:.1e}" if floor_tolerance > 0 else "WITHOUT Floor (Pure Quartic Law)"
    print(f"\n=======================================================")
    print(f"DENSITY ANALYSIS (P1): {floor_str}")
    print(f"=======================================================")
    print(f"Density range: {mesh_configs[0][0]+mesh_configs[0][1]} to {mesh_configs[-1][0]+mesh_configs[-1][1]} nodes")
    print(f"Initial neighborhood K: {init_neighborhood_size}\n")
    
    results_p1 = []
    results_l1 = []
    
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
        tol_h_p1 = max(ref_tolerance * (h_mean / ref_h)**4, floor_tolerance)
        
        res_p1 = evaluate_point_grid_6_p1(
            coords=coords,
            vectors=vectors,
            search_tree=search_tree,
            eval_points=eval_points,
            tolerance=tol_h_p1,
            neighborhood_size=init_neighborhood_size,
            adaptive=True,
            step_k=4
        )
        
        res_p1['n_front'] = n_front
        res_p1['n_int'] = n_int
        res_p1['n_total'] = n_total
        res_p1['h_mean'] = h_mean
        res_p1['h_medio'] = h_mean
        res_p1['tol_h'] = tol_h_p1
        res_p1['label'] = label
        results_p1.append(res_p1)
        
        if evaluate_l1:
            tol_h_l1 = ref_tolerance * (h_mean / ref_h)
            res_l1_item = evaluate_point_grid(
                coords=coords,
                vectors=vectors,
                search_tree=search_tree,
                eval_points=eval_points,
                tolerance=tol_h_l1,
                neighborhood_size=8,
                adaptive=True
            )
            res_l1_item['n_total'] = n_total
            res_l1_item['h_mean'] = h_mean
            res_l1_item['tol_h'] = tol_h_l1
            res_l1_item['label'] = label
            results_l1.append(res_l1_item)
        
        print(f"N_total = {n_total:5d} (Front={n_front:3d}, Int={n_int:4d}, h={h_mean:6.4f}) | "
              f"Tol(h)={tol_h_p1:8.2e} | |det(A)|: {res_p1['mean_det']:8.2e} | "
              f"K mean: {res_p1['mean_k']:4.1f} | "
              f"E Err RMS: {res_p1['vector_err_rms']:.4e} | "
              f"Rot Err RMS: {res_p1['curl_err_rms']:.4e}")
              
    return results_p1, results_l1


def generate_parametric_plots_6_p1(
    res_tolerance, 
    res_density_no_floor, 
    res_density_with_floor, 
    res_density_l1=None, 
    output_dir=REPORTS_DIR
):
    """Generate high-resolution plots for P1 parametric analyses."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Tolerance Variation (P1)
    tols = [r['tolerance'] for r in res_tolerance]
    e_v_rms = [r['vector_err_rms'] for r in res_tolerance]
    e_v_max = [r['vector_err_max'] for r in res_tolerance]
    e_r_rms = [r['curl_err_rms'] for r in res_tolerance]
    e_r_max = [r['curl_err_max'] for r in res_tolerance]
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.plot(tols, e_v_rms, 'b-o', linewidth=2, label=r"RMS Error $\vec{E}$")
    ax1.plot(tols, e_v_max, 'b--s', linewidth=1.5, alpha=0.7, label=r"Max Error $\vec{E}$")
    ax1.plot(tols, e_r_rms, 'r-^', linewidth=2, label=r"RMS Error $\nabla \times \vec{E}$")
    ax1.plot(tols, e_r_max, 'r--d', linewidth=1.5, alpha=0.7, label=r"Max Error $\nabla \times \vec{E}$")
    ax1.set_xlabel(r"Determinant Tolerance $Tol_{det}$ (Log scale)", fontsize=11)
    ax1.set_ylabel("Error Magnitude (Log scale)", fontsize=11)
    ax1.set_title(r"Tolerance Sensitivity Analysis (Basis $\mathcal{P}^1$ - 6 nodes)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=10)
    fig.tight_layout()
    plot_tol_path = os.path.join(output_dir, "analise_tolerancia_P1.png")
    fig.savefig(plot_tol_path, dpi=300)
    plt.close(fig)
    print(f"P1 tolerance plot saved to: {plot_tol_path}")
    
    # Plot 2: Convergence WITHOUT Floor
    h_vals = np.array([r['h_mean'] for r in res_density_no_floor])
    ev_rms_no = np.array([r['vector_err_rms'] for r in res_density_no_floor])
    er_rms_no = np.array([r['curl_err_rms'] for r in res_density_no_floor])
    
    p_e_no = np.polyfit(np.log(h_vals), np.log(ev_rms_no), 1)
    p_rot_no = np.polyfit(np.log(h_vals), np.log(er_rms_no), 1)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(h_vals, ev_rms_no, 'b-o', linewidth=2.2, label=f"RMS Error $\\vec{{E}}$ (Rate = {p_e_no[0]:.2f})")
    ax.loglog(h_vals, er_rms_no, 'r-s', linewidth=2.2, label=f"RMS Error $\\nabla \\times \\vec{{E}}$ (Rate = {p_rot_no[0]:.2f})")
    
    h_ref_line = np.linspace(h_vals.min(), h_vals.max(), 100)
    c_e_no = ev_rms_no[0] / (h_vals[0]**2)
    c_rot_no = er_rms_no[0] / (h_vals[0]**1)
    ax.loglog(h_ref_line, c_e_no * (h_ref_line**2), 'k--', alpha=0.5, label=r"Theoretical Reference $\mathcal{O}(h^2)$")
    ax.loglog(h_ref_line, c_rot_no * (h_ref_line**1), 'k:', alpha=0.6, label=r"Theoretical Reference $\mathcal{O}(h^1)$")
    
    ax.set_xlabel(r"Characteristic Spacing $h$ (Log scale)", fontsize=11)
    ax.set_ylabel("RMS Interpolation Error (Log scale)", fontsize=11)
    ax.set_title(r"Convergence WITHOUT Tolerance Floor ($Tol_{det} \propto h^4$)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    plot_no_floor_path = os.path.join(output_dir, "analise_densidade_convergencia_P1_sem_piso.png")
    fig.savefig(plot_no_floor_path, dpi=300)
    plt.close(fig)
    print(f"Convergence without floor plot saved to: {plot_no_floor_path}")
    
    # Plot 3: Convergence WITH Floor
    ev_rms_with = np.array([r['vector_err_rms'] for r in res_density_with_floor])
    er_rms_with = np.array([r['curl_err_rms'] for r in res_density_with_floor])
    
    p_e_with = np.polyfit(np.log(h_vals), np.log(ev_rms_with), 1)
    p_rot_with = np.polyfit(np.log(h_vals), np.log(er_rms_with), 1)
    
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.loglog(h_vals, ev_rms_with, 'b-o', linewidth=2.2, label=f"RMS Error $\\vec{{E}}$ (Rate = {p_e_with[0]:.2f} $\\approx \\mathcal{{O}}(h^2)$)")
    ax.loglog(h_vals, er_rms_with, 'r-s', linewidth=2.2, label=f"RMS Error $\\nabla \\times \\vec{{E}}$ (Rate = {p_rot_with[0]:.2f} $\\approx \\mathcal{{O}}(h^1)$)")
    
    c_e_with = ev_rms_with[-1] / (h_vals[-1]**2)
    c_rot_with = er_rms_with[-1] / (h_vals[-1]**1)
    ax.loglog(h_ref_line, c_e_with * (h_ref_line**2), 'k--', alpha=0.5, label=r"Theoretical Reference $\mathcal{O}(h^2)$")
    ax.loglog(h_ref_line, c_rot_with * (h_ref_line**1), 'k:', alpha=0.6, label=r"Theoretical Reference $\mathcal{O}(h^1)$")
    
    ax.set_xlabel(r"Characteristic Spacing $h$ (Log scale)", fontsize=11)
    ax.set_ylabel("RMS Interpolation Error (Log scale)", fontsize=11)
    ax.set_title(r"Convergence WITH Tolerance Floor ($Tol_{piso} = 1.5 \times 10^{-5}$)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    plot_with_floor_path = os.path.join(output_dir, "analise_densidade_convergencia_P1_com_piso.png")
    fig.savefig(plot_with_floor_path, dpi=300)
    plt.close(fig)
    print(f"Convergence with floor plot saved to: {plot_with_floor_path}")


def generate_parametric_markdown_report_6_p1(
    res_tolerance, 
    res_density_no_floor, 
    res_density_with_floor, 
    res_density_l1=None, 
    report_path=DEFAULT_REPORT_PATH
):
    """Generate Markdown report for P1 parametric analysis."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    lines = []
    lines.append("# Technical Report: Parametric Analysis of Complete Linear Basis $\\mathcal{P}^1$ (6 Nodes)\n\n")
    lines.append("![Tolerance Analysis](analise_tolerancia_P1.png)\n\n")
    lines.append("![Convergence without Floor](analise_densidade_convergencia_P1_sem_piso.png)\n\n")
    lines.append("![Convergence with Floor](analise_densidade_convergencia_P1_com_piso.png)\n\n")
    
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
    
    res_tol = run_tolerance_analysis_6_p1(coords, vectors, search_tree, eval_points)
    res_no_floor, res_l1 = run_density_analysis_6_p1(eval_points=eval_points, floor_tolerance=0.0, evaluate_l1=True)
    res_with_floor, _ = run_density_analysis_6_p1(eval_points=eval_points, floor_tolerance=1.5e-5, evaluate_l1=False)
    
    generate_parametric_plots_6_p1(res_tol, res_no_floor, res_with_floor, res_l1)
    generate_parametric_markdown_report_6_p1(res_tol, res_no_floor, res_with_floor, res_l1)


# Backward compatibility aliases
executar_analise_tolerancia_6_P1 = run_tolerance_analysis_6_p1
executar_analise_densidade_6_P1 = run_density_analysis_6_p1
gerar_graficos_relatorio_6_P1 = generate_parametric_plots_6_p1
gerar_relatorio_markdown_6_P1 = generate_parametric_markdown_report_6_p1


if __name__ == "__main__":
    main()
