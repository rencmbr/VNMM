import os
import sys
import time
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.vnmm_assembler_2d import assemble_vnmm_matrices_2d
from src.eigen_solver_cavity_2d import apply_pec_boundary_conditions, ANALYTICAL_REFERENCE_MODES


def solve_unregularized_eigenvalues(
    Nx=21, 
    Ny=21, 
    Lx=np.pi, 
    Ly=np.pi, 
    Ncx=10, 
    Ncy=10, 
    basis="P1", 
    points_per_dir=2, 
    support_mode="gauss_point",
    tol_zero=1e-2,
    num_eigenvalues=10
):
    """
    Solves the purely curl-curl eigenvalue problem (no divergence regularization, s_div = 0.0),
    filtering out near-zero gradient nullspace modes (lambda < tol_zero).
    """
    coords, vectors, is_boundary = generate_cavity_mesh(
        Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, interior_type="alternating"
    )
    
    K, M = assemble_vnmm_matrices_2d(
        coords=coords, 
        vectors=vectors, 
        basis=basis, 
        s_div=0.0, 
        Ncx=Ncx, 
        Ncy=Ncy, 
        Lx=Lx, 
        Ly=Ly, 
        points_per_dir=points_per_dir, 
        support_mode=support_mode
    )
    
    K_red, M_red, int_idx = apply_pec_boundary_conditions(K, M, is_boundary)
    
    try:
        vals, vecs = la.eigh(K_red.toarray(), M_red.toarray())
    except Exception:
        vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
        vals = np.real(vals)
        vecs = np.real(vecs)
        
    positive_mask = vals > tol_zero
    positive_vals = np.sort(vals[positive_mask])
    
    n_null = np.sum(~positive_mask)
    num_eigenvals = positive_vals[:num_eigenvalues]
    
    ref_vals = np.array([item[2] for item in ANALYTICAL_REFERENCE_MODES[:len(num_eigenvals)]])
    ref_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:len(num_eigenvals)]])
    
    kc_num = np.sqrt(np.maximum(num_eigenvals, 0.0))
    errors_lambda = np.abs(num_eigenvals - ref_vals) / ref_vals * 100.0
    errors_kc = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    return {
        'Nx': Nx,
        'Ny': Ny,
        'Nc': Ncx,
        'p': points_per_dir,
        'num_cells': Ncx * Ncy,
        'total_gauss_points': Ncx * Ncy * points_per_dir * points_per_dir,
        'discarded_null_modes': int(n_null),
        'positive_modes_count': int(len(positive_vals)),
        'all_eigenvalues': vals,
        'positive_eigenvalues': num_eigenvals,
        'analytical_eigenvalues': ref_vals,
        'numerical_kc': kc_num,
        'analytical_kc': ref_kc,
        'rel_errors_lambda_pct': errors_lambda,
        'rel_errors_kc_pct': errors_kc,
        'mean_error_lambda_pct': float(np.mean(errors_lambda)),
        'mean_error_kc_pct': float(np.mean(errors_kc)),
        'max_error_kc_pct': float(np.max(errors_kc)),
        # Backward compatibility aliases
        'n_celulas': Ncx * Ncy,
        'total_pontos_gauss': Ncx * Ncy * points_per_dir * points_per_dir,
        'n_nulos_descartados': int(n_null),
        'n_positivos': int(len(positive_vals)),
        'autovalores_todos': vals,
        'autovalores_positivos': num_eigenvals,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': errors_lambda,
        'erros_kc_pct': errors_kc,
        'erro_medio_lambda_pct': float(np.mean(errors_lambda)),
        'erro_medio_kc_pct': float(np.mean(errors_kc)),
        'erro_max_kc_pct': float(np.max(errors_kc))
    }


def run_full_unregularized_study():
    print("==================================================================================")
    print("  STUDY OF 2D VNMM WITHOUT DIVERGENCE REGULARIZATION (s_div = 0.0)")
    print("  Gauss-point support (EFG style) | Near-zero eigenvalue filter (lambda > 0.01)")
    print("==================================================================================\n")
    
    cell_list = [10, 15, 20, 30]
    p_list = [2, 3, 4]
    results = []
    
    for Nc in cell_list:
        for p in p_list:
            t0 = time.time()
            res = solve_unregularized_eigenvalues(
                Nx=21, Ny=21, 
                Ncx=Nc, Ncy=Nc, 
                points_per_dir=p, 
                support_mode="gauss_point",
                tol_zero=1e-2,
                num_eigenvalues=10
            )
            t_exec = time.time() - t0
            res['runtime_sec'] = t_exec
            results.append(res)
            
            print(f"Cells: {Nc:2d}x{Nc:2d} | Gauss: {p}x{p} ({p*p:2d} pts) | Total Pts: {res['total_gauss_points']:5d} | "
                  f"Discarded Zeros: {res['discarded_null_modes']:3d} | "
                  f"1st λ: {res['positive_eigenvalues'][0]:.4f} | "
                  f"Mean kc Err: {res['mean_error_kc_pct']:5.2f}% | Max kc Err: {res['max_error_kc_pct']:5.2f}% | "
                  f"Time: {t_exec:.2f}s")
                  
    return results


def generate_report_and_plots(results, output_dir=REPORTS_DIR):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "relatorio_sem_regularizacao_divergente.md")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    res_s0 = results[0]
    
    mode_names = ["TE10", "TE01", "TE11", "TE20", "TE02", "TE21", "TE12", "TE22", "TE30", "TE03"]
    indices = np.arange(1, 11)
    bar_width = 0.35
    
    ax.bar(indices - bar_width/2, res_s0['analytical_kc'], bar_width, label="Analytical Table 4-1", color="#1f77b4", alpha=0.85)
    ax.bar(indices + bar_width/2, res_s0['numerical_kc'], bar_width, label="VNMM 2D (s_div = 0.0)", color="#d62728", alpha=0.85)
    
    ax.set_xticks(indices)
    ax.set_xticklabels(mode_names, fontsize=10, fontweight="bold")
    ax.set_ylabel(r"Cutoff Wavenumber $k_c$ [rad/m]", fontsize=11)
    ax.set_title("VNMM 2D Spectrum Without Divergence Regularization ($s_{\\text{div}} = 0.0$)", fontsize=12, fontweight="bold")
    ax.grid(True, axis='y', linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    fig.tight_layout()
    
    chart_path = os.path.join(output_dir, "espectro_sem_regularizacao_s0.png")
    fig.savefig(chart_path, dpi=300)
    plt.close(fig)
    print(f"Chart saved to: {chart_path}")
    
    lines = []
    lines.append("# Study of 2D VNMM Without Divergence Regularization ($s_{\\text{div}} = 0.0$)\n\n")
    lines.append("This report examines the purely curl-curl vector eigenvalue problem ($s_{\\text{div}} = 0.0$) "
                 "using Gauss-point support (EFG style) and filtering static near-zero modes.\n\n")
                 
    lines.append("## 1. Results by Quadrature Discretization\n\n")
    lines.append("| Cells ($N_c \\times N_c$) | Gauss ($p \\times p$) | Total Gauss Pts | Discarded Zeros | 1st $\\lambda$ | 2nd $\\lambda$ | 3rd $\\lambda$ | Mean $k_c$ Error (%) | Max $k_c$ Error (%) |\n")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    for r in results:
        l1 = r['positive_eigenvalues'][0] if len(r['positive_eigenvalues']) > 0 else 0.0
        l2 = r['positive_eigenvalues'][1] if len(r['positive_eigenvalues']) > 1 else 0.0
        l3 = r['positive_eigenvalues'][2] if len(r['positive_eigenvalues']) > 2 else 0.0
        lines.append(f"| ${r['Nc']} \\times {r['Nc']}$ | ${r['p']} \\times {r['p']}$ | {r['total_gauss_points']} | {r['discarded_null_modes']} | {l1:.4f} | {l2:.4f} | {l3:.4f} | **{r['mean_error_kc_pct']:5.2f}%** | {r['max_error_kc_pct']:5.2f}% |\n")
        
    lines.append("\n## 2. Spectrum of First 10 Filtered Modes ($N_c = 10 \\times 10, p = 2 \\times 2$)\n\n")
    lines.append("| Mode ($TE_{nm}$) | $\\lambda_{\\text{analytical}}$ | $\\lambda_{\\text{VNMM}}$ | $k_{c, \\text{analytical}}$ | $k_{c, \\text{VNMM}}$ | Error $k_c$ (%) | Diagnosis |\n")
    lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n")
    
    diagnoses = [
        "Physical (TE10)",
        "Physical (TE01)",
        "Spurious / Intermediate gradient",
        "Spurious / Intermediate gradient",
        "Physical (TE11)",
        "Physical (TE20)",
        "Physical (TE02)",
        "Spurious / Intermediate gradient",
        "Physical (TE21)",
        "Physical (TE12)"
    ]
    for i in range(10):
        m_name = mode_names[i]
        l_ref = res_s0['analytical_eigenvalues'][i]
        l_num = res_s0['positive_eigenvalues'][i]
        kc_r = res_s0['analytical_kc'][i]
        kc_n = res_s0['numerical_kc'][i]
        e_kc = res_s0['rel_errors_kc_pct'][i]
        diag = diagnoses[i]
        lines.append(f"| ${m_name}$ | {l_ref:6.2f} | {l_num:7.4f} | {kc_r:6.3f} | {kc_n:6.3f} | **{e_kc:5.2f}%** | {diag} |\n")
        
    lines.append("\n![Spectrum Without Regularization](espectro_sem_regularizacao_s0.png)\n\n")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
        
    print(f"Report saved to: {report_path}")


def main():
    res = run_full_unregularized_study()
    generate_report_and_plots(res)


# Backward compatibility aliases
resolver_autovalores_sem_regularizacao = solve_unregularized_eigenvalues
executar_estudo_completo_sem_regularizacao = run_full_unregularized_study
gerar_relatorio_e_graficos = generate_report_and_plots

if __name__ == "__main__":
    main()
