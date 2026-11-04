import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d, verify_mlpg_coverage_2d
from scripts.mlpg_source_problem_2d import assemble_and_solve_mlpg_mms
from scripts.mlpg_cavity_eigenvalues_2d import solve_mlpg_cavity_eigenvalues

# Publication figure style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 7.5,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})


def run_alpha_s_parametric_study():
    alpha_vals = [
        0.30, 0.40, 0.50, 0.60, 0.70, 0.707, 0.75, 0.80, 0.85, 0.90,
        0.95, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50, 1.60, 1.80, 2.00
    ]

    print("=" * 115)
    print("ESTUDO PARAMETRICO RIGOROSO DE ALPHA_S NO VNMM-MLPG 2D (DIRETORES ALEATORIOS)")
    print("Comparacao de Nuvens: Jitter 0% (Regular), 25% e 50% | alpha_s in [0.30, 2.00]")
    print("=" * 115)

    Nx, Ny = 21, 21
    jitters = [0.0, 0.25, 0.50]
    data_by_jitter = {}
    cache_file = os.path.join(ROOT_DIR, "scripts", "alpha_s_parametric_data.json")

    import json
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data_by_jitter = json.load(f)
            print(f"Loaded cached parametric data from {cache_file}")
        except Exception:
            data_by_jitter = {}

    if not data_by_jitter or len(data_by_jitter) < len(jitters):
        data_by_jitter = {}
        for jit in jitters:
            jit_key = f"jitter_{int(jit*100)}"
            print(f"\n>>> EXECUTANDO JITTER = {jit*100:.0f}% (DIRETORES ALEATORIOS, SEED=42) <<<")
            coords, vecs, bnds = generate_cavity_mesh(
                Nx, Ny, interior_type="random", jitter_frac=jit, seed=42
            )

            header = (
                f"{'alpha_s':>7} | {'Coverage':>8} | {'r_s>r_u':>7} | "
                f"{'MMS L2(E)':>10} | {'MMS H(curl)':>11} | "
                f"{'Cav Mean kc':>11} | {'Cav Max kc':>10} | {'Cav cond(K)':>11} | {'Spur':>4}"
            )
            print(header)
            print("-" * 110)

            results = []
            for a in alpha_vals:
                # 1. Domain coverage and r_s / r_u checks
                subs = build_mlpg_subdomains_2d(coords, vecs, is_boundary=bnds, alpha_s=a)
                cov, _ = verify_mlpg_coverage_2d(subs, Lx=np.pi, Ly=np.pi, sample_density=50)
                ratios = [s.radius_s / s.radius_u for s in subs]
                viol_count = sum(1 for r in ratios if r > 1.0)

                # 2. MMS Problem (21x21, s_div=6.0, Dirichlet)
                res_mms = assemble_and_solve_mlpg_mms(
                    coords=coords, vectors=vecs, is_boundary=bnds,
                    alpha_s=a, s_div=6.0, boundary_condition='dirichlet'
                )
                l2_err = res_mms['err_L2_E'] * 100.0
                curl_err = res_mms['err_L2_curl'] * 100.0

                # 3. Cavity Eigenvalues (21x21, s_div=6.0)
                res_cav = solve_mlpg_cavity_eigenvalues(
                    Nx=Nx, Ny=Ny, s_div=6.0, alpha_s=a, num_eigenvalues=10,
                    interior_type="random", jitter_frac=jit, seed=42, spur_threshold=1000.0
                )
                cav_mean_kc = res_cav['mean_error_kc_pct']
                cav_max_kc = res_cav['max_error_kc_pct']
                cav_cond = res_cav['cond_K']
                spur_band = res_cav['spur_in_band']

                row_str = (
                    f"{a:7.3f} | {cov*100.0:7.1f}% | {viol_count:7d} | "
                    f"{l2_err:9.3f}% | {curl_err:10.3f}% | "
                    f"{cav_mean_kc:10.2f}% | {cav_max_kc:9.2f}% | {cav_cond:11.2e} | {spur_band:4d}"
                )
                print(row_str)

                results.append({
                    'alpha_s': a,
                    'coverage': cov * 100.0,
                    'rs_gt_ru': viol_count,
                    'viol_pct': viol_count / len(coords) * 100.0,
                    'mean_rs_ru': float(np.mean(ratios)),
                    'mms_l2': l2_err,
                    'mms_curl': curl_err,
                    'cav_mean_kc': cav_mean_kc,
                    'cav_max_kc': cav_max_kc,
                    'cav_cond': cav_cond,
                    'spur': spur_band
                })

            data_by_jitter[jit_key] = results

            with open(cache_file, "w") as f:
                json.dump(data_by_jitter, f, indent=2)
            print(f"Saved parametric data cache to {cache_file}")

    # Generate 2-Panel Publication Figure: (a) Coverage, (b) MMS Convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    alphas = np.array(alpha_vals)

    curve_styles = [
        ("jitter_0", '#1f77b4', '-', 'o', 'Jitter 0% (Regular Lattice)'),
        ("jitter_25", '#2ca02c', '--', 's', 'Jitter 25% (Perturbed)')
    ]

    # -------------------------------------------------------------
    # (a) Domain Integration Coverage
    # -------------------------------------------------------------
    ax1.axvspan(0.70, 1.20, color='#2ecc71', alpha=0.18, label=r'Operational Window $[0.7, 1.2]$')
    ax1.axvline(0.7071, color='#e74c3c', linestyle='--', linewidth=1.4, label=r'Regular Limit $\alpha_s = \sqrt{2}/2 \approx 0.7071$')
    ax1.axhline(100.0, color='#7f8c8d', linestyle=':', linewidth=1.0)

    for k, col, ls, marker, lbl in curve_styles:
        covs = np.array([r['coverage'] for r in data_by_jitter[k]])
        ax1.plot(alphas, covs, marker=marker, linestyle=ls, color=col, linewidth=2.0, markersize=6, label=lbl)

    ax1.set_xlabel(r'Subdomain Radius Scaling Factor $\alpha_s$', fontsize=10.5)
    ax1.set_ylabel(r'Domain Integration Coverage (%)', fontsize=10.5)
    ax1.set_title(r'(a) Global Domain Coverage $\bigcup \Omega_s^I \supseteq \Omega$', fontsize=11.5)
    ax1.set_xlim(0.25, 2.05)
    ax1.set_ylim(10, 105)
    ax1.tick_params(labelsize=9.5)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='lower right', framealpha=0.92, fontsize=9.0)

    # -------------------------------------------------------------
    # (b) MMS Source Problem Convergence
    # -------------------------------------------------------------
    ax2.axvspan(0.70, 1.20, color='#2ecc71', alpha=0.18, label=r'Operational Window $[0.7, 1.2]$')
    ax2.axvline(0.7071, color='#e74c3c', linestyle='--', linewidth=1.4, label=r'Regular Limit $\alpha_s = \sqrt{2}/2$')

    for k, col, ls, marker, lbl in curve_styles:
        mms_err = np.array([r['mms_l2'] for r in data_by_jitter[k]])
        ax2.plot(alphas, mms_err, marker=marker, linestyle=ls, color=col, linewidth=2.0, markersize=6, label=lbl)
        # Mark alpha_s = 0.80
        val_08 = mms_err[alphas == 0.80][0]
        ax2.plot([0.80], [val_08], '*', color=col, markersize=11)

    ax2.plot([], [], '*', color='#2c3e50', markersize=11, label=r'Adopted Standard $\alpha_s = 0.80$')
    ax2.set_xlabel(r'Subdomain Radius Scaling Factor $\alpha_s$', fontsize=10.5)
    ax2.set_ylabel(r'MMS Relative $L^2(\vec{E})$ Error (%)', fontsize=10.5)
    ax2.set_title(r'(b) MMS Convergence ($21 \times 21$, Random Directors)', fontsize=11.5)
    ax2.set_xlim(0.25, 2.05)
    ax2.tick_params(labelsize=9.5)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left', framealpha=0.92, fontsize=9.0)

    plt.tight_layout()

    repo_root = os.path.abspath(os.path.join(ROOT_DIR, ".."))
    out_fig = os.path.join(repo_root, "paper", "mlpg-vnmm", "fig_mlpg_alpha_s_sensitivity.png")
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"\nSaved publication figure: {out_fig}")

    return data_by_jitter


if __name__ == "__main__":
    run_alpha_s_parametric_study()
