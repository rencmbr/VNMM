#!/usr/bin/env python3
"""
parametric_analysis_alpha_s_mlpg_2d.py
======================================
Comprehensive parametric sensitivity study of the subdomain radius factor alpha_s
for the 2D VNMM-MLPG framework across both:
1. The Manufactured Solution (MMS) boundary-value problem (Pure Dirichlet, 17x17)
2. The Resonant Cavity eigenvalue problem (PEC, s_div=15.0, 21x21)

The parameter alpha_s varies across [0.30, 2.00], deliberately extending well below
the sub-critical coverage threshold (alpha_s < 0.707) and well above the stencil
containment limit (alpha_s > 1.20).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

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
    'legend.fontsize': 9,
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
    print("ESTUDO PARAMETRICO RIGOROSO DE ALPHA_S NO VNMM-MLPG 2D")
    print("Faixa recomendada na literatura: [0.70, 1.20] | Faixa investigada: [0.30, 2.00]")
    print("=" * 115)

    coords_mms, vecs_mms, bnd_mms = generate_cavity_mesh(17, 17)

    results = []

    header = (
        f"{'alpha_s':>7} | {'Coverage':>8} | {'r_s>r_u':>7} | "
        f"{'MMS L2(E)':>10} | {'MMS H(curl)':>11} | {'MMS DoF':>9} | "
        f"{'Cav Mean kc':>11} | {'Cav Max kc':>10} | {'Cav cond(K)':>11} | {'Spur':>4}"
    )
    print(header)
    print("-" * 115)

    for a in alpha_vals:
        # Domain coverage and r_s / r_u checks
        subs = build_mlpg_subdomains_2d(coords_mms, vecs_mms, is_boundary=bnd_mms, alpha_s=a)
        cov, _ = verify_mlpg_coverage_2d(subs, Lx=np.pi, Ly=np.pi, sample_density=50)
        ratios = [s.radius_s / s.radius_u for s in subs]
        viol_count = sum(1 for r in ratios if r > 1.0)

        # 1. MMS Problem
        res_mms = assemble_and_solve_mlpg_mms(
            coords=coords_mms, vectors=vecs_mms, is_boundary=bnd_mms,
            alpha_s=a, s_div=6.0, boundary_condition='dirichlet'
        )
        l2_err = res_mms['err_L2_E'] * 100.0
        curl_err = res_mms['err_L2_curl'] * 100.0
        dof_err = res_mms['err_dof_rel'] * 100.0

        # 2. Cavity Eigenvalues
        res_cav = solve_mlpg_cavity_eigenvalues(
            Nx=21, Ny=21, s_div=15.0, alpha_s=a, num_eigenvalues=10
        )
        cav_mean_kc = res_cav['mean_error_kc_pct']
        cav_max_kc = res_cav['max_error_kc_pct']
        cav_cond = res_cav['cond_K']
        spur_band = res_cav['spur_in_band']

        row_str = (
            f"{a:7.3f} | {cov*100.0:7.1f}% | {viol_count:7d} | "
            f"{l2_err:9.3f}% | {curl_err:10.3f}% | {dof_err:8.3f}% | "
            f"{cav_mean_kc:10.2f}% | {cav_max_kc:9.2f}% | {cav_cond:11.2e} | {spur_band:4d}"
        )
        print(row_str)

        results.append({
            'alpha_s': a,
            'coverage': cov * 100.0,
            'rs_gt_ru': viol_count,
            'mean_rs_ru': float(np.mean(ratios)),
            'max_rs_ru': float(np.max(ratios)),
            'mms_l2': l2_err,
            'mms_curl': curl_err,
            'mms_dof': dof_err,
            'cav_mean_kc': cav_mean_kc,
            'cav_max_kc': cav_max_kc,
            'cav_cond': cav_cond,
            'spur': spur_band
        })

    print("=" * 115)

    # Generate Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=300)

    alphas = np.array([r['alpha_s'] for r in results])
    mms_l2 = np.array([r['mms_l2'] for r in results])
    mms_curl = np.array([r['mms_curl'] for r in results])
    cav_mean = np.array([r['cav_mean_kc'] for r in results])
    cav_max = np.array([r['cav_max_kc'] for r in results])

    # Plot 1: MMS Source Problem
    ax1.axvspan(0.70, 1.20, color='#2ecc71', alpha=0.18, label=r'Recommended Range $[0.7, 1.2]$')
    ax1.axvline(0.7071, color='#e74c3c', linestyle='--', linewidth=1.2, label=r'Critical Coverage $\alpha_s = \sqrt{2}/2$')
    ax1.plot(alphas, mms_l2, 'o-', color='#1f77b4', linewidth=2.0, markersize=5, label=r'$L^2(\vec{E})$ Error (%)')
    ax1.plot(alphas, mms_curl, 's--', color='#e67e22', linewidth=1.8, markersize=5, label=r'$H(\mathbf{curl})$ Error (%)')

    ax1.set_xlabel(r'Subdomain Radius Scaling Factor $\alpha_s$')
    ax1.set_ylabel(r'MMS Relative Error (%)')
    ax1.set_title(r'(a) MMS Source Problem Convergence ($17 \times 17$)')
    ax1.set_xlim(0.25, 2.05)
    ax1.set_ylim(0, 36)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left', framealpha=0.92, fontsize=8.5)

    # Plot 2: Cavity Eigenvalue Resonances
    ax2.axvspan(0.70, 1.20, color='#2ecc71', alpha=0.18, label=r'Recommended Range $[0.7, 1.2]$')
    ax2.axvline(0.7071, color='#e74c3c', linestyle='--', linewidth=1.2, label=r'Critical Coverage $\alpha_s = \sqrt{2}/2$')
    ax2.plot(alphas, cav_mean, 'd-', color='#2c3e50', linewidth=2.0, markersize=5, label=r'Mean $k_c$ Error (%) [First 10 Modes]')
    ax2.plot(alphas, cav_max, '^--', color='#9b59b6', linewidth=1.8, markersize=5, label=r'Max $k_c$ Error (%)')

    ax2.set_xlabel(r'Subdomain Radius Scaling Factor $\alpha_s$')
    ax2.set_ylabel(r'Cavity Cutoff Wavenumber Error (%)')
    ax2.set_title(r'(b) PEC Resonant Cavity Spectrum ($21 \times 21, s_{\mathrm{div}}=15.0$)')
    ax2.set_xlim(0.25, 2.05)
    ax2.set_ylim(0, 24)
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.legend(loc='upper left', framealpha=0.92, fontsize=8.5)

    out_fig = os.path.join(ROOT_DIR, "paper", "mlpg-vnmm", "fig_mlpg_alpha_s_sensitivity.png")
    fig.savefig(out_fig)
    plt.close(fig)
    print(f"\nSaved publication figure: {out_fig}")

    return results


if __name__ == "__main__":
    run_alpha_s_parametric_study()
