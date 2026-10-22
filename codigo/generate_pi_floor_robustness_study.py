#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robustness and Numerical Stabilization Study Under Stochastic Perturbations:
Evaluation of the Role of the Tolerance Floor (Tol_floor) in the Complete P1 Basis (6 nodes)
over the Domain Omega = [0, pi] x [0, pi].

Generates the publication figure 'figura_robustez_perturbacao_estocastica.pdf'/'.png'
and consolidated data for Table 2 of the manuscript.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "relatorios")
os.makedirs(REPORTS_DIR, exist_ok=True)

sys.path.insert(0, CODE_DIR)

from support_nodes_vnmm_2d_6_p1 import find_support_nodes_vnmm_2d_6_p1
from shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1
from generate_convergence_l1_vs_p1_pi_figure import generate_pi_nodal_mesh, exact_field_te11

# Global Publication Styling (Elsevier standard)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.0,
    "figure.titlesize": 12.5,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.45,
    "lines.markersize": 6.5,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

COLOR_NAVY = "#0f4c81"
COLOR_CRIMSON = "#c0392b"
COLOR_TEAL = "#16a085"
COLOR_GRAY = "#7f8c8d"


def evaluate_p1_grid(coords, vectors, tree, test_points, tol_det, global_projections):
    """Evaluates interpolation errors on a test grid for P1 basis."""
    err_E = []
    err_rot = []
    dets = []
    ks = []
    conds = []
    successes = 0
    
    for point in test_points:
        e_exact, rot_exact = exact_field_te11(point)
        sel_p1, det_a, a_mat, k_eff = find_support_nodes_vnmm_2d_6_p1(
            point, coords, vectors, tree, k=12, tol_det=tol_det, adaptive=True, step_k=4, k_max=16
        )
        if len(sel_p1) == 6 and det_a > 0:
            successes += 1
            phi_p1, rot_phi_p1, _ = shape_functions_vnmm_2d_6_p1(point, coords, vectors, sel_p1, matrix_a=a_mat)
            e_p1 = global_projections[sel_p1]
            e_interp = phi_p1 @ e_p1
            rot_interp = float(np.dot(rot_phi_p1, e_p1))
            
            err_E.append(np.linalg.norm(e_interp - e_exact))
            err_rot.append(abs(rot_interp - rot_exact))
            dets.append(det_a)
            ks.append(k_eff)
            
            try:
                cond_a = np.linalg.cond(a_mat)
            except Exception:
                cond_a = 1e6
            conds.append(cond_a)
        else:
            err_E.append(10.0)
            err_rot.append(50.0)
            dets.append(0.0)
            ks.append(k_eff)
            conds.append(1e6)
            
    n_pts = len(test_points)
    return {
        'success_rate': (successes / n_pts) * 100.0,
        'err_e_rms': float(np.sqrt(np.mean(np.array(err_E)**2))),
        'err_rot_rms': float(np.sqrt(np.mean(np.array(err_rot)**2))),
        'det_avg': float(np.mean(dets)),
        'k_avg': float(np.mean(ks)),
        'cond_med': float(np.median(conds)),
        # Dual-key compatibility
        'taxa_sucesso': (successes / n_pts) * 100.0,
        'err_E_rms': float(np.sqrt(np.mean(np.array(err_E)**2))),
        'det_med': float(np.mean(dets)),
        'k_med': float(np.mean(ks))
    }


def generate_pi_floor_robustness_study():
    """Executes the stochastic robustness study and generates figures."""
    print("=" * 80)
    print("STOCHASTIC ROBUSTNESS AND TOLERANCE FLOOR STUDY IN [0, pi]^2")
    print("=" * 80)
    
    # 1600 evaluation points grid
    n_pts_side = 40
    xs = np.linspace(0.04 * np.pi, 0.96 * np.pi, n_pts_side)
    ys = np.linspace(0.04 * np.pi, 0.96 * np.pi, n_pts_side)
    X, Y = np.meshgrid(xs, ys)
    test_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # 6 mesh configurations
    mesh_configs = [
        (24, 60, "Sparse (N=84)"),
        (36, 150, "Medium-Sparse (N=186)"),
        (56, 360, "Medium (N=416)"),
        (84, 800, "Medium-Dense (N=884)"),
        (128, 1800, "Dense (N=1928)"),
        (192, 4000, "Very Dense (N=4192)")
    ]
    h_ref = (4.0 * np.pi) / 24.0
    tol_floor = 1.5e-5
    
    # -------------------------------------------------------------------------
    # PART 1: Convergence with vs without floor
    # -------------------------------------------------------------------------
    print("\n>>> PART 1: Mesh refinement with and without tolerance floor...")
    conv_data = []
    
    for n_boundary, n_int, label in mesh_configs:
        coords, vectors, h_avg = generate_pi_nodal_mesh(n_boundary, n_int, seed=42)
        tree = KDTree(coords)
        
        # Analytical projections
        e_nodes = np.zeros_like(coords)
        for i, pt in enumerate(coords):
            e_nodes[i], _ = exact_field_te11(pt)
        global_projections = np.sum(e_nodes * vectors, axis=1)
        
        # Pure quartic threshold (WITHOUT floor)
        tol_without = 0.005 * (h_avg / h_ref)**4
        res_without = evaluate_p1_grid(coords, vectors, tree, test_points, tol_without, global_projections)
        
        # Stabilized law (WITH floor)
        tol_with = max(0.08 * (h_avg / h_ref)**4, tol_floor)
        res_with = evaluate_p1_grid(coords, vectors, tree, test_points, tol_with, global_projections)
        
        item = {
            'n_boundary': n_boundary,
            'n_int': n_int,
            'n_total': len(coords),
            'h_avg': h_avg,
            'label': label,
            'tol_without': tol_without,
            'tol_with': tol_with,
            'res_without': res_without,
            'res_with': res_with,
            # Dual-key compatibility
            'tol_sem': tol_without,
            'tol_com': tol_with,
            'res_sem': res_without,
            'res_com': res_with
        }
        conv_data.append(item)
        print(f"Mesh {label:20s} | N={item['n_total']:5d} | h={h_avg:.4f} m")
        print(f"  WITHOUT floor (Tol={tol_without:.2e}): E_RMS = {res_without['err_e_rms']:.4e} | Rot_RMS = {res_without['err_rot_rms']:.4e} | K={res_without['k_avg']:.1f} | cond={res_without['cond_med']:.1f}")
        print(f"  WITH floor    (Tol={tol_with:.2e}): E_RMS = {res_with['err_e_rms']:.4e} | Rot_RMS = {res_with['err_rot_rms']:.4e} | K={res_with['k_avg']:.1f} | cond={res_with['cond_med']:.1f}")

    # -------------------------------------------------------------------------
    # PART 2: Tolerance sweep on very dense mesh
    # -------------------------------------------------------------------------
    print("\n>>> PART 2: Tolerance sweep on very dense mesh (N = 4192, h = 0.0654 m)...")
    coords_dense, vectors_dense, h_dense = generate_pi_nodal_mesh(192, 4000, seed=42)
    tree_dense = KDTree(coords_dense)
    
    e_nodes_d = np.zeros_like(coords_dense)
    for i, pt in enumerate(coords_dense):
        e_nodes_d[i], _ = exact_field_te11(pt)
    proj_dense = np.sum(e_nodes_d * vectors_dense, axis=1)
    
    tol_sweep = [1.0e-6, 3.0e-6, 8.0e-6, 1.5e-5, 3.0e-5, 6.0e-5, 1.0e-4]
    sweep_data = []
    
    for tol in tol_sweep:
        res = evaluate_p1_grid(coords_dense, vectors_dense, tree_dense, test_points, tol, proj_dense)
        res['tol'] = tol
        sweep_data.append(res)
        print(f"Tol = {tol:8.1e} | Success: {res['success_rate']:5.1f}% | K_avg: {res['k_avg']:4.1f} | "
              f"Error E: {res['err_e_rms']:.4e} | Error Rot: {res['err_rot_rms']:.4e} | cond: {res['cond_med']:6.1f}")
              
    # -------------------------------------------------------------------------
    # PART 3: Publication Figure Generation
    # -------------------------------------------------------------------------
    h_vals = np.array([d['h_avg'] for d in conv_data])
    e_rms_without = np.array([d['res_without']['err_e_rms'] for d in conv_data])
    rot_rms_without = np.array([d['res_without']['err_rot_rms'] for d in conv_data])
    
    e_rms_with = np.array([d['res_with']['err_e_rms'] for d in conv_data])
    rot_rms_with = np.array([d['res_with']['err_rot_rms'] for d in conv_data])
    
    fig, (ax_conv, ax_tol) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.15, top=0.92, wspace=0.32)
    
    # Subplot (a)
    ax_conv.loglog(h_vals, rot_rms_without, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                   markeredgewidth=1.8, label=r"Curl RMS: Pure Quartic ($\mathrm{Tol}_{\mathrm{det}} \propto h^4$)")
    ax_conv.loglog(h_vals, rot_rms_with, 'd-', color=COLOR_TEAL, markerfacecolor=COLOR_TEAL,
                   label=r"Curl RMS: With Floor ($\mathrm{Tol}_{\mathrm{floor}} = 1.5 \times 10^{-5}$)")
                   
    ax_conv.loglog(h_vals, e_rms_without, '^--', color=COLOR_GRAY, markerfacecolor='white',
                   markeredgewidth=1.8, label=r"$\vec{E}$ RMS: Pure Quartic")
    ax_conv.loglog(h_vals, e_rms_with, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                   label=r"$\vec{E}$ RMS: With Floor ($\mathrm{Tol}_{\mathrm{floor}} = 1.5 \times 10^{-5}$)")
                   
    href = np.linspace(h_vals.min() * 0.85, h_vals.max() * 1.15, 60)
    c_h2 = e_rms_with[2] / (h_vals[2]**2)
    c_h1 = rot_rms_with[2] / (h_vals[2]**1)
    ax_conv.loglog(href, c_h2 * (href**2), 'k:', alpha=0.55, label=r"Theoretical $\mathcal{O}(h^2)$ slope")
    ax_conv.loglog(href, c_h1 * (href**1), 'gray', linestyle='-.', alpha=0.45, label=r"Theoretical $\mathcal{O}(h^1)$ slope")
    
    ax_conv.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{avg}}$ [m]")
    ax_conv.set_ylabel(r"RMS Interpolation Error")
    ax_conv.set_title(r"(a) Convergence Under Stochastic Perturbations")
    ax_conv.set_ylim(4.0e-4, 1.5)
    ax_conv.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_conv.legend(loc="lower right", fontsize=8.2, framealpha=0.92)
    
    # Subplot (b)
    tols_arr = np.array([d['tol'] for d in sweep_data])
    err_e_sw = np.array([d['err_e_rms'] for d in sweep_data])
    err_rot_sw = np.array([d['err_rot_rms'] for d in sweep_data])
    ks_sw = np.array([d['k_avg'] for d in sweep_data])
    
    ax_tol.set_xscale('log')
    ax_tol.set_yscale('log')
    
    l1 = ax_tol.plot(tols_arr, err_rot_sw, 'd-', color=COLOR_CRIMSON, linewidth=2.0,
                     markerfacecolor=COLOR_CRIMSON, label=r"Curl RMS $\|(\nabla \times \vec{E})_z - (\nabla \times \vec{E}^h)_z\|$")
    l2 = ax_tol.plot(tols_arr, err_e_sw, 'o-', color=COLOR_NAVY, linewidth=2.0,
                     markerfacecolor=COLOR_NAVY, label=r"Field RMS $\|\vec{E} - \vec{E}^h\|$")
                     
    ax_tol.set_xlabel(r"Determinant Threshold Criterion $\mathrm{Tol}_{\mathrm{det}}$")
    ax_tol.set_ylabel(r"RMS Error")
    ax_tol.set_title(r"(b) Stabilization vs. Tolerance Floor ($N = 4{,}192$)")
    ax_tol.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_tol.set_ylim(5.0e-4, 0.25)
    
    ax_k = ax_tol.twinx()
    l3 = ax_k.plot(tols_arr, ks_sw, 's--', color="#27ae60", linewidth=1.8,
                   markerfacecolor='white', markeredgewidth=1.8, label=r"Support Neighborhood $K_{\mathrm{avg}}$")
    ax_k.set_ylabel(r"Average Support Neighbors $K_{\mathrm{avg}}$", color="#27ae60")
    ax_k.tick_params(axis='y', labelcolor="#27ae60")
    ax_k.set_ylim(6.0, 16.0)
    
    ax_tol.axvspan(1.5e-5, 2.0e-4, color='#27ae60', alpha=0.12, label="Optimal Stabilization Regime")
    ax_tol.axvline(1.5e-5, color=COLOR_TEAL, linestyle=':', linewidth=1.6)
    ax_tol.text(1.7e-5, 0.12, r"$\mathrm{Tol}_{\mathrm{floor}} = 1.5 \times 10^{-5}$",
                fontsize=8.5, color=COLOR_TEAL, fontweight="bold")
                
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax_tol.legend(lines, labels, loc="upper right", fontsize=8.2, framealpha=0.92)
    
    fig.tight_layout()
    
    pdf_path = os.path.join(REPORTS_DIR, "figura_robustez_perturbacao_estocastica.pdf")
    png_path = os.path.join(REPORTS_DIR, "figura_robustez_perturbacao_estocastica.png")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    print(f"\nSaved figures to:\n  PDF: {pdf_path}\n  PNG: {png_path}")


# Backward compatibility aliases
avaliar_grade_P1 = evaluate_p1_grid
gerar_estudo_robustez_piso_pi = generate_pi_floor_robustness_study

if __name__ == "__main__":
    generate_pi_floor_robustness_study()
