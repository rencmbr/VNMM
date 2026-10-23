#!/usr/bin/env python3
"""
s_div_sensitivity_study.py
==========================
Performs parametric sweep of div-curl penalty parameter s_div for resonant cavity base case
(Nx = Ny = 21, 361 active DoFs, 20x20 integration cells, 3x3 Gauss points).
Generates data and console summary corresponding to the manuscript table.
"""

import os
import sys
import numpy as np
import scipy.linalg as la

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.vnmm_assembler_2d import assemble_vnmm_matrices_2d
from src.eigen_solver_cavity_2d import apply_pec_boundary_conditions, ANALYTICAL_REFERENCE_MODES


def run_s_div_sweep(
    s_div_list=None,
    Nx=21,
    Ny=21,
    Ncx=20,
    Ncy=20,
    points_per_dir=3,
    basis="P1",
    # Compatibility aliases
    pontos_por_dir=None,
    base=None
):
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    if base is not None:
        basis = base

    if s_div_list is None:
        s_div_list = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 50.0]

    coords, vectors, is_boundary = generate_cavity_mesh(
        Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi, interior_type="alternating"
    )

    # Assemble K_curl and K_div separately for efficiency
    K0, M = assemble_vnmm_matrices_2d(
        coords=coords, vectors=vectors, basis=basis, s_div=0.0,
        Ncx=Ncx, Ncy=Ncy, Lx=np.pi, Ly=np.pi, points_per_dir=points_per_dir, support_mode="gauss_point"
    )
    K1, _ = assemble_vnmm_matrices_2d(
        coords=coords, vectors=vectors, basis=basis, s_div=1.0,
        Ncx=Ncx, Ncy=Ncy, Lx=np.pi, Ly=np.pi, points_per_dir=points_per_dir, support_mode="gauss_point"
    )

    K_curl_red, M_red, _ = apply_pec_boundary_conditions(K0, M, is_boundary)
    K_div_red = apply_pec_boundary_conditions(K1 - K0, M, is_boundary)[0]

    K_curl_mat = K_curl_red.toarray()
    K_div_mat = K_div_red.toarray()
    M_mat = M_red.toarray()

    analytical_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:10]])

    results = []

    print(f"{'s_div':>6} | {'Spur In Band':>12} | {'Lowest Spur':>12} | {'Mean Err kc (%)':>15} | {'Max Err kc (%)':>14} | {'cond(K_red)':>12} | {'Max R_div (%)':>13}")
    print("-" * 95)

    for s in s_div_list:
        K_mat = K_curl_mat + s * K_div_mat
        vals, vecs = la.eigh(K_mat, M_mat)
        
        # Filter static null eigenvalues
        vals_pos = vals[vals > 0.05]
        vecs_pos = vecs[:, vals > 0.05]

        # Evaluate R_div for first 35 modes
        spur_modes = []
        r_divs = []
        for i in range(min(35, len(vals_pos))):
            v = vecs_pos[:, i]
            e_c = v.T @ K_curl_mat @ v
            e_d = v.T @ K_div_mat @ v
            r = (e_d / e_c) * 100.0 if e_c > 1e-12 else 99999.0
            r_divs.append(r)
            if r > 10.0:  # Spurious mode criterion with dominant divergence residue
                spur_modes.append((i + 1, vals_pos[i]))

        spur_in_10 = sum(1 for m in spur_modes if m[0] <= 10)
        lowest_spur_val = spur_modes[0][1] if spur_modes else None

        first10 = vals_pos[:10]
        kc_num = np.sqrt(first10)
        err_kc = np.abs(kc_num - analytical_kc) / analytical_kc * 100.0

        phys_r_divs = [r_divs[i] for i in range(10) if r_divs[i] <= 10.0]
        max_phys_r = max(phys_r_divs) if phys_r_divs else 0.0

        cond_K = np.linalg.cond(K_mat) if s > 0 else np.nan

        spur_str = f"{lowest_spur_val:.2f}" if lowest_spur_val is not None else "> 50"
        cond_str = f"{cond_K:.2e}" if not np.isnan(cond_K) else "Singular"

        print(f"{s:6.1f} | {spur_in_10:12d} | {spur_str:>12} | {np.mean(err_kc):15.2f} | {np.max(err_kc):14.2f} | {cond_str:>12} | {max_phys_r:13.2f}")

        results.append({
            's_div': s,
            'spur_in_10': spur_in_10,
            'lowest_spur': lowest_spur_val,
            'mean_err_kc': np.mean(err_kc),
            'max_err_kc': np.max(err_kc),
            'cond_K': cond_K,
            'max_phys_r': max_phys_r
        })

    return results


# Backward compatibility alias
executar_varredura_s_div = run_s_div_sweep


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parametric sweep of s_div in 2D VNMM")
    parser.add_argument("--dense", "--densa", action="store_true", help="Run only on dense mesh (Nx=29, 729 DoFs, 3x3 Gauss)")
    parser.add_argument("--all", "--todas", action="store_true", help="Run on baseline mesh (361 DoFs) and dense mesh (729 DoFs)")
    args = parser.parse_args()

    if args.dense:
        print("\n>>> REFINED / DENSE CASE: Nx = Ny = 29 (729 active DoFs, 14x14 cells, 3x3 Gauss)")
        run_s_div_sweep(Nx=29, Ny=29, Ncx=14, Ncy=14, points_per_dir=3)
    elif args.all:
        print("\n>>> 1. BASELINE CASE: Nx = Ny = 21 (361 active DoFs, 10x10 cells, 2x2 Gauss)")
        run_s_div_sweep(Nx=21, Ny=21, Ncx=10, Ncy=10, points_per_dir=2)
        print("\n>>> 2. DENSE CASE: Nx = Ny = 29 (729 active DoFs, 14x14 cells, 3x3 Gauss)")
        run_s_div_sweep(Nx=29, Ny=29, Ncx=14, Ncy=14, points_per_dir=3)
    else:
        print("\n>>> BASELINE CASE: Nx = Ny = 21 (361 active DoFs, 10x10 cells, 2x2 Gauss)")
        run_s_div_sweep(Nx=21, Ny=21, Ncx=10, Ncy=10, points_per_dir=2)
