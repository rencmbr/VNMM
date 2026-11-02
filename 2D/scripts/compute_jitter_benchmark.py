#!/usr/bin/env python3
"""
compute_jitter_benchmark.py
===========================
Computes eigenvalue convergence on 6 EABE grids (15, 22, 29, 36, 43, 49)
for VNMM-Gauss and VNMM-MLPG under nodal position jitter:
jitter_frac = 0.0 (0%), 0.25 (25%), 0.50 (50%).
"""

import os
import sys
import time
import json
import numpy as np
import scipy.linalg as la

DIR_2D = os.path.abspath("2D")
sys.path.insert(0, DIR_2D)
sys.path.insert(0, os.path.join(DIR_2D, "src"))
sys.path.insert(0, os.path.join(DIR_2D, "scripts"))

from cavity_mesh_2d import generate_cavity_mesh
from mlpg_subdomains import build_mlpg_subdomains_2d
from mlpg_assembler_2d import assemble_mlpg_matrices_2d
from eigen_solver_cavity_2d import assemble_vnmm_matrices_2d, apply_pec_boundary_conditions, ANALYTICAL_REFERENCE_MODES
from mlpg_cavity_eigenvalues_2d import solve_mlpg_cavity_eigenvalues

grids = [15, 22, 29, 36, 43, 49]
jitters = [0.0, 0.25, 0.50]
ref_kc = np.array([m[3] for m in ANALYTICAL_REFERENCE_MODES[:10]])

results = {
    'grids': grids,
    'jitters': jitters,
    'h': [float(np.pi / (N - 1)) for N in grids],
    'gauss': {},
    'mlpg': {}
}

print("=" * 90)
print("COMPUTING JITTER BENCHMARK (0%, 25%, 50%) FOR GAUSS AND MLPG")
print("=" * 90)

for j in jitters:
    j_key = f"{int(j*100)}%"
    results['gauss'][j_key] = []
    results['mlpg'][j_key] = []
    
    print(f"\n--- JITTER LEVEL: {j_key} (jitter_frac = {j}) ---")
    
    for N in grids:
        h = np.pi / (N - 1)
        h0 = np.pi / 20.0
        s_div_mlpg = float(15.0 * (h0 / h)**1.5)
        
        # 1. VNMM-Gauss
        t0 = time.time()
        coords_g, vectors_g, is_bnd_g = generate_cavity_mesh(
            Nx=N, Ny=N, Lx=np.pi, Ly=np.pi, interior_type="random", jitter_frac=j, seed=42
        )
        Kg, Mg = assemble_vnmm_matrices_2d(
            coords=coords_g, vectors=vectors_g, basis="P1", s_div=6.0,
            Ncx=N-1, Ncy=N-1, points_per_dir=3, support_mode="gauss_point"
        )
        t_asm_g = time.time() - t0
        
        Kg_red, Mg_red, int_idx_g = apply_pec_boundary_conditions(Kg, Mg, is_bnd_g)
        
        t0 = time.time()
        vals_g, _ = la.eigh(Kg_red.toarray(), Mg_red.toarray())
        t_sol_g = time.time() - t0
        
        pos_g = vals_g[vals_g > 0.05][:10]
        kc_g = np.sqrt(pos_g)
        err_kc_g = np.abs(kc_g - ref_kc[:len(kc_g)]) / ref_kc[:len(kc_g)] * 100.0
        mean_err_g = float(np.mean(err_kc_g))
        max_err_g = float(np.max(err_kc_g))
        
        # 2. VNMM-MLPG
        t0 = time.time()
        res_m = solve_mlpg_cavity_eigenvalues(
            Nx=N, Ny=N, Lx=np.pi, Ly=np.pi, alpha_s=0.80, s_div=s_div_mlpg,
            num_eigenvalues=10, interior_type="random", jitter_frac=j, seed=42
        )
        t_tot_m = time.time() - t0
        mean_err_m = float(res_m['mean_error_kc_pct'])
        max_err_m = float(res_m['max_error_kc_pct'])
        
        g_data = {
            'N': N, 'h': float(h), 'dofs': len(int_idx_g),
            'mean_err_kc': mean_err_g, 'max_err_kc': max_err_g,
            't_asm': t_asm_g, 't_sol': t_sol_g, 'nnz': int(Kg_red.nnz)
        }
        m_data = {
            'N': N, 'h': float(h), 'dofs': res_m['interior_dofs'],
            'mean_err_kc': mean_err_m, 'max_err_kc': max_err_m,
            's_div': s_div_mlpg,
            't_asm': res_m['time_assembly'], 't_sol': res_m['time_solve'],
            'spur_in_band': res_m['spur_in_band']
        }
        
        results['gauss'][j_key].append(g_data)
        results['mlpg'][j_key].append(m_data)
        
        print(f"Grid {N:2d}x{N:2d} (h={h:.4f}): Gauss err={mean_err_g:6.2f}%, MLPG err={mean_err_m:6.2f}% (s_div={s_div_mlpg:5.1f})")

# Compute convergence rates (slopes in log-log scale)
rates = {'gauss': {}, 'mlpg': {}}
log_h = np.log([np.pi / (N - 1) for N in grids])

for method in ['gauss', 'mlpg']:
    for j_key in ['0%', '25%', '50%']:
        errs = [d['mean_err_kc'] for d in results[method][j_key]]
        log_err = np.log(errs)
        # Linear regression slope: err ~ C * h^p => log(err) = p*log(h) + const
        slope, intercept = np.polyfit(log_h, log_err, 1)
        rates[method][j_key] = {
            'rate_p': float(slope),
            'errors': errs
        }
        print(f"Convergence rate {method.upper()} [{j_key}]: p = {slope:.2f}")

results['rates'] = rates

with open("2D/scripts/jitter_benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults successfully saved to 2D/scripts/jitter_benchmark_results.json")
