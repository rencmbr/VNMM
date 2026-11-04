#!/usr/bin/env python3
"""
compute_alpha_jitter_study.py
=============================
Computes cavity eigenvalue errors across all 6 EABE grids (15, 22, 29, 36, 43, 49)
varying alpha_s in [0.70, 0.80, 0.90, 1.00] for jitter levels 0%, 25%, 50%.
"""

import os
import sys
import time
import json
import numpy as np

DIR_2D = os.path.abspath("2D")
sys.path.insert(0, DIR_2D)
sys.path.insert(0, os.path.join(DIR_2D, "src"))
sys.path.insert(0, os.path.join(DIR_2D, "scripts"))

from mlpg_cavity_eigenvalues_2d import solve_mlpg_cavity_eigenvalues

grids = [15, 22, 29, 36, 43, 49]
alphas = [0.70, 0.80, 0.90, 1.00]
jitters = [0.0, 0.25, 0.50]

results = {
    'grids': grids,
    'alphas': alphas,
    'jitters': jitters,
    'h': [float(np.pi / (N - 1)) for N in grids],
    'data': {},
    'rates': {}
}

print("=" * 100)
print("COMPUTING FULL ALPHA_S X JITTER X MESH REFINEMENT STUDY")
print("=" * 100)

for jit in jitters:
    j_key = f"{int(jit*100)}%"
    results['data'][j_key] = {}
    results['rates'][j_key] = {}
    
    print(f"\n=================== JITTER LEVEL: {j_key} ===================")
    
    for a in alphas:
        a_key = f"{a:.2f}"
        results['data'][j_key][a_key] = []
        print(f"\n--- alpha_s = {a:.2f} ---")
        
        for N in grids:
            h = np.pi / (N - 1)
            h0 = np.pi / 20.0
            s_div = float(15.0 * (h0 / h)**1.5)
            
            t0 = time.time()
            res = solve_mlpg_cavity_eigenvalues(
                Nx=N, Ny=N, Lx=np.pi, Ly=np.pi,
                alpha_s=a, s_div=s_div, num_eigenvalues=10,
                interior_type="random", jitter_frac=jit, seed=42
            )
            elapsed = time.time() - t0
            
            mean_err = float(res['mean_error_kc_pct'])
            max_err = float(res['max_error_kc_pct'])
            spur = res['spur_in_band']
            cond = float(res['cond_K'])
            
            results['data'][j_key][a_key].append({
                'N': N, 'h': float(h),
                'mean_err_kc': mean_err, 'max_err_kc': max_err,
                'spur': spur, 'cond_K': cond, 'time': elapsed
            })
            
            print(f"Grid {N:2d}x{N:2d}: mean_err={mean_err:6.2f}%, max_err={max_err:6.2f}%, spur={spur:d}, cond={cond:.1e} ({elapsed:.2f}s)")
            
        # Compute convergence rate p
        log_h = np.log([np.pi / (N - 1) for N in grids])
        log_err = np.log([d['mean_err_kc'] for d in results['data'][j_key][a_key]])
        slope, _ = np.polyfit(log_h, log_err, 1)
        results['rates'][j_key][a_key] = float(slope)
        print(f"--> Convergence rate alpha_s={a:.2f} [{j_key}]: p = {slope:.2f}")

with open("2D/scripts/alpha_jitter_study_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults successfully saved to 2D/scripts/alpha_jitter_study_results.json")
