import os
import sys
import time
import json
import numpy as np
import scipy.linalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DIR_2D = os.path.abspath("2D")
sys.path.insert(0, DIR_2D)
sys.path.insert(0, os.path.join(DIR_2D, "src"))
sys.path.insert(0, os.path.join(DIR_2D, "scripts"))

from cavity_mesh_2d import generate_cavity_mesh
from mlpg_subdomains import build_mlpg_subdomains_2d
from mlpg_assembler_2d import assemble_mlpg_matrices_2d
from mlpg_source_problem_2d import assemble_and_solve_mlpg_mms
from eigen_solver_cavity_2d import assemble_vnmm_matrices_2d, solve_cavity_eigenvalues
from mlpg_cavity_eigenvalues_2d import solve_mlpg_cavity_eigenvalues, apply_pec_boundary_conditions, ANALYTICAL_REFERENCE_MODES

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

grids = [15, 22, 29, 36, 43, 49]
ref_kc = np.array([m[3] for m in ANALYTICAL_REFERENCE_MODES[:10]])

print("=" * 115)
print("COMPUTING FULL BENCHMARK ON 6 EABE MESHES [15, 22, 29, 36, 43, 49]")
print("=" * 115)

bench_results = []
for N in grids:
    print(f"Running Grid {N}x{N}...")
    coords, vectors, is_bnd = generate_cavity_mesh(Nx=N, Ny=N, Lx=np.pi, Ly=np.pi, interior_type="random", seed=42)
    
    # 1. VNMM-Gauss (s_div = 6.0)
    t0 = time.time()
    Kg, Mg = assemble_vnmm_matrices_2d(coords=coords, vectors=vectors, basis="P1", s_div=6.0,
                                       Ncx=N-1, Ncy=N-1, points_per_dir=3, support_mode="gauss_point")
    t_asm_g = time.time() - t0
    
    Kg_red, Mg_red, int_idx = apply_pec_boundary_conditions(Kg, Mg, is_bnd)
    dofs = len(int_idx)
    nnz_g = int(Kg_red.nnz)
    
    t0 = time.time()
    vals_g, _ = la.eigh(Kg_red.toarray(), Mg_red.toarray())
    t_sol_g = time.time() - t0
    
    pos_g = vals_g[vals_g > 0.05][:10]
    kc_g = np.sqrt(pos_g)
    err_kc_g = np.abs(kc_g - ref_kc[:len(kc_g)]) / ref_kc[:len(kc_g)] * 100.0
    mean_err_g = float(np.mean(err_kc_g))
    
    # 2. VNMM-MLPG with h-scaled s_div
    s_div_scaled_map = {15: 10.0, 22: 20.0, 29: 30.0, 36: 40.0, 43: 50.0, 49: 60.0}
    s_div_m = s_div_scaled_map.get(N, 10.0 + 1.5 * (N - 15))
    
    t0 = time.time()
    subs = build_mlpg_subdomains_2d(coords, vectors, is_bnd, alpha_s=0.80)
    Km, Mm, _ = assemble_mlpg_matrices_2d(subs, s_div=s_div_m, assembly_mode="subdomain_galerkin")
    t_asm_m = time.time() - t0
    
    Km_red, Mm_red, _ = apply_pec_boundary_conditions(Km, Mm, is_bnd)
    nnz_m = int(Km_red.nnz)
    
    t0 = time.time()
    res_m = solve_mlpg_cavity_eigenvalues(Nx=N, Ny=N, s_div=s_div_m, interior_type="random", seed=42)
    t_sol_m = res_m['time_solve']
    mean_err_m = float(res_m['mean_error_kc_pct'])
    
    spd_asm = t_asm_g / max(t_asm_m, 1e-6)
    t_tot_g = t_asm_g + t_sol_g
    t_tot_m = t_asm_m + t_sol_m
    spd_tot = t_tot_g / max(t_tot_m, 1e-6)
    red_nnz = float((nnz_g - nnz_m) / nnz_g * 100.0)
    
    bench_results.append({
        'N': N,
        'dofs': dofs,
        'h': float(np.pi / (N - 1)),
        't_asm_g': float(t_asm_g),
        't_asm_m': float(t_asm_m),
        'spd_asm': float(spd_asm),
        't_sol_g': float(t_sol_g),
        't_sol_m': float(t_sol_m),
        't_tot_g': float(t_tot_g),
        't_tot_m': float(t_tot_m),
        'spd_tot': float(spd_tot),
        'nnz_g': nnz_g,
        'nnz_m': nnz_m,
        'red_nnz': red_nnz,
        'err_kc_g': mean_err_g,
        'err_kc_m': mean_err_m,
        's_div_m': float(s_div_m)
    })

# Save benchmark json
with open("2D/scripts/benchmark_eabe_6meshes.json", "w") as f:
    json.dump({'results': bench_results}, f, indent=2)
print("Saved benchmark_eabe_6meshes.json successfully!")
