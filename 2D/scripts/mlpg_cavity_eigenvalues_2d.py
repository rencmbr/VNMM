#!/usr/bin/env python3
"""
mlpg_cavity_eigenvalues_2d.py
==============================
Fase 5 do Roteiro de Validacao do VNMM-MLPG 2D.
Validacao Espectral Fisica: Cavidade Ressonante PEC 2D.

Resolve o problema generalizado de autovalores:
    K_red * c = lambda * M_red * c
com K_red e M_red obtidos pelo metodo MLPG puramente sem malha (cell-free)
e condicoes de contorno de condutor eletrico perfeito (PEC, E_t = 0).

Caracteristicas:
1. Extracao e classificacao dos primeiros 10 modos fisicos (TE10, TE01, TE11, ...)
   comparados diretamente com a Tabela 4-1 de Luilly Ortiz (UFMG, 2023).
2. Analise de sensibilidade do parametro de penalizacao s_div e purificacao espectral.
3. Benchmarking de tempo de montagem: VNMM-MLPG (cell-free) vs. VNMM-Gauss (celulas).
"""

import os
import sys
import time
import numpy as np
import scipy.linalg as la
from typing import Dict, Any, List, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d
from src.mlpg_assembler_2d import assemble_mlpg_matrices_2d
from src.eigen_solver_cavity_2d import (
    apply_pec_boundary_conditions,
    ANALYTICAL_REFERENCE_MODES
)
from src.vnmm_assembler_2d import assemble_vnmm_matrices_2d


def solve_mlpg_cavity_eigenvalues(
    Nx: int = 21,
    Ny: int = 21,
    Lx: float = np.pi,
    Ly: float = np.pi,
    alpha_s: float = 0.80,
    s_div: float = 6.0,
    num_eigenvalues: int = 10,
    interior_type: str = "random",
    jitter_frac: float = 0.0,
    seed: int = 42,
    assembly_mode: str = "subdomain_galerkin",
    spur_threshold: float = 50.0,
    s_div_scaling: Optional[str] = None,
    h_ref: float = np.pi / 20.0,
    alpha_stab: float = 0.0
) -> Dict[str, Any]:
    """
    Resolve o problema espectral da cavidade ressonante retangular 2D com VNMM-MLPG.
    
    Parametros:
    - Nx, Ny: Numero de divisoes nodais em x e y.
    - Lx, Ly: Dimensoes da cavidade (padrao: pi x pi).
    - alpha_s: Fator do raio do subdominio local r_s = alpha_s * h.
    - s_div: Parametro de penalizacao do divergente (regularizacao).
    - num_eigenvalues: Quantidade de autovalores fisicos desejados.
    - interior_type: Padrao de orientacao dos diretores internos ('alternating', 'diagonal', 'random', 'crossed').
    - jitter_frac: Fracao de perturbacao estocastica das posicoes nodais (0.0 = regular).
    - seed: Semente para reprodutibilidade estocastica.
    - assembly_mode: 'subdomain_galerkin' (recomendado) ou 'row_petrov_galerkin'.
    - spur_threshold: Limiar percentual de R_div (%) para classificar modo como espurio.
    
    Retorna:
    - Dicionario com autovalores numericos, analiticos, erros em lambda e kc,
      tempos de montagem e solucao, e metricas de qualidade espectral.
    """
    t_start_mesh = time.time()
    coords, vectors, is_bnd = generate_cavity_mesh(
        Nx=Nx,
        Ny=Ny,
        Lx=Lx,
        Ly=Ly,
        interior_type=interior_type,
        jitter_frac=jitter_frac,
        seed=seed
    )
    t_mesh = time.time() - t_start_mesh

    # 1. Construcao dos subdominios locais MLPG
    t_start_sub = time.time()
    subdomains = build_mlpg_subdomains_2d(
        coords=coords,
        vectors=vectors,
        is_boundary=is_bnd,
        alpha_s=alpha_s
    )
    t_sub = time.time() - t_start_sub

    # 2. Montagem de K_curl, K_div e M
    t_start_asm = time.time()
    K0, M, _ = assemble_mlpg_matrices_2d(
        subdomains=subdomains,
        mu_r=1.0,
        epsilon_r=1.0,
        s_div=0.0,
        assembly_mode=assembly_mode,
        alpha_stab=alpha_stab
    )
    K1, _, _ = assemble_mlpg_matrices_2d(
        subdomains=subdomains,
        mu_r=1.0,
        epsilon_r=1.0,
        s_div=1.0,
        assembly_mode=assembly_mode,
        alpha_stab=alpha_stab
    )
    t_asm = time.time() - t_start_asm

    # 3. Aplicacao das condicoes de contorno PEC
    K_curl_red, M_red, int_indices = apply_pec_boundary_conditions(K0, M, is_bnd)
    K_div_red = apply_pec_boundary_conditions(K1 - K0, M, is_bnd)[0]

    K_curl_mat = K_curl_red.toarray()
    K_div_mat = K_div_red.toarray()
    M_mat = M_red.toarray()
    if s_div_scaling == "h_inverse":
        h_nom = max(Lx / (Nx - 1), Ly / (Ny - 1))
        effective_s_div = s_div * (h_ref / h_nom)
    else:
        effective_s_div = s_div

    K_mat = K_curl_mat + effective_s_div * K_div_mat

    # 4. Solucao do problema generalizado de autovalores
    t_start_sol = time.time()
    vals, vecs = la.eigh(K_mat, M_mat)
    t_sol = time.time() - t_start_sol

    # 5. Filtragem e classificacao fisica vs. espuria via R_div
    vals_pos = vals[vals > 0.05]
    vecs_pos = vecs[:, vals > 0.05]

    physical_modes = []
    spurious_modes = []

    for i in range(len(vals_pos)):
        v = vecs_pos[:, i]
        e_c = float(v.T @ K_curl_mat @ v)
        e_d = float(v.T @ K_div_mat @ v)
        r_div = (e_d / e_c) * 100.0 if e_c > 1e-12 else 99999.0
        
        mode_data = {
            'index': i + 1,
            'lambda': float(vals_pos[i]),
            'kc': float(np.sqrt(max(vals_pos[i], 0.0))),
            'e_curl': e_c,
            'e_div': e_d,
            'r_div': r_div
        }
        
        if r_div <= spur_threshold:
            physical_modes.append(mode_data)
        else:
            spurious_modes.append(mode_data)

    # Coleta dos primeiros num_eigenvalues modos fisicos
    selected_modes = physical_modes[:num_eigenvalues]
    num_vals = np.array([m['lambda'] for m in selected_modes])
    num_kc = np.array([m['kc'] for m in selected_modes])

    ref_data = ANALYTICAL_REFERENCE_MODES[:len(selected_modes)]
    ref_vals = np.array([r[2] for r in ref_data])
    ref_kc = np.array([r[3] for r in ref_data])
    ref_labels = [f"TE{r[0]}{r[1]}" for r in ref_data]

    rel_err_lambda = np.abs(num_vals - ref_vals) / ref_vals * 100.0
    rel_err_kc = np.abs(num_kc - ref_kc) / ref_kc * 100.0

    # Spurious modes dentro da faixa dos 10 primeiros autovalores brutos
    raw_first10 = vals_pos[:num_eigenvalues]
    raw_r_divs = []
    for i in range(len(raw_first10)):
        v = vecs_pos[:, i]
        ec = float(v.T @ K_curl_mat @ v)
        ed = float(v.T @ K_div_mat @ v)
        r = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
        raw_r_divs.append(r)
        
    spur_in_band = sum(1 for r in raw_r_divs if r > spur_threshold)
    lowest_spur_val = spurious_modes[0]['lambda'] if spurious_modes else None

    cond_K = float(np.linalg.cond(K_mat)) if s_div > 0 else np.nan

    return {
        'Nx': Nx,
        'Ny': Ny,
        'total_nodes': len(coords),
        'interior_dofs': len(int_indices),
        'boundary_nodes': int(np.sum(is_bnd)),
        's_div': effective_s_div,
        's_div_base': s_div,
        'alpha_s': alpha_s,
        'time_mesh': t_mesh,
        'time_subdomains': t_sub,
        'time_assembly': t_asm,
        'time_solve': t_sol,
        'total_time': t_mesh + t_sub + t_asm + t_sol,
        'cond_K': cond_K,
        'modes': selected_modes,
        'numerical_eigenvalues': num_vals,
        'analytical_eigenvalues': ref_vals,
        'numerical_kc': num_kc,
        'analytical_kc': ref_kc,
        'mode_labels': ref_labels,
        'rel_errors_lambda_pct': rel_err_lambda,
        'rel_errors_kc_pct': rel_err_kc,
        'mean_error_kc_pct': float(np.mean(rel_err_kc)) if len(rel_err_kc) > 0 else 0.0,
        'max_error_kc_pct': float(np.max(rel_err_kc)) if len(rel_err_kc) > 0 else 0.0,
        'num_physical_found': len(physical_modes),
        'num_spurious_found': len(spurious_modes),
        'spur_in_band': spur_in_band,
        'lowest_spur_val': lowest_spur_val
    }


def run_mlpg_s_div_sensitivity_study(
    Nx: int = 21,
    Ny: int = 21,
    s_div_list: Optional[List[float]] = None,
    alpha_s: float = 0.80,
    r_div_threshold: float = 50.0,
    interior_type: str = "random",
    seed: int = 42,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """
    Executa uma varredura parametrica de s_div para avaliar o deslocamento de modos espurios no VNMM-MLPG.
    
    Criterio de classificacao de modos espurios:
    Modos verdadeiramente espurios oriundos do vazamento do nucleo discreto de rotacional (gradientes)
    apresentam energia de divergencia dominante em relacao a energia de rotacional (R_div >> 1,
    adotando-se o limiar R_div > r_div_threshold = 50.0% em consonancia com EABE).
    Modos fisicos solenoidais apresentam R_div << 1 (tipicamente < 3% em s_div calibrado).
    """
    if s_div_list is None:
        s_div_list = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    coords, vectors, is_bnd = generate_cavity_mesh(
        Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi, interior_type=interior_type, seed=seed
    )
    subdomains = build_mlpg_subdomains_2d(
        coords=coords, vectors=vectors, is_boundary=is_bnd, alpha_s=alpha_s
    )

    K0, M, _ = assemble_mlpg_matrices_2d(
        subdomains=subdomains, s_div=0.0, assembly_mode="subdomain_galerkin"
    )
    K1, _, _ = assemble_mlpg_matrices_2d(
        subdomains=subdomains, s_div=1.0, assembly_mode="subdomain_galerkin"
    )

    K_curl_red, M_red, int_indices = apply_pec_boundary_conditions(K0, M, is_bnd)
    K_div_red = apply_pec_boundary_conditions(K1 - K0, M, is_bnd)[0]

    K_curl_mat = K_curl_red.toarray()
    K_div_mat = K_div_red.toarray()
    M_mat = M_red.toarray()

    analytical_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:10]])

    results = []

    if verbose:
        print(f"\n{'='*98}")
        print(f"ESTUDO DE SENSIBILIDADE DO PARAMETRO s_div (VNMM-MLPG)")
        print(f"Grade: {Nx}x{Ny} ({len(int_indices)} DoFs internos livres) | Subdominio alpha_s = {alpha_s} | Limiar Spur: R_div > {r_div_threshold}%")
        print(f"{'='*98}")
        print(f"{'s_div':>6} | {'Spur In Band':>12} | {'Lowest Spur':>12} | {'Mean Err kc (%)':>15} | {'Max Err kc (%)':>14} | {'cond(K_red)':>12} | {'Max R_div (%)':>13}")
        print("-" * 98)

    for s in s_div_list:
        K_mat = K_curl_mat + s * K_div_mat
        vals, vecs = la.eigh(K_mat, M_mat)

        vals_pos = vals[vals > 0.05]
        vecs_pos = vecs[:, vals > 0.05]

        spur_modes = []
        r_divs = []
        for i in range(len(vals_pos)):
            v = vecs_pos[:, i]
            e_c = v.T @ K_curl_mat @ v
            e_d = v.T @ K_div_mat @ v
            r = (e_d / e_c) * 100.0 if e_c > 1e-12 else 99999.0
            r_divs.append(r)
            if r > r_div_threshold:
                spur_modes.append((i + 1, vals_pos[i]))

        spur_in_10 = sum(1 for m in spur_modes if m[0] <= 10)
        lowest_spur_val = spur_modes[0][1] if spur_modes else None

        first10 = vals_pos[:10]
        kc_num = np.sqrt(first10)
        err_kc = np.abs(kc_num - analytical_kc) / analytical_kc * 100.0

        r_10 = r_divs[:10]
        max_band_r = max(r_10) if r_10 else 0.0
        phys_r_divs = [r_divs[i] for i in range(10) if r_divs[i] <= 10.0]
        max_phys_r = max(phys_r_divs) if phys_r_divs else None

        cond_K = np.linalg.cond(K_mat) if s > 0 else np.nan
        spur_str = f"{lowest_spur_val:.2f}" if lowest_spur_val is not None else "> 50"
        cond_str = f"{cond_K:.2e}" if not np.isnan(cond_K) else "Singular"
        r_str = f"{max_band_r:.1f}" if max_band_r < 1e4 else "> 10^4"

        if verbose:
            print(f"{s:6.1f} | {spur_in_10:12d} | {spur_str:>12} | {np.mean(err_kc):15.2f} | {np.max(err_kc):14.2f} | {cond_str:>12} | {r_str:>13}")

        results.append({
            's_div': s,
            'spur_in_10': spur_in_10,
            'spur_in_band': spur_in_10,
            'lowest_spur': lowest_spur_val,
            'mean_err_kc': float(np.mean(err_kc)),
            'max_err_kc': float(np.max(err_kc)),
            'cond_K': cond_K,
            'max_band_r': max_band_r,
            'max_phys_r': max_phys_r if max_phys_r is not None else 0.0
        })

    return results


def run_mlpg_vs_gauss_benchmark(
    Nx: int = 21,
    Ny: int = 21,
    s_div: float = 6.0,
    interior_type: str = "random",
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executa comparacao de desempenho (benchmarking) entre VNMM-MLPG (cell-free)
    e VNMM-Gauss (celulas cartesianas de fundo).
    """
    coords, vectors, is_bnd = generate_cavity_mesh(
        Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi, interior_type=interior_type, seed=seed
    )

    # 1. VNMM-Gauss (celulas de fundo 20x20, 3x3 pontos de Gauss)
    t0_gauss = time.time()
    K_gauss, M_gauss = assemble_vnmm_matrices_2d(
        coords=coords,
        vectors=vectors,
        basis="P1",
        s_div=s_div,
        Ncx=Nx - 1,
        Ncy=Ny - 1,
        points_per_dir=3,
        support_mode="gauss_point"
    )
    t_gauss = time.time() - t0_gauss

    # 2. VNMM-MLPG (cell-free, N subdominios circulares analiticos)
    t0_mlpg = time.time()
    subdomains = build_mlpg_subdomains_2d(
        coords=coords, vectors=vectors, is_boundary=is_bnd, alpha_s=0.80
    )
    K_mlpg, M_mlpg, _ = assemble_mlpg_matrices_2d(
        subdomains=subdomains,
        s_div=s_div,
        assembly_mode="subdomain_galerkin"
    )
    t_mlpg = time.time() - t0_mlpg

    speedup = t_gauss / t_mlpg if t_mlpg > 0 else np.nan

    bench = {
        'Nx': Nx,
        'Ny': Ny,
        'total_nodes': len(coords),
        'gauss_cells': (Nx - 1) * (Ny - 1),
        'gauss_eval_points': (Nx - 1) * (Ny - 1) * 9,
        'time_gauss_sec': t_gauss,
        'time_mlpg_sec': t_mlpg,
        'speedup_factor': speedup,
        'nnz_K_gauss': K_gauss.nnz,
        'nnz_K_mlpg': K_mlpg.nnz,
        'nnz_M_gauss': M_gauss.nnz,
        'nnz_M_mlpg': M_mlpg.nnz
    }

    if verbose:
        print(f"\n{'='*75}")
        print(f"BENCHMARK COMPUTACIONAL: VNMM-MLPG (CELL-FREE) vs. VNMM-GAUSS (CELULAS)")
        print(f"{'='*75}")
        print(f"Dimensoes da Grade:               {Nx} x {Ny} ({len(coords)} nos totais)")
        print(f"Celulas de Fundo Gauss:           {bench['gauss_cells']} celulas (3x3 = {bench['gauss_eval_points']} pontos)")
        print(f"Subdominios Locais MLPG:          {len(subdomains)} discos analiticos Omega_s^I")
        print(f"Tempo Montagem VNMM-Gauss:        {t_gauss:.4f} s")
        print(f"Tempo Montagem VNMM-MLPG:         {t_mlpg:.4f} s")
        print(f"Fator de Aceleracao (Speedup):    {speedup:.2f}x mais rapido!")
        print(f"Elementos Nao-Nulos K (Gauss):    {K_gauss.nnz}")
        print(f"Elementos Nao-Nulos K (MLPG):     {K_mlpg.nnz}")
        print(f"{'='*75}")

    return bench


def main():
    print("=" * 85)
    print("VNMM-MLPG 2D - FASE 5: VALIDACAO ESPECTRAL DA CAVIDADE RESSONANTE PEC 2D")
    print("=" * 85)

    # 1. Caso Base: 21x21, s_div=6.0
    res = solve_mlpg_cavity_eigenvalues(
        Nx=21,
        Ny=21,
        s_div=6.0,
        alpha_s=0.80,
        num_eigenvalues=10
    )

    print(f"\nResultados do Caso Base (Nx=Ny=21, s_div=6.0, alpha_s=0.80):")
    print(f"DoFs Internos Ativos: {res['interior_dofs']}")
    print(f"Tempo Total:          {res['total_time']:.4f} s (Montagem: {res['time_assembly']:.4f} s, Solucao: {res['time_solve']:.4f} s)")
    print(f"Condicionamento K:    {res['cond_K']:.2e}")
    print(f"Modos Fisicos:        {res['num_physical_found']} encontrados (Espurios na banda <= 10: {res['spur_in_band']})")
    print()
    print(f"{'Modo':^6} | {'lambda_num':>11} | {'lambda_ref':>11} | {'kc_num':>9} | {'kc_ref':>9} | {'Erro kc (%)':>11} | {'R_div (%)':>9}")
    print("-" * 78)

    for i in range(len(res['numerical_eigenvalues'])):
        lbl = res['mode_labels'][i]
        l_num = res['numerical_eigenvalues'][i]
        l_ref = res['analytical_eigenvalues'][i]
        k_num = res['numerical_kc'][i]
        k_ref = res['analytical_kc'][i]
        err_k = res['rel_errors_kc_pct'][i]
        r_d = res['modes'][i]['r_div']
        print(f"{lbl:^6} | {l_num:11.4f} | {l_ref:11.4f} | {k_num:9.4f} | {k_ref:9.4f} | {err_k:10.2f}% | {r_d:8.2f}%")

    print("-" * 78)
    print(f"Erro Relativo Medio em kc: {res['mean_error_kc_pct']:.2f}%")
    print(f"Erro Relativo Maximo em kc: {res['max_error_kc_pct']:.2f}%")

    # 2. Estudo de Sensibilidade de s_div
    run_mlpg_s_div_sensitivity_study(Nx=21, Ny=21, alpha_s=0.80, verbose=True)

    # 3. Benchmarking de Desempenho
    run_mlpg_vs_gauss_benchmark(Nx=21, Ny=21, s_div=6.0, verbose=True)


def run_mlpg_cavity_convergence_study(
    grid_sizes: Optional[List[int]] = None,
    s_div_base: float = 6.0,
    s_div_scaling: Optional[str] = None,
    alpha_s: float = 0.80,
    interior_type: str = "random",
    seed: int = 42,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executa estudo de convergencia dos autovalores da cavidade PEC através de uma sequencia de malhas.
    Permite avaliar tanto s_div fixo (padrao) quanto s_div com escalonamento assintotico h_inverse.
    """
    if grid_sizes is None:
        grid_sizes = [15, 21, 29, 36, 43]

    results = []
    if verbose:
        print("\n" + "=" * 90)
        print("ESTUDO DE CONVERGENCIA ESPECTRAL DA CAVIDADE (VNMM-MLPG)")
        print(f"s_div base = {s_div_base} | Scaling = {s_div_scaling} | alpha_s = {alpha_s} | Diretores: {interior_type}")
        print("=" * 90)
        print(f"{'Mesh':>8} | {'h [m]':>8} | {'s_div':>7} | {'Mean Err kc (%)':>16} | {'Max Err kc (%)':>15} | {'Spur in band':>13}")
        print("-" * 90)

    for N in grid_sizes:
        h = np.pi / (N - 1)
        res = solve_mlpg_cavity_eigenvalues(
            Nx=N, Ny=N,
            s_div=s_div_base,
            s_div_scaling=s_div_scaling,
            alpha_s=alpha_s,
            interior_type=interior_type,
            seed=seed
        )
        mean_err = res['mean_error_kc_pct']
        max_err = res['max_error_kc_pct']
        s_used = res['s_div']
        spur = res['spur_in_band']
        grid_str = f"{N}x{N}"
        if verbose:
            print(f"{grid_str:>8} | {h:8.4f} | {s_used:7.1f} | {mean_err:16.2f} | {max_err:15.2f} | {spur:13d}")

        results.append({
            'N': N,
            'h': h,
            's_div': s_used,
            'mean_error_kc': mean_err,
            'max_error_kc': max_err,
            'spur_in_band': spur,
            'res': res
        })

    return {'grid_sizes': grid_sizes, 'results': results}


if __name__ == "__main__":
    main()
