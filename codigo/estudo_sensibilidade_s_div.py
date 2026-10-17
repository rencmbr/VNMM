#!/usr/bin/env python3
"""
estudo_sensibilidade_s_div.py
=============================
Realiza a varredura paramétrica da penalidade div-curl s_div para o caso base da cavidade ressonante
(Nx = Ny = 21, 361 DoFs ativos, 10x10 células de integração, 2x2 pontos de Gauss).
Gera os dados e a tabela LaTeX correspondente para inclusão no artigo.
"""

import os
import sys
import numpy as np
import scipy.linalg as la

DIRETORIO_CODIGO = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_CODIGO)
if DIRETORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRETORIO_RAIZ)

from src.malha_cavidade import gerar_malha_cavidade
from src.montador_vnmm import montar_matrizes_vnmm_2d
from src.eigen_solver_cavity import aplicar_condicao_pec, MODOS_ANALITICOS_REF


def executar_varredura_s_div(
    s_div_list=None,
    Nx=21,
    Ny=21,
    Ncx=10,
    Ncy=10,
    pontos_por_dir=2,
    base="P1"
):
    if s_div_list is None:
        s_div_list = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 20.0, 50.0]

    coords, vectors, is_boundary = gerar_malha_cavidade(
        Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi, tipo_interior="alternado"
    )

    # Monta K_curl e K_div separadamente para eficiência
    K0, M = montar_matrizes_vnmm_2d(
        coords=coords, vectors=vectors, base=base, s_div=0.0,
        Ncx=Ncx, Ncy=Ncy, Lx=np.pi, Ly=np.pi, pontos_por_dir=pontos_por_dir, modo_suporte="ponto_gauss"
    )
    K1, _ = montar_matrizes_vnmm_2d(
        coords=coords, vectors=vectors, base=base, s_div=1.0,
        Ncx=Ncx, Ncy=Ncy, Lx=np.pi, Ly=np.pi, pontos_por_dir=pontos_por_dir, modo_suporte="ponto_gauss"
    )

    K_curl_red, M_red, _ = aplicar_condicao_pec(K0, M, is_boundary)
    K_div_red = aplicar_condicao_pec(K1 - K0, M, is_boundary)[0]

    K_curl_mat = K_curl_red.toarray()
    K_div_mat = K_div_red.toarray()
    M_mat = M_red.toarray()

    analytical_kc = np.array([item[3] for item in MODOS_ANALITICOS_REF[:10]])

    resultados = []

    print(f"{'s_div':>6} | {'Spur In Band':>12} | {'Lowest Spur':>12} | {'Mean Err kc (%)':>15} | {'Max Err kc (%)':>14} | {'cond(K_red)':>12} | {'Max R_div (%)':>13}")
    print("-" * 95)

    for s in s_div_list:
        K_mat = K_curl_mat + s * K_div_mat
        vals, vecs = la.eigh(K_mat, M_mat)
        
        # Filtra autovalores nulos estáticos
        vals_pos = vals[vals > 0.05]
        vecs_pos = vecs[:, vals > 0.05]

        # Avalia R_div para os primeiros 35 modos
        spur_modes = []
        r_divs = []
        for i in range(min(35, len(vals_pos))):
            v = vecs_pos[:, i]
            e_c = v.T @ K_curl_mat @ v
            e_d = v.T @ K_div_mat @ v
            r = (e_d / e_c) * 100.0 if e_c > 1e-12 else 99999.0
            r_divs.append(r)
            if r > 10.0:  # Critério de modo espúrio com divergência residual dominante
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

        resultados.append({
            's_div': s,
            'spur_in_10': spur_in_10,
            'lowest_spur': lowest_spur_val,
            'mean_err_kc': np.mean(err_kc),
            'max_err_kc': np.max(err_kc),
            'cond_K': cond_K,
            'max_phys_r': max_phys_r
        })

    return resultados


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Varredura paramétrica de s_div no VNMM 2D")
    parser.add_argument("--densa", action="store_true", help="Executa apenas na malha densa (Nx=29, 729 DoFs, 3x3 Gauss)")
    parser.add_argument("--todas", action="store_true", help="Executa na malha base (361 DoFs) e na densa (729 DoFs)")
    args = parser.parse_args()

    if args.densa:
        print("\n>>> CASO REFINADO / DENSO: Nx = Ny = 29 (729 DoFs ativos, 14x14 células, 3x3 Gauss)")
        executar_varredura_s_div(Nx=29, Ny=29, Ncx=14, Ncy=14, pontos_por_dir=3)
    elif args.todas:
        print("\n>>> 1. CASO BASE: Nx = Ny = 21 (361 DoFs ativos, 10x10 células, 2x2 Gauss)")
        executar_varredura_s_div(Nx=21, Ny=21, Ncx=10, Ncy=10, pontos_por_dir=2)
        print("\n>>> 2. CASO DENSO: Nx = Ny = 29 (729 DoFs ativos, 14x14 células, 3x3 Gauss)")
        executar_varredura_s_div(Nx=29, Ny=29, Ncx=14, Ncy=14, pontos_por_dir=3)
    else:
        print("\n>>> CASO BASE: Nx = Ny = 21 (361 DoFs ativos, 10x10 células, 2x2 Gauss)")
        executar_varredura_s_div(Nx=21, Ny=21, Ncx=10, Ncy=10, pontos_por_dir=2)
