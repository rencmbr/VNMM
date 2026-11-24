#!/usr/bin/env python3
"""
study_alpha_gamma_and_beta2_2d.py
=================================
Option 1 diagnostic study reusing cached base operators from
study_fine_mesh_beta_cstab_2d.py to test:
1. Alternative orthogonal stabilization scaling laws:
       alpha_stab(h) = alpha_0 * (h / h0)^gamma
   with gamma in {0.0, 0.5, 1.0, 1.5}, alpha_0 in {0.02, 0.05, 0.10, 0.20}
   (and beta = 1.5), anchored at h0 = pi/20.
2. Higher divergence penalty exponents beta in {1.5, 2.0, 2.5}
   paired with: unstabilized (alpha_0=0), linear stabilization (gamma=1.0, alpha_0=0.10),
   and quadratic baseline (gamma=2.0, c_stab=70 -> alpha_0=0.175).

Tracks signed mode errors (TE10, TE01, TE11, TE22), mean cutoff error,
spurious mode counts, conditioning cond_2(K_red), and convergence rates.
Saves output to 2D/scripts/alpha_gamma_beta2_study.json.
"""

import os
import sys
import time
import json
import numpy as np
from scipy.sparse.linalg import eigsh

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.study_fine_mesh_beta_cstab_2d import (
    GRIDS,
    JITTERS,
    LX,
    H0,
    S_DIV_0,
    SPUR_THRESHOLD,
    ANALYTICAL_REFERENCE_MODES,
    build_base_operators,
    compute_sparse_cond2,
    compute_rates,
)

GAMMAS = [0.0, 0.5, 1.0, 1.5]
ALPHA0S_GAMMA = [0.02, 0.05, 0.10, 0.20]
HIGH_BETAS = [1.5, 2.0, 2.5]


def solve_with_signed(base_ops, s_div: float, alpha_stab: float, num_modes: int = 10):
    """Solves sparse eigenproblem and returns both unsigned mean and signed TE errors."""
    K_curl = base_ops["K_curl"]
    K_div = base_ops["K_div"]
    K_stab = base_ops["K_stab"]
    M = base_ops["M"]

    K_red = (K_curl + s_div * K_div + alpha_stab * K_stab).tocsr()
    K_red.sum_duplicates()
    n_dof = K_red.shape[0]

    selected = []
    spur_list = []
    vals_pos = np.array([])
    vecs_pos = np.empty((n_dof, 0))

    for k_req in [25, 45]:
        k_use = min(k_req, n_dof - 2)
        vals, vecs = eigsh(K_red, k=k_use, M=M, sigma=-0.05, which="LM", tol=1e-10)
        order = np.argsort(vals)
        vals = np.real(vals[order])
        vecs = np.real(vecs[:, order])

        mask = vals > 0.05
        vals_pos = vals[mask]
        vecs_pos = vecs[:, mask]

        physical, spurious = [], []
        for i in range(len(vals_pos)):
            v = vecs_pos[:, i]
            ec = float(v @ (K_curl @ v))
            ed = float(v @ (K_div @ v))
            r_div = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
            entry = {
                "kc": float(np.sqrt(max(vals_pos[i], 0.0))),
                "r_div": float(r_div),
            }
            if r_div <= SPUR_THRESHOLD:
                physical.append(entry)
            else:
                spurious.append(entry)

        selected = physical[:num_modes]
        spur_list = spurious
        if len(selected) == num_modes or k_use >= n_dof - 2:
            break

    spur_in_band = 0
    for i in range(min(num_modes, len(vals_pos))):
        v = vecs_pos[:, i]
        ec = float(v @ (K_curl @ v))
        ed = float(v @ (K_div @ v))
        r = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
        if r > SPUR_THRESHOLD:
            spur_in_band += 1

    ref_kc = np.array([r[3] for r in ANALYTICAL_REFERENCE_MODES[: len(selected)]])
    num_kc = np.array([m["kc"] for m in selected])
    signed_kc = (num_kc - ref_kc) / ref_kc * 100.0
    abs_kc = np.abs(signed_kc)
    cond_K = compute_sparse_cond2(K_red)

    return {
        "n_phys_found": len(selected),
        "mean_err_kc": float(np.mean(abs_kc)),
        "max_err_kc": float(np.max(abs_kc)),
        "signed_TE10": float(signed_kc[0]),
        "signed_TE11": float(signed_kc[2]),
        "signed_TE22": float(signed_kc[7]),
        "spur_in_band": int(spur_in_band),
        "max_r_div_phys": float(max(m["r_div"] for m in selected)),
        "cond_K": float(cond_K),
    }


def main():
    t0 = time.time()
    print("=" * 105)
    print("OPTION 1 STUDY: ALTERNATIVE gamma (alpha_stab) SCALINGS & HIGH beta in {1.5, 2.0, 2.5}")
    print("=" * 105)

    cache = {(N, j): build_base_operators(N, j) for j in JITTERS for N in GRIDS}

    # Part 1: gamma in {0.0, 0.5, 1.0, 1.5} x alpha_0 in {0.02, 0.05, 0.10, 0.20} at beta = 1.5
    gamma_study = {}
    gamma_summary = {}
    for j in JITTERS:
        j_key = f"{int(j*100)}%"
        gamma_study[j_key] = {}
        gamma_summary[j_key] = {}
        for gamma in GAMMAS:
            g_key = f"{gamma:.1f}"
            gamma_study[j_key][g_key] = {}
            gamma_summary[j_key][g_key] = {}
            for a0 in ALPHA0S_GAMMA:
                a_key = f"{a0:.2f}"
                entries = []
                for N in GRIDS:
                    h_nom = LX / float(N - 1)
                    s_div = float(S_DIV_0 * (H0 / h_nom) ** 1.5)
                    alpha_stab = float(a0 * (h_nom / H0) ** gamma)
                    res = solve_with_signed(cache[(N, j)], s_div=s_div, alpha_stab=alpha_stab)
                    res.update({"N": N, "h": float(h_nom), "s_div": s_div, "alpha_stab": alpha_stab})
                    entries.append(res)
                gamma_study[j_key][g_key][a_key] = entries
                gamma_summary[j_key][g_key][a_key] = compute_rates(entries)

    # Part 2: high beta in {1.5, 2.0, 2.5} with 3 stabilization presets
    presets = [
        ("unstab", 0.0, 2.0),
        ("lin_0.10", 0.10, 1.0),
        ("quad_c70", 0.175, 2.0),
    ]
    beta_study = {}
    beta_summary = {}
    for j in JITTERS:
        j_key = f"{int(j*100)}%"
        beta_study[j_key] = {}
        beta_summary[j_key] = {}
        for beta in HIGH_BETAS:
            b_key = f"{beta:.1f}"
            beta_study[j_key][b_key] = {}
            beta_summary[j_key][b_key] = {}
            for p_name, a0, gamma in presets:
                entries = []
                for N in GRIDS:
                    h_nom = LX / float(N - 1)
                    s_div = float(S_DIV_0 * (H0 / h_nom) ** beta)
                    alpha_stab = float(a0 * (h_nom / H0) ** gamma)
                    res = solve_with_signed(cache[(N, j)], s_div=s_div, alpha_stab=alpha_stab)
                    res.update({"N": N, "h": float(h_nom), "s_div": s_div, "alpha_stab": alpha_stab})
                    entries.append(res)
                beta_study[j_key][b_key][p_name] = entries
                beta_summary[j_key][b_key][p_name] = compute_rates(entries)

    out_path = os.path.join(CURRENT_DIR, "alpha_gamma_beta2_study.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "grids": GRIDS,
                "gammas": GAMMAS,
                "alpha0s": ALPHA0S_GAMMA,
                "high_betas": HIGH_BETAS,
                "gamma_study": gamma_study,
                "gamma_summary": gamma_summary,
                "beta_study": beta_study,
                "beta_summary": beta_summary,
            },
            f,
            indent=2,
        )
    print(f"Saved results to {out_path} in {time.time() - t0:.1f}s")

    for j_key in ["0%", "25%"]:
        print("\n" + "=" * 116)
        print(f"PART 1: alpha_stab = alpha_0 * (h/h0)^gamma (beta = 1.5, jitter = {j_key})")
        print("=" * 116)
        print("  gamma | alpha_0 |   15^2 |   29^2 |   49^2 |   65^2 |   81^2 | TE10(15->81)       | p_15_81 | p_36_81 | p(65->81) | spur")
        print("-" * 116)
        for gamma in GAMMAS:
            g_key = f"{gamma:.1f}"
            for a0 in ALPHA0S_GAMMA:
                a_key = f"{a0:.2f}"
                ents = gamma_study[j_key][g_key][a_key]
                summ = gamma_summary[j_key][g_key][a_key]
                e15, e29, e49, e65, e81 = ents[0], ents[2], ents[5], ents[6], ents[7]
                max_spur = max(e["spur_in_band"] for e in ents)
                te_str = f"{e15['signed_TE10']:+5.1f}%->{e81['signed_TE10']:+5.2f}%"
                print(
                    f"  {g_key:>5} | {a_key:>7} | "
                    f"{e15['mean_err_kc']:5.2f}% | {e29['mean_err_kc']:5.2f}% | "
                    f"{e49['mean_err_kc']:5.2f}% | {e65['mean_err_kc']:5.2f}% | {e81['mean_err_kc']:5.2f}% | "
                    f"{te_str:>18} | {summ['p_global8']:7.2f} | {summ['p_fine_36_81']:7.2f} | "
                    f"{summ['p_local']['65->81']:9.2f} | {max_spur:4d}"
                )

    for j_key in ["0%", "25%"]:
        print("\n" + "=" * 116)
        print(f"PART 2: HIGH DIVERGENCE EXPONENT beta in {{1.5, 2.0, 2.5}} (jitter = {j_key})")
        print("=" * 116)
        print("  beta |   preset |   49^2 |   65^2 |   81^2 | TE10(49->81)     | p_15_81 | p_36_81 | p(65->81) | cond_K(81^2) | spur")
        print("-" * 116)
        for beta in HIGH_BETAS:
            b_key = f"{beta:.1f}"
            for p_name, _, _ in presets:
                ents = beta_study[j_key][b_key][p_name]
                summ = beta_summary[j_key][b_key][p_name]
                e49, e65, e81 = ents[5], ents[6], ents[7]
                max_spur = max(e["spur_in_band"] for e in ents)
                te_str = f"{e49['signed_TE10']:+5.2f}%->{e81['signed_TE10']:+5.2f}%"
                print(
                    f"  {b_key:>4} | {p_name:>8} | "
                    f"{e49['mean_err_kc']:5.2f}% | {e65['mean_err_kc']:5.2f}% | {e81['mean_err_kc']:5.2f}% | "
                    f"{te_str:>16} | {summ['p_global8']:7.2f} | {summ['p_fine_36_81']:7.2f} | "
                    f"{summ['p_local']['65->81']:9.2f} | {e81['cond_K']:12.2e} | {max_spur:4d}"
                )


if __name__ == "__main__":
    main()
