#!/usr/bin/env python3
"""
study_eigenfunction_accuracy_2d.py
==================================
Evaluates modal eigenfunction accuracy of the calibrated linear VNMM-MLPG
formulation (beta = 1.5, gamma = 1, c_stab = 4.0, alpha_s = 0.80, seed = 42)
across 8 grid resolutions (15x15 to 81x81) under 0% and 25% nodal jitter.

Computes:
1. Multiplicity-weighted broken-L2 field error (eps_{L2, h}), broken H(curl)
   semi-norm error (eps_{curl, h}), and broken-norm Modal Assurance Criterion
   (MAC) for normalized cavity modes TE10 (projected onto the 2D degenerate
   fundamental subspace), TE11, and TE22 (MAC-matched across lowest physical
   modes to handle coarse-grid mode-order swaps).
2. Head-to-head N = 49 comparison against unstabilized VNMM-MLPG (c_stab = 0)
   and calibrated VNMM-Gauss (3x3 cell quadrature, s_div = 6.0).
3. Reconstructed global modal field E_global(x) (Eq. 19) on a 75x75 visualization
   grid at N = 81 for TE10, TE11, and TE22 (0% and 25% jitter).

Saves structured results to 2D/scripts/eigenfunction_accuracy_study.json.
"""

import os
import sys
import json
import time
import numpy as np
from scipy.sparse.linalg import eigsh

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.mlpg_assembler_2d import get_disk_quadrature_points_2d
from src.eigen_solver_cavity_2d import assemble_vnmm_matrices_2d
from scripts.study_fine_mesh_beta_cstab_2d import (
    GRIDS,
    JITTERS,
    ALPHA_S,
    SEED,
    LX,
    H0,
    S_DIV_0,
    SPUR_THRESHOLD,
    build_base_operators,
)

MODE_SPECS = [
    ("TE10", 1, 0),
    ("TE11", 1, 1),
    ("TE22", 2, 2),
]


def exact_te_field(pts: np.ndarray, n: int, m: int) -> np.ndarray:
    """Analytical PEC cavity TE_nm electric field on [0, pi]^2."""
    x = pts[:, 0]
    y = pts[:, 1]
    ex = -float(m) * np.cos(n * x) * np.sin(m * y)
    ey = float(n) * np.sin(n * x) * np.cos(m * y)
    return np.column_stack([ex, ey])


def exact_te_curl(pts: np.ndarray, n: int, m: int) -> np.ndarray:
    """Analytical z-component of curl(E_nm) on [0, pi]^2."""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(n * n + m * m) * np.cos(n * x) * np.cos(m * y)


def build_quad_cache(subs, n_r: int = 4, n_theta: int = 12):
    """Precomputes quadrature points, weights, shape functions, and curls per disk."""
    cache = []
    for sub in subs:
        pts, wts = get_disk_quadrature_points_2d(
            sub.center,
            sub.radius_s,
            n_r=n_r,
            n_theta=n_theta,
            domain_bounds=getattr(sub, "domain_bounds", None),
        )
        n_all = sub.eval_shape_functions(pts)
        curls = sub.eval_shape_curl()
        cache.append((pts, wts, n_all, curls, sub.stencil_nodes))
    return cache


def eval_broken_field_from_cache(e_full: np.ndarray, quad_cache):
    """Evaluates local P1 vector fields and constant curls on each subdomain."""
    e_list = []
    c_list = []
    for _, _, n_all, curls, st in quad_cache:
        e_st = e_full[st]
        e_list.append(np.einsum("mqk,k->mq", n_all, e_st))
        c_list.append(float(np.dot(curls, e_st)))
    return e_list, c_list


def broken_l2_inner(e1_list, e2_list, quad_cache) -> float:
    """Multiplicity-weighted broken L2 inner product summed over all active disks."""
    total = 0.0
    for e1, e2, (_, wts, _, _, _) in zip(e1_list, e2_list, quad_cache):
        total += float(np.sum(wts * np.sum(e1 * e2, axis=1)))
    return total


def broken_curl_metrics(c_h_list, c_ex_list, quad_cache):
    """Returns broken L2 norm of (curl E_h - curl E_ex) divided by ||curl E_ex||."""
    num = 0.0
    den = 0.0
    for c_h, c_ex, (_, wts, _, _, _) in zip(c_h_list, c_ex_list, quad_cache):
        num += float(np.sum(wts * (c_h - c_ex) ** 2))
        den += float(np.sum(wts * (c_ex ** 2)))
    return float(np.sqrt(num / max(den, 1e-30)))


def normalized_exact_broken_modes(quad_cache):
    """Builds unit broken-L2 normalized exact fields and curls for TE10, TE11, TE22."""
    ex_e = {}
    ex_c = {}
    for name, n, m in MODE_SPECS:
        e_l = [exact_te_field(pts, n, m) for pts, _, _, _, _ in quad_cache]
        c_l = [exact_te_curl(pts, n, m) for pts, _, _, _, _ in quad_cache]
        nrm = float(np.sqrt(broken_l2_inner(e_l, e_l, quad_cache)))
        ex_e[name] = [arr / nrm for arr in e_l]
        ex_c[name] = [arr / nrm for arr in c_l]
    return ex_e, ex_c


def extract_physical_modes(K_red, M_red, K_curl, K_div, int_idx, n_total: int, k_req: int = 35):
    """Solves Shift-and-Invert Arnoldi and returns physical modes sorted by lambda."""
    n_dof = K_red.shape[0]
    k_use = min(k_req, n_dof - 2)
    vals, vecs = eigsh(K_red, k=k_use, M=M_red, sigma=-0.05, which="LM", tol=1e-10)
    order = np.argsort(vals)
    vals = np.real(vals[order])
    vecs = np.real(vecs[:, order])

    phys = []
    for i in range(len(vals)):
        if vals[i] <= 0.05:
            continue
        v = vecs[:, i]
        if K_curl is not None and K_div is not None:
            ec = float(v @ (K_curl @ v))
            ed = float(v @ (K_div @ v))
            r_div = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
            if r_div > SPUR_THRESHOLD:
                continue
        else:
            r_div = 0.0
        e_full = np.zeros(n_total, dtype=float)
        e_full[int_idx] = v
        phys.append({
            "rank_phys": len(phys) + 1,
            "lam": float(vals[i]),
            "kc": float(np.sqrt(vals[i])),
            "r_div": float(r_div),
            "e_full": e_full,
        })
        if len(phys) >= 16:
            break
    return phys


def evaluate_three_modes(phys, quad_cache, ex_e, ex_c):
    """
    Computes normalized broken-L2 error, broken-H(curl) error, and MAC for
    TE10 (2D subspace alignment over modes 1..2), TE11, and TE22 (MAC-matched).
    Also returns aligned full DoF vectors for field reconstruction.
    """
    broken_e = []
    broken_c = []
    for item in phys:
        el, cl = eval_broken_field_from_cache(item["e_full"], quad_cache)
        nrm = float(np.sqrt(broken_l2_inner(el, el, quad_cache)))
        broken_e.append([arr / nrm for arr in el])
        broken_c.append([c / nrm for c in cl])

    results = {}
    aligned_dofs = {}

    # 1. TE10 via optimal linear combination in the 2D fundamental doublet
    e0, e1 = broken_e[0], broken_e[1]
    c0, c1 = broken_c[0], broken_c[1]
    raw0_e, _ = eval_broken_field_from_cache(phys[0]["e_full"], quad_cache)
    raw1_e, _ = eval_broken_field_from_cache(phys[1]["e_full"], quad_cache)
    scale0 = 1.0 / float(np.sqrt(broken_l2_inner(raw0_e, raw0_e, quad_cache)))
    scale1 = 1.0 / float(np.sqrt(broken_l2_inner(raw1_e, raw1_e, quad_cache)))

    G = np.array([
        [broken_l2_inner(e0, e0, quad_cache), broken_l2_inner(e0, e1, quad_cache)],
        [broken_l2_inner(e1, e0, quad_cache), broken_l2_inner(e1, e1, quad_cache)],
    ])
    rhs = np.array([
        broken_l2_inner(e0, ex_e["TE10"], quad_cache),
        broken_l2_inner(e1, ex_e["TE10"], quad_cache),
    ])
    alpha = np.linalg.solve(G, rhs)
    eh_10 = [alpha[0] * a + alpha[1] * b for a, b in zip(e0, e1)]
    ch_10 = [alpha[0] * a + alpha[1] * b for a, b in zip(c0, c1)]
    e_dof_10 = alpha[0] * scale0 * phys[0]["e_full"] + alpha[1] * scale1 * phys[1]["e_full"]

    nrm_10 = float(np.sqrt(broken_l2_inner(eh_10, eh_10, quad_cache)))
    eh_10 = [arr / nrm_10 for arr in eh_10]
    ch_10 = [c / nrm_10 for c in ch_10]
    e_dof_10 = e_dof_10 / nrm_10
    ip_10 = broken_l2_inner(eh_10, ex_e["TE10"], quad_cache)
    if ip_10 < 0.0:
        eh_10 = [-arr for arr in eh_10]
        ch_10 = [-c for c in ch_10]
        e_dof_10 = -e_dof_10
        ip_10 = -ip_10

    results["TE10"] = {
        "err_L2_pct": float(100.0 * np.sqrt(max(2.0 * (1.0 - ip_10), 0.0))),
        "err_curl_pct": float(100.0 * broken_curl_metrics(ch_10, ex_c["TE10"], quad_cache)),
        "mac": float(ip_10 * ip_10),
        "matched_rank": "1--2 (2D proj.)",
        "lam": float(0.5 * (phys[0]["lam"] + phys[1]["lam"])),
    }
    aligned_dofs["TE10"] = e_dof_10

    # 2. Simple modes TE11 and TE22 via maximum broken-L2 MAC
    for name in ("TE11", "TE22"):
        best_idx = -1
        best_abs_ip = -1.0
        best_sign = 1.0
        for idx, el in enumerate(broken_e):
            ip = broken_l2_inner(el, ex_e[name], quad_cache)
            if abs(ip) > best_abs_ip:
                best_abs_ip = abs(ip)
                best_idx = idx
                best_sign = 1.0 if ip >= 0.0 else -1.0

        raw_e, _ = eval_broken_field_from_cache(phys[best_idx]["e_full"], quad_cache)
        sc = best_sign / float(np.sqrt(broken_l2_inner(raw_e, raw_e, quad_cache)))
        ch = [best_sign * c for c in broken_c[best_idx]]

        results[name] = {
            "err_L2_pct": float(100.0 * np.sqrt(max(2.0 * (1.0 - best_abs_ip), 0.0))),
            "err_curl_pct": float(100.0 * broken_curl_metrics(ch, ex_c[name], quad_cache)),
            "mac": float(best_abs_ip * best_abs_ip),
            "matched_rank": int(phys[best_idx]["rank_phys"]),
            "lam": float(phys[best_idx]["lam"]),
        }
        aligned_dofs[name] = sc * phys[best_idx]["e_full"]

    return results, aligned_dofs


def reconstruct_global_field_on_grid(subs, e_full: np.ndarray, pts_grid: np.ndarray):
    """
    Evaluates the multiplicity-weighted global field E_global(x) (Eq. 19) on a
    regular evaluation grid. For uncovered interstitial pixels (M(x)=0 under
    25% jitter), uses nearest-subdomain extrapolation strictly for contour continuity
    while returning the exact covered pixel fraction.
    """
    n_pts = len(pts_grid)
    e_sum = np.zeros((n_pts, 2), dtype=float)
    m_cnt = np.zeros(n_pts, dtype=float)
    centers = np.array([sub.center for sub in subs])

    for sub in subs:
        diff = pts_grid - sub.center
        mask = (diff[:, 0] ** 2 + diff[:, 1] ** 2) <= (sub.radius_s ** 2 + 1e-14)
        if not np.any(mask):
            continue
        n_all = sub.eval_shape_functions(pts_grid[mask])
        e_st = e_full[sub.stencil_nodes]
        e_sum[mask] += np.einsum("mqk,k->mq", n_all, e_st)
        m_cnt[mask] += 1.0

    valid = m_cnt > 0
    covered_frac = 100.0 * float(np.mean(valid))
    e_glob = np.zeros_like(e_sum)
    e_glob[valid] = e_sum[valid] / m_cnt[valid, None]

    if not np.all(valid):
        miss_idx = np.where(~valid)[0]
        from scipy.spatial import KDTree
        tree = KDTree(centers)
        _, nearest_sub_idx = tree.query(pts_grid[miss_idx])
        for loc_k, p_idx in enumerate(miss_idx):
            sub = subs[int(nearest_sub_idx[loc_k])]
            n_pt = sub.eval_shape_functions(pts_grid[p_idx : p_idx + 1])
            e_st = e_full[sub.stencil_nodes]
            e_glob[p_idx] = np.einsum("mqk,k->mq", n_pt, e_st)[0]

    return e_glob, covered_frac


def fit_rate(h_vals, err_vals):
    p, _ = np.polyfit(np.log(h_vals), np.log(err_vals), 1)
    return float(p)


def main():
    t_start = time.time()
    print("=" * 78)
    print("VNMM-MLPG EIGENFUNCTION ACCURACY & MODAL FIELD STUDY (15x15 to 81x81)")
    print("=" * 78)

    output = {
        "grids": GRIDS,
        "alpha_s": ALPHA_S,
        "seed": SEED,
        "calibration": {"beta": 1.5, "gamma": 1.0, "c_stab": 4.0},
        "refinement": {},
        "rates": {},
        "comparison_N49": {},
        "modal_fields_N81": {},
    }

    n_vis = 75
    gx = np.linspace(0.0, LX, n_vis)
    XX, YY = np.meshgrid(gx, gx)
    vis_pts = np.column_stack([XX.ravel(), YY.ravel()])
    output["modal_fields_N81"]["grid_n"] = n_vis
    output["modal_fields_N81"]["exact"] = {}

    for name, n, m in MODE_SPECS:
        # Exact field on visualization grid normalized by ||E_ex||_L2([0,pi]^2) so it matches broken-L2 scaling
        # Exact continuous L2 norm of E_nm on [0, pi]^2 is pi * sqrt((n^2 + m^2)/4) for n,m>=1, and pi*m/sqrt(2) for n=0 or m=0.
        # Equivalently, normalize by peak exact amplitude after broken-L2 scaling so color scale is cleanly [0, 1].
        e_vis = exact_te_field(vis_pts, n, m)
        e_mag = np.linalg.norm(e_vis, axis=1)
        scale_vis = float(np.max(e_mag))
        e_vis_unit = e_vis / scale_vis
        output["modal_fields_N81"]["exact"][name] = {
            "Ex": np.round(e_vis_unit[:, 0].reshape(n_vis, n_vis), 4).tolist(),
            "Ey": np.round(e_vis_unit[:, 1].reshape(n_vis, n_vis), 4).tolist(),
        }

    for jitter in JITTERS:
        j_key = f"{int(jitter * 100)}%"
        print(f"\n--- Jitter = {j_key} ---")
        rows = []

        for N in GRIDS:
            h_nom = LX / (N - 1)
            s_div = S_DIV_0 * (H0 / h_nom) ** 1.5
            alpha_stab = 4.0 * (h_nom / LX)
            ops = build_base_operators(N, jitter)
            subs = ops["subs"]
            n_total = len(subs)
            is_bnd = np.array([s.is_boundary for s in subs], dtype=bool)
            int_idx = np.where(~is_bnd)[0]

            K_red = (ops["K_curl"] + s_div * ops["K_div"] + alpha_stab * ops["K_stab"]).tocsr()
            K_red.sum_duplicates()
            phys = extract_physical_modes(
                K_red, ops["M"], ops["K_curl"], ops["K_div"], int_idx, n_total, k_req=35
            )
            quad_cache = build_quad_cache(subs, n_r=4, n_theta=12)
            ex_e, ex_c = normalized_exact_broken_modes(quad_cache)
            metrics, aligned_dofs = evaluate_three_modes(phys, quad_cache, ex_e, ex_c)

            row = {"N": int(N), "h": float(h_nom), "s_div": float(s_div), "alpha_stab": float(alpha_stab), "modes": metrics}
            rows.append(row)

            m10, m11, m22 = metrics["TE10"], metrics["TE11"], metrics["TE22"]
            print(
                f"  N={N:2d}: TE10(L2={m10['err_L2_pct']:5.2f}%, curl={m10['err_curl_pct']:5.2f}%, MAC={m10['mac']:.5f}) | "
                f"TE11(L2={m11['err_L2_pct']:5.2f}%, curl={m11['err_curl_pct']:5.2f}%, MAC={m11['mac']:.5f}) | "
                f"TE22(rk={m22['matched_rank']:2d}, L2={m22['err_L2_pct']:5.2f}%, curl={m22['err_curl_pct']:5.2f}%, MAC={m22['mac']:.5f})"
            )

            # Head-to-head N = 49 comparison: MLPG(c=4.0) vs MLPG(c=0) vs VNMM-Gauss(s_div=6.0)
            if N == 49:
                K_unstab = (ops["K_curl"] + s_div * ops["K_div"]).tocsr()
                K_unstab.sum_duplicates()
                phys_c0 = extract_physical_modes(
                    K_unstab, ops["M"], ops["K_curl"], ops["K_div"], int_idx, n_total, k_req=35
                )
                metrics_c0, _ = evaluate_three_modes(phys_c0, quad_cache, ex_e, ex_c)

                coords = np.array([s.center for s in subs])
                dirs = np.array([s.director for s in subs])
                Kg, Mg = assemble_vnmm_matrices_2d(
                    coords=coords, vectors=dirs, basis="P1", s_div=6.0,
                    Ncx=N - 1, Ncy=N - 1, points_per_dir=3, support_mode="gauss_point",
                )
                Kg_red = Kg[int_idx, :][:, int_idx].tocsr()
                Mg_red = Mg[int_idx, :][:, int_idx].tocsr()
                phys_g = extract_physical_modes(Kg_red, Mg_red, None, None, int_idx, n_total, k_req=35)
                metrics_g, _ = evaluate_three_modes(phys_g, quad_cache, ex_e, ex_c)

                output["comparison_N49"][j_key] = {
                    "MLPG_lin_c4": metrics,
                    "MLPG_unstab_c0": metrics_c0,
                    "VNMM_Gauss": metrics_g,
                }
                print("    N=49 comparison (TE10 / TE11 / TE22 L2 | curl):")
                for lbl, md in [("MLPG lin c=4.0", metrics), ("MLPG c=0", metrics_c0), ("VNMM-Gauss", metrics_g)]:
                    print(
                        f"      {lbl:15s}: "
                        f"TE10({md['TE10']['err_L2_pct']:5.2f}%|{md['TE10']['err_curl_pct']:5.2f}%)  "
                        f"TE11({md['TE11']['err_L2_pct']:5.2f}%|{md['TE11']['err_curl_pct']:5.2f}%)  "
                        f"TE22({md['TE22']['err_L2_pct']:5.2f}%|{md['TE22']['err_curl_pct']:5.2f}%)"
                    )

            # Reconstruct 75x75 modal fields at N = 81
            if N == 81:
                output["modal_fields_N81"][j_key] = {}
                for name, n, m in MODE_SPECS:
                    e_glob, cov_pct = reconstruct_global_field_on_grid(subs, aligned_dofs[name], vis_pts)
                    # Convert broken-L2 unit normalization to peak exact magnitude = 1.0 using analytical inner product factor
                    # Exact E_vis at vis_pts gives exact broken-L2 norm ratio from quad_cache
                    e_ex_q = [exact_te_field(pts, n, m) for pts, _, _, _, _ in quad_cache]
                    ex_q_norm = float(np.sqrt(broken_l2_inner(e_ex_q, e_ex_q, quad_cache)))
                    ex_peak = float(np.max(np.linalg.norm(exact_te_field(vis_pts, n, m), axis=1)))
                    e_vis_scaled = e_glob * (ex_q_norm / ex_peak)
                    output["modal_fields_N81"][j_key][name] = {
                        "Ex": np.round(e_vis_scaled[:, 0].reshape(n_vis, n_vis), 4).tolist(),
                        "Ey": np.round(e_vis_scaled[:, 1].reshape(n_vis, n_vis), 4).tolist(),
                        "covered_pct": round(cov_pct, 2),
                        "err_L2_pct": round(metrics[name]["err_L2_pct"], 2),
                        "mac": round(metrics[name]["mac"], 5),
                    }

        output["refinement"][j_key] = rows
        h_all = np.array([r["h"] for r in rows])
        h_29_81 = h_all[2:]
        rates_j = {}
        for name, _, _ in MODE_SPECS:
            l2_all = [r["modes"][name]["err_L2_pct"] for r in rows]
            c_all = [r["modes"][name]["err_curl_pct"] for r in rows]
            rates_j[name] = {
                "p_L2_all8": fit_rate(h_all, l2_all),
                "p_curl_all8": fit_rate(h_all, c_all),
                "p_L2_29_81": fit_rate(h_29_81, l2_all[2:]),
                "p_curl_29_81": fit_rate(h_29_81, c_all[2:]),
            }
        output["rates"][j_key] = rates_j
        print(f"  Fitted rates ({j_key}):")
        for name, rt in rates_j.items():
            print(
                f"    {name:4s}: p_L2(15..81)={rt['p_L2_all8']:.2f}, p_curl(15..81)={rt['p_curl_all8']:.2f} | "
                f"p_L2(29..81)={rt['p_L2_29_81']:.2f}, p_curl(29..81)={rt['p_curl_29_81']:.2f}"
            )

    out_path = os.path.join(CURRENT_DIR, "eigenfunction_accuracy_study.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved eigenfunction study results to: {out_path}")
    print(f"Total elapsed time: {time.time() - t_start:.2f} s")


if __name__ == "__main__":
    main()
