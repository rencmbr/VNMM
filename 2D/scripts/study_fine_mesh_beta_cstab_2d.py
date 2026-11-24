#!/usr/bin/env python3
"""
study_fine_mesh_beta_cstab_2d.py
================================
Extends the VNMM-MLPG 2D PEC cavity convergence study to 65x65 and 81x81 grids
and performs a coupled sweep over divergence penalty exponent beta in {0.0, 0.5, 1.0, 1.5}
and orthogonal stabilization constant c_stab in {0, 20, 40, 70, 100} under
0% (regular) and 25% nodal jitter (seed=42).

Key optimizations:
- Reuses sparse base operators (K_curl, K_div, K_stab, M) per (N, jitter).
- Uses sparse Shift-and-Invert solver (scipy.sparse.linalg.eigsh) cross-validated
  against dense scipy.linalg.eigh on N=22.
- Computes continuous multiplicity Rayleigh-quotient shifts across all 8 grids.
- Saves full structured results to 2D/scripts/fine_mesh_beta_cstab_study.json.
"""

import os
import sys
import time
import json
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigsh

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d
from src.mlpg_assembler_2d import (
    assemble_mlpg_matrices_2d,
    get_disk_quadrature_points_2d,
)
from src.eigen_solver_cavity_2d import (
    apply_pec_boundary_conditions,
    ANALYTICAL_REFERENCE_MODES,
)

GRIDS = [15, 22, 29, 36, 43, 49, 65, 81]
JITTERS = [0.0, 0.25]
BETAS = [0.0, 0.5, 1.0, 1.5]
CSTABS = [0, 20, 40, 70, 100]
ALPHA_S = 0.80
SEED = 42
LX = np.pi
LY = np.pi
H0 = np.pi / 20.0
S_DIV_0 = 15.0
SPUR_THRESHOLD = 50.0


def sym(A):
    return (0.5 * (A + A.T)).tocsr()


def build_base_operators(N: int, jitter_frac: float):
    """Generates mesh, subdomains, and reduced sparse base matrices once per (N, jitter)."""
    t0 = time.time()
    coords, vectors, is_bnd = generate_cavity_mesh(
        Nx=N,
        Ny=N,
        Lx=LX,
        Ly=LY,
        interior_type="random",
        jitter_frac=jitter_frac,
        seed=SEED,
    )
    subs = build_mlpg_subdomains_2d(
        coords=coords,
        vectors=vectors,
        is_boundary=is_bnd,
        alpha_s=ALPHA_S,
    )
    t_prep = time.time() - t0

    t0 = time.time()
    K0, M, _ = assemble_mlpg_matrices_2d(
        subdomains=subs,
        mu_r=1.0,
        epsilon_r=1.0,
        s_div=0.0,
        alpha_stab=0.0,
        assembly_mode="subdomain_galerkin",
    )
    K1, _, _ = assemble_mlpg_matrices_2d(
        subdomains=subs,
        mu_r=1.0,
        epsilon_r=1.0,
        s_div=1.0,
        alpha_stab=0.0,
        assembly_mode="subdomain_galerkin",
    )
    K2, _, _ = assemble_mlpg_matrices_2d(
        subdomains=subs,
        mu_r=1.0,
        epsilon_r=1.0,
        s_div=0.0,
        alpha_stab=1.0,
        assembly_mode="subdomain_galerkin",
    )
    t_asm_base = time.time() - t0

    K_curl_red, M_red, int_idx = apply_pec_boundary_conditions(K0, M, is_bnd)
    K_div_full = K1 - K0
    K_stab_full = K2 - K0
    K_div_red = K_div_full[int_idx, :][:, int_idx]
    K_stab_red = K_stab_full[int_idx, :][:, int_idx]

    return {
        "subs": subs,
        "int_dofs": int(len(int_idx)),
        "t_prep": float(t_prep),
        "t_asm_base": float(t_asm_base),
        "K_curl": sym(K_curl_red),
        "K_div": sym(K_div_red),
        "K_stab": sym(K_stab_red),
        "M": sym(M_red),
    }


def compute_sparse_cond2(K_mat):
    """Computes exact 2-norm condition number lambda_max / lambda_min of SPD matrix K_mat."""
    n = K_mat.shape[0]
    if n <= 1:
        return 1.0
    lmax = float(eigsh(K_mat, k=1, which="LA", tol=1e-8, return_eigenvectors=False)[0])
    try:
        lmin = float(
            eigsh(K_mat, k=1, sigma=-1e-9, which="LM", tol=1e-9, return_eigenvectors=False)[0]
        )
    except Exception:
        lmin = float(
            eigsh(K_mat, k=1, which="SA", tol=1e-7, maxiter=5000, return_eigenvectors=False)[0]
        )
    if lmin <= 1e-15:
        return float("inf")
    return float(lmax / lmin)


def solve_sparse_modes(base_ops, s_div: float, alpha_stab: float, num_modes: int = 10):
    """Solves generalized eigenproblem via sparse Shift-and-Invert and classifies physical modes."""
    K_curl = base_ops["K_curl"]
    K_div = base_ops["K_div"]
    K_stab = base_ops["K_stab"]
    M = base_ops["M"]

    K_red = (K_curl + s_div * K_div + alpha_stab * K_stab).tocsr()
    K_red.sum_duplicates()
    n_dof = K_red.shape[0]

    t0 = time.time()
    selected = []
    spur_list = []
    vals_pos = np.array([])
    vecs_pos = np.empty((n_dof, 0))

    for k_req in [25, 45, 70]:
        k_use = min(k_req, n_dof - 2)
        # Negative shift sigma=-0.05 guarantees (K_red - sigma*M) is strictly SPD
        vals, vecs = eigsh(K_red, k=k_use, M=M, sigma=-0.05, which="LM", tol=1e-10)
        order = np.argsort(vals)
        vals = np.real(vals[order])
        vecs = np.real(vecs[:, order])

        mask = vals > 0.05
        vals_pos = vals[mask]
        vecs_pos = vecs[:, mask]

        physical = []
        spurious = []
        for i in range(len(vals_pos)):
            v = vecs_pos[:, i]
            ec = float(v @ (K_curl @ v))
            ed = float(v @ (K_div @ v))
            es = float(alpha_stab * (v @ (K_stab @ v))) if alpha_stab > 0.0 else 0.0
            r_div = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
            r_stab = (es / ec) * 100.0 if ec > 1e-12 else 0.0
            entry = {
                "idx": i + 1,
                "lambda": float(vals_pos[i]),
                "kc": float(np.sqrt(max(vals_pos[i], 0.0))),
                "r_div": float(r_div),
                "r_stab": float(r_stab),
            }
            if r_div <= SPUR_THRESHOLD:
                physical.append(entry)
            else:
                spurious.append(entry)

        selected = physical[:num_modes]
        spur_list = spurious
        if len(selected) == num_modes or k_use >= n_dof - 2:
            break

    t_solve = time.time() - t0

    # Spurious count inside first num_modes raw positive eigenvalues
    spur_in_band = 0
    for i in range(min(num_modes, len(vals_pos))):
        v = vecs_pos[:, i]
        ec = float(v @ (K_curl @ v))
        ed = float(v @ (K_div @ v))
        r = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
        if r > SPUR_THRESHOLD:
            spur_in_band += 1

    ref_data = ANALYTICAL_REFERENCE_MODES[: len(selected)]
    ref_kc = np.array([r[3] for r in ref_data])
    num_kc = np.array([m["kc"] for m in selected])
    err_kc = np.abs(num_kc - ref_kc) / ref_kc * 100.0

    cond_K = compute_sparse_cond2(K_red)

    return {
        "n_phys_found": len(selected),
        "mean_err_kc": float(np.mean(err_kc)) if len(err_kc) else float("nan"),
        "max_err_kc": float(np.max(err_kc)) if len(err_kc) else float("nan"),
        "err_TE10": float(err_kc[0]) if len(err_kc) > 0 else float("nan"),
        "err_TE01": float(err_kc[1]) if len(err_kc) > 1 else float("nan"),
        "err_TE11": float(err_kc[2]) if len(err_kc) > 2 else float("nan"),
        "err_TE22": float(err_kc[7]) if len(err_kc) > 7 else float("nan"),
        "spur_in_band": int(spur_in_band),
        "lowest_spur_lambda": float(spur_list[0]["lambda"]) if spur_list else None,
        "max_r_div_phys": float(max(m["r_div"] for m in selected)) if selected else float("nan"),
        "mean_r_stab_phys": float(np.mean([m["r_stab"] for m in selected])) if selected else 0.0,
        "cond_K": float(cond_K),
        "nnz": int(K_red.nnz),
        "t_solve": float(t_solve),
    }


def cross_validate_sparse_solver(base_ops_22):
    """Asserts sparse eigsh and sparse cond_2 match dense la.eigh and np.linalg.cond on N=22."""
    h = LX / 21.0
    s_div = float(S_DIV_0 * (H0 / h) ** 1.5)
    alpha_stab = float(70.0 * (h / LX) ** 2)

    K_red = (
        base_ops_22["K_curl"]
        + s_div * base_ops_22["K_div"]
        + alpha_stab * base_ops_22["K_stab"]
    ).tocsr()
    M_red = base_ops_22["M"]

    vals_d, _ = la.eigh(K_red.toarray(), M_red.toarray())
    kc_d = np.sqrt(vals_d[vals_d > 0.05][:10])
    cond_d = float(np.linalg.cond(K_red.toarray()))

    res_s = solve_sparse_modes(base_ops_22, s_div=s_div, alpha_stab=alpha_stab)
    vals_s, _ = eigsh(K_red, k=25, M=M_red, sigma=-0.05, which="LM", tol=1e-10)
    vals_s = np.sort(np.real(vals_s))
    kc_s = np.sqrt(vals_s[vals_s > 0.05][:10])

    max_kc_diff = float(np.max(np.abs(kc_d - kc_s)))
    cond_rel_diff = abs(res_s["cond_K"] - cond_d) / cond_d

    print(
        f"[Cross-Validation N=22, beta=1.5, c_stab=70] "
        f"max |kc_dense - kc_sparse| = {max_kc_diff:.3e}, "
        f"cond rel diff = {cond_rel_diff:.3e}"
    )
    assert max_kc_diff < 1e-9, f"Sparse eigsh mismatch: {max_kc_diff}"
    assert cond_rel_diff < 1e-5, f"Sparse cond mismatch: {cond_rel_diff}"


def compute_multiplicity_shifts(cache_subs):
    """Computes continuous multiplicity M(x) Rayleigh quotient shift across all 8 grids."""
    modes = [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2), (3, 0), (0, 3)]
    out = {}
    for j in JITTERS:
        j_key = f"{int(j*100)}%"
        out[j_key] = []
        for N in GRIDS:
            subs = cache_subs[(N, j)]["subs"]
            # Cache quadrature points/weights per subdomain once across the 10 modes
            num_M = np.zeros(len(modes))
            den_M = np.zeros(len(modes))
            for sub in subs:
                pts, wts = get_disk_quadrature_points_2d(
                    sub.center,
                    sub.radius_s,
                    n_r=4,
                    n_theta=12,
                    domain_bounds=getattr(sub, "domain_bounds", None),
                )
                x, y = pts[:, 0], pts[:, 1]
                for idx_m, (n, m) in enumerate(modes):
                    Ex = m * np.cos(n * x) * np.sin(m * y)
                    Ey = -n * np.sin(n * x) * np.cos(m * y)
                    curl_z = -(n ** 2 + m ** 2) * np.cos(n * x) * np.cos(m * y)
                    num_M[idx_m] += float(np.sum(wts * (curl_z ** 2)))
                    den_M[idx_m] += float(np.sum(wts * (Ex ** 2 + Ey ** 2)))

            errs = []
            for idx_m, (n, m) in enumerate(modes):
                kc_exact = float(np.sqrt(n ** 2 + m ** 2))
                kc_M = float(np.sqrt(num_M[idx_m] / den_M[idx_m]))
                errs.append(abs(kc_M - kc_exact) / kc_exact * 100.0)

            out[j_key].append(
                {
                    "N": N,
                    "h": float(LX / (N - 1)),
                    "mean_shift_pct": float(np.mean(errs)),
                    "max_shift_pct": float(np.max(errs)),
                }
            )
    return out


def compute_rates(grid_entries):
    """Computes orig6 (15..49), global8 (15..81), fine (36..81), local pairwise rates, and p_cond."""
    h = np.array([e["h"] for e in grid_entries])
    err = np.array([e["mean_err_kc"] for e in grid_entries])
    cond = np.array([e["cond_K"] for e in grid_entries])

    p_orig6 = float(np.polyfit(np.log(h[:6]), np.log(err[:6]), 1)[0])
    p_global8 = float(np.polyfit(np.log(h), np.log(err), 1)[0])
    # Fine regime starts at index 3 (N=36) through index 7 (N=81)
    p_fine = float(np.polyfit(np.log(h[3:]), np.log(err[3:]), 1)[0])
    p_cond = float(np.polyfit(np.log(h), np.log(cond), 1)[0])

    local = {}
    for idx in range(1, len(grid_entries)):
        n_prev, n_curr = grid_entries[idx - 1]["N"], grid_entries[idx]["N"]
        val = float(np.log(err[idx - 1] / err[idx]) / np.log(h[idx - 1] / h[idx]))
        local[f"{n_prev}->{n_curr}"] = val

    return {
        "p_orig6": p_orig6,
        "p_global8": p_global8,
        "p_fine_36_81": p_fine,
        "p_cond": p_cond,
        "p_local": local,
    }


def main():
    t_total_start = time.time()
    print("=" * 100)
    print("FINE-MESH (UP TO 81x81) & COUPLED (beta, c_stab) STUDY FOR VNMM-MLPG 2D (seed=42)")
    print("=" * 100)

    cache = {}
    for j in JITTERS:
        for N in GRIDS:
            t0 = time.time()
            ops = build_base_operators(N, j)
            cache[(N, j)] = ops
            print(
                f"Prepared base operators: N={N:2d}x{N:<2d}, jitter={int(j*100):2d}%, "
                f"DoFs={ops['int_dofs']:4d}, prep={ops['t_prep']:.2f}s, asm_base={ops['t_asm_base']:.2f}s"
            )

    cross_validate_sparse_solver(cache[(22, 0.0)])

    print("\nComputing continuous multiplicity M(x) Rayleigh quotient shifts across 8 grids...")
    mult_shifts = compute_multiplicity_shifts(cache)

    sweep = {}
    summary = {}
    for j in JITTERS:
        j_key = f"{int(j*100)}%"
        sweep[j_key] = {}
        summary[j_key] = {}
        for beta in BETAS:
            b_key = f"{beta:.1f}"
            sweep[j_key][b_key] = {}
            summary[j_key][b_key] = {}
            for c_stab in CSTABS:
                c_key = str(c_stab)
                entries = []
                for N in GRIDS:
                    h_nom = LX / float(N - 1)
                    s_div = float(S_DIV_0 * (H0 / h_nom) ** beta)
                    alpha_stab = float(c_stab * (h_nom / LX) ** 2)
                    res = solve_sparse_modes(cache[(N, j)], s_div=s_div, alpha_stab=alpha_stab)
                    res.update(
                        {
                            "N": N,
                            "dofs": cache[(N, j)]["int_dofs"],
                            "h": float(h_nom),
                            "s_div": s_div,
                            "alpha_stab": alpha_stab,
                        }
                    )
                    entries.append(res)
                sweep[j_key][b_key][c_key] = entries
                summary[j_key][b_key][c_key] = compute_rates(entries)

    # Assert exact reproduction of manuscript Table 8 at beta=1.5, c_stab=70, jitter=0% (15..49)
    expected_t8 = [17.67, 7.11, 3.34, 2.87, 1.39, 1.29]
    got_t8 = [round(e["mean_err_kc"], 2) for e in sweep["0%"]["1.5"]["70"][:6]]
    assert got_t8 == expected_t8, f"Table 8 mismatch: got {got_t8}, expected {expected_t8}"
    print(f"\n[Verified Table 8 reproduction (0%, beta=1.5, c_stab=70, 15..49)]: {got_t8}")

    out_data = {
        "grids": GRIDS,
        "jitters": JITTERS,
        "betas": BETAS,
        "c_stabs": CSTABS,
        "alpha_s": ALPHA_S,
        "seed": SEED,
        "multiplicity_shifts": mult_shifts,
        "sweep": sweep,
        "summary": summary,
    }
    out_path = os.path.join(CURRENT_DIR, "fine_mesh_beta_cstab_study.json")
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved complete results to: {out_path} (total elapsed: {time.time() - t_total_start:.1f}s)")

    # Print Axis A report (beta = 1.5 across all 8 grids)
    for j_key in ["0%", "25%"]:
        print("\n" + "=" * 105)
        print(f"AXIS A: FINE-MESH EXTENSION FOR BASELINE beta = 1.5 (jitter = {j_key})")
        print("=" * 105)
        header = "  c_stab |   15^2 |   22^2 |   29^2 |   36^2 |   43^2 |   49^2 |   65^2 |   81^2 | p_orig6 | p_glob8 | p_36_81 | p(49->65) | p(65->81)"
        print(header)
        print("-" * len(header))
        for c_key in [str(c) for c in CSTABS]:
            ents = sweep[j_key]["1.5"][c_key]
            summ = summary[j_key]["1.5"][c_key]
            errs = " | ".join(f"{e['mean_err_kc']:5.2f}%" for e in ents)
            print(
                f"  {int(c_key):6d} | {errs} | "
                f"{summ['p_orig6']:7.2f} | {summ['p_global8']:7.2f} | {summ['p_fine_36_81']:7.2f} | "
                f"{summ['p_local']['49->65']:9.2f} | {summ['p_local']['65->81']:9.2f}"
            )
        print("  Multiplicity shift lambda^(M):")
        m_str = " | ".join(f"{m['mean_shift_pct']:5.3f}%" for m in mult_shifts[j_key])
        print(f"  M(x)   | {m_str}")

    # Print Axis B report (coupled beta x c_stab on fine meshes)
    for j_key in ["0%", "25%"]:
        print("\n" + "=" * 112)
        print(f"AXIS B: COUPLED (beta x c_stab) SWEEP ON FINE MESHES (jitter = {j_key})")
        print("=" * 112)
        print("  beta | c_stab |   49^2 |   65^2 |   81^2 | max_spur(15..81) | max_Rdiv(81) | p_36_81 | p(65->81) | cond_K(81^2) | p_cond")
        print("-" * 112)
        for b_key in [f"{b:.1f}" for b in BETAS]:
            for c_key in [str(c) for c in CSTABS]:
                ents = sweep[j_key][b_key][c_key]
                summ = summary[j_key][b_key][c_key]
                e49, e65, e81 = ents[5], ents[6], ents[7]
                max_spur = max(e["spur_in_band"] for e in ents)
                print(
                    f"  {b_key:>4} | {int(c_key):6d} | "
                    f"{e49['mean_err_kc']:5.2f}% | {e65['mean_err_kc']:5.2f}% | {e81['mean_err_kc']:5.2f}% | "
                    f"{max_spur:16d} | {e81['max_r_div_phys']:11.2f}% | "
                    f"{summ['p_fine_36_81']:7.2f} | {summ['p_local']['65->81']:9.2f} | "
                    f"{e81['cond_K']:12.2e} | {summ['p_cond']:6.2f}"
                )


if __name__ == "__main__":
    main()
