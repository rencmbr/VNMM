"""
Experimental test on branch test/vnmm-mlpg:
Evaluates adding the artificial-boundary contour flux operator K_bnd over L_{sI}
to the MLPG6 local weak form:
    (K_vol - K_bnd + kappa * M) e = F_vol
for the conventional volumetric-source MMS (without analytical contour fluxes on L_{sI}).

Compares:
  1. Baseline SSG (K_vol + kappa * M) e = F_vol (no K_bnd)
  2. Subdomain Galerkin (6x6) with K_bnd using local P1 stencil at each quadrature point x_q in L_{sI}
  3. Row-by-row MLPG6 (1x6) with K_bnd using local P1 stencil at each x_q in L_{sI}
  4. Subdomain Galerkin (6x6) with K_bnd using average of neighboring subdomains J != I containing x_q
Also verifies:
  - Static affine patch test (k0=0, J_s=0) with ZERO analytical fluxes on L_{sI}
  - Asymmetry ratio ||A - A^T||_F / ||A||_F
  - GMRES preconditioned by the original symmetric matrix (K_vol + kappa * M)
"""

import os
import sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve, splu, LinearOperator, gmres
from scipy.spatial import KDTree
from typing import Dict, Any, List, Tuple, Optional

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
SRC_DIR = os.path.join(DIR_2D, "src")
for p in [DIR_2D, ROOT_DIR, CURRENT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.mlpg_subdomains import build_mlpg_subdomains_2d, MLPGSubdomain2D
from src.mlpg_assembler_2d import (
    compute_subdomain_matrices_analytic,
    get_disk_quadrature_points_2d,
)
from src.gauss_quadrature_2d import get_gauss_points_weights_1d
from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from scripts.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1
from scripts.mlpg_source_problem_2d import (
    exact_solution_mms,
    exact_curl_mms,
    source_mms,
)


def get_internal_arc_quadrature_2d(
    center: np.ndarray,
    radius: float,
    domain_bounds: Tuple[float, float, float, float],
    n_gauss_per_arc: int = 8,
    n_full_circle: int = 24,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns quadrature points, arc-length weights, outward normals n, and
    counter-clockwise tangents t along L_{sI} = partial(Omega_s^I) intersect Omega.
    Uses exact circle-box intersection angles so truncated arcs integrate polynomials exactly.
    """
    xc, yc = float(center[0]), float(center[1])
    xmin, xmax, ymin, ymax = domain_bounds
    r = float(radius)

    cut_angles = [0.0, 2.0 * np.pi]
    # Intersections with vertical walls x = xmin, xmax
    for xw in (xmin, xmax):
        dx = (xw - xc) / r
        if abs(dx) < 1.0 - 1e-12:
            th = np.arccos(dx)
            cut_angles.extend([th % (2.0 * np.pi), (-th) % (2.0 * np.pi)])
    # Intersections with horizontal walls y = ymin, ymax
    for yw in (ymin, ymax):
        dy = (yw - yc) / r
        if abs(dy) < 1.0 - 1e-12:
            th = np.arcsin(dy)
            cut_angles.extend([th % (2.0 * np.pi), (np.pi - th) % (2.0 * np.pi)])

    cut_angles = sorted(set(round(a, 12) for a in cut_angles))
    # Deduplicate close angles
    unique_cuts = [cut_angles[0]]
    for a in cut_angles[1:]:
        if a - unique_cuts[-1] > 1e-10:
            unique_cuts.append(a)
    if unique_cuts[-1] < 2.0 * np.pi - 1e-10:
        unique_cuts.append(2.0 * np.pi)

    pts_list, wts_list, n_list, t_list = [], [], [], []
    # Full circle case: no cuts inside (0, 2pi)
    if len(unique_cuts) == 2:
        th_mid = (np.arange(n_full_circle) + 0.5) * (2.0 * np.pi / n_full_circle)
        ds = r * (2.0 * np.pi / n_full_circle)
        for th in th_mid:
            px = xc + r * np.cos(th)
            py = yc + r * np.sin(th)
            if (xmin + 1e-12 < px < xmax - 1e-12) and (ymin + 1e-12 < py < ymax - 1e-12):
                pts_list.append([px, py])
                wts_list.append(ds)
                n_list.append([np.cos(th), np.sin(th)])
                t_list.append([-np.sin(th), np.cos(th)])
    else:
        xi, wxi = get_gauss_points_weights_1d(n_gauss_per_arc)
        for a0, a1 in zip(unique_cuts[:-1], unique_cuts[1:]):
            amid = 0.5 * (a0 + a1)
            mx = xc + r * np.cos(amid)
            my = yc + r * np.sin(amid)
            if not ((xmin + 1e-12 < mx < xmax - 1e-12) and (ymin + 1e-12 < my < ymax - 1e-12)):
                continue
            half_d = 0.5 * (a1 - a0)
            for s, ws in zip(xi, wxi):
                th = amid + half_d * s
                pts_list.append([xc + r * np.cos(th), yc + r * np.sin(th)])
                wts_list.append(r * half_d * ws)
                n_list.append([np.cos(th), np.sin(th)])
                t_list.append([-np.sin(th), np.cos(th)])

    if not pts_list:
        return (
            np.zeros((0, 2)),
            np.zeros((0,)),
            np.zeros((0, 2)),
            np.zeros((0, 2)),
        )
    return (
        np.array(pts_list, dtype=float),
        np.array(wts_list, dtype=float),
        np.array(n_list, dtype=float),
        np.array(t_list, dtype=float),
    )


def get_physical_boundary_segments_quadrature_2d(
    center: np.ndarray,
    radius: float,
    domain_bounds: Tuple[float, float, float, float],
    n_gauss: int = 6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns quadrature points, arc-length weights, outward normals n, and
    counter-clockwise tangents t along Gamma_s^I = Omega_s^I intersect partial(Omega).
    """
    xc, yc = float(center[0]), float(center[1])
    xmin, xmax, ymin, ymax = domain_bounds
    r = float(radius)
    xi, wxi = get_gauss_points_weights_1d(n_gauss)

    pts_list, wts_list, n_list, t_list = [], [], [], []

    # Vertical walls: x = xmin (n=(-1,0), t=(0,-1)) and x = xmax (n=(1,0), t=(0,1))
    for xw, nx in ((xmin, -1.0), (xmax, 1.0)):
        dx = abs(xw - xc)
        if dx < r - 1e-12:
            half_y = float(np.sqrt(max(r * r - dx * dx, 0.0)))
            y0 = max(ymin, yc - half_y)
            y1 = min(ymax, yc + half_y)
            if y1 - y0 > 1e-12:
                ymid = 0.5 * (y0 + y1)
                hy = 0.5 * (y1 - y0)
                for s, ws in zip(xi, wxi):
                    pts_list.append([xw, ymid + hy * s])
                    wts_list.append(hy * ws)
                    n_list.append([nx, 0.0])
                    t_list.append([0.0, nx])

    # Horizontal walls: y = ymin (n=(0,-1), t=(1,0)) and y = ymax (n=(0,1), t=(-1,0))
    for yw, ny in ((ymin, -1.0), (ymax, 1.0)):
        dy = abs(yw - yc)
        if dy < r - 1e-12:
            half_x = float(np.sqrt(max(r * r - dy * dy, 0.0)))
            x0 = max(xmin, xc - half_x)
            x1 = min(xmax, xc + half_x)
            if x1 - x0 > 1e-12:
                xmid = 0.5 * (x0 + x1)
                hx = 0.5 * (x1 - x0)
                for s, ws in zip(xi, wxi):
                    pts_list.append([xmid + hx * s, yw])
                    wts_list.append(hx * ws)
                    n_list.append([0.0, ny])
                    t_list.append([-ny, 0.0])

    if not pts_list:
        return (
            np.zeros((0, 2)),
            np.zeros((0,)),
            np.zeros((0, 2)),
            np.zeros((0, 2)),
        )
    return (
        np.array(pts_list, dtype=float),
        np.array(wts_list, dtype=float),
        np.array(n_list, dtype=float),
        np.array(t_list, dtype=float),
    )


def evaluate_flux_stencil_at_point(
    pt: np.ndarray,
    sub_i: MLPGSubdomain2D,
    subdomains: List[MLPGSubdomain2D],
    coords: np.ndarray,
    vectors: np.ndarray,
    tree: KDTree,
    flux_mode: str = "point_stencil",
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Returns (node_indices, c_curl_vec, c_div_vec) such that at point `pt` in L_{sI}:
        (curl E_h)_z(pt) = sum_k c_curl_vec[k] * e[node_indices[k]]
        (div  E_h)(pt)   = sum_k c_div_vec[k]  * e[node_indices[k]]

    flux_mode:
      - "point_stencil": fresh 6-node P1 stencil selected around `pt` (via KDTree)
      - "neighbor_avg": average of all subdomains J != I whose disk covers `pt`
                        (fallback to nearest J != I if none covers `pt`)
    """
    if flux_mode == "point_stencil":
        sextet, _, mat_A, _ = support_nodes_vnmm_2d_6_p1(
            P=pt,
            nodes_coords=coords,
            nodes_vectors=vectors,
            search_tree=tree,
            K=12,
            Tol_det=1e-4,
            adaptive=True,
        )
        _, curl_phi, beta = shape_functions_vnmm_2d_6_p1(
            P=pt,
            nodes_coords=coords,
            nodes_vectors=vectors,
            selected_nodes=sextet,
            matrix_a=mat_A,
        )
        div_phi = beta[2, :] + beta[5, :]
        return list(sextet), curl_phi, div_phi

    elif flux_mode == "neighbor_avg":
        # Query nearby nodes and collect subdomains J != I containing pt
        dists, idxs = tree.query(pt, k=min(12, len(subdomains)))
        covering = []
        for idx in np.atleast_1d(idxs):
            j = int(idx)
            if j == sub_i.node_idx:
                continue
            sub_j = subdomains[j]
            if np.linalg.norm(pt - sub_j.center) <= sub_j.radius_s + 1e-12:
                covering.append(sub_j)
        if not covering:
            # Fallback to nearest neighbor J != I
            for idx in np.atleast_1d(idxs):
                j = int(idx)
                if j != sub_i.node_idx:
                    covering.append(subdomains[j])
                    break

        w_avg = 1.0 / float(len(covering))
        acc_curl: Dict[int, float] = {}
        acc_div: Dict[int, float] = {}
        for sub_j in covering:
            c_curl = sub_j.eval_shape_curl()
            c_div = sub_j.eval_shape_div()
            for a_idx, g_node in enumerate(sub_j.stencil_nodes):
                g_node = int(g_node)
                acc_curl[g_node] = acc_curl.get(g_node, 0.0) + w_avg * float(c_curl[a_idx])
                acc_div[g_node] = acc_div.get(g_node, 0.0) + w_avg * float(c_div[a_idx])

        nodes = sorted(acc_curl.keys())
        c_curl_arr = np.array([acc_curl[g] for g in nodes], dtype=float)
        c_div_arr = np.array([acc_div[g] for g in nodes], dtype=float)
        return nodes, c_curl_arr, c_div_arr

    else:
        raise ValueError(f"Unknown flux_mode: {flux_mode}")


def assemble_mlpg_system_with_boundary_flux(
    coords: np.ndarray,
    vectors: np.ndarray,
    is_bnd: np.ndarray,
    s_div: float = 6.0,
    kappa: float = 1.0,
    mu_r: float = 1.0,
    eps_r: float = 1.0,
    alpha_s: float = 0.8,
    alpha_stab: float = 0.0,
    assembly_mode: str = "subdomain_galerkin",
    include_k_bnd: bool = True,
    include_phys_bnd_flux: bool = False,
    flux_mode: str = "point_stencil",
    source_fn=None,
) -> Dict[str, Any]:
    """
    Assembles K_vol, K_bnd, M, and F_vol on [0, pi]^2.
    """
    N = len(coords)
    domain_bounds = (0.0, np.pi, 0.0, np.pi)
    subdomains = build_mlpg_subdomains_2d(
        coords=coords,
        vectors=vectors,
        is_boundary=is_bnd,
        alpha_s=alpha_s,
    )
    tree = KDTree(coords)
    is_sub_gal = assembly_mode == "subdomain_galerkin"

    rows_v, cols_v, data_kv, data_mv = [], [], [], []
    rows_b, cols_b, data_kb = [], [], []
    F_vol = np.zeros(N, dtype=float)

    for sub in subdomains:
        k_loc, m_loc = compute_subdomain_matrices_analytic(
            subdomain=sub,
            mu_r=mu_r,
            epsilon_r=eps_r,
            s_div=s_div,
            alpha_stab=alpha_stab,
            matrix_format="6x6" if is_sub_gal else "1x6",
        )
        idxs = sub.stencil_nodes
        if is_sub_gal:
            for a in range(6):
                ga = int(idxs[a])
                for b in range(6):
                    gb = int(idxs[b])
                    rows_v.append(ga)
                    cols_v.append(gb)
                    data_kv.append(float(k_loc[a, b]))
                    data_mv.append(float(m_loc[a, b]))
        else:
            ga = int(sub.node_idx)
            for b in range(6):
                gb = int(idxs[b])
                rows_v.append(ga)
                cols_v.append(gb)
                data_kv.append(float(k_loc[b]))
                data_mv.append(float(m_loc[b]))

        # Volumetric source F_vol
        if source_fn is not None:
            pts_d, wts_d = get_disk_quadrature_points_2d(
                sub.center, sub.radius_s, n_r=4, n_theta=12, domain_bounds=domain_bounds
            )
            N_all = sub.eval_shape_functions(pts_d)  # (M, 2, 6)
            J_pts = np.array([source_fn(px, py) for px, py in pts_d])
            dot_JN = np.einsum("mq,mqa->ma", J_pts, N_all)
            f_loc = np.sum(wts_d[:, np.newaxis] * dot_JN, axis=0)
            if is_sub_gal:
                for a in range(6):
                    F_vol[int(idxs[a])] += float(f_loc[a])
            else:
                local_i = int(np.where(idxs == sub.node_idx)[0][0])
                F_vol[int(sub.node_idx)] += float(f_loc[local_i])

        # Boundary flux operator K_bnd over L_{sI} (and optionally Gamma_s^I for closed partial(D_I))
        if include_k_bnd:
            pts_arc, wts_arc, n_arc, t_arc = get_internal_arc_quadrature_2d(
                sub.center, sub.radius_s, domain_bounds
            )
            if include_phys_bnd_flux:
                pts_g, wts_g, n_g, t_g = get_physical_boundary_segments_quadrature_2d(
                    sub.center, sub.radius_s, domain_bounds
                )
                if len(pts_g) > 0:
                    pts_arc = np.vstack([pts_arc, pts_g]) if len(pts_arc) > 0 else pts_g
                    wts_arc = np.concatenate([wts_arc, wts_g]) if len(wts_arc) > 0 else wts_g
                    n_arc = np.vstack([n_arc, n_g]) if len(n_arc) > 0 else n_g
                    t_arc = np.vstack([t_arc, t_g]) if len(t_arc) > 0 else t_g
            if len(pts_arc) > 0:
                N_arc = sub.eval_shape_functions(pts_arc)  # (Q, 2, 6)
                test_indices = range(6) if is_sub_gal else [int(np.where(idxs == sub.node_idx)[0][0])]
                for q in range(len(pts_arc)):
                    wq = float(wts_arc[q])
                    nq = n_arc[q]
                    tq = t_arc[q]
                    f_nodes, c_curl_ext, c_div_ext = evaluate_flux_stencil_at_point(
                        pts_arc[q], sub, subdomains, coords, vectors, tree, flux_mode=flux_mode
                    )
                    for a in test_indices:
                        ga = int(idxs[a]) if is_sub_gal else int(sub.node_idx)
                        Wa = N_arc[q, :, a]
                        Wt = float(np.dot(Wa, tq))
                        Wn = float(np.dot(Wa, nq))
                        for k_idx, gb in enumerate(f_nodes):
                            val = wq * (
                                (1.0 / mu_r) * Wt * float(c_curl_ext[k_idx])
                                + (s_div / mu_r) * Wn * float(c_div_ext[k_idx])
                            )
                            rows_b.append(ga)
                            cols_b.append(int(gb))
                            data_kb.append(val)

    K_vol = coo_matrix((data_kv, (rows_v, cols_v)), shape=(N, N)).tocsr()
    M_mat = coo_matrix((data_mv, (rows_v, cols_v)), shape=(N, N)).tocsr()
    K_bnd = coo_matrix((data_kb, (rows_b, cols_b)), shape=(N, N)).tocsr()
    return {
        "K_vol": K_vol,
        "K_bnd": K_bnd,
        "M": M_mat,
        "F_vol": F_vol,
        "subdomains": subdomains,
    }


def solve_dirichlet_and_eval_errors(
    A: csr_matrix,
    F: np.ndarray,
    coords: np.ndarray,
    vectors: np.ndarray,
    is_bnd: np.ndarray,
    subdomains: List[MLPGSubdomain2D],
    exact_fn,
    exact_curl_fn,
    A_sym_precond: Optional[csr_matrix] = None,
) -> Dict[str, Any]:
    N = len(coords)
    e_exact = np.array([float(np.dot(exact_fn(x, y), t)) for (x, y), t in zip(coords, vectors)])
    int_idx = np.where(~is_bnd)[0]
    bnd_idx = np.where(is_bnd)[0]

    A_ii = A[int_idx, :][:, int_idx].tocsc()
    A_ib = A[int_idx, :][:, bnd_idx]
    rhs = F[int_idx] - A_ib.dot(e_exact[bnd_idx])

    e_sol = np.zeros(N, dtype=float)
    e_sol[bnd_idx] = e_exact[bnd_idx]
    e_sol[int_idx] = spsolve(A_ii, rhs)

    # Asymmetry ratio of A_ii
    diff_sym = A_ii - A_ii.T
    asym_ratio = float(np.linalg.norm(diff_sym.data) / max(np.linalg.norm(A_ii.data), 1e-30))

    # Optional GMRES preconditioned by symmetric (K_vol + kappa M)_ii
    gmres_iters = None
    if A_sym_precond is not None:
        P_ii = A_sym_precond[int_idx, :][:, int_idx].tocsc()
        lu_P = splu(P_ii)
        M_pre = LinearOperator(A_ii.shape, matvec=lu_P.solve)
        it_count = [0]

        def cb(_):
            it_count[0] += 1

        x_gmres, info = gmres(A_ii, rhs, M=M_pre, rtol=1e-10, atol=0.0, restart=50, maxiter=200, callback=cb, callback_type="pr_norm")
        gmres_iters = it_count[0] if info == 0 else -info

    # Evaluate broken L2 and broken curl errors
    num_L2, den_L2 = 0.0, 0.0
    num_curl, den_curl = 0.0, 0.0
    domain_bounds = (0.0, np.pi, 0.0, np.pi)
    for sub in subdomains:
        e_loc = e_sol[sub.stencil_nodes]
        pts, wts = get_disk_quadrature_points_2d(
            sub.center, sub.radius_s, n_r=4, n_theta=12, domain_bounds=domain_bounds
        )
        N_all = sub.eval_shape_functions(pts)
        E_h = np.einsum("mqa,a->mq", N_all, e_loc)
        E_ex = np.array([exact_fn(px, py) for px, py in pts])
        num_L2 += float(np.sum(wts * np.sum((E_h - E_ex) ** 2, axis=1)))
        den_L2 += float(np.sum(wts * np.sum(E_ex ** 2, axis=1)))

        curl_h = float(np.dot(sub.eval_shape_curl(), e_loc))
        curl_ex = np.array([exact_curl_fn(px, py) for px, py in pts])
        num_curl += float(np.sum(wts * ((curl_h - curl_ex) ** 2)))
        den_curl += float(np.sum(wts * (curl_ex ** 2)))

    err_inf = float(np.max(np.abs(e_sol[int_idx] - e_exact[int_idx])) / max(np.max(np.abs(e_exact[int_idx])), 1e-30))
    return {
        "err_L2": float(np.sqrt(num_L2 / max(den_L2, 1e-30))),
        "err_curl": float(np.sqrt(num_curl / max(den_curl, 1e-30))),
        "err_inf": err_inf,
        "asym_ratio": asym_ratio,
        "gmres_iters": gmres_iters,
    }


def run_experiments():
    # 1. Static affine patch test (k0=0, J_s=0) WITHOUT analytical contour fluxes on partial(D_I)
    print("=" * 95)
    print("  1. STATIC AFFINE PATCH TEST (k0=0, J_s=0, ZERO ANALYTICAL FLUXES ON partial(D_I))")
    print("=" * 95)
    affine_E = lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y])
    affine_curl = lambda x, y: 0.7
    coords_p, vecs_p, bnd_p = generate_cavity_mesh(Nx=11, Ny=11, jitter_frac=0.0, interior_type="random", seed=42)

    for label, amode, inc_b, inc_phys, fmode in [
        ("Baseline SSG (no K_bnd)", "subdomain_galerkin", False, False, "neighbor_avg"),
        ("Subdomain Galerkin + K_bnd (L_sI only, neighbor_avg)", "subdomain_galerkin", True, False, "neighbor_avg"),
        ("Subdomain Galerkin + K_bnd (L_sI + Gamma_s, neighbor_avg)", "subdomain_galerkin", True, True, "neighbor_avg"),
        ("Subdomain Galerkin + K_bnd (L_sI + Gamma_s, point_stencil)", "subdomain_galerkin", True, True, "point_stencil"),
    ]:
        asm = assemble_mlpg_system_with_boundary_flux(
            coords_p, vecs_p, bnd_p, s_div=6.0, kappa=0.0,
            assembly_mode=amode, include_k_bnd=inc_b,
            include_phys_bnd_flux=inc_phys, flux_mode=fmode, source_fn=None
        )
        A = asm["K_vol"] - asm["K_bnd"]
        res = solve_dirichlet_and_eval_errors(
            A, asm["F_vol"], coords_p, vecs_p, bnd_p, asm["subdomains"], affine_E, affine_curl
        )
        print(f"  {label:<58s} | err_inf = {res['err_inf']:.3e} | asym = {res['asym_ratio']:.3f}")

    # 2. Conventional Volumetric-Source MMS (kappa=1.0, J_s=(2+kappa)*E*, NO analytical fluxes)
    print("\n" + "=" * 95)
    print("  2. CONVENTIONAL VOLUMETRIC-SOURCE MMS (kappa=1.0, NO ANALYTICAL FLUXES, 15x15 to 49x49)")
    print("=" * 95)
    grid_sizes = [15, 22, 29, 36, 43, 49]
    kappa = 1.0
    src_fn = lambda x, y: source_mms(x, y, kappa=kappa)
    h0 = np.pi / 14.0

    configs = [
        ("C1. Subdomain Galerkin + K_bnd (L_sI + Gamma_s, neighbor_avg, s_div=6.0)", False, False),
        ("C2. Subdomain Galerkin + K_bnd (L_sI + Gamma_s, neighbor_avg, scaled s_div)", True, False),
        ("C3. Subdomain Galerkin + K_bnd (L_sI + Gamma_s, neighbor_avg, scaled s_div + c_stab=70)", True, True),
    ]

    for label, use_scaled_sdiv, use_stab in configs:
        print(f"\n>>> {label} <<<")
        h_vals, eL2_vals, eCurl_vals = [], [], []
        for N in grid_sizes:
            h_nom = np.pi / float(N - 1)
            s_div_val = 15.0 * ((h0 / h_nom) ** 1.5) if use_scaled_sdiv else 6.0
            alpha_stab_val = 70.0 * ((h_nom / np.pi) ** 2) if use_stab else 0.0
            coords, vecs, bnd = generate_cavity_mesh(Nx=N, Ny=N, jitter_frac=0.0, interior_type="random", seed=42)
            asm = assemble_mlpg_system_with_boundary_flux(
                coords, vecs, bnd, s_div=s_div_val, kappa=kappa, alpha_stab=alpha_stab_val,
                assembly_mode="subdomain_galerkin", include_k_bnd=True,
                include_phys_bnd_flux=True, flux_mode="neighbor_avg", source_fn=src_fn
            )
            A_sym = asm["K_vol"] + kappa * asm["M"]
            A = A_sym - asm["K_bnd"]
            res = solve_dirichlet_and_eval_errors(
                A, asm["F_vol"], coords, vecs, bnd, asm["subdomains"],
                exact_solution_mms, exact_curl_mms,
                A_sym_precond=A_sym,
            )
            h_vals.append(h_nom)
            eL2_vals.append(res["err_L2"])
            eCurl_vals.append(res["err_curl"])
            print(
                f"  {N:2d}x{N:<2d} (h={h_nom:.4f}): L2={res['err_L2']*100:6.2f}%, "
                f"curl={res['err_curl']*100:6.2f}%, asym={res['asym_ratio']:.3f} | gmres_it={res['gmres_iters']}"
            )
        p_L2, _ = np.polyfit(np.log(h_vals), np.log(eL2_vals), 1)
        p_curl, _ = np.polyfit(np.log(h_vals), np.log(eCurl_vals), 1)
        print(f"  Fitted rates: p_L2 = {p_L2:.2f}, p_curl = {p_curl:.2f}")


if __name__ == "__main__":
    run_experiments()
