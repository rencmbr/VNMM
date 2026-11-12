"""
Cálculo de Modos Ressonantes e Autovalores TEz em Cavidades Carregadas com Dielétrico
=====================================================================================
Arquivo: VNMM_Dielectric_Cavity_Eigenvalues.py

Objetivo:
---------
Calcular os autovalores (lambda = k_0^2) e os números de onda de corte (k_c = sqrt(lambda))
para os modos ressonantes TE_z em uma cavidade retangular [0, Lx] x [0, Ly] com paredes PEC,
parcialmente preenchida com um bloco dielétrico (eps_r1 = 1.0 para x < x_int, eps_r2 = 4.0 para x > x_int).

Metodologia:
------------
1. Solução Analítica Exata:
   Resolve as equações transcendentais de dispersão de interface para os modos com m = 0, 1, 2,...
   - Para m = 0: k_x1 * cot(k_x1 * d) + k_x2 * cot(k_x2 * (Lx - d)) = 0
   - Para m >= 1: beta_1/eps1 * tan(beta_1 * d) + beta_2/eps2 * tan(beta_2 * (Lx - d)) = 0 (ou ramo hiperbólico)
2. Discretização VNMM 2D com Partição de Domínio:
   - Base linear completa P^1 (6 nós por estêncil).
   - Nós de contorno PEC escalonados (staggered midpoints) sem nós nos vértices.
   - Nós sobre a interface Gamma_int com diretores puramente tangenciais.
   - Estênceis de integração estritamente restritos a cada material.
3. Montagem Fraca e Purificação Espectral:
   - K = K_curl + s_div * K_div
   - M = M_eps (pesado por eps_r(x))
   - Critério de pureza física via razão de energia de divergência:
     R_div = (v^T K_div v) / (v^T K_curl v) * 100% < 5.0%
"""

import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.optimize import root_scalar
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

DIR_EXPERIMENTOS = Path(__file__).resolve().parent
DIR_2D = DIR_EXPERIMENTOS.parent
ROOT_DIR = DIR_2D.parent

for p in [str(DIR_2D), str(ROOT_DIR), str(DIR_EXPERIMENTOS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from scripts.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1
from src.gauss_quadrature_2d import get_gauss_points_weights_1d


# ==============================================================================
# 1. SOLUÇÃO ANALÍTICA TRANSCENDENTAL
# ==============================================================================

def compute_analytical_dielectric_cavity_modes(Lx=np.pi, Ly=np.pi, d=np.pi/2.0, eps1=1.0, eps2=4.0, max_modes=10):
    """
    Calcula as raízes analíticas exatas da equação transcendental para a cavidade
    retangular com placa dielétrica em [d, Lx].
    """
    all_modes = []

    # 1.1 Modos m = 0 (sem variação em y)
    def f_m0(k0):
        kx1 = k0 * np.sqrt(eps1)
        kx2 = k0 * np.sqrt(eps2)
        t1 = np.tan(kx1 * d)
        t2 = np.tan(kx2 * (Lx - d))
        if abs(t1) < 1e-12 or abs(t2) < 1e-12:
            return 1e12
        return kx1 / t1 + kx2 / t2

    k_grid = np.linspace(0.05, 5.0, 3000)
    for i in range(len(k_grid) - 1):
        ka, kb = k_grid[i], k_grid[i + 1]
        fa, fb = f_m0(ka), f_m0(kb)
        if fa * fb < 0 and abs(fa - fb) < 100:
            try:
                sol = root_scalar(f_m0, bracket=[ka, kb])
                if sol.converged:
                    k0_val = sol.root
                    if not any(abs(k0_val - m['k0']) < 1e-4 for m in all_modes):
                        all_modes.append({
                            'm': 0,
                            'k0': k0_val,
                            'lambda': k0_val**2,
                            'desc': f"TE(0, {len([x for x in all_modes if x['m'] == 0]) + 1})"
                        })
            except Exception:
                pass

    # 1.2 Modos m >= 1 (com variação senoidal em y)
    for m in range(1, 5):
        ky = m * np.pi / Ly

        def f_m(k0):
            val1 = k0**2 * eps1 - ky**2
            val2 = k0**2 * eps2 - ky**2
            if val2 <= 0:
                return 1e12
            beta2 = np.sqrt(val2)

            if val1 > 0:
                beta1 = np.sqrt(val1)
                t1 = np.tan(beta1 * d)
                if abs(t1) < 1e-12:
                    return 1e12
                term1 = (beta1 / eps1) * t1
            else:
                alpha1 = np.sqrt(-val1)
                t1 = np.tanh(alpha1 * d)
                term1 = -(alpha1 / eps1) * t1

            t2 = np.tan(beta2 * (Lx - d))
            if abs(t2) < 1e-12:
                return 1e12
            term2 = (beta2 / eps2) * t2
            return term1 + term2

        for i in range(len(k_grid) - 1):
            ka, kb = k_grid[i], k_grid[i + 1]
            fa, fb = f_m(ka), f_m(kb)
            if fa * fb < 0 and abs(fa - fb) < 50:
                try:
                    sol = root_scalar(f_m, bracket=[ka, kb])
                    if sol.converged:
                        k0_val = sol.root
                        if not any(abs(k0_val - x['k0']) < 1e-4 for x in all_modes):
                            all_modes.append({
                                'm': m,
                                'k0': k0_val,
                                'lambda': k0_val**2,
                                'desc': f"TE({m}, {len([x for x in all_modes if x['m'] == m]) + 1})"
                            })
                except Exception:
                    pass

    # Ordenar por lambda em ordem crescente
    all_modes.sort(key=lambda x: x['lambda'])
    return all_modes[:max_modes]


# ==============================================================================
# 2. GERAÇÃO DE MALHA ESCALONADA (STAGGERED) PARA CAVIDADE BIMATERIAL
# ==============================================================================

def generate_staggered_dielectric_mesh(Lx=np.pi, Ly=np.pi, x_int=np.pi/2.0, Nx1=11, Nx2=11, Ny=21):
    """
    Gera a nuvem de nós com condições estritas de contorno PEC escalonadas
    e diretores tangenciais na interface Gamma_int.
    """
    dx1 = x_int / (Nx1 - 1)
    dx2 = (Lx - x_int) / (Nx2 - 1)
    dy = Ly / (Ny - 1)

    coords = []
    vectors = []
    is_pec = []
    mat_ids = []

    # 1. Nós internos da Região 1 (Material 1)
    for j in range(1, Ny - 1):
        y = j * dy
        for i in range(1, Nx1 - 1):
            x = i * dx1
            coords.append([x, y])
            vec = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)]) if (i + j) % 2 == 0 else np.array([-1.0/np.sqrt(2), 1.0/np.sqrt(2)])
            vectors.append(vec)
            is_pec.append(False)
            mat_ids.append(1)

    # 2. Nós internos da Região 2 (Material 2)
    for j in range(1, Ny - 1):
        y = j * dy
        for i in range(1, Nx2 - 1):
            x = x_int + i * dx2
            coords.append([x, y])
            vec = np.array([1.0/np.sqrt(2), 1.0/np.sqrt(2)]) if (i + j) % 2 == 0 else np.array([-1.0/np.sqrt(2), 1.0/np.sqrt(2)])
            vectors.append(vec)
            is_pec.append(False)
            mat_ids.append(2)

    # 3. Nós sobre a Interface Gamma_int (x = x_int)
    for j in range(1, Ny - 1):
        y = j * dy
        coords.append([x_int, y])
        vectors.append(np.array([0.0, 1.0]))  # Tangente vertical à interface
        is_pec.append(False)
        mat_ids.append(0)  # Interface ativa

    # 4. Nós de contorno PEC escalonados (sem nós de quina)
    # y = 0 (Mat 1)
    for i in range(Nx1 - 1):
        coords.append([(i + 0.5) * dx1, 0.0])
        vectors.append(np.array([1.0, 0.0]))
        is_pec.append(True)
        mat_ids.append(1)

    # y = 0 (Mat 2)
    for i in range(Nx2 - 1):
        coords.append([x_int + (i + 0.5) * dx2, 0.0])
        vectors.append(np.array([1.0, 0.0]))
        is_pec.append(True)
        mat_ids.append(2)

    # y = Ly (Mat 1)
    for i in range(Nx1 - 1):
        coords.append([(i + 0.5) * dx1, Ly])
        vectors.append(np.array([1.0, 0.0]))
        is_pec.append(True)
        mat_ids.append(1)

    # y = Ly (Mat 2)
    for i in range(Nx2 - 1):
        coords.append([x_int + (i + 0.5) * dx2, Ly])
        vectors.append(np.array([1.0, 0.0]))
        is_pec.append(True)
        mat_ids.append(2)

    # x = 0 (Parede vertical esquerda)
    for j in range(Ny - 1):
        coords.append([0.0, (j + 0.5) * dy])
        vectors.append(np.array([0.0, 1.0]))
        is_pec.append(True)
        mat_ids.append(1)

    # x = Lx (Parede vertical direita)
    for j in range(Ny - 1):
        coords.append([Lx, (j + 0.5) * dy])
        vectors.append(np.array([0.0, 1.0]))
        is_pec.append(True)
        mat_ids.append(2)

    return (
        np.array(coords, dtype=float),
        np.array(vectors, dtype=float),
        np.array(is_pec, dtype=bool),
        np.array(mat_ids, dtype=int)
    )


# ==============================================================================
# 3. RESOLVEDOR ESPECTRAL VNMM 2D PARA CAVIDADE CARREGADA
# ==============================================================================

def solve_dielectric_cavity_eigenvalues(
    Lx=np.pi,
    Ly=np.pi,
    x_int=np.pi/2.0,
    eps1=1.0,
    eps2=4.0,
    Nx1=11,
    Nx2=11,
    Ny=21,
    Ncx=16,
    Ncy=16,
    points_per_dir=2,
    s_div=6.0,
    Tol_det=1e-5
):
    """
    Executa a montagem variacional e resolve o problema espectral generalizado.
    """
    coords, vectors, is_pec, mat_ids = generate_staggered_dielectric_mesh(
        Lx=Lx, Ly=Ly, x_int=x_int, Nx1=Nx1, Nx2=Nx2, Ny=Ny
    )

    total_nodes = len(coords)
    pec_count = np.sum(is_pec)
    active_count = np.sum(~is_pec)

    # Subconjuntos de busca espacial restritos por material
    subsets = {}
    for m in [1, 2]:
        cand = np.where((mat_ids == m) | (mat_ids == 0))[0]
        subsets[m] = {
            'indices': cand,
            'coords': coords[cand],
            'vectors': vectors[cand],
            'tree': KDTree(coords[cand])
        }

    # Quadratura numérica de Gauss em células de fundo
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)
    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)

    rows_Kc, cols_Kc, data_Kc = [], [], []
    rows_Kd, cols_Kd, data_Kd = [], [], []
    rows_M, cols_M, data_M = [], [], []

    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy_c = y1 - y0
        yc = 0.5 * (y0 + y1)
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx_c = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_J = 0.25 * dx_c * dy_c

            target_mat = 1 if xc < x_int else 2
            eps_r = eps1 if target_mat == 1 else eps2
            sub = subsets[target_mat]

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx_c * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy_c * eta
                    Pg = np.array([xg, yg])
                    wg = wi * wj * det_J

                    sel_local, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
                        P=Pg,
                        nodes_coords=sub['coords'],
                        nodes_vectors=sub['vectors'],
                        search_tree=sub['tree'],
                        K=12,
                        Tol_det=Tol_det,
                        adaptive=True
                    )
                    sel_global = sub['indices'][sel_local]

                    beta = np.linalg.inv(matrix_A)
                    Phi_g = beta[0:2, :]
                    rot_Phi = beta[4, :] - beta[3, :]
                    div_Phi = beta[2, :] + beta[5, :]

                    rot_outer = np.outer(rot_Phi, rot_Phi) * wg
                    div_outer = np.outer(div_Phi, div_Phi) * wg
                    vec_dot = (Phi_g.T @ Phi_g) * (eps_r * wg)

                    for a in range(6):
                        Ia = sel_global[a]
                        for b in range(6):
                            Ib = sel_global[b]
                            rows_Kc.append(Ia); cols_Kc.append(Ib); data_Kc.append(rot_outer[a, b])
                            rows_Kd.append(Ia); cols_Kd.append(Ib); data_Kd.append(div_outer[a, b])
                            rows_M.append(Ia); cols_M.append(Ib); data_M.append(vec_dot[a, b])

    Kc = csr_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(total_nodes, total_nodes))
    Kd = csr_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(total_nodes, total_nodes))
    M = csr_matrix((data_M, (rows_M, cols_M)), shape=(total_nodes, total_nodes))

    # Redução por condições homogêneas de Dirichlet nas paredes PEC (E_t = 0)
    int_idx = np.where(~is_pec)[0]
    Kc_red = 0.5 * (Kc[int_idx, :][:, int_idx].toarray() + Kc[int_idx, :][:, int_idx].toarray().T)
    Kd_red = 0.5 * (Kd[int_idx, :][:, int_idx].toarray() + Kd[int_idx, :][:, int_idx].toarray().T)
    M_red = 0.5 * (M[int_idx, :][:, int_idx].toarray() + M[int_idx, :][:, int_idx].toarray().T)

    K_red = Kc_red + s_div * Kd_red

    # Resolução do problema generalizado de autovalores
    vals, vecs = la.eigh(K_red, M_red)

    # Filtrar e purificar modos físicos livres de divergência (R_div < 5.0%)
    vals_pos = vals[vals > 0.05]
    vecs_pos = vecs[:, vals > 0.05]

    physical_modes = []
    for i in range(len(vals_pos)):
        v = vecs_pos[:, i]
        e_c = v.T @ Kc_red @ v
        e_d = v.T @ Kd_red @ v
        e_m = v.T @ M_red @ v
        r_div = (e_d / e_c) * 100.0 if e_c > 1e-12 else 99999.0
        lambda_curl = e_c / e_m
        kc = np.sqrt(lambda_curl)

        if r_div < 5.0:
            # Reconstruir autovetor global
            v_global = np.zeros(total_nodes)
            v_global[int_idx] = v
            physical_modes.append({
                'lambda': lambda_curl,
                'kc': kc,
                'r_div': r_div,
                'eigenvector': v_global
            })

    return {
        'coords': coords,
        'vectors': vectors,
        'is_pec': is_pec,
        'mat_ids': mat_ids,
        'int_indices': int_idx,
        'subsets': subsets,
        'physical_modes': physical_modes,
        'total_nodes': total_nodes,
        'pec_nodes': pec_count,
        'active_nodes': active_count
    }


# ==============================================================================
# 4. RECONSTRUÇÃO DE CAMPO E VISUALIZAÇÃO GRÁFICA
# ==============================================================================

def plot_dielectric_resonant_modes(
    results,
    analytical_modes,
    Lx=np.pi,
    Ly=np.pi,
    x_int=np.pi/2.0,
    output_path=None,
    show=False
):
    """
    Gera painel visual com a comparação dos 4 primeiros modos ressonantes físicos.
    """
    coords = results['coords']
    vectors = results['vectors']
    mat_ids = results['mat_ids']
    modes = results['physical_modes']
    subsets = results['subsets']

    # Grade fina de avaliação para reconstrução do campo
    nx_eval, ny_eval = 30, 30
    xg = np.linspace(0.02 * Lx, 0.98 * Lx, nx_eval)
    yg = np.linspace(0.02 * Ly, 0.98 * Ly, ny_eval)
    Xg, Yg = np.meshgrid(xg, yg)
    eval_pts = np.column_stack([Xg.ravel(), Yg.ravel()])

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes_flat = axes.flatten()

    num_plot = min(4, len(modes))
    for m_idx in range(num_plot):
        ax = axes_flat[m_idx]
        mode = modes[m_idx]
        e_vec = mode['eigenvector']
        ref = analytical_modes[m_idx] if m_idx < len(analytical_modes) else None

        # Reconstruir E no grid de avaliação
        E_grid = np.zeros((len(eval_pts), 2))
        for p_i, P in enumerate(eval_pts):
            t_mat = 1 if P[0] < x_int else 2
            sub = subsets[t_mat]

            sel_loc, _, mat_A, _ = support_nodes_vnmm_2d_6_p1(
                P=P,
                nodes_coords=sub['coords'],
                nodes_vectors=sub['vectors'],
                search_tree=sub['tree'],
                K=12,
                Tol_det=1e-5,
                adaptive=True
            )
            sel_glob = sub['indices'][sel_loc]
            beta = np.linalg.inv(mat_A)
            Phi = beta[0:2, :]
            E_grid[p_i] = Phi @ e_vec[sel_glob]

        E_norm = np.linalg.norm(E_grid, axis=1).reshape((ny_eval, nx_eval))
        E_max = np.max(E_norm) if np.max(E_norm) > 1e-12 else 1.0
        E_norm /= E_max

        # Fundo sombreado das duas regiões
        ax.fill_between([0.0, x_int], 0.0, Ly, color="#E3F2FD", alpha=0.4, zorder=1)
        ax.fill_between([x_int, Lx], 0.0, Ly, color="#FFF3E0", alpha=0.4, zorder=1)
        ax.axvline(x=x_int, color="#C62828", linestyle="--", linewidth=1.8, zorder=2)

        # Mapa de intensidade do campo elétrico
        cf = ax.contourf(Xg, Yg, E_norm, levels=25, cmap="magma", alpha=0.85, zorder=2)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label="$\\|\\mathbf{E}\\| / \\|\\mathbf{E}\\|_{\\max}$")

        # Quiver do campo vetorial
        step = 2
        ax.quiver(
            Xg[::step, ::step], Yg[::step, ::step],
            E_grid[:, 0].reshape((ny_eval, nx_eval))[::step, ::step],
            E_grid[:, 1].reshape((ny_eval, nx_eval))[::step, ::step],
            color="white", alpha=0.9, width=0.0035, scale=25.0, zorder=3
        )

        ref_str = f"Ref: $\\lambda={ref['lambda']:.4f}, k_c={ref['k0']:.4f}$" if ref else ""
        err_kc = abs(mode['kc'] - ref['k0']) / ref['k0'] * 100.0 if ref else 0.0

        ax.set_title(
            f"Modo {m_idx + 1}: VNMM $\\lambda = {mode['lambda']:.4f}$ ($k_c = {mode['kc']:.4f}$)\n"
            f"{ref_str} (Erro: {err_kc:.2f}%) | $R_{{\\mathrm{{div}}}} = {mode['r_div']:.2f}\\%$",
            fontsize=10.5, fontweight="bold"
        )
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_xlim(0, Lx)
        ax.set_ylim(0, Ly)
        ax.set_aspect("equal")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico de modos ressonantes salvo com sucesso em: {output_path}")

    if show:
        plt.show()


# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    print("=" * 85)
    print("   ESPECTRO DE AUTOVALORES TEz EM CAVIDADE CARREGADA COM DIELÉTRICO (VNMM 2D)")
    print("=" * 85)

    Lx = np.pi
    Ly = np.pi
    x_int = np.pi / 2.0
    eps1 = 1.0
    eps2 = 4.0

    print(f"\nGeometria da Cavidade: [0, {Lx:.3f}] x [0, {Ly:.3f}] m")
    print(f"- Região 1 (x < {x_int:.3f}): eps_r1 = {eps1}")
    print(f"- Região 2 (x > {x_int:.3f}): eps_r2 = {eps2} (Dielétrico)")

    # 1. Obter modos analíticos exatos
    print("\n--- 1. Cálculo dos Autovalores Analíticos Exatos (Equações Transcendentais) ---")
    analytical_modes = compute_analytical_dielectric_cavity_modes(
        Lx=Lx, Ly=Ly, d=x_int, eps1=eps1, eps2=eps2, max_modes=8
    )

    print(f"{'Modo':<8} | {'Tipo':<10} | {'lambda_exato':<14} | {'kc_exato':<12}")
    print("-" * 52)
    for idx, m in enumerate(analytical_modes, 1):
        print(f"{idx:<8} | {m['desc']:<10} | {m['lambda']:<14.6f} | {m['k0']:<12.6f}")

    # 2. Resolução com VNMM 2D
    print("\n--- 2. Resolução Numérica via VNMM 2D com Restrição Estrita de Suportes ---")
    results = solve_dielectric_cavity_eigenvalues(
        Lx=Lx, Ly=Ly, x_int=x_int, eps1=eps1, eps2=eps2,
        Nx1=11, Nx2=11, Ny=21,
        Ncx=16, Ncy=16,
        points_per_dir=2,
        s_div=6.0
    )

    phys_modes = results['physical_modes']
    print(f"Total de nós: {results['total_nodes']} (PEC: {results['pec_nodes']}, Ativos: {results['active_nodes']})")
    print(f"Modos físicos purificados identificados (R_div < 5%): {len(phys_modes)}")

    print("\n" + "=" * 85)
    print(f"{'Modo':<5} | {'lambda_exato':<13} | {'lambda_VNMM':<13} | {'kc_exato':<10} | {'kc_VNMM':<10} | {'Erro kc (%)':<12} | {'R_div (%)':<10}")
    print("=" * 85)

    errors_kc = []
    for i in range(min(len(analytical_modes), len(phys_modes))):
        ref = analytical_modes[i]
        num = phys_modes[i]
        err_kc = abs(num['kc'] - ref['k0']) / ref['k0'] * 100.0
        errors_kc.append(err_kc)

        print(f"{i+1:<5} | {ref['lambda']:<13.6f} | {num['lambda']:<13.6f} | {ref['k0']:<10.6f} | {num['kc']:<10.6f} | {err_kc:<12.2f} | {num['r_div']:<10.3f}")

    print("-" * 85)
    print(f"Erro Médio em kc : {np.mean(errors_kc):.2f}%")
    print(f"Erro Máximo em kc: {np.max(errors_kc):.2f}%")
    print("=" * 85)

    # 3. Gerar e salvar gráfico dos modos
    plot_path = DIR_EXPERIMENTOS / "VNMM_Dielectric_Cavity_Modes.png"
    print("\nGerando mapas de contorno dos modos ressonantes...")
    plot_dielectric_resonant_modes(
        results=results,
        analytical_modes=analytical_modes,
        Lx=Lx, Ly=Ly, x_int=x_int,
        output_path=str(plot_path),
        show=False
    )
    print("\nSimulação concluída com sucesso!")


if __name__ == "__main__":
    main()
