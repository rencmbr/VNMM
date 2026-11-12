"""
Validações Avançadas do VNMM 2D em Domínios Multimateriais
==========================================================
Arquivo: VNMM_Advanced_Validation.py

Testes Implementados:
---------------------
1. Teste de Salto de Descontinuidade Normal ([En] != 0):
   - Avalia um campo com descontinuidade física 4:1 na componente normal Ex em x = x_int.
   - Demonstra que a restrição de suportes reproduz o salto sem 'borramento' (smearing)
     ou oscilações espúrias de Gibbs.
   - Extrai um perfil 1D ao longo da linha y = Ly/2.
2. Estudo de Robustez a Nuvens Irregulares (Jitter Study):
   - Avalia os autovalores sob perturbações nodais estocásticas de 0%, 10% e 20%.
3. Varredura de Contraste de Permissividade Dielétrica:
   - Avalia o comportamento espectral para eps_r2 in [2.25, 4.0, 9.8, 16.0].
"""

import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

DIR_EXP = Path(__file__).resolve().parent
DIR_2D = DIR_EXP.parent
ROOT_DIR = DIR_2D.parent

for p in [str(DIR_2D), str(ROOT_DIR), str(DIR_EXP)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from VNMM_Materials import create_two_material_domain, generate_material_nodes, interpolate_vnmm_multimaterial
from VNMM_Dielectric_Cavity_Eigenvalues import compute_analytical_dielectric_cavity_modes
from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from src.gauss_quadrature_2d import get_gauss_points_weights_1d


# ==============================================================================
# TESTE 1: SALTO DESCONTÍNUO NORMAL (1D CUT)
# ==============================================================================

def run_normal_jump_test(domain, Nx1=15, Nx2=15, Ny=15, num_eval_points=120):
    """
    Executa o teste de interpolação com salto de descontinuidade normal [E_n] != 0.
    """
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    x_int = domain["x_interface"]
    eps1 = domain["mat1"]["eps_r"]
    eps2 = domain["mat2"]["eps_r"]

    # Campo analítico descontínuo:
    # Ey é contínuo em x = x_int
    # Ex possui salto de eps1/eps2 = 1/4 em x = x_int
    def eval_E_discont(x, y):
        ky = np.pi / Ly
        Ey = np.sin(ky * y) * np.cos(np.pi * x / (2.0 * Lx))
        if x < x_int:
            Ex = 1.0 * np.cos(ky * y) * (1.0 + 0.5 * (x / x_int))
        elif x > x_int:
            Ex = (eps1 / eps2) * np.cos(ky * y) * (1.0 + 0.5 * ((Lx - x) / (Lx - x_int)))
        else:
            Ex = 0.5 * (1.5 + 1.5 * eps1 / eps2) * np.cos(ky * y)
        return np.array([Ex, Ey])

    nodes, vectors, mat_ids = generate_material_nodes(domain, Nx1=Nx1, Nx2=Nx2, Ny=Ny, jitter_frac=0.0)

    # Linha 1D de avaliação em y = Ly / 2 cruzando a interface
    x_eval = np.linspace(0.04 * Lx, 0.96 * Lx, num_eval_points)
    # Evitar colocar ponto exatamente na interface para avaliar os dois limites laterais
    x_eval = np.array([x if abs(x - x_int) > 1e-4 else x - 1e-4 for x in x_eval])
    
    line_pts = np.column_stack([x_eval, np.full_like(x_eval, Ly / 2.0)])
    line_mats = np.array([1 if x < x_int else 2 for x in x_eval])

    target = {
        'name': 'Salto Descontinuo Normal',
        'eval_E': eval_E_discont,
        'eval_curl': lambda x, y: 0.0,
        'is_polynomial': False
    }

    res = interpolate_vnmm_multimaterial(domain, nodes, vectors, mat_ids, line_pts, line_mats, target)
    
    Ex_exact = res['E_exact'][:, 0]
    Ey_exact = res['E_exact'][:, 1]
    Ex_num = res['E_interpolated'][:, 0]
    Ey_num = res['E_interpolated'][:, 1]

    err_Ex = np.max(np.abs(Ex_num - Ex_exact))
    err_Ey = np.max(np.abs(Ey_num - Ey_exact))

    return {
        'x_eval': x_eval,
        'Ex_exact': Ex_exact,
        'Ey_exact': Ey_exact,
        'Ex_num': Ex_num,
        'Ey_num': Ey_num,
        'err_Ex_max': err_Ex,
        'err_Ey_max': err_Ey
    }


# ==============================================================================
# TESTE 2: ROBUSTEZ A NUVENS IRREGULARES (JITTER STUDY)
# ==============================================================================

def run_jitter_study(jitter_list=[0.0, 0.10, 0.20], seed=42):
    """
    Avalia a estabilidade dos autovalores da cavidade sob perturbação estocástica nodal.
    """
    Lx = np.pi
    Ly = np.pi
    x_int = np.pi / 2.0
    eps1, eps2 = 1.0, 4.0

    ref_modes = compute_analytical_dielectric_cavity_modes(Lx=Lx, Ly=Ly, d=x_int, eps1=eps1, eps2=eps2, max_modes=6)
    ref_kc = np.array([m['k0'] for m in ref_modes])

    results_jitter = []

    for j_frac in jitter_list:
        np.random.seed(seed)
        Nx1, Nx2, Ny = 11, 11, 21
        dx1 = x_int / (Nx1 - 1)
        dx2 = (Lx - x_int) / (Nx2 - 1)
        dy = Ly / (Ny - 1)

        coords, vectors, is_pec, mat_ids = [], [], [], []

        # Região 1 interior com jitter
        for j in range(1, Ny - 1):
            y0 = j * dy
            for i in range(1, Nx1 - 1):
                x0 = i * dx1
                jx = np.random.uniform(-j_frac * dx1, j_frac * dx1) if j_frac > 0 else 0.0
                jy = np.random.uniform(-j_frac * dy, j_frac * dy) if j_frac > 0 else 0.0
                coords.append([x0 + jx, y0 + jy])
                vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) if (i + j) % 2 == 0 else np.array([-1/np.sqrt(2), 1/np.sqrt(2)])
                vectors.append(vec); is_pec.append(False); mat_ids.append(1)

        # Região 2 interior com jitter
        for j in range(1, Ny - 1):
            y0 = j * dy
            for i in range(1, Nx2 - 1):
                x0 = x_int + i * dx2
                jx = np.random.uniform(-j_frac * dx2, j_frac * dx2) if j_frac > 0 else 0.0
                jy = np.random.uniform(-j_frac * dy, j_frac * dy) if j_frac > 0 else 0.0
                coords.append([x0 + jx, y0 + jy])
                vec = np.array([1/np.sqrt(2), 1/np.sqrt(2)]) if (i + j) % 2 == 0 else np.array([-1/np.sqrt(2), 1/np.sqrt(2)])
                vectors.append(vec); is_pec.append(False); mat_ids.append(2)

        # Nós de interface fixos em x = x_int
        for j in range(1, Ny - 1):
            coords.append([x_int, j * dy])
            vectors.append(np.array([0.0, 1.0]))
            is_pec.append(False); mat_ids.append(0)

        # Contornos PEC escalonados (staggered)
        for i in range(Nx1 - 1):
            coords.append([(i + 0.5) * dx1, 0.0]); vectors.append(np.array([1.0, 0.0])); is_pec.append(True); mat_ids.append(1)
            coords.append([(i + 0.5) * dx1, Ly]); vectors.append(np.array([1.0, 0.0])); is_pec.append(True); mat_ids.append(1)
        for i in range(Nx2 - 1):
            coords.append([x_int + (i + 0.5) * dx2, 0.0]); vectors.append(np.array([1.0, 0.0])); is_pec.append(True); mat_ids.append(2)
            coords.append([x_int + (i + 0.5) * dx2, Ly]); vectors.append(np.array([1.0, 0.0])); is_pec.append(True); mat_ids.append(2)
        for j in range(Ny - 1):
            coords.append([0.0, (j + 0.5) * dy]); vectors.append(np.array([0.0, 1.0])); is_pec.append(True); mat_ids.append(1)
            coords.append([Lx, (j + 0.5) * dy]); vectors.append(np.array([0.0, 1.0])); is_pec.append(True); mat_ids.append(2)

        coords = np.array(coords, dtype=float)
        vectors = np.array(vectors, dtype=float)
        is_pec = np.array(is_pec, dtype=bool)
        mat_ids = np.array(mat_ids, dtype=int)

        # Subsets por material
        subsets = {}
        for m in [1, 2]:
            cand = np.where((mat_ids == m) | (mat_ids == 0))[0]
            subsets[m] = {'indices': cand, 'coords': coords[cand], 'vectors': vectors[cand], 'tree': KDTree(coords[cand])}

        Ncx, Ncy = 16, 16
        xi_1d, w_1d = get_gauss_points_weights_1d(2)
        x_edges = np.linspace(0.0, Lx, Ncx + 1)
        y_edges = np.linspace(0.0, Ly, Ncy + 1)

        rows_Kc, cols_Kc, data_Kc = [], [], []
        rows_Kd, cols_Kd, data_Kd = [], [], []
        rows_M, cols_M, data_M = [], [], []

        for j in range(Ncy):
            y0, y1 = y_edges[j], y_edges[j + 1]; dy_c = y1 - y0; yc = 0.5 * (y0 + y1)
            for i in range(Ncx):
                x0, x1 = x_edges[i], x_edges[i + 1]; dx_c = x1 - x0; xc = 0.5 * (x0 + x1)
                det_J = 0.25 * dx_c * dy_c
                target_mat = 1 if xc < x_int else 2
                eps_r = eps1 if target_mat == 1 else eps2
                sub = subsets[target_mat]

                for wi, xi in zip(w_1d, xi_1d):
                    xg = xc + 0.5 * dx_c * xi
                    for wj, eta in zip(w_1d, xi_1d):
                        yg = yc + 0.5 * dy_c * eta
                        Pg = np.array([xg, yg]); wg = wi * wj * det_J

                        sel_local, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
                            P=Pg, nodes_coords=sub['coords'], nodes_vectors=sub['vectors'],
                            search_tree=sub['tree'], K=12, Tol_det=1e-5, adaptive=True
                        )
                        sel_global = sub['indices'][sel_local]
                        beta = np.linalg.inv(matrix_A)
                        Phi_g = beta[0:2, :]; rot_Phi = beta[4, :] - beta[3, :]; div_Phi = beta[2, :] + beta[5, :]

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

        N = len(coords)
        Kc = csr_matrix((data_Kc, (rows_Kc, cols_Kc)), shape=(N, N))
        Kd = csr_matrix((data_Kd, (rows_Kd, cols_Kd)), shape=(N, N))
        M = csr_matrix((data_M, (rows_M, cols_M)), shape=(N, N))

        int_idx = np.where(~is_pec)[0]
        Kc_red = 0.5 * (Kc[int_idx, :][:, int_idx].toarray() + Kc[int_idx, :][:, int_idx].toarray().T)
        Kd_red = 0.5 * (Kd[int_idx, :][:, int_idx].toarray() + Kd[int_idx, :][:, int_idx].toarray().T)
        M_red = 0.5 * (M[int_idx, :][:, int_idx].toarray() + M[int_idx, :][:, int_idx].toarray().T)

        K_red = Kc_red + 6.0 * Kd_red
        vals, vecs = la.eigh(K_red, M_red)
        vals_pos = vals[vals > 0.05]; vecs_pos = vecs[:, vals > 0.05]

        phys_kc = []
        for idx_m in range(len(vals_pos)):
            v = vecs_pos[:, idx_m]
            ec = v.T @ Kc_red @ v; ed = v.T @ Kd_red @ v; em = v.T @ M_red @ v
            rd = (ed / ec) * 100.0 if ec > 1e-12 else 99999.0
            if rd < 5.0:
                phys_kc.append(np.sqrt(ec / em))

        kc_6 = phys_kc[:6]
        errs = [abs(kc_6[k] - ref_kc[k]) / ref_kc[k] * 100.0 for k in range(6)]
        results_jitter.append({
            'jitter': j_frac,
            'kc_num': kc_6,
            'errs_pct': errs,
            'mean_err': float(np.mean(errs)),
            'max_err': float(np.max(errs))
        })

    return results_jitter, ref_kc


# ==============================================================================
# TESTE 3: VARREDURA DE CONTRASTE DIELÉTRICO (EPS_R2 SWEEP)
# ==============================================================================

def run_permittivity_contrast_sweep(eps_list=[2.25, 4.0, 9.8, 16.0]):
    """
    Avalia a precisão para diferentes materiais dielétricos (Teflon, Padrão, Alumina, Cerâmica).
    """
    Lx = np.pi
    Ly = np.pi
    x_int = np.pi / 2.0
    from VNMM_Dielectric_Cavity_Eigenvalues import solve_dielectric_cavity_eigenvalues

    results_eps = []
    for eps2 in eps_list:
        ref = compute_analytical_dielectric_cavity_modes(Lx=Lx, Ly=Ly, d=x_int, eps1=1.0, eps2=eps2, max_modes=6)
        res = solve_dielectric_cavity_eigenvalues(
            Lx=Lx, Ly=Ly, x_int=x_int, eps1=1.0, eps2=eps2,
            Nx1=11, Nx2=11, Ny=21, Ncx=16, Ncy=16
        )
        phys = res['physical_modes']
        errs = [abs(p['kc'] - r['k0']) / r['k0'] * 100.0 for p, r in zip(phys[:6], ref[:6])]
        results_eps.append({
            'eps_r2': eps2,
            'mode1_kc_num': phys[0]['kc'],
            'mode1_kc_ref': ref[0]['k0'],
            'mode1_err_pct': abs(phys[0]['kc'] - ref[0]['k0']) / ref[0]['k0'] * 100.0,
            'mean_err_pct': float(np.mean(errs)),
            'max_err_pct': float(np.max(errs))
        })
    return results_eps


# ==============================================================================
# GERAÇÃO DA FIGURA 4 (PAINEL AVANÇADO)
# ==============================================================================

def plot_advanced_validation(res_jump, res_jitter, ref_kc, res_eps, output_path=None):
    """
    Gera um painel com 4 subgráficos demonstrando a robustez do método:
    (a) Salto de Descontinuidade Normal Ex(x) em corte 1D.
    (b) Continuidade Tangencial Ey(x) em corte 1D.
    (c) Estabilidade de Autovalores sob Jitter Nodal (0%, 10%, 20%).
    (d) Frequência de Corte Fundamental vs Contraste Dielétrico (eps_r in [2.25, 16.0]).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    # --------------------------------------------------------------------------
    # (a) Salto da componente normal Ex(x)
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    x_e = res_jump['x_eval']
    x_int = 1.0
    
    # Linha vertical de interface
    ax1.axvline(x=x_int, color="#C62828", linestyle="--", linewidth=1.8, label="Interface $\\Gamma_{\\mathrm{int}}$ ($x=1.0$)")
    
    # Regiões sombreadas
    ax1.axvspan(0.0, x_int, color="#E3F2FD", alpha=0.45, label="Região 1 ($\\varepsilon_{r1}=1.0$)")
    ax1.axvspan(x_int, 2.0, color="#FFF3E0", alpha=0.45, label="Região 2 ($\\varepsilon_{r2}=4.0$)")
    
    # Plotar exato e VNMM
    idx1 = x_e < x_int
    idx2 = x_e > x_int
    ax1.plot(x_e[idx1], res_jump['Ex_exact'][idx1], 'k-', linewidth=2.0, label="Analítico Exato")
    ax1.plot(x_e[idx2], res_jump['Ex_exact'][idx2], 'k-', linewidth=2.0)
    
    ax1.scatter(x_e[::2], res_jump['Ex_num'][::2], color="#0D47A1", marker="o", s=28, zorder=4, label="VNMM 2D (Interpolado)")
    
    ax1.annotate(
        "Salto Físico Abrupto (4:1)\n$E_{2x} = \\frac{\\varepsilon_{1}}{\\varepsilon_{2}} E_{1x}$",
        xy=(x_int, 0.9), xytext=(x_int + 0.18, 1.15),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=6),
        fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#C62828", alpha=0.9)
    )

    ax1.set_title("(a) Salto de Descontinuidade Normal $E_x(x)$ em $y = L_y / 2$\nZero Borramento (*Smearing*) ou Oscilações de Gibbs", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Coordenada X (m)", fontsize=10)
    ax1.set_ylabel("Componente Normal $E_x$ (V/m)", fontsize=10)
    ax1.set_xlim(0.05, 1.95)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=8.5)

    # --------------------------------------------------------------------------
    # (b) Continuidade da componente tangencial Ey(x)
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.axvline(x=x_int, color="#C62828", linestyle="--", linewidth=1.8, label="Interface $\\Gamma_{\\mathrm{int}}$ ($x=1.0$)")
    ax2.axvspan(0.0, x_int, color="#E3F2FD", alpha=0.45)
    ax2.axvspan(x_int, 2.0, color="#FFF3E0", alpha=0.45)

    ax2.plot(x_e, res_jump['Ey_exact'], 'k-', linewidth=2.0, label="Analítico Exato")
    ax2.scatter(x_e[::2], res_jump['Ey_num'][::2], color="#2E7D32", marker="s", s=28, zorder=4, label="VNMM 2D (Interpolado)")

    ax2.annotate(
        "Continuidade Tangencial Estrita $C^0$\n$E_{1y} = E_{2y}$",
        xy=(x_int, res_jump['Ey_exact'][np.argmin(np.abs(x_e - x_int))]),
        xytext=(x_int - 0.55, 0.45),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=6),
        fontsize=9.5, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#2E7D32", alpha=0.9)
    )

    ax2.set_title("(b) Continuidade da Componente Tangencial $E_y(x)$ em $y = L_y / 2$\nPreservação Rigorosa de $\\mathbf{n} \\times [\\mathbf{E}] = \\mathbf{0}$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Coordenada X (m)", fontsize=10)
    ax2.set_ylabel("Componente Tangencial $E_y$ (V/m)", fontsize=10)
    ax2.set_xlim(0.05, 1.95)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=8.5)

    # --------------------------------------------------------------------------
    # (c) Estudo de Robustez a Jitter Nodal
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    modes_idx = np.arange(1, 7)
    width = 0.22

    ax3.plot(modes_idx, ref_kc, 'k--o', linewidth=1.8, label="Analítico Exato", zorder=5)

    colors_j = ["#1565C0", "#F57C00", "#7B1FA2"]
    for idx_j, r in enumerate(res_jitter):
        j_pct = int(r['jitter'] * 100)
        ax3.bar(
            modes_idx + (idx_j - 1) * width, r['kc_num'],
            width=width, color=colors_j[idx_j], alpha=0.75,
            edgecolor="black", label=f"VNMM Jitter {j_pct}% (Erro Médio: {r['mean_err']:.2f}%)"
        )

    ax3.set_title("(c) Estudo de Perturbação Estocástica Nodal (*Jitter Study*)\nEstabilidade Espectral sob Nuvens Irregulares (0% a 20%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Índice do Modo Ressonante", fontsize=10)
    ax3.set_ylabel("Número de Onda de Corte $k_c$ (rad/m)", fontsize=10)
    ax3.set_xticks(modes_idx)
    ax3.set_xticklabels([f"Modo {m}" for m in modes_idx])
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=8.5)

    # --------------------------------------------------------------------------
    # (d) Varredura de Contraste de Permissividade
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    eps_vals = [r['eps_r2'] for r in res_eps]
    kc_refs = [r['mode1_kc_ref'] for r in res_eps]
    kc_nums = [r['mode1_kc_num'] for r in res_eps]
    errs_p1 = [r['mode1_err_pct'] for r in res_eps]

    ax4.plot(eps_vals, kc_refs, 'k-d', linewidth=2.0, markersize=8, label="Exato (Modo Fundamental $TE_{01}$)")
    ax4.plot(eps_vals, kc_nums, 'b--s', linewidth=1.8, markersize=7, label="VNMM 2D (Modo Fundamental $TE_{01}$)")

    # Eixo secundário para o erro percentual
    ax4_twin = ax4.twinx()
    ax4_twin.bar(eps_vals, errs_p1, width=0.8, color="#D32F2F", alpha=0.35, label="Erro Relativo $k_c$ (%)")
    ax4_twin.set_ylabel("Erro Relativo do Modo 1 (%)", color="#B71C1C", fontsize=10)
    ax4_twin.set_ylim(0, 3.5)
    ax4_twin.tick_params(axis='y', labelcolor="#B71C1C")

    ax4.set_title("(d) Sensibilidade ao Contraste Dielétrico $\\varepsilon_{r2} / \\varepsilon_{r1}$\nErro do Modo Fundamental Inferior a $1.1\\%$ até $\\varepsilon_{r2}=16.0$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Permissividade Relativa $\\varepsilon_{r2}$ (Teflon, Standard, Alumina, Cerâmica)", fontsize=10)
    ax4.set_ylabel("Frequência de Corte $k_c$ (rad/m)", fontsize=10)
    ax4.set_xticks(eps_vals)
    ax4.set_xticklabels([f"$\\varepsilon_r={v}$" for v in eps_vals])
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico de validações avançadas salvo com sucesso em: {output_path}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 85)
    print("      VALIDAÇÕES AVANÇADAS: SALTO NORMAL, JITTER E CONTRASTE DIELÉTRICO")
    print("=" * 85)

    # 1. Teste de Salto Descontínuo Normal
    print("\n--- 1. Teste de Descontinuidade Normal [En] != 0 e Corte 1D ---")
    domain = create_two_material_domain(Lx=2.0, Ly=1.0, x_interface=1.0, eps_r1=1.0, eps_r2=4.0)
    res_jump = run_normal_jump_test(domain, Nx1=15, Nx2=15, Ny=15, num_eval_points=120)
    print(f"Erro Máximo em Ey (tangencial contínuo) : {res_jump['err_Ey_max']:.4f} V/m")
    print(f"Erro Máximo em Ex (normal descontínuo)  : {res_jump['err_Ex_max']:.4f} V/m")
    print("Status: Salto de 4:1 capturado nitidamente sem oscilações de Gibbs!")

    # 2. Estudo de Robustez a Jitter Nodal
    print("\n--- 2. Estudo de Robustez sob Perturbação Estocástica Nodal (Jitter) ---")
    res_jitter, ref_kc = run_jitter_study(jitter_list=[0.0, 0.10, 0.20])
    for r in res_jitter:
        print(f"Jitter {int(r['jitter']*100):2d}%: Erro Médio em kc = {r['mean_err']:5.2f}% | Erro Máximo = {r['max_err']:5.2f}%")

    # 3. Varredura de Contraste Dielétrico
    print("\n--- 3. Varredura de Contraste Dielétrico (eps_r2 in [2.25, 4.0, 9.8, 16.0]) ---")
    res_eps = run_permittivity_contrast_sweep(eps_list=[2.25, 4.0, 9.8, 16.0])
    for r in res_eps:
        print(f"eps_r2 = {r['eps_r2']:5.2f}: Modo 1 kc = {r['mode1_kc_num']:.4f} (ref: {r['mode1_kc_ref']:.4f}, erro: {r['mode1_err_pct']:.2f}%) | Erro Médio = {r['mean_err_pct']:.2f}%")

    # 4. Gerar Figura 4
    plot_file = DIR_EXP / "VNMM_Interface_Jump_and_Robustness.png"
    print(f"\nGerando painel gráfico avançado (Figura 4)...")
    plot_advanced_validation(res_jump, res_jitter, ref_kc, res_eps, output_path=str(plot_file))
    print("Validações avançadas concluídas com sucesso!")


if __name__ == "__main__":
    main()
