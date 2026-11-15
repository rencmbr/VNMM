"""
Espalhamento Eletromagnético 2D (TEz) por Cilindro PEC com PML Canônica via VNMM
===============================================================================
Arquivo: VNMM_PML_Scattering_Cylinder.py

Objetivo:
---------
Validar o VNMM 2D em um problema determinístico de espalhamento em frequência fixa
(sistema linear complexo A * u = b) com Camada Perfeitamente Casada (UPML) para
truncamento de domínio aberto.

Configuração Física:
--------------------
- Domínio computacional: [-L, L] x [-L, L], com L = 1.5 m.
- Região física (vácuo): [-L0, L0] x [-L0, L0], com L0 = 1.0 m.
- Camada PML (espessura d_pml = 0.5 m): |x| > L0 ou |y| > L0.
- Cilindro espalhador PEC centrado em (0, 0) com raio a = 0.30 m.
- Onda plana incidente TEz: E_inc = y_hat * exp(-j * k0 * x), com lambda0 = 1.0 m (k0 = 2*pi).
- Formulação de campo espalhado (E_s):
  curl( (1/mu_zz) curl(E_s) ) - k0^2 * eps_tensor * E_s = 0
  com condição de Dirichlet não-homogênea no cilindro PEC:
    t . E_s = - t . E_inc   (em r = a)
  e parede PEC homogênea na borda externa da PML (t . E_s = 0 em |x|=L, |y|=L).

Validação:
----------
Compara a solução numérica do VNMM na região física (r > a, |x|,|y| < L0) com a
Série Analítica Exata de Mie (Funções de Bessel/Hankel).
"""

import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import jv, jvp, hankel2, h2vp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DIR_EXP = Path(__file__).resolve().parent
DIR_2D = DIR_EXP.parent
ROOT_DIR = DIR_2D.parent
DIR_FIG = DIR_EXP / "figures"
DIR_FIG.mkdir(parents=True, exist_ok=True)

for p in [str(DIR_2D), str(ROOT_DIR), str(DIR_EXP)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from src.gauss_quadrature_2d import get_gauss_points_weights_1d


# ==============================================================================
# 1. SOLUÇÃO ANALÍTICA EXATA (SÉRIE DE MIE 2D TEz PARA CILINDRO PEC)
# ==============================================================================

def analytical_mie_pec_cylinder_tez(x, y, a=0.3, k0=2.0*np.pi, N_terms=35):
    """
    Calcula o campo elétrico espalhado exato E_s = (Ex_s, Ey_s) para uma onda plana
    TEz incidente E_inc = y_hat * exp(-j * k0 * x) sobre um cilindro PEC de raio a.
    Convenção temporal: exp(+j * omega * t) -> onda saindo usa hankel2.
    """
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)

    # Evitar divisão por zero dentro do cilindro
    r_safe = np.where(r < 1e-12, 1e-12, r)

    # Para E_inc = y_hat * exp(-j k0 x), pela Lei de Faraday eta0 * H_z^inc = +exp(-j k0 x)
    # Em TEz: curl(H_z z_hat) = (1/r dHz/dphi) r_hat - (dHz/dr) phi_hat = j * omega * eps0 * E
    # Logo: E = (1 / (j k0)) * [ (1/r dHz_norm/dphi) r_hat - (dHz_norm/dr) phi_hat ]
    # onde Hz_norm = eta0 * Hz, com Hz_norm^inc = +exp(-j k0 x) = sum j^(-n) J_n(k0 r) e^(j n phi)
    Er_s = np.zeros_like(r_safe, dtype=complex)
    Ephi_s = np.zeros_like(r_safe, dtype=complex)

    for n in range(-N_terms, N_terms + 1):
        jn_p_a = jvp(n, k0 * a, 1)
        hn_p_a = h2vp(n, k0 * a, 1)
        an = -((-1j) ** n) * (jn_p_a / hn_p_a)

        hn_r = hankel2(n, k0 * r_safe)
        hn_p_r = h2vp(n, k0 * r_safe, 1)
        exp_n = np.exp(1j * n * phi)

        # Hz_norm^s = +sum an * H_n^(2)(k0 r) e^(j n phi)
        Er_s += (1.0 / (1j * k0 * r_safe)) * (1j * n * an * hn_r * exp_n)
        Ephi_s += -(1.0 / (1j * k0)) * (k0 * an * hn_p_r * exp_n)

    Ex_s = Er_s * np.cos(phi) - Ephi_s * np.sin(phi)
    Ey_s = Er_s * np.sin(phi) + Ephi_s * np.cos(phi)
    return Ex_s, Ey_s


# ==============================================================================
# 2. PARÂMETROS DA CAMADA PML UNIAXIAL (UPML CANÔNICA)
# ==============================================================================

def pml_parameters(x, y, L0=1.0, L=1.5, k0=2.0*np.pi, sigma_max=8.0, p_order=2):
    """
    Retorna (inv_mu_zz, eps_xx, eps_yy) da UPML uniaxial no ponto (x, y).
    Na região física (|x| <= L0 e |y| <= L0), retorna (1.0, 1.0, 1.0).
    """
    d_pml = L - L0
    sx = 1.0 + 0.0j
    sy = 1.0 + 0.0j

    if abs(x) > L0:
        dist_x = (abs(x) - L0) / d_pml
        sx = 1.0 - 1j * (sigma_max / k0) * (dist_x ** p_order)

    if abs(y) > L0:
        dist_y = (abs(y) - L0) / d_pml
        sy = 1.0 - 1j * (sigma_max / k0) * (dist_y ** p_order)

    inv_mu_zz = 1.0 / (sx * sy)
    eps_xx = sy / sx
    eps_yy = sx / sy
    return inv_mu_zz, eps_xx, eps_yy


# ==============================================================================
# 3. GERAÇÃO DA NUVEM DE PONTOS E SOLVER VNMM 2D
# ==============================================================================

def generate_pml_scattering_cloud(L=1.5, a=0.3, h=0.075, N_cyl=36):
    """
    Gera a nuvem de pontos nodais em [-L, L]^2 com furo circular PEC de raio a.
    """
    coords = []
    vectors = []
    node_type = []  # 'inner', 'pec_outer', 'pec_cyl'

    xs = np.arange(-L + h, L - 0.5 * h, h)
    ys = np.arange(-L + h, L - 0.5 * h, h)

    # 1. Nós internos (excluindo o interior e a vizinhança imediata do cilindro)
    min_dist_cyl = a + 0.45 * h
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            r = np.hypot(x, y)
            if r <= min_dist_cyl:
                continue
            ang = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
            coords.append([x, y])
            vectors.append([np.cos(ang), np.sin(ang)])
            node_type.append('inner')

    # 2. Nós escalonados na fronteira externa PEC (|x|=L, |y|=L)
    n_edge = int(round(2.0 * L / h))
    h_edge = 2.0 * L / n_edge
    for k in range(n_edge):
        pos = -L + (k + 0.5) * h_edge
        # Topo e Base
        coords.append([pos, +L]); vectors.append([1.0, 0.0]); node_type.append('pec_outer')
        coords.append([pos, -L]); vectors.append([1.0, 0.0]); node_type.append('pec_outer')
        # Direita e Esquerda
        coords.append([+L, pos]); vectors.append([0.0, 1.0]); node_type.append('pec_outer')
        coords.append([-L, pos]); vectors.append([0.0, 1.0]); node_type.append('pec_outer')

    # 3. Nós sobre o cilindro PEC (r = a) com diretores tangenciais ao círculo
    thetas = np.linspace(0.0, 2.0 * np.pi, N_cyl, endpoint=False)
    for th in thetas:
        xb = a * np.cos(th)
        yb = a * np.sin(th)
        tx, ty = -np.sin(th), np.cos(th)
        coords.append([xb, yb])
        vectors.append([tx, ty])
        node_type.append('pec_cyl')

    return (
        np.array(coords, dtype=float),
        np.array(vectors, dtype=float),
        np.array(node_type)
    )


def solve_vnmm_pml_scattering(L=1.8, L0=1.0, a=0.35, k0=3.5, h=0.05, sigma_max=8.0, s_div=1.5):
    coords, vectors, node_type = generate_pml_scattering_cloud(L=L, a=a, h=h, N_cyl=52)
    total_nodes = len(coords)
    tree = KDTree(coords)

    # Células cartesianas de fundo para integração numérica de Gauss
    Nc = int(round(2.0 * L / h))
    x_edges = np.linspace(-L, L, Nc + 1)
    y_edges = np.linspace(-L, L, Nc + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(2)

    rows, cols, data = [], [], []

    for j in range(Nc):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy_c = y1 - y0
        yc = 0.5 * (y0 + y1)
        for i in range(Nc):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx_c = x1 - x0
            xc = 0.5 * (x0 + x1)
            # Pular célula se o centro estiver estritamente dentro do cilindro PEC
            if np.hypot(xc, yc) < a * 0.85:
                continue

            det_J = 0.25 * dx_c * dy_c

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx_c * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy_c * eta
                    # Verificar se o ponto de Gauss está fora do cilindro PEC
                    if np.hypot(xg, yg) < a:
                        continue

                    wg = wi * wj * det_J
                    Pg = np.array([xg, yg])

                    sel, _, mat_A, _ = support_nodes_vnmm_2d_6_p1(
                        P=Pg,
                        nodes_coords=coords,
                        nodes_vectors=vectors,
                        search_tree=tree,
                        K=12,
                        Tol_det=1e-5,
                        adaptive=True
                    )
                    beta = np.linalg.inv(mat_A)
                    Phi_g = beta[0:2, :]
                    rot_Phi = beta[4, :] - beta[3, :]
                    div_Phi = beta[2, :] + beta[5, :]

                    inv_mu_zz, eps_xx, eps_yy = pml_parameters(
                        xg, yg, L0=L0, L=L, k0=k0, sigma_max=sigma_max, p_order=2
                    )
                    eps_tensor = np.array([[eps_xx, 0.0], [0.0, eps_yy]], dtype=complex)

                    # Operador fraco: (1/mu_zz) rot.rot - k0^2 Phi^T eps Phi + s_div div.div
                    K_loc = (
                        inv_mu_zz * np.outer(rot_Phi, rot_Phi)
                        - (k0 ** 2) * (Phi_g.T @ eps_tensor @ Phi_g)
                        + s_div * np.outer(div_Phi, div_Phi)
                    ) * wg

                    for a_idx in range(6):
                        Ia = sel[a_idx]
                        for b_idx in range(6):
                            Ib = sel[b_idx]
                            rows.append(Ia)
                            cols.append(Ib)
                            data.append(K_loc[a_idx, b_idx])

    A_mat = csr_matrix((data, (rows, cols)), shape=(total_nodes, total_nodes), dtype=complex)

    # Condições de contorno essenciais (Dirichlet):
    # - pec_outer: u_i = 0
    # - pec_cyl:   u_i = - t_i . E_inc(x_i) = - ty * exp(-j * k0 * x_i)
    is_dirichlet = (node_type != 'inner')
    u_dir = np.zeros(total_nodes, dtype=complex)

    cyl_idx = np.where(node_type == 'pec_cyl')[0]
    for idx in cyl_idx:
        xi_c, yi_c = coords[idx]
        tx, ty = vectors[idx]
        # Onda incidente: E_inc = (0, exp(-j k0 x))
        Ey_inc = np.exp(-1j * k0 * xi_c)
        u_dir[idx] = -ty * Ey_inc

    inner_idx = np.where(~is_dirichlet)[0]
    dir_idx = np.where(is_dirichlet)[0]

    A_ii = A_mat[inner_idx, :][:, inner_idx]
    A_id = A_mat[inner_idx, :][:, dir_idx]
    b_rhs = -A_id @ u_dir[dir_idx]

    u_inner = spsolve(A_ii, b_rhs)

    u_total = np.zeros(total_nodes, dtype=complex)
    u_total[dir_idx] = u_dir[dir_idx]
    u_total[inner_idx] = u_inner

    return {
        'coords': coords,
        'vectors': vectors,
        'node_type': node_type,
        'tree': tree,
        'u_s': u_total,
        'total_nodes': total_nodes,
        'active_nodes': len(inner_idx)
    }


# ==============================================================================
# 4. VISUALIZAÇÃO E COMPARAÇÃO COM SÉRIE DE MIE
# ==============================================================================

def evaluate_and_plot(res, L=1.5, L0=1.0, a=0.3, k0=2.0*np.pi):
    coords = res['coords']
    vectors = res['vectors']
    node_type = res['node_type']
    tree = res['tree']
    u_s = res['u_s']

    # 1. Plot da Geometria e Nuvem de Pontos
    fig, ax = plt.subplots(figsize=(7.5, 7.0))
    pml_outer = patches.Rectangle((-L, -L), 2*L, 2*L, facecolor="#FFE0B2", edgecolor="black", linewidth=2.0, alpha=0.6, label="Camada Absorvente (UPML)")
    phys_reg = patches.Rectangle((-L0, -L0), 2*L0, 2*L0, facecolor="#E3F2FD", edgecolor="#1565C0", linestyle="--", linewidth=1.8, alpha=0.85, label="Região Física (Vácuo)")
    cyl_patch = patches.Circle((0, 0), a, facecolor="#B71C1C", edgecolor="black", linewidth=1.5, alpha=0.85, label=f"Cilindro PEC (r={a:.2f} m)")
    ax.add_patch(pml_outer)
    ax.add_patch(phys_reg)
    ax.add_patch(cyl_patch)

    is_in = node_type == 'inner'
    is_po = node_type == 'pec_outer'
    is_pc = node_type == 'pec_cyl'

    ax.scatter(coords[is_in, 0], coords[is_in, 1], s=8, color="#0D47A1", alpha=0.65, label="Nós Ativos (±45°)")
    ax.scatter(coords[is_po, 0], coords[is_po, 1], s=14, color="#424242", label="Nós PEC Externos (u=0)")
    ax.quiver(coords[is_pc, 0], coords[is_pc, 1], vectors[is_pc, 0], vectors[is_pc, 1],
              color="#FFD54F", scale=18, width=0.005, label="Nós Cilindro PEC (E_t=-E_t^inc)")

    ax.set_title("Espalhamento 2D com UPML Canônica\nNuvem de Pontos e Camada Absorvente", fontsize=12, fontweight='bold')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_xlim(-L - 0.15, L + 0.15)
    ax.set_ylim(-L - 0.15, L + 0.15)
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.92)
    plt.tight_layout()
    fig_geom = DIR_FIG / "Geometria_Nuvem_PML_Espalhamento.png"
    plt.savefig(fig_geom, dpi=200)
    plt.close()
    print(f"Figura de geometria salva em: {fig_geom}")

    # 2. Avaliação do campo na grade fina para comparação com Mie
    ng = 91
    xg = np.linspace(-L + 0.02, L - 0.02, ng)
    yg = np.linspace(-L + 0.02, L - 0.02, ng)
    Xg, Yg = np.meshgrid(xg, yg)
    Ex_num = np.zeros_like(Xg, dtype=complex)
    Ey_num = np.zeros_like(Yg, dtype=complex)

    for r_i in range(ng):
        for c_i in range(ng):
            px, py = Xg[r_i, c_i], Yg[r_i, c_i]
            if np.hypot(px, py) < a:
                Ex_num[r_i, c_i] = np.nan
                Ey_num[r_i, c_i] = np.nan
                continue
            sel, _, mat_A, _ = support_nodes_vnmm_2d_6_p1(
                P=np.array([px, py]),
                nodes_coords=coords,
                nodes_vectors=vectors,
                search_tree=tree,
                K=12,
                Tol_det=1e-5,
                adaptive=True
            )
            beta = np.linalg.inv(mat_A)
            E_vec = beta[0:2, :] @ u_s[sel]
            Ex_num[r_i, c_i] = E_vec[0]
            Ey_num[r_i, c_i] = E_vec[1]

    # Solução analítica exata de Mie
    Ex_ana, Ey_ana = analytical_mie_pec_cylinder_tez(Xg, Yg, a=a, k0=k0)
    mask_phys = (np.abs(Xg) <= L0) & (np.abs(Yg) <= L0) & (np.hypot(Xg, Yg) >= a + 0.04)

    err_l2 = np.sqrt(
        np.sum(np.abs(Ex_num[mask_phys] - Ex_ana[mask_phys])**2 + np.abs(Ey_num[mask_phys] - Ey_ana[mask_phys])**2)
        / np.sum(np.abs(Ex_ana[mask_phys])**2 + np.abs(Ey_ana[mask_phys])**2)
    ) * 100.0
    print(f"Erro L2 Relativo na Região Física (VNMM vs Série Exata de Mie): {err_l2:.2f}%")

    # Campo total: E_total = E_inc + E_s
    Ey_inc = np.exp(-1j * k0 * Xg)
    E_tot_mag = np.sqrt(np.abs(Ex_num)**2 + np.abs(Ey_num + Ey_inc)**2)

    # Plot de 3 painéis: Re(Ey_s) VNMM | Re(Ey_s) Exato Mie | |E_total| VNMM
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # Painel 1: Re(Ey^s) VNMM
    im0 = axes[0].pcolormesh(Xg, Yg, np.real(Ey_num), cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto")
    axes[0].add_patch(patches.Rectangle((-L0, -L0), 2*L0, 2*L0, fill=False, edgecolor="black", linestyle="--", linewidth=1.2))
    axes[0].add_patch(patches.Circle((0, 0), a, facecolor="gray", edgecolor="black"))
    axes[0].set_title("Campo Espalhado Re(Ey^s) - VNMM + UPML", fontsize=10, fontweight='bold')
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)"); axes[0].set_aspect('equal')
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Painel 2: Re(Ey^s) Exato Mie (mascarado fora da região física para comparação)
    Ey_ana_plot = np.where(np.hypot(Xg, Yg) < a, np.nan, np.real(Ey_ana))
    im1 = axes[1].pcolormesh(Xg, Yg, Ey_ana_plot, cmap="RdBu_r", vmin=-1.0, vmax=1.0, shading="auto")
    axes[1].add_patch(patches.Rectangle((-L0, -L0), 2*L0, 2*L0, fill=False, edgecolor="black", linestyle="--", linewidth=1.2))
    axes[1].add_patch(patches.Circle((0, 0), a, facecolor="gray", edgecolor="black"))
    axes[1].set_title(f"Referência Analítica Exata (Mie)\nErro L2 na Região Física = {err_l2:.2f}%", fontsize=10, fontweight='bold')
    axes[1].set_xlabel("x (m)"); axes[1].set_aspect('equal')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Painel 3: |E_total| na Região Física (e decaimento de |E^s| na PML)
    E_show = np.where((np.abs(Xg) <= L0) & (np.abs(Yg) <= L0), E_tot_mag, np.sqrt(np.abs(Ex_num)**2 + np.abs(Ey_num)**2))
    im2 = axes[2].pcolormesh(Xg, Yg, E_show, cmap="inferno", vmin=0.0, vmax=2.0, shading="auto")
    axes[2].add_patch(patches.Rectangle((-L0, -L0), 2*L0, 2*L0, fill=False, edgecolor="cyan", linestyle="--", linewidth=1.4))
    axes[2].add_patch(patches.Circle((0, 0), a, facecolor="white", edgecolor="black"))
    axes[2].set_title("|E_total| (Física) & Atenuação |E^s| (PML)", fontsize=10, fontweight='bold')
    axes[2].set_xlabel("x (m)"); axes[2].set_aspect('equal')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Espalhamento TEz por Cilindro PEC (a={a:.2f} m, k0={k0:.2f} rad/m) via VNMM 2D com UPML",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig_modes = DIR_FIG / "VNMM_PML_Espalhamento_Cilindro.png"
    plt.savefig(fig_modes, dpi=220)
    plt.close()
    print(f"Figura de espalhamento salva em: {fig_modes}")


def main():
    print("=" * 75)
    print("  TESTE CANÔNICO DE ESPALHAMENTO TEz COM PML VIA VNMM 2D")
    print("  Cilindro PEC (a = 0.35 m) | k0 = 3.5 rad/m | L0 = 1.0 m | L = 1.8 m")
    print("=" * 75)

    res = solve_vnmm_pml_scattering(L=1.8, L0=1.0, a=0.35, k0=3.5, h=0.05, sigma_max=8.0, s_div=1.5)
    print(f"Nós totais: {res['total_nodes']} (Active: {res['active_nodes']})")
    evaluate_and_plot(res, L=1.8, L0=1.0, a=0.35, k0=3.5)


if __name__ == "__main__":
    main()
