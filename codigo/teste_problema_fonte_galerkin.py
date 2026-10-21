"""
Teste de Problema de Fonte (Driven Problem) com Solução Manufaturada (MMS)
para a formulação de Galerkin do VNMM 2D.

Equação Governante (BVP de Maxwell com regularização da divergência):
    curl(curl E) - s_div * grad(div E) + kappa * E = f(x)   em Omega = [0, pi]^2
    E . t = E* . t                                          em dOmega

Solução Manufaturada Exata:
    E*(x, y) = [  sin(x) cos(y) ]
               [ -cos(x) sin(y) ]

Propriedades Analíticas:
    div(E*) = cos(x)cos(y) - cos(x)cos(y) = 0
    (curl E*)_z = sin(x)sin(y) - (-sin(x)sin(y)) = 2 sin(x) sin(y)
    curl(curl E*) = 2 E*
    (curl E*)_z = 0 em toda a fronteira dOmega (pois sin(0)=sin(pi)=0)
    Portanto, termo de contorno natural na integração por partes anula-se identicamente!

Termo de Fonte Volumétrico:
    f(x, y) = (2 + kappa) * E*(x, y)

Medidas de Erro sob Refinamento (h -> 0):
    1. Erro relativo L2 do campo: ||E_h - E*||_L2 / ||E*||_L2
    2. Erro relativo no rotacional: ||rot(E_h) - rot(E*)||_L2 / ||rot(E*)||_L2
    3. Erro relativo nos DoFs discretos: ||e_sol - e*||_inf / ||e*||_inf
    4. Resíduo de divergência L2: ||div(E_h)||_L2
    5. Taxas empíricas de convergência assintótica O(h^p).
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# Configuração de caminhos
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_ATUAL)
DIRETORIO_SRC = os.path.join(DIRETORIO_RAIZ, "src")
for p in [DIRETORIO_RAIZ, DIRETORIO_ATUAL, DIRETORIO_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.malha_cavidade import gerar_malha_cavidade
from src.quadratura_gauss import obter_pontos_pesos_gauss_1d
from nos_suporte_vnmm_2d_3_L1 import nos_suporte_vnmm_2d_3_L1
from nos_suporte_vnmm_2d_6_P1 import nos_suporte_vnmm_2d_6_P1


def solucao_exata_mms(x, y):
    """Campo vetorial exato E*(x, y)."""
    return np.array([np.sin(x) * np.cos(y), -np.cos(x) * np.sin(y)])


def rotacional_exato_mms(x, y):
    """(curl E*)_z = dEy/dx - dEx/dy = 2 * sin(x) * sin(y)."""
    return 2.0 * np.sin(x) * np.sin(y)


def divergencia_exata_mms(x, y):
    """div(E*) = 0."""
    return 0.0


def fonte_mms(x, y, kappa=1.0):
    """f(x, y) = (2 + kappa) * E*(x, y)."""
    return (2.0 + kappa) * solucao_exata_mms(x, y)


def montar_e_resolver_mms(
    coords,
    vectors,
    is_boundary,
    base="P1",
    s_div=6.0,
    kappa=1.0,
    Ncx=None,
    Ncy=None,
    Lx=np.pi,
    Ly=np.pi,
    pontos_por_dir=3,
    tolerancia_det=None,
    tol_piso=None
):
    """
    Monta as matrizes do sistema de Galerkin e o vetor de carga da fonte volumétrica:
        A = K_curl + s_div * K_div + kappa * M
        F_i = integral_Omega N_i . f(x) dOmega
    Aplica condições essenciais de Dirichlet nos nós de fronteira e resolve.
    Retorna erros em norma L2, H(curl) e DoF.
    """
    N_total = len(coords)
    arvore = KDTree(coords)
    is_P1 = base.upper() in ["P1", "6_P1", "6NOS"]

    N_lado = int(np.round(np.sqrt(N_total)))
    h_char = max(Lx / max(N_lado - 1, 1), Ly / max(N_lado - 1, 1))
    h_ref = np.pi / 20.0

    if Ncx is None:
        Ncx = max(N_lado - 1, 6)
    if Ncy is None:
        Ncy = max(N_lado - 1, 6)

    if tolerancia_det is None:
        tol_base = 1e-4 * (h_char / h_ref)**4 if is_P1 else 1e-4 * (h_char / h_ref)
        tolerancia_det = max(tol_base, tol_piso) if tol_piso is not None else tol_base

    # Células de integração de Gauss
    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    xi_1d, w_1d = obter_pontos_pesos_gauss_1d(pontos_por_dir)

    rows_A, cols_A, data_A = [], [], []
    F = np.zeros(N_total, dtype=float)
    gauss_info = []

    for j in range(Ncy):
        y0, y1 = y_edges[j], y_edges[j + 1]
        dy = y1 - y0
        yc = 0.5 * (y0 + y1)
        for i in range(Ncx):
            x0, x1 = x_edges[i], x_edges[i + 1]
            dx = x1 - x0
            xc = 0.5 * (x0 + x1)
            det_J = 0.25 * dx * dy

            for wi, xi in zip(w_1d, xi_1d):
                xg = xc + 0.5 * dx * xi
                for wj, eta in zip(w_1d, xi_1d):
                    yg = yc + 0.5 * dy * eta
                    Pg = np.array([xg, yg])
                    wg = wi * wj * det_J

                    if is_P1:
                        nos, det_val, A_mat, _ = nos_suporte_vnmm_2d_6_P1(
                            P=Pg,
                            nodes_coords=coords,
                            nodes_vectors=vectors,
                            arvore_busca=arvore,
                            K=12,
                            Tol_det=tolerancia_det,
                            adaptativo=True,
                            passo_K=4
                        )
                        beta = np.linalg.inv(A_mat)
                        Phi_g = beta[0:2, :]                   # (2, 6)
                        rot_Phi = beta[4, :] - beta[3, :]      # (6,)
                        div_Phi = beta[2, :] + beta[5, :]      # (6,)
                        n_supp = 6
                    else:
                        nos, det_val, A_mat, _ = nos_suporte_vnmm_2d_3_L1(
                            P=Pg,
                            nodes_coords=coords,
                            nodes_vectors=vectors,
                            arvore_busca=arvore,
                            K=8,
                            Tol_det=tolerancia_det,
                            adaptativo=True,
                            passo_K=2
                        )
                        beta = np.linalg.inv(A_mat)
                        Phi_g = beta[0:2, :]                   # (2, 3)
                        rot_Phi = -2.0 * beta[2, :]             # (3,)
                        div_Phi = np.zeros(3, dtype=float)     # L1 é identicamente solenoidal
                        n_supp = 3

                    # Termo de fonte nos pontos de Gauss
                    f_val = fonte_mms(xg, yg, kappa=kappa)

                    # Contribuições locais:
                    # K_curl = rot_Phi * rot_Phi^T
                    # K_div  = div_Phi * div_Phi^T
                    # M      = Phi_g^T * Phi_g
                    rot_outer = np.outer(rot_Phi, rot_Phi) * wg
                    div_outer = np.outer(div_Phi, div_Phi) * (s_div * wg)
                    mass_local = (Phi_g.T @ Phi_g) * (kappa * wg)
                    a_local = rot_outer + div_outer + mass_local

                    # Vetor de carga local: F_local = Phi_g^T * f_val * wg
                    f_local = (Phi_g.T @ f_val) * wg

                    gauss_info.append({
                        'Pg': Pg,
                        'wg': wg,
                        'nos': nos,
                        'Phi_g': Phi_g,
                        'rot_Phi': rot_Phi,
                        'div_Phi': div_Phi
                    })

                    for a in range(n_supp):
                        Ia = nos[a]
                        F[Ia] += f_local[a]
                        for b in range(n_supp):
                            Ib = nos[b]
                            val = a_local[a, b]
                            if abs(val) > 1e-18:
                                rows_A.append(Ia)
                                cols_A.append(Ib)
                                data_A.append(val)

    A_mat = coo_matrix((data_A, (rows_A, cols_A)), shape=(N_total, N_total)).tocsr()
    A_mat = 0.5 * (A_mat + A_mat.T)

    # 1. Avalia DoFs exatos em todos os nós
    e_exact = np.zeros(N_total, dtype=float)
    for k in range(N_total):
        E_pt = solucao_exata_mms(coords[k, 0], coords[k, 1])
        e_exact[k] = np.dot(E_pt, vectors[k])

    idx_bnd = np.where(is_boundary)[0]
    idx_int = np.where(~is_boundary)[0]

    # Partição Dirichlet
    A_ii = A_mat[idx_int, :][:, idx_int]
    A_ib = A_mat[idx_int, :][:, idx_bnd]
    e_bnd = e_exact[idx_bnd]
    rhs_int = F[idx_int] - A_ib @ e_bnd

    # Solução
    e_sol_int = spsolve(A_ii, rhs_int)
    e_sol = np.zeros(N_total, dtype=float)
    e_sol[idx_bnd] = e_bnd
    e_sol[idx_int] = e_sol_int

    # Erro nos DoFs
    erro_dof_inf = np.max(np.abs(e_sol_int - e_exact[idx_int]))
    norma_dof_inf = np.max(np.abs(e_exact[idx_int]))
    erro_dof_rel = erro_dof_inf / max(norma_dof_inf, 1e-14)

    # Integração de erros em L2 e H(curl) nos pontos de Gauss
    err_L2_E_sq = 0.0
    norm_L2_E_sq = 0.0
    err_L2_rot_sq = 0.0
    norm_L2_rot_sq = 0.0
    err_L2_div_sq = 0.0

    for g in gauss_info:
        Pg = g['Pg']
        wg = g['wg']
        nos = g['nos']
        Phi_g = g['Phi_g']
        rot_Phi = g['rot_Phi']
        div_Phi = g['div_Phi']

        e_supp = e_sol[nos]
        E_h = Phi_g @ e_supp
        rot_h = np.dot(rot_Phi, e_supp)
        div_h = np.dot(div_Phi, e_supp)

        E_ex = solucao_exata_mms(Pg[0], Pg[1])
        rot_ex = rotacional_exato_mms(Pg[0], Pg[1])
        div_ex = divergencia_exata_mms(Pg[0], Pg[1])

        err_L2_E_sq += np.sum((E_h - E_ex)**2) * wg
        norm_L2_E_sq += np.sum(E_ex**2) * wg

        err_L2_rot_sq += ((rot_h - rot_ex)**2) * wg
        norm_L2_rot_sq += (rot_ex**2) * wg

        err_L2_div_sq += ((div_h - div_ex)**2) * wg

    err_L2_E_rel = np.sqrt(err_L2_E_sq) / np.sqrt(norm_L2_E_sq)
    err_L2_rot_rel = np.sqrt(err_L2_rot_sq) / np.sqrt(norm_L2_rot_sq)
    res_div_L2 = np.sqrt(err_L2_div_sq)

    return {
        'h': h_char,
        'N_total': N_total,
        'N_int': len(idx_int),
        'N_bnd': len(idx_bnd),
        'erro_dof_rel': erro_dof_rel,
        'err_L2_E_rel': err_L2_E_rel,
        'err_L2_rot_rel': err_L2_rot_rel,
        'res_div_L2': res_div_L2
    }


def executar_estudo_convergencia_mms():
    """
    Executa a sequência de refinamento sob malha regular e calcula
    as taxas assintóticas de convergência O(h^p) para P1 e L1.
    """
    print("=" * 78)
    print(" ESTUDO DE CONVERGÊNCIA: PROBLEMA DE FONTE MANUFATURADO (MMS) - VNMM 2D")
    print("=" * 78)
    print("Equação: curl(curl E) - s_div * grad(div E) + kappa * E = f(x) em [0, pi]^2")
    print("Solução: E* = [sin(x)cos(y), -cos(x)sin(y)]^T (solenoidal, rot=2*sin(x)sin(y))")
    print("Parâmetros: s_div = 6.0, kappa = 1.0, Quadratura: 3x3 Gauss por célula conforme")
    print("-" * 78)

    tamanhos = [9, 13, 17, 21, 25, 29]
    
    for base in ["P1", "L1"]:
        print(f"\n>>> Base: {base} <<<")
        resultados = []
        for N in tamanhos:
            coords, vectors, is_boundary = gerar_malha_cavidade(Nx=N, Ny=N, jitter_frac=0.0)
            res = montar_e_resolver_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_boundary,
                base=base,
                s_div=6.0,
                kappa=1.0,
                Ncx=N - 1,
                Ncy=N - 1,
                pontos_por_dir=3
            )
            resultados.append(res)
            print(f"N={N:2d}x{N:2d} (DoFs={res['N_total']:4d}, h={res['h']:.4f}): "
                  f"Err_L2(E) = {res['err_L2_E_rel']*100:6.3f}%, "
                  f"Err_H(curl) = {res['err_L2_rot_rel']*100:6.3f}%, "
                  f"Err_DoF = {res['erro_dof_rel']*100:6.3f}%, "
                  f"Div_res = {res['res_div_L2']:.2e}")

        # Cálculo das taxas de convergência
        h_vals = np.array([r['h'] for r in resultados])
        e_L2 = np.array([r['err_L2_E_rel'] for r in resultados])
        e_curl = np.array([r['err_L2_rot_rel'] for r in resultados])
        e_dof = np.array([r['erro_dof_rel'] for r in resultados])

        taxa_L2, _ = np.polyfit(np.log(h_vals), np.log(e_L2), 1)
        taxa_curl, _ = np.polyfit(np.log(h_vals), np.log(e_curl), 1)
        taxa_dof, _ = np.polyfit(np.log(h_vals), np.log(e_dof), 1)

        print(f"\n--- Taxas Assintóticas Globais de Convergência O(h^p) [{base}] ---")
        print(f"Taxa L2 (Campo E):       p = {taxa_L2:.2f}")
        print(f"Taxa H(curl) (Rotacional): p = {taxa_curl:.2f}")
        print(f"Taxa DoF (Graus de lib):  p = {taxa_dof:.2f}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    executar_estudo_convergencia_mms()
