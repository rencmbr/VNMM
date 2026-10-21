"""
Teste de Patch Test de Galerkin para o Método Nodal sem Malha Vetorial (VNMM 2D).
Verifica a reprodução exata de campos vetoriais afins através da solução
do sistema linear de Galerkin K * e = F com condições de contorno de Dirichlet essenciais.

Campos testados:
1. Translação pura em X: E = [1, 0]^T (Harmônico: rot=0, div=0)
2. Translação pura em Y: E = [0, 1]^T (Harmônico: rot=0, div=0)
3. Deformação normal (strain longitudinal): E = [x, -y]^T (Harmônico: rot=0, div=0)
4. Deformação cisalhante (shear strain): E = [y, x]^T (Harmônico: rot=0, div=0)
5. Rotação de corpo rígido (Pure Curl): E = [-y, x]^T (rot=2, div=0)
6. Expansão volumétrica (Pure Div): E = [x, y]^T (rot=0, div=2)
7. Campo afim geral completo: E = [a1 + b1*x + b2*y, a2 + b3*x + b4*y]^T

Compara:
- Base completa P1 (6 nós/termos)
- Base incompleta L1 (3 nós/termos, Nedelec de 1ª ordem)
Em malha regular e nuvem irregular com perturbação estocástica (25% jitter).
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix
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


def montar_sistema_patch_test(
    coords,
    vectors,
    campo_alvo,
    base="P1",
    s_div=6.0,
    Ncx=10,
    Ncy=10,
    Lx=np.pi,
    Ly=np.pi,
    pontos_por_dir=3,
    tolerancia_det=None,
    tol_piso=None
):
    """
    Monta a matriz de rigidez K = K_curl + s_div * K_div e o vetor de carga F
    consistente com o campo afim alvo E_target:
        rot_exact = (dEy/dx - dEx/dy) = const
        div_exact = (dEx/dx + dEy/dy) = const
        
    F_i = integral_Omega [ (rot N_i)_z * rot_exact + s_div * (div N_i) * div_exact ] dOmega
    """
    N_total = len(coords)
    arvore = KDTree(coords)
    
    is_P1 = base.upper() in ["P1", "6_P1", "6NOS"]
    
    # Derivadas analíticas do campo afim alvo
    rot_exact = campo_alvo["rot"]
    div_exact = campo_alvo["div"]
    
    N_lado = int(np.round(np.sqrt(N_total)))
    h_char = max(Lx / max(N_lado - 1, 1), Ly / max(N_lado - 1, 1))
    h_ref = np.pi / 20.0
    if tolerancia_det is None:
        tol_base = 1e-4 * (h_char / h_ref)**4 if is_P1 else 1e-4 * (h_char / h_ref)
        tolerancia_det = max(tol_base, tol_piso) if tol_piso is not None else tol_base

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    xi_1d, w_1d = obter_pontos_pesos_gauss_1d(pontos_por_dir)
    
    rows_K, cols_K, data_K = [], [], []
    F = np.zeros(N_total, dtype=float)
    
    # Armazena dados dos pontos de Gauss para pós-processamento
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
                        
                    gauss_info.append({
                        'Pg': Pg,
                        'wg': wg,
                        'nos': nos,
                        'Phi_g': Phi_g,
                        'rot_Phi': rot_Phi,
                        'div_Phi': div_Phi
                    })
                    
                    # Contribuição local para rigidez K = K_curl + s_div * K_div
                    rot_outer = np.outer(rot_Phi, rot_Phi) * wg
                    div_outer = np.outer(div_Phi, div_Phi) * (s_div * wg)
                    k_local = rot_outer + div_outer
                    
                    # Contribuição local para o vetor de carga F
                    f_local = (rot_Phi * rot_exact + s_div * div_Phi * div_exact) * wg
                    
                    for a in range(n_supp):
                        Ia = nos[a]
                        F[Ia] += f_local[a]
                        for b in range(n_supp):
                            Ib = nos[b]
                            val = k_local[a, b]
                            if abs(val) > 1e-18:
                                rows_K.append(Ia)
                                cols_K.append(Ib)
                                data_K.append(val)
                                
    K = coo_matrix((data_K, (rows_K, cols_K)), shape=(N_total, N_total)).tocsr()
    K = 0.5 * (K + K.T)
    return K, F, gauss_info


def executar_patch_test_modo(
    coords,
    vectors,
    is_boundary,
    campo_alvo,
    base="P1",
    s_div=6.0,
    Ncx=10,
    Ncy=10,
    Lx=np.pi,
    Ly=np.pi
):
    """
    Executa o Patch Test para um modo afim específico:
    1. Monta K e F
    2. Impõe Dirichlet essencial nos nós da fronteira: e_bnd = E_target(x_k) . t_k
    3. Resolve K_int,int * e_int = F_int - K_int,bnd * e_bnd
    4. Avalia erro nos DoFs internos e erros de campo reconstruído nos pontos de Gauss
    """
    N_total = len(coords)
    
    # 1. Avalia DoFs exatos em todos os nós
    E_func = campo_alvo["func"]
    e_exact = np.zeros(N_total, dtype=float)
    for k in range(N_total):
        E_pt = E_func(coords[k, 0], coords[k, 1])
        e_exact[k] = np.dot(E_pt, vectors[k])
        
    idx_bnd = np.where(is_boundary)[0]
    idx_int = np.where(~is_boundary)[0]
    
    # 2. Monta o sistema de Galerkin
    K, F, gauss_info = montar_sistema_patch_test(
        coords=coords,
        vectors=vectors,
        campo_alvo=campo_alvo,
        base=base,
        s_div=s_div,
        Ncx=Ncx,
        Ncy=Ncy,
        Lx=Lx,
        Ly=Ly
    )
    
    # 3. Particionamento e imposição de Dirichlet
    K_ii = K[idx_int, :][:, idx_int]
    K_ib = K[idx_int, :][:, idx_bnd]
    
    e_bnd = e_exact[idx_bnd]
    rhs_int = F[idx_int] - K_ib @ e_bnd
    
    # 4. Solução do sistema linear
    e_sol_int = spsolve(K_ii, rhs_int)
    
    # 5. Vetor solução global montado
    e_sol = np.zeros(N_total, dtype=float)
    e_sol[idx_bnd] = e_bnd
    e_sol[idx_int] = e_sol_int
    
    # 6. Erro nos graus de liberdade internos
    erro_dof_abs = np.max(np.abs(e_sol_int - e_exact[idx_int]))
    norma_dof_ref = np.max(np.abs(e_exact[idx_int]))
    erro_dof_rel = erro_dof_abs / max(norma_dof_ref, 1e-14)
    
    # 7. Erro de reconstrução de campo nos pontos de quadratura
    err_L2_E_sq = 0.0
    norm_L2_E_sq = 0.0
    err_L2_rot_sq = 0.0
    err_L2_div_sq = 0.0
    area_total = 0.0
    
    rot_exact = campo_alvo["rot"]
    div_exact = campo_alvo["div"]
    
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
        
        E_ex = E_func(Pg[0], Pg[1])
        
        err_L2_E_sq += wg * np.sum((E_h - E_ex)**2)
        norm_L2_E_sq += wg * np.sum(E_ex**2)
        err_L2_rot_sq += wg * (rot_h - rot_exact)**2
        err_L2_div_sq += wg * (div_h - div_exact)**2
        area_total += wg
        
    err_L2_E = np.sqrt(err_L2_E_sq / max(norm_L2_E_sq, 1e-14))
    err_L2_rot = np.sqrt(err_L2_rot_sq / area_total)
    err_L2_div = np.sqrt(err_L2_div_sq / area_total)
    
    return {
        'erro_dof_abs': erro_dof_abs,
        'erro_dof_rel': erro_dof_rel,
        'err_L2_E': err_L2_E,
        'err_L2_rot': err_L2_rot,
        'err_L2_div': err_L2_div,
        'n_dof_int': len(idx_int),
        'n_dof_bnd': len(idx_bnd)
    }


def executar_bateria_patch_tests():
    """
    Executa a bateria completa do Patch Test para todas as 7 configurações de campo afim
    em duas topologias de nuvem:
    1. Grade Regular (Nx=Ny=11, diretores alternados a 45°/135°)
    2. Nuvem Irregular com Jitter (Nx=Ny=11, jitter 25%, diretores aleatórios)
    """
    modos_afins = [
        {
            "nome": "1. Translacao X: E = [1, 0]^T",
            "func": lambda x, y: np.array([1.0, 0.0]),
            "rot": 0.0,
            "div": 0.0,
            "harmonico": True
        },
        {
            "nome": "2. Translacao Y: E = [0, 1]^T",
            "func": lambda x, y: np.array([0.0, 1.0]),
            "rot": 0.0,
            "div": 0.0,
            "harmonico": True
        },
        {
            "nome": "3. Strain Normal (Harmonico): E = [x, -y]^T",
            "func": lambda x, y: np.array([x, -y]),
            "rot": 0.0,
            "div": 0.0,
            "harmonico": True
        },
        {
            "nome": "4. Strain Cisalhante (Harmonico): E = [y, x]^T",
            "func": lambda x, y: np.array([y, x]),
            "rot": 0.0,
            "div": 0.0,
            "harmonico": True
        },
        {
            "nome": "5. Rotacao Corpo Rigido (Pure Curl): E = [-y, x]^T",
            "func": lambda x, y: np.array([-y, x]),
            "rot": 2.0,
            "div": 0.0,
            "harmonico": False
        },
        {
            "nome": "6. Expansao Volumetrica (Pure Div): E = [x, y]^T",
            "func": lambda x, y: np.array([x, y]),
            "rot": 0.0,
            "div": 2.0,
            "harmonico": False
        },
        {
            "nome": "7. Campo Afim Completo Arbitrario",
            "func": lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y]),
            "rot": 0.4 - (-0.3),  # b3 - b2 = 0.7
            "div": 0.8 + 1.2,     # b1 + b4 = 2.0
            "harmonico": False
        }
    ]
    
    topologias = [
        {
            "nome": "MALHA REGULAR (11x11, Diretores Alternados 45°/135°)",
            "jitter": 0.0,
            "tipo_int": "alternado"
        },
        {
            "nome": "NUVEM IRREGULAR (11x11, 25% Jitter, Diretores Aleatorios)",
            "jitter": 0.25,
            "tipo_int": "aleatorio"
        }
    ]
    
    print("=" * 95)
    print("        BATERIA DE PATCH TEST DE GALERKIN - VNMM 2D (P1 COMPLETO vs L1 INCOMPLETO)")
    print("=" * 95)
    
    for topo in topologias:
        coords, vectors, is_bnd = gerar_malha_cavidade(
            Nx=11,
            Ny=11,
            Lx=np.pi,
            Ly=np.pi,
            tipo_interior=topo["tipo_int"],
            jitter_frac=topo["jitter"],
            seed=42
        )
        
        n_tot = len(coords)
        n_bnd = np.sum(is_bnd)
        n_int = n_tot - n_bnd
        
        print(f"\n>>> TOPOLOGIA: {topo['nome']}")
        print(f"    Nós totais: {n_tot} | Nós contorno (Dirichlet): {n_bnd} | Nós internos (livres): {n_int}")
        print("-" * 95)
        print(f"{'Modo Afim Testado':<42} | {'Base P1 (DoF Rel)':<17} | {'Base L1 (DoF Rel)':<17} | {'Veredito':<10}")
        print("-" * 95)
        
        for modo in modos_afins:
            # Executa com base P1
            res_P1 = executar_patch_test_modo(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                campo_alvo=modo,
                base="P1",
                s_div=6.0,
                Ncx=10,
                Ncy=10
            )
            
            # Executa com base L1
            res_L1 = executar_patch_test_modo(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                campo_alvo=modo,
                base="L1",
                s_div=6.0,
                Ncx=10,
                Ncy=10
            )
            
            pass_P1 = res_P1['erro_dof_rel'] < 1e-10
            status_P1 = "PASS (P1)" if pass_P1 else "FAIL (P1)"
            
            str_P1 = f"{res_P1['erro_dof_rel']:.2e}"
            str_L1 = f"{res_L1['erro_dof_rel']:.2e}"
            
            veredito = "P1 PASS" if pass_P1 else "FAIL"
            if res_L1['erro_dof_rel'] > 1e-3 and pass_P1:
                veredito += " (L1 FAIL)"
            elif res_L1['erro_dof_rel'] < 1e-10 and pass_P1:
                veredito += " (L1 PASS)"
                
            print(f"{modo['nome']:<42} | {str_P1:<17} | {str_L1:<17} | {veredito:<10}")
            
    print("\n" + "=" * 95)
    print("DETALHAMENTO DOS ERROS DE CAMPO RECONSTRUÍDO (Base P1 - Nuvem Irregular 25% Jitter):")
    print("-" * 95)
    print(f"{'Modo':<35} | {'Erro DoF Rel':<14} | {'Erro L2 Campo':<14} | {'Erro L2 Rot':<14} | {'Erro L2 Div':<14}")
    print("-" * 95)
    
    # Re-executa na nuvem irregular detalhando erros de campo
    coords_irr, vectors_irr, is_bnd_irr = gerar_malha_cavidade(
        Nx=11, Ny=11, Lx=np.pi, Ly=np.pi, tipo_interior="aleatorio", jitter_frac=0.25, seed=42
    )
    for modo in modos_afins:
        res = executar_patch_test_modo(
            coords=coords_irr, vectors=vectors_irr, is_boundary=is_bnd_irr,
            campo_alvo=modo, base="P1", s_div=6.0, Ncx=10, Ncy=10
        )
        print(f"{modo['nome']:<35} | {res['erro_dof_rel']:<14.2e} | {res['err_L2_E']:<14.2e} | {res['err_L2_rot']:<14.2e} | {res['err_L2_div']:<14.2e}")
    print("=" * 95)


if __name__ == "__main__":
    executar_bateria_patch_tests()
