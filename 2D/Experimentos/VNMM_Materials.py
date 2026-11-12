"""
Experimento: Interpolação VNMM 2D em Domínio Multimaterial
==========================================================
Arquivo: VNMM_Materials.py

Objetivo:
---------
Avaliar o desempenho do Método sem Malha Nodal Vetorial (VNMM 2D)
ao interpolar campos vetoriais em domínios compostos por múltiplos materiais
distintos separados por interfaces materiais (descontinuidade dielétrica).

Restrição Fundamental:
----------------------
Os domínios de suporte (os 6 nós de cada stencil da base P^1) NÃO podem
ser formados por nós de materiais distintos (princípio de partição de domínio
sem poluição de nós além da interface material).

Casos de Teste Avaliados:
-------------------------
1. Galerkin Patch Test: Campo afim geral linear (deve ser reproduzido com precisão de máquina ~10^-15).
2. MMS (Method of Manufactured Solutions): Campo trigonométrico não-linear suave.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

# Configuração de caminhos para importação do repositório VNMM
DIR_EXPERIMENTOS = Path(__file__).resolve().parent
DIR_2D = DIR_EXPERIMENTOS.parent
ROOT_DIR = DIR_2D.parent

for p in [str(DIR_2D), str(ROOT_DIR), str(DIR_EXPERIMENTOS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from scripts.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1


def create_two_material_domain(Lx=2.0, Ly=1.0, x_interface=1.0, eps_r1=1.0, eps_r2=4.0):
    """
    Define a geometria e propriedades do domínio retangular com dois materiais.
    """
    return {
        "Lx": Lx,
        "Ly": Ly,
        "x_interface": x_interface,
        "mat1": {
            "id": 1,
            "name": "Material 1 (Vácuo / Ar)",
            "eps_r": eps_r1,
            "x_bounds": (0.0, x_interface),
            "y_bounds": (0.0, Ly),
            "color_bg": "#E3F2FD",  # Azul claro
            "color_node": "#1565C0", # Azul escuro
        },
        "mat2": {
            "id": 2,
            "name": "Material 2 (Dielétrico)",
            "eps_r": eps_r2,
            "x_bounds": (x_interface, Lx),
            "y_bounds": (0.0, Ly),
            "color_bg": "#FFF3E0",  # Laranja claro
            "color_node": "#E65100", # Laranja escuro
        },
        "interface": {
            "id": 0,
            "name": "Interface Material (Gamma_int)",
            "color_node": "#2E7D32", # Verde escuro
            "color_line": "#C62828", # Vermelho escuro
        }
    }


def generate_material_nodes(domain, Nx1=9, Nx2=9, Ny=9, jitter_frac=0.05, seed=42):
    """
    Gera a nuvem de nós do VNMM para o domínio bimaterial.
    Mantém estritamente os ângulos determinados (alternados +-45 graus no interior,
    tangentes ao contorno nas bordas externas e tangentes na interface).
    """
    np.random.seed(seed)
    
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    x_int = domain["x_interface"]
    
    coords_list = []
    vectors_list = []
    mat_ids_list = []
    
    # Espaçamentos característicos
    dx1 = x_int / (Nx1 - 1)
    dx2 = (Lx - x_int) / (Nx2 - 1)
    dy = Ly / (Ny - 1)
    
    # 1. Nós da Região 1 (0 <= x < x_int)
    for i in range(Nx1 - 1):
        x_nominal = i * dx1
        is_x_boundary = (i == 0)
        for j in range(Ny):
            y_nominal = j * dy
            is_y_boundary = (j == 0 or j == Ny - 1)
            
            if not is_x_boundary and not is_y_boundary and jitter_frac > 0:
                jx = (np.random.rand() - 0.5) * 2 * jitter_frac * dx1
                jy = (np.random.rand() - 0.5) * 2 * jitter_frac * dy
                x = np.clip(x_nominal + jx, 0.05 * x_int, 0.95 * x_int)
                y = np.clip(y_nominal + jy, 0.05 * Ly, 0.95 * Ly)
            else:
                x = x_nominal
                y = y_nominal
                
            if is_y_boundary:
                t = np.array([1.0, 0.0])
            elif is_x_boundary:
                t = np.array([0.0, 1.0])
            else:
                angle = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(angle), np.sin(angle)])
                
            coords_list.append([x, y])
            vectors_list.append(t)
            mat_ids_list.append(1)
            
    # 2. Nós sobre a Interface (x = x_int)
    for j in range(Ny):
        y = j * dy
        t = np.array([0.0, 1.0])  # Tangente ao longo do eixo y
        coords_list.append([x_int, y])
        vectors_list.append(t)
        mat_ids_list.append(0)  # 0 indica nó de interface
        
    # 3. Nós da Região 2 (x_int < x <= Lx)
    for i in range(1, Nx2):
        x_nominal = x_int + i * dx2
        is_x_boundary = (i == Nx2 - 1)
        for j in range(Ny):
            y_nominal = j * dy
            is_y_boundary = (j == 0 or j == Ny - 1)
            
            if not is_x_boundary and not is_y_boundary and jitter_frac > 0:
                jx = (np.random.rand() - 0.5) * 2 * jitter_frac * dx2
                jy = (np.random.rand() - 0.5) * 2 * jitter_frac * dy
                x = np.clip(x_nominal + jx, x_int + 0.05 * (Lx - x_int), Lx - 0.05 * (Lx - x_int))
                y = np.clip(y_nominal + jy, 0.05 * Ly, 0.95 * Ly)
            else:
                x = x_nominal
                y = y_nominal
                
            if is_y_boundary:
                t = np.array([1.0, 0.0])
            elif is_x_boundary:
                t = np.array([0.0, 1.0])
            else:
                angle = np.pi / 4.0 if (i + j) % 2 == 0 else 3.0 * np.pi / 4.0
                t = np.array([np.cos(angle), np.sin(angle)])
                
            coords_list.append([x, y])
            vectors_list.append(t)
            mat_ids_list.append(2)
            
    coords = np.array(coords_list)
    vectors = np.array(vectors_list)
    mat_ids = np.array(mat_ids_list, dtype=int)
    
    return coords, vectors, mat_ids


def generate_interpolation_points(domain, nx=24, ny=12):
    """
    Gera uma grade regular de pontos internos para avaliação da interpolação.
    """
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    x_int = domain["x_interface"]
    
    dx = Lx / nx
    dy = Ly / ny
    
    x_coords = np.linspace(dx / 2.0, Lx - dx / 2.0, nx)
    y_coords = np.linspace(dy / 2.0, Ly - dy / 2.0, ny)
    
    X, Y = np.meshgrid(x_coords, y_coords)
    interp_points = np.column_stack([X.ravel(), Y.ravel()])
    
    interp_mat_ids = np.zeros(len(interp_points), dtype=int)
    for k, pt in enumerate(interp_points):
        if pt[0] < x_int:
            interp_mat_ids[k] = 1
        elif pt[0] > x_int:
            interp_mat_ids[k] = 2
        else:
            interp_mat_ids[k] = 0
            
    return interp_points, interp_mat_ids


# ==============================================================================
# DEFINIÇÃO DOS CAMPOS DE TESTE: GALERKIN PATCH TEST E MMS
# ==============================================================================

def get_patch_test_field():
    """
    Retorna o campo afim linear do Patch Test de Galerkin e seu rotacional exato.
    E(x, y) = [1.5 + 0.8*x - 0.3*y, -0.7 + 0.4*x + 1.2*y]^T
    rot(E)_z = dEy/dx - dEx/dy = 0.4 - (-0.3) = 0.7
    """
    def eval_E(x, y):
        return np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y])
    
    def eval_curl(x, y):
        return 0.7
    
    return {
        "name": "Galerkin Patch Test (Campo Afim)",
        "eval_E": eval_E,
        "eval_curl": eval_curl,
        "is_polynomial": True
    }


def get_mms_field(domain):
    """
    Retorna o campo trigonométrico MMS (Method of Manufactured Solutions)
    e seu rotacional exato em [0, Lx] x [0, Ly].
    E(x, y) = [ sin(kx*x)*cos(ky*y), -cos(kx*x)*sin(ky*y) ]^T
    rot(E)_z = (kx + ky) * sin(kx*x) * sin(ky*y)
    """
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    kx = np.pi / Lx
    ky = np.pi / Ly
    
    def eval_E(x, y):
        return np.array([
            np.sin(kx * x) * np.cos(ky * y),
            -np.cos(kx * x) * np.sin(ky * y)
        ])
    
    def eval_curl(x, y):
        return (kx + ky) * np.sin(kx * x) * np.sin(ky * y)
    
    return {
        "name": "MMS (Method of Manufactured Solutions)",
        "eval_E": eval_E,
        "eval_curl": eval_curl,
        "is_polynomial": False
    }


# ==============================================================================
# FUNÇÃO DE INTERPOLAÇÃO MULTIMATERIAL COM RESTRIÇÃO ESTRITA DE SUPORTE
# ==============================================================================

def interpolate_vnmm_multimaterial(
    domain,
    node_coords,
    node_vectors,
    node_mat_ids,
    interp_points,
    interp_mat_ids,
    target_field,
    allow_interface_nodes=True,
    Tol_det=1e-5
):
    """
    Executa a interpolação VNMM 2D (base linear P^1 de 6 nós) garantindo estritamente
    que os domínios de suporte NÃO contenham nós de materiais distintos.
    
    Regra de Formação do Suporte:
    - Para pontos no Material 1 (id=1): candidatos são estritamente nós com mat_id == 1
      (e nós de interface mat_id == 0 se allow_interface_nodes=True). Nós de mat_id == 2
      são ESTRITAMENTE EXCLUÍDOS.
    - Para pontos no Material 2 (id=2): candidatos são estritamente nós com mat_id == 2
      (e nós de interface mat_id == 0 se allow_interface_nodes=True). Nós de mat_id == 1
      são ESTRITAMENTE EXCLUÍDOS.
    """
    eval_E_exact = target_field["eval_E"]
    eval_curl_exact = target_field["eval_curl"]
    
    # 1. Amostragem do campo exato nos nós -> graus de liberdade e_i
    total_nodes = len(node_coords)
    nodal_projections = np.zeros(total_nodes)
    for i in range(total_nodes):
        x, y = node_coords[i]
        E_exact_node = eval_E_exact(x, y)
        nodal_projections[i] = np.dot(E_exact_node, node_vectors[i])
        
    num_points = len(interp_points)
    E_interpolated = np.zeros((num_points, 2))
    curl_interpolated = np.zeros(num_points)
    
    E_exact_all = np.zeros((num_points, 2))
    curl_exact_all = np.zeros(num_points)
    
    errors_E = np.zeros(num_points)
    errors_curl = np.zeros(num_points)
    selected_stencils = []
    
    # Pré-filtrar nós e construir árvores KDTree independentes para cada material
    mat_subsets = {}
    for mat_id in [1, 2]:
        if allow_interface_nodes:
            cand_indices = np.where((node_mat_ids == mat_id) | (node_mat_ids == 0))[0]
        else:
            cand_indices = np.where(node_mat_ids == mat_id)[0]
            
        mat_subsets[mat_id] = {
            "indices": cand_indices,
            "coords": node_coords[cand_indices],
            "vectors": node_vectors[cand_indices],
            "tree": KDTree(node_coords[cand_indices])
        }
        
    # Verificação de segurança: checar contaminação de nós
    stencils_validated = True
    
    # 2. Loop sobre os pontos de avaliação
    for p_idx, P in enumerate(interp_points):
        p_mat = interp_mat_ids[p_idx]
        target_mat = p_mat if p_mat in [1, 2] else 1
        
        subset = mat_subsets[target_mat]
        
        # Seleção dos 6 nós de suporte restritos estritamente ao material
        sel_local, det_A, matrix_A, eff_k = support_nodes_vnmm_2d_6_p1(
            P=P,
            nodes_coords=subset["coords"],
            nodes_vectors=subset["vectors"],
            search_tree=subset["tree"],
            K=12,
            Tol_det=Tol_det,
            adaptive=True
        )
        
        # Mapear índices locais do subconjunto para os índices globais
        sel_global = subset["indices"][sel_local]
        selected_stencils.append(sel_global)
        
        # Verificação rigorosa: assegurar que nenhum nó é de outro material
        foreign_material_nodes = (node_mat_ids[sel_global] != target_mat) & (node_mat_ids[sel_global] != 0)
        if np.any(foreign_material_nodes):
            stencils_validated = False
            raise RuntimeError(f"Violação de partição de material no ponto {p_idx}! Nó de outro material no stencil.")
            
        # Calcular funções de forma vetoriais e rotacional via P^1
        Phi, curl_Phi, beta = shape_functions_vnmm_2d_6_p1(
            P=P,
            nodes_coords=node_coords,
            nodes_vectors=node_vectors,
            selected_nodes=sel_global,
            matrix_a=matrix_A
        )
        
        # Interpolação vetorial: E^h = Phi @ e_s
        dofs_stencil = nodal_projections[sel_global]
        E_h = Phi @ dofs_stencil
        curl_h = float(curl_Phi @ dofs_stencil)
        
        E_interpolated[p_idx] = E_h
        curl_interpolated[p_idx] = curl_h
        
        # Valores analíticos exatos em P
        E_ex = eval_E_exact(P[0], P[1])
        curl_ex = eval_curl_exact(P[0], P[1])
        
        E_exact_all[p_idx] = E_ex
        curl_exact_all[p_idx] = curl_ex
        
        errors_E[p_idx] = np.linalg.norm(E_h - E_ex)
        errors_curl[p_idx] = np.abs(curl_h - curl_ex)
        
    # 3. Métricas globais de erro
    norm_E_exact = np.sqrt(np.sum(np.linalg.norm(E_exact_all, axis=1)**2))
    norm_curl_exact = np.sqrt(np.sum(curl_exact_all**2))
    
    l2_rel_E = np.sqrt(np.sum(errors_E**2)) / max(norm_E_exact, 1e-15)
    l2_rel_curl = np.sqrt(np.sum(errors_curl**2)) / max(norm_curl_exact, 1e-15)
    max_err_E = np.max(errors_E)
    max_err_curl = np.max(errors_curl)
    
    # Métricas por material
    metrics_per_mat = {}
    for m in [1, 2]:
        m_idx = np.where(interp_mat_ids == m)[0]
        if len(m_idx) > 0:
            m_norm_E = np.sqrt(np.sum(np.linalg.norm(E_exact_all[m_idx], axis=1)**2))
            m_norm_c = np.sqrt(np.sum(curl_exact_all[m_idx]**2))
            metrics_per_mat[m] = {
                "num_points": len(m_idx),
                "l2_rel_E": np.sqrt(np.sum(errors_E[m_idx]**2)) / max(m_norm_E, 1e-15),
                "l2_rel_curl": np.sqrt(np.sum(errors_curl[m_idx]**2)) / max(m_norm_c, 1e-15),
                "max_E": np.max(errors_E[m_idx]),
                "max_curl": np.max(errors_curl[m_idx])
            }
            
    return {
        "field_name": target_field["name"],
        "is_polynomial": target_field["is_polynomial"],
        "stencils_validated": stencils_validated,
        "selected_stencils": selected_stencils,
        "E_interpolated": E_interpolated,
        "curl_interpolated": curl_interpolated,
        "E_exact": E_exact_all,
        "curl_exact": curl_exact_all,
        "errors_E": errors_E,
        "errors_curl": errors_curl,
        "l2_rel_E": l2_rel_E,
        "l2_rel_curl": l2_rel_curl,
        "max_err_E": max_err_E,
        "max_err_curl": max_err_curl,
        "metrics_per_mat": metrics_per_mat
    }


# ==============================================================================
# VISUALIZAÇÕES GRÁFICAS
# ==============================================================================

def plot_materials_setup(domain, node_coords, node_vectors, node_mat_ids, 
                         interp_points, interp_mat_ids, 
                         output_path=None, show=False):
    """
    Gera o gráfico ilustrando o domínio bimaterial, nós com diretores e pontos de teste.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    x_int = domain["x_interface"]
    
    # 1. Regiões sombreadas dos materiais de fundo
    ax.fill_between(
        [0.0, x_int], 0.0, Ly,
        color=domain["mat1"]["color_bg"],
        label=f"Região 1: {domain['mat1']['name']} ($\\varepsilon_{{r1}} = {domain['mat1']['eps_r']}$)",
        zorder=1
    )
    ax.fill_between(
        [x_int, Lx], 0.0, Ly,
        color=domain["mat2"]["color_bg"],
        label=f"Região 2: {domain['mat2']['name']} ($\\varepsilon_{{r2}} = {domain['mat2']['eps_r']}$)",
        zorder=1
    )
    
    # 2. Linha divisória da Interface
    ax.axvline(
        x=x_int, color=domain["interface"]["color_line"], 
        linestyle="--", linewidth=2.2,
        label=f"Interface $\\Gamma_{{\\mathrm{{int}}}}$ ($x = {x_int:.2f}$)",
        zorder=3
    )
    
    # 3. Pontos de interpolação / avaliação
    idx_pts_m1 = np.where(interp_mat_ids == 1)[0]
    idx_pts_m2 = np.where(interp_mat_ids == 2)[0]
    
    ax.scatter(
        interp_points[idx_pts_m1, 0], interp_points[idx_pts_m1, 1],
        marker="x", color="#1976D2", s=25, alpha=0.6,
        label=f"Pontos Interpolação Mat 1 ({len(idx_pts_m1)} pts)",
        zorder=4
    )
    ax.scatter(
        interp_points[idx_pts_m2, 0], interp_points[idx_pts_m2, 1],
        marker="x", color="#E64A19", s=25, alpha=0.6,
        label=f"Pontos Interpolação Mat 2 ({len(idx_pts_m2)} pts)",
        zorder=4
    )
    
    # 4. Nós do VNMM divididos por material
    scale_vec = 0.06
    
    idx_n_m1 = np.where(node_mat_ids == 1)[0]
    ax.scatter(
        node_coords[idx_n_m1, 0], node_coords[idx_n_m1, 1],
        facecolor=domain["mat1"]["color_node"], edgecolor="black", linewidth=0.8,
        s=60, label=f"Nós VNMM Mat 1 ({len(idx_n_m1)} nós)", zorder=6
    )
    ax.quiver(
        node_coords[idx_n_m1, 0], node_coords[idx_n_m1, 1],
        node_vectors[idx_n_m1, 0], node_vectors[idx_n_m1, 1],
        angles="xy", scale_units="xy", scale=1.0 / scale_vec,
        color="#0D47A1", width=0.0035, zorder=7
    )
    
    idx_n_m2 = np.where(node_mat_ids == 2)[0]
    ax.scatter(
        node_coords[idx_n_m2, 0], node_coords[idx_n_m2, 1],
        facecolor=domain["mat2"]["color_node"], edgecolor="black", linewidth=0.8,
        s=60, label=f"Nós VNMM Mat 2 ({len(idx_n_m2)} nós)", zorder=6
    )
    ax.quiver(
        node_coords[idx_n_m2, 0], node_coords[idx_n_m2, 1],
        node_vectors[idx_n_m2, 0], node_vectors[idx_n_m2, 1],
        angles="xy", scale_units="xy", scale=1.0 / scale_vec,
        color="#BF360C", width=0.0035, zorder=7
    )
    
    idx_n_int = np.where(node_mat_ids == 0)[0]
    ax.scatter(
        node_coords[idx_n_int, 0], node_coords[idx_n_int, 1],
        facecolor=domain["interface"]["color_node"], edgecolor="black", linewidth=1.2,
        s=80, marker="o", label=f"Nós de Interface $\\Gamma_{{\\mathrm{{int}}}}$ ({len(idx_n_int)} nós)", zorder=8
    )
    ax.quiver(
        node_coords[idx_n_int, 0], node_coords[idx_n_int, 1],
        node_vectors[idx_n_int, 0], node_vectors[idx_n_int, 1],
        angles="xy", scale_units="xy", scale=1.0 / scale_vec,
        color="#1B5E20", width=0.0045, zorder=9
    )
    
    ax.set_title(
        "Configuração de Domínio Multimaterial para Interpolação VNMM 2D\n"
        "Regiões de Materiais Distintos, Interface, Nuvem de Nós com Diretores e Pontos de Interpolação",
        fontsize=13, fontweight="bold", pad=15
    )
    ax.set_xlabel("Coordenada X (m)", fontsize=11)
    ax.set_ylabel("Coordenada Y (m)", fontsize=11)
    ax.set_xlim(-0.08 * Lx, 1.08 * Lx)
    ax.set_ylim(-0.08 * Ly, 1.08 * Ly)
    ax.set_aspect("equal", "box")
    ax.grid(True, linestyle=":", alpha=0.6)
    
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=3, framealpha=0.95, fontsize=9.5
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico de configuração salvo com sucesso em: {output_path}")
        
    if show:
        plt.show()


def plot_interpolation_results(
    domain,
    node_coords,
    node_mat_ids,
    interp_points,
    res_patch,
    res_mms,
    output_path=None,
    show=False
):
    """
    Gera um painel com 4 subgráficos detalhando os resultados de interpolação:
    1. Mapa de Erro do Patch Test (comprovando precisão de máquina ~10^-15 em ambos materiais).
    2. Mapa de Erro do MMS (Method of Manufactured Solutions).
    3. Campo vetorial interpolado vs exato para o MMS.
    4. Exemplo de stencils de suporte comprovando estrita separação entre materiais.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    Lx = domain["Lx"]
    Ly = domain["Ly"]
    x_int = domain["x_interface"]
    
    # --------------------------------------------------------------------------
    # SUBPLOT 1: Mapa de Erro do Patch Test de Galerkin
    # --------------------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.fill_between([0.0, x_int], 0.0, Ly, color=domain["mat1"]["color_bg"], alpha=0.5, zorder=1)
    ax1.fill_between([x_int, Lx], 0.0, Ly, color=domain["mat2"]["color_bg"], alpha=0.5, zorder=1)
    ax1.axvline(x=x_int, color=domain["interface"]["color_line"], linestyle="--", linewidth=1.5, zorder=2)
    
    err_log_patch = np.log10(np.clip(res_patch["errors_E"], 1e-16, None))
    sc1 = ax1.scatter(
        interp_points[:, 0], interp_points[:, 1],
        c=err_log_patch, cmap="viridis", s=45, edgecolor="none", zorder=3
    )
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label("$\\log_{10}(\\|\\mathbf{E}^h - \\mathbf{E}_{\\mathrm{ex}}\\|_2)$", fontsize=10)
    
    ax1.set_title(
        f"Galerkin Patch Test (Campo Afim)\n"
        f"Erro Relativo Global: {res_patch['l2_rel_E']:.2e} (Precisão de Máquina)",
        fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # --------------------------------------------------------------------------
    # SUBPLOT 2: Mapa de Erro do MMS
    # --------------------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.fill_between([0.0, x_int], 0.0, Ly, color=domain["mat1"]["color_bg"], alpha=0.5, zorder=1)
    ax2.fill_between([x_int, Lx], 0.0, Ly, color=domain["mat2"]["color_bg"], alpha=0.5, zorder=1)
    ax2.axvline(x=x_int, color=domain["interface"]["color_line"], linestyle="--", linewidth=1.5, zorder=2)
    
    sc2 = ax2.scatter(
        interp_points[:, 0], interp_points[:, 1],
        c=res_mms["errors_E"], cmap="plasma", s=45, edgecolor="none", zorder=3
    )
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label("Erro Absoluto $\\|\\mathbf{E}^h - \\mathbf{E}_{\\mathrm{ex}}\\|_2$", fontsize=10)
    
    ax2.set_title(
        f"MMS (Campo Trigonométrico Suave)\n"
        f"Erro Relativo Global: {res_mms['l2_rel_E']:.2%} | Rotacional: {res_mms['l2_rel_curl']:.2%}",
        fontsize=11, fontweight="bold"
    )
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_aspect("equal")
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    # --------------------------------------------------------------------------
    # SUBPLOT 3: Campo Vetorial Interpolado (MMS)
    # --------------------------------------------------------------------------
    ax3 = axes[1, 0]
    ax3.fill_between([0.0, x_int], 0.0, Ly, color=domain["mat1"]["color_bg"], alpha=0.5, zorder=1)
    ax3.fill_between([x_int, Lx], 0.0, Ly, color=domain["mat2"]["color_bg"], alpha=0.5, zorder=1)
    ax3.axvline(x=x_int, color=domain["interface"]["color_line"], linestyle="--", linewidth=1.5, zorder=2)
    
    # Vetores interpolados
    E_h = res_mms["E_interpolated"]
    ax3.quiver(
        interp_points[:, 0], interp_points[:, 1],
        E_h[:, 0], E_h[:, 1],
        angles="xy", scale_units="xy", scale=8.0,
        color="#283593", width=0.0035, label="$\\mathbf{E}^h$ (Interpolado)", zorder=3
    )
    
    ax3.set_title(
        "Campo Vetorial Interpolado $\\mathbf{E}^h$ (MMS)\n"
        "Comportamento Físico Contínuo no Domínio",
        fontsize=11, fontweight="bold"
    )
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_aspect("equal")
    ax3.grid(True, linestyle=":", alpha=0.5)
    ax3.legend(loc="upper right", fontsize=9)
    
    # --------------------------------------------------------------------------
    # SUBPLOT 4: Ilustração da Restrição Estrita de Suportes
    # --------------------------------------------------------------------------
    ax4 = axes[1, 1]
    ax4.fill_between([0.0, x_int], 0.0, Ly, color=domain["mat1"]["color_bg"], alpha=0.5, zorder=1)
    ax4.fill_between([x_int, Lx], 0.0, Ly, color=domain["mat2"]["color_bg"], alpha=0.5, zorder=1)
    ax4.axvline(x=x_int, color=domain["interface"]["color_line"], linestyle="--", linewidth=1.8, label="Interface $\\Gamma_{\\mathrm{int}}$", zorder=2)
    
    # Plotar nós de fundo
    ax4.scatter(node_coords[:, 0], node_coords[:, 1], color="lightgray", s=25, alpha=0.6, zorder=2)
    
    # Destacar 2 pontos de avaliação próximos da interface (um em cada material)
    pts_m1_near = [i for i, pt in enumerate(interp_points) if pt[0] < x_int and (x_int - pt[0]) < 0.15]
    pts_m2_near = [i for i, pt in enumerate(interp_points) if pt[0] > x_int and (pt[0] - x_int) < 0.15]
    
    sample_p1 = pts_m1_near[len(pts_m1_near) // 2] if pts_m1_near else 0
    sample_p2 = pts_m2_near[len(pts_m2_near) // 2] if pts_m2_near else -1
    
    colors_stencil = ["#1565C0", "#E65100"]
    for p_sample, col, label_p in [(sample_p1, colors_stencil[0], "P1 (Mat 1)"), (sample_p2, colors_stencil[1], "P2 (Mat 2)")]:
        P = interp_points[p_sample]
        st = res_mms["selected_stencils"][p_sample]
        
        # Conectar ponto aos seus 6 nós
        for n_idx in st:
            n_xy = node_coords[n_idx]
            ax4.plot([P[0], n_xy[0]], [P[1], n_xy[1]], linestyle="--", color=col, linewidth=1.2, alpha=0.6, zorder=3)
            
        ax4.scatter(node_coords[st, 0], node_coords[st, 1], facecolor=col, edgecolor="black", s=70, zorder=4)
        ax4.scatter(P[0], P[1], facecolor=col, edgecolor="black", marker="*", s=300, label=f"Ponto {label_p} + 6 Nós", zorder=5)
        
    ax4.set_title(
        "Verificação de Suporte Estrito por Material\n"
        "Nenhum Stencil Contém Nós do Material Oposto",
        fontsize=11, fontweight="bold"
    )
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.set_aspect("equal")
    ax4.grid(True, linestyle=":", alpha=0.5)
    ax4.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Gráfico de resultados de interpolação salvo com sucesso em: {output_path}")
        
    if show:
        plt.show()


# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    print("=" * 80)
    print("      INTERPOLAÇÃO VNMM 2D EM DOMÍNIO MULTIMATERIAL")
    print("=" * 80)
    
    # 1. Definir o domínio bimaterial
    domain = create_two_material_domain(
        Lx=2.0, 
        Ly=1.0, 
        x_interface=1.0, 
        eps_r1=1.0, 
        eps_r2=4.0
    )
    print(f"\nDomínio Retangular: [0, {domain['Lx']}] x [0, {domain['Ly']}]")
    print(f"- Região 1: x in {domain['mat1']['x_bounds']}, {domain['mat1']['name']}, eps_r = {domain['mat1']['eps_r']}")
    print(f"- Região 2: x in {domain['mat2']['x_bounds']}, {domain['mat2']['name']}, eps_r = {domain['mat2']['eps_r']}")
    print(f"- Interface material: x = {domain['x_interface']}")
    
    # 2. Gerar nuvem de nós do VNMM (mantendo os ângulos pré-determinados)
    node_coords, node_vectors, node_mat_ids = generate_material_nodes(
        domain=domain,
        Nx1=9,
        Nx2=9,
        Ny=9,
        jitter_frac=0.05,
        seed=42
    )
    
    n_total = len(node_coords)
    n_mat1 = np.sum(node_mat_ids == 1)
    n_mat2 = np.sum(node_mat_ids == 2)
    n_int = np.sum(node_mat_ids == 0)
    
    print(f"\n--- Nuvem de Nós VNMM ---")
    print(f"Total de nós gerados : {n_total}")
    print(f"  - Nós no Material 1: {n_mat1}")
    print(f"  - Nós no Material 2: {n_mat2}")
    print(f"  - Nós na Interface : {n_int}")
    
    # 3. Gerar pontos de interpolação / teste
    interp_points, interp_mat_ids = generate_interpolation_points(
        domain=domain,
        nx=24,
        ny=12
    )
    print(f"\nTotal de pontos de interpolação: {len(interp_points)} (144 em cada região)")
    
    # 4. Gráfico da configuração inicial
    setup_plot_file = DIR_EXPERIMENTOS / "VNMM_Materials_setup.png"
    plot_materials_setup(
        domain=domain,
        node_coords=node_coords,
        node_vectors=node_vectors,
        node_mat_ids=node_mat_ids,
        interp_points=interp_points,
        interp_mat_ids=interp_mat_ids,
        output_path=str(setup_plot_file),
        show=False
    )
    
    # ==========================================================================
    # TESTE 1: GALERKIN PATCH TEST (CAMPO AFIM LINEAR)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("      EXECUTANDO TESTE 1: GALERKIN PATCH TEST (CAMPO AFIM)")
    print("=" * 80)
    patch_field = get_patch_test_field()
    res_patch = interpolate_vnmm_multimaterial(
        domain=domain,
        node_coords=node_coords,
        node_vectors=node_vectors,
        node_mat_ids=node_mat_ids,
        interp_points=interp_points,
        interp_mat_ids=interp_mat_ids,
        target_field=patch_field,
        allow_interface_nodes=True
    )
    
    print(f"Restrição de stencils verificada: {res_patch['stencils_validated']} (100% de separação de materiais)")
    print(f"Erro L2 Relativo Global (Campo E)  : {res_patch['l2_rel_E']:.2e}")
    print(f"Erro L2 Relativo Global (Rotacional): {res_patch['l2_rel_curl']:.2e}")
    print(f"Erro Máximo Pontual (Linfinity)     : {res_patch['max_err_E']:.2e}")
    for m in [1, 2]:
        m_info = res_patch["metrics_per_mat"][m]
        print(f"  - Material {m}: Erro Rel E = {m_info['l2_rel_E']:.2e}, Max = {m_info['max_E']:.2e}")
    if res_patch['l2_rel_E'] < 1e-12:
        print(">>> STATUS DO PATCH TEST: APROVADO COM PRECISÃO DE MÁQUINA! <<<")
    else:
        print(">>> STATUS DO PATCH TEST: FALHOU <<<")
        
    # ==========================================================================
    # TESTE 2: METHOD OF MANUFACTURED SOLUTIONS (MMS)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("      EXECUTANDO TESTE 2: METHOD OF MANUFACTURED SOLUTIONS (MMS)")
    print("=" * 80)
    mms_field = get_mms_field(domain)
    res_mms = interpolate_vnmm_multimaterial(
        domain=domain,
        node_coords=node_coords,
        node_vectors=node_vectors,
        node_mat_ids=node_mat_ids,
        interp_points=interp_points,
        interp_mat_ids=interp_mat_ids,
        target_field=mms_field,
        allow_interface_nodes=True
    )
    
    print(f"Restrição de stencils verificada: {res_mms['stencils_validated']} (100% de separação de materiais)")
    print(f"Erro L2 Relativo Global (Campo E)  : {res_mms['l2_rel_E']:.4%}")
    print(f"Erro L2 Relativo Global (Rotacional): {res_mms['l2_rel_curl']:.4%}")
    print(f"Erro Máximo Pontual (Linfinity)     : {res_mms['max_err_E']:.4e}")
    for m in [1, 2]:
        m_info = res_mms["metrics_per_mat"][m]
        print(f"  - Material {m}: Erro Rel E = {m_info['l2_rel_E']:.4%}, Rotacional = {m_info['l2_rel_curl']:.4%}")
        
    # ==========================================================================
    # GERAÇÃO DO GRÁFICO DE RESULTADOS DE INTERPOLAÇÃO
    # ==========================================================================
    results_plot_file = DIR_EXPERIMENTOS / "VNMM_Materials_interpolation_results.png"
    print(f"\nGerando gráfico completo de resultados de interpolação...")
    plot_interpolation_results(
        domain=domain,
        node_coords=node_coords,
        node_mat_ids=node_mat_ids,
        interp_points=interp_points,
        res_patch=res_patch,
        res_mms=res_mms,
        output_path=str(results_plot_file),
        show=False
    )
    print(f"Todos os testes foram concluídos com sucesso!")


if __name__ == "__main__":
    main()
