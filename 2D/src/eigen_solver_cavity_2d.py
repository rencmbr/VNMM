import os
import sys
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import eigs

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(SRC_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh_2d import generate_cavity_mesh
from src.vnmm_assembler_2d import assemble_vnmm_matrices_2d

# Analytical reference eigenvalues and cutoff wavenumbers from Table 4-1 (Luilly Ortiz, 2023)
# lambda = n^2 + m^2 (TE_nm with n, m >= 0, (n,m) != (0,0)), kc = sqrt(lambda)
ANALYTICAL_REFERENCE_MODES = [
    (1, 0, 1.0, 1.0),            # TE10
    (0, 1, 1.0, 1.0),            # TE01
    (1, 1, 2.0, np.sqrt(2.0)),   # TE11
    (2, 0, 4.0, 2.0),            # TE20
    (0, 2, 4.0, 2.0),            # TE02
    (2, 1, 5.0, np.sqrt(5.0)),   # TE21
    (1, 2, 5.0, np.sqrt(5.0)),   # TE12
    (2, 2, 8.0, np.sqrt(8.0)),   # TE22
    (3, 0, 9.0, 3.0),            # TE30
    (0, 3, 9.0, 3.0),            # TE03
    (3, 1, 10.0, np.sqrt(10.0)), # TE31
    (1, 3, 10.0, np.sqrt(10.0))  # TE13
]

# Compatibility alias
MODOS_ANALITICOS_REF = ANALYTICAL_REFERENCE_MODES


def apply_pec_boundary_conditions(K, M, is_boundary):
    """
    Applies homogeneous Dirichlet boundary conditions (PEC walls) by eliminating
    degrees of freedom corresponding to boundary nodes, where E_t = c_k = 0.
    
    Returns reduced submatrices K_red and M_red, and an array of internal node indices.
    """
    interior_indices = np.where(~is_boundary)[0]
    
    K_red = K[interior_indices, :][:, interior_indices]
    M_red = M[interior_indices, :][:, interior_indices]
    
    return K_red, M_red, interior_indices


def solve_generalized_eigenvalue_problem(
    K_red, 
    M_red, 
    num_eigenvalues=10, 
    sigma=0.5, 
    method="eigh",
    # Compatibility aliases
    num_autovalores=None,
    metodo=None
):
    """
    Solves the generalized eigenvalue problem:
        K_red * c_red = lambda * M_red * c_red
        
    Returns ascendingly sorted eigenvalues and associated eigenvectors.
    """
    if num_autovalores is not None:
        num_eigenvalues = num_autovalores
    if metodo is not None:
        method = metodo

    N_int = K_red.shape[0]
    
    if method == "eigh" or N_int < 300:
        # Exact dense solve via la.eigh (Cholesky), falling back to QZ (la.eig) if M_red is not SPD
        try:
            vals, vecs = la.eigh(K_red.toarray(), M_red.toarray())
        except (la.LinAlgError, np.linalg.LinAlgError):
            vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
            vals = np.real(vals)
            vecs = np.real(vecs)
    else:
        # Sparse shift-and-invert solver
        k_search = min(num_eigenvalues + 6, N_int - 2)
        try:
            vals, vecs = eigs(
                A=K_red, 
                M=M_red, 
                k=k_search, 
                sigma=sigma, 
                which='LM'
            )
            vals = np.real(vals)
            vecs = np.real(vecs)
        except Exception:
            vals, vecs = la.eig(K_red.toarray(), M_red.toarray())
            vals = np.real(vals)
            vecs = np.real(vecs)
        
    # Filter non-zero positive eigenvalues
    positive_mask = vals > 1e-4
    valid_vals = vals[positive_mask]
    valid_vecs = vecs[:, positive_mask]
    
    sorted_idx = np.argsort(valid_vals)
    sorted_vals = valid_vals[sorted_idx][:num_eigenvalues]
    sorted_vecs = valid_vecs[:, sorted_idx][:, :num_eigenvalues]
    
    return sorted_vals, sorted_vecs


def solve_cavity_eigenvalues(
    Nx=21, 
    Ny=21, 
    Lx=np.pi, 
    Ly=np.pi, 
    Ncx=None, 
    Ncy=None, 
    basis="P1", 
    interior_type="alternating", 
    jitter_frac=0.0, 
    num_eigenvalues=10, 
    sigma=0.5,
    s_div=4.0,
    points_per_dir=3,
    det_tolerance=None,
    floor_tolerance=None,
    support_mode="gauss_point",
    seed=42,
    # Compatibility aliases
    base=None,
    tipo_interior=None,
    num_autovalores=None,
    pontos_por_dir=None,
    tolerancia_det=None,
    tol_piso=None,
    modo_suporte=None
):
    """
    Complete 2D VNMM eigenvalue solver pipeline for rectangular PEC cavities.
    
    Pipeline Steps:
    1. Generates nodal distribution with tangential vectors on PEC boundaries.
    2. Assembles sparse regularized K and M matrices via background integration cells.
    3. Reduces system by enforcing homogeneous tangential Dirichlet conditions (E_t = 0).
    4. Solves generalized algebraic eigenvalue problem.
    5. Computes error metrics compared to analytical reference values.
    """
    if base is not None:
        basis = base
    if tipo_interior is not None:
        interior_type = tipo_interior
    if num_autovalores is not None:
        num_eigenvalues = num_autovalores
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    if tolerancia_det is not None:
        det_tolerance = tolerancia_det
    if tol_piso is not None:
        floor_tolerance = tol_piso
    if modo_suporte is not None:
        support_mode = modo_suporte

    if Ncx is None:
        Ncx = max(4, int(np.round(0.6 * Nx)))
    if Ncy is None:
        Ncy = max(4, int(np.round(0.6 * Ny)))
        
    h_char = max(Lx / max(Nx - 1, 1), Ly / max(Ny - 1, 1))
    h_ref = np.pi / 20.0
    if det_tolerance is None:
        tol_base = 1e-4 * (h_char / h_ref)**4 if basis.upper() in ["P1", "6_P1"] else 1e-4 * (h_char / h_ref)
        det_tolerance = max(tol_base, floor_tolerance) if floor_tolerance is not None else tol_base
        
    # 1. Nodal discretization
    coords, vectors, is_boundary = generate_cavity_mesh(
        Nx=Nx, 
        Ny=Ny, 
        Lx=Lx, 
        Ly=Ly, 
        interior_type=interior_type, 
        jitter_frac=jitter_frac, 
        seed=seed
    )
    
    # 2. Assemble global matrices K and M
    K, M, support_stats = assemble_vnmm_matrices_2d(
        coords=coords, 
        vectors=vectors, 
        basis=basis, 
        det_tolerance=det_tolerance,
        floor_tolerance=floor_tolerance,
        s_div=s_div,
        Ncx=Ncx,
        Ncy=Ncy,
        Lx=Lx,
        Ly=Ly,
        points_per_dir=points_per_dir,
        support_mode=support_mode,
        return_support_details=True
    )
    
    # 3. Enforce homogeneous Dirichlet (PEC walls)
    K_red, M_red, interior_indices = apply_pec_boundary_conditions(K, M, is_boundary)
    
    # 4. Spectral solution
    num_eigenvals, eigenvectors_red = solve_generalized_eigenvalue_problem(
        K_red=K_red, 
        M_red=M_red, 
        num_eigenvalues=num_eigenvalues, 
        sigma=sigma
    )
    
    # 5. Comparison against analytical benchmarks
    ref_vals = np.array([item[2] for item in ANALYTICAL_REFERENCE_MODES[:len(num_eigenvals)]])
    ref_kc = np.array([item[3] for item in ANALYTICAL_REFERENCE_MODES[:len(num_eigenvals)]])
    
    kc_num = np.sqrt(num_eigenvals)
    
    rel_errors_lambda_pct = np.abs(num_eigenvals - ref_vals) / ref_vals * 100.0
    rel_errors_kc_pct = np.abs(kc_num - ref_kc) / ref_kc * 100.0
    
    # Global eigenvector reconstruction
    total_nodes = len(coords)
    global_eigenvectors = np.zeros((total_nodes, len(num_eigenvals)), dtype=float)
    global_eigenvectors[interior_indices, :] = eigenvectors_red
    
    h_max = max(Lx / (Nx - 1), Ly / (Ny - 1))
    
    result = {
        'Nx': Nx,
        'Ny': Ny,
        'total_nodes': total_nodes,
        'interior_nodes': len(interior_indices),
        'boundary_nodes': int(np.sum(is_boundary)),
        'h_max': h_max,
        'basis': basis,
        'coords': coords,
        'vectors': vectors,
        'is_boundary': is_boundary,
        'numerical_eigenvalues': num_eigenvals,
        'analytical_eigenvalues': ref_vals,
        'numerical_kc': kc_num,
        'analytical_kc': ref_kc,
        'rel_errors_lambda_pct': rel_errors_lambda_pct,
        'rel_errors_kc_pct': rel_errors_kc_pct,
        'mean_error_lambda_pct': float(np.mean(rel_errors_lambda_pct)),
        'max_error_lambda_pct': float(np.max(rel_errors_lambda_pct)),
        'mean_error_kc_pct': float(np.mean(rel_errors_kc_pct)),
        'max_error_kc_pct': float(np.max(rel_errors_kc_pct)),
        'eigenvectors': global_eigenvectors,
        'K_red': K_red,
        'M_red': M_red,
        'mean_k': support_stats['mean_k'],
        'max_k': support_stats['max_k'],
        'min_k': support_stats['min_k'],
        'mean_det': support_stats['mean_det'],
        'min_det': support_stats['min_det'],
        'max_det': support_stats['max_det'],
        'num_integration_points': support_stats['num_integration_points'],
        'num_support_nodes': support_stats['n_supp'],
        'support_stats': support_stats,
        # Portuguese backward compatibility aliases
        'N_total': total_nodes,
        'N_internos': len(interior_indices),
        'N_fronteira': int(np.sum(is_boundary)),
        'base': basis,
        'autovalores_numericos': num_eigenvals,
        'autovalores_analiticos': ref_vals,
        'kc_numerico': kc_num,
        'kc_analitico': ref_kc,
        'erros_lambda_pct': rel_errors_lambda_pct,
        'erros_kc_pct': rel_errors_kc_pct,
        'erro_medio_lambda_pct': float(np.mean(rel_errors_lambda_pct)),
        'erro_max_lambda_pct': float(np.max(rel_errors_lambda_pct)),
        'erro_medio_kc_pct': float(np.mean(rel_errors_kc_pct)),
        'erro_max_kc_pct': float(np.max(rel_errors_kc_pct)),
        'autovetores': global_eigenvectors,
        'k_medio': support_stats['k_medio'],
        'k_max': support_stats['k_max'],
        'k_min': support_stats['k_min'],
        'det_medio': support_stats['det_medio'],
        'det_min': support_stats['det_min'],
        'det_max': support_stats['det_max'],
        'num_pontos_integracao': support_stats['num_pontos_integracao'],
        'n_nos_suporte_selecionados': support_stats['n_supp'],
        'stats_suporte': support_stats
    }
    return result


# Backward compatibility aliases
aplicar_condicao_pec = apply_pec_boundary_conditions
aplicar_condicoes_contorno_pec = apply_pec_boundary_conditions
resolver_problema_autovalores = solve_generalized_eigenvalue_problem
resolver_problema_autovalor_generalizado = solve_generalized_eigenvalue_problem
resolver_autovalores_cavidade = solve_cavity_eigenvalues
MODOS_ANALITICOS_REF = ANALYTICAL_REFERENCE_MODES
