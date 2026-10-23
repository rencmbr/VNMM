"""
Galerkin Patch Test for Vector Nodal Meshless Method (VNMM 2D).
Verifies exact reproduction of affine vector fields by solving the
Galerkin linear system K * e = F with essential Dirichlet boundary conditions.

Fields tested:
1. Pure translation along X: E = [1, 0]^T (Harmonic: curl=0, div=0)
2. Pure translation along Y: E = [0, 1]^T (Harmonic: curl=0, div=0)
3. Normal strain (longitudinal): E = [x, -y]^T (Harmonic: curl=0, div=0)
4. Shear strain: E = [y, x]^T (Harmonic: curl=0, div=0)
5. Rigid body rotation (Pure Curl): E = [-y, x]^T (curl=2, div=0)
6. Volumetric expansion (Pure Div): E = [x, y]^T (curl=0, div=2)
7. Full general affine field: E = [a1 + b1*x + b2*y, a2 + b3*x + b4*y]^T

Compares:
- Complete basis P1 (6 nodes/terms)
- Incomplete basis L1 (3 nodes/terms, 1st order Nedelec)
On regular Cartesian grid and irregular nodal cloud with stochastic perturbation (25% jitter).
"""

import os
import sys
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Path configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
for p in [ROOT_DIR, CURRENT_DIR, SRC_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.cavity_mesh import generate_cavity_mesh, gerar_malha_cavidade
from src.gauss_quadrature import get_gauss_points_weights_1d
from scripts.support_nodes_vnmm_2d_3_l1 import support_nodes_vnmm_2d_3_l1
from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1


def assemble_patch_test_system(
    coords,
    vectors,
    target_field,
    basis="P1",
    s_div=6.0,
    Ncx=10,
    Ncy=10,
    Lx=np.pi,
    Ly=np.pi,
    points_per_dir=3,
    det_tolerance=None,
    floor_tolerance=None,
    # Backward compatibility
    campo_alvo=None,
    base=None,
    pontos_por_dir=None,
    tolerancia_det=None,
    tol_piso=None
):
    """
    Assembles stiffness matrix K = K_curl + s_div * K_div and load vector F
    consistent with affine target field E_target:
        rot_exact = (dEy/dx - dEx/dy) = const
        div_exact = (dEx/dx + dEy/dy) = const
        
    F_i = integral_Omega [ (curl N_i)_z * rot_exact + s_div * (div N_i) * div_exact ] dOmega
    """
    if campo_alvo is not None:
        target_field = campo_alvo
    if base is not None:
        basis = base
    if pontos_por_dir is not None:
        points_per_dir = pontos_por_dir
    if tolerancia_det is not None:
        det_tolerance = tolerancia_det
    if tol_piso is not None:
        floor_tolerance = tol_piso

    total_nodes = len(coords)
    tree = KDTree(coords)
    
    is_P1 = basis.upper() in ["P1", "6_P1", "6NOS"]
    
    rot_exact = target_field["rot"]
    div_exact = target_field["div"]
    
    side_nodes = int(np.round(np.sqrt(total_nodes)))
    h_char = max(Lx / max(side_nodes - 1, 1), Ly / max(side_nodes - 1, 1))
    h_ref = np.pi / 20.0
    if det_tolerance is None:
        tol_base = 1e-4 * (h_char / h_ref)**4 if is_P1 else 1e-4 * (h_char / h_ref)
        det_tolerance = max(tol_base, floor_tolerance) if floor_tolerance is not None else tol_base

    x_edges = np.linspace(0.0, Lx, Ncx + 1)
    y_edges = np.linspace(0.0, Ly, Ncy + 1)
    xi_1d, w_1d = get_gauss_points_weights_1d(points_per_dir)
    
    rows_K, cols_K, data_K = [], [], []
    F = np.zeros(total_nodes, dtype=float)
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
                        supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_2d_6_p1(
                            P=Pg,
                            nodes_coords=coords,
                            nodes_vectors=vectors,
                            search_tree=tree,
                            K=12,
                            Tol_det=det_tolerance,
                            adaptive=True,
                            step_K=4
                        )
                        beta = np.linalg.inv(A_mat)
                        Phi_g = beta[0:2, :]
                        rot_Phi = beta[4, :] - beta[3, :]
                        div_Phi = beta[2, :] + beta[5, :]
                        n_supp = 6
                    else:
                        supp_nodes, det_val, A_mat, _ = support_nodes_vnmm_2d_3_l1(
                            P=Pg,
                            nodes_coords=coords,
                            nodes_vectors=vectors,
                            search_tree=tree,
                            K=8,
                            Tol_det=det_tolerance,
                            adaptive=True,
                            step_K=2
                        )
                        beta = np.linalg.inv(A_mat)
                        Phi_g = beta[0:2, :]
                        rot_Phi = -2.0 * beta[2, :]
                        div_Phi = np.zeros(3, dtype=float)
                        n_supp = 3
                        
                    gauss_info.append({
                        'Pg': Pg,
                        'wg': wg,
                        'nodes': supp_nodes,
                        'Phi_g': Phi_g,
                        'rot_Phi': rot_Phi,
                        'div_Phi': div_Phi,
                        'nos': supp_nodes
                    })
                    
                    rot_outer = np.outer(rot_Phi, rot_Phi) * wg
                    div_outer = np.outer(div_Phi, div_Phi) * (s_div * wg)
                    k_local = rot_outer + div_outer
                    
                    f_local = (rot_Phi * rot_exact + s_div * div_Phi * div_exact) * wg
                    
                    for a in range(n_supp):
                        Ia = supp_nodes[a]
                        F[Ia] += f_local[a]
                        for b in range(n_supp):
                            Ib = supp_nodes[b]
                            val = k_local[a, b]
                            if abs(val) > 1e-18:
                                rows_K.append(Ia)
                                cols_K.append(Ib)
                                data_K.append(val)
                                
    K = coo_matrix((data_K, (rows_K, cols_K)), shape=(total_nodes, total_nodes)).tocsr()
    K = 0.5 * (K + K.T)
    return K, F, gauss_info


def run_patch_test_mode(
    coords,
    vectors,
    is_boundary,
    target_field,
    basis="P1",
    s_div=6.0,
    Ncx=10,
    Ncy=10,
    Lx=np.pi,
    Ly=np.pi,
    # Backward compatibility
    campo_alvo=None,
    base=None
):
    """
    Executes Patch Test for a specific affine mode:
    1. Assembles K and F
    2. Enforces essential Dirichlet on boundary nodes: e_bnd = E_target(x_k) . t_k
    3. Solves K_int,int * e_int = F_int - K_int,bnd * e_bnd
    4. Evaluates error in interior DoFs and field reconstruction errors at Gauss points.
    """
    if campo_alvo is not None:
        target_field = campo_alvo
    if base is not None:
        basis = base

    total_nodes = len(coords)
    
    E_func = target_field["func"]
    e_exact = np.zeros(total_nodes, dtype=float)
    for k in range(total_nodes):
        E_pt = E_func(coords[k, 0], coords[k, 1])
        e_exact[k] = np.dot(E_pt, vectors[k])
        
    idx_bnd = np.where(is_boundary)[0]
    idx_int = np.where(~is_boundary)[0]
    
    K, F, gauss_info = assemble_patch_test_system(
        coords=coords,
        vectors=vectors,
        target_field=target_field,
        basis=basis,
        s_div=s_div,
        Ncx=Ncx,
        Ncy=Ncy,
        Lx=Lx,
        Ly=Ly
    )
    
    K_ii = K[idx_int, :][:, idx_int]
    K_ib = K[idx_int, :][:, idx_bnd]
    
    e_bnd = e_exact[idx_bnd]
    rhs_int = F[idx_int] - K_ib @ e_bnd
    
    e_sol_int = spsolve(K_ii, rhs_int)
    
    e_sol = np.zeros(total_nodes, dtype=float)
    e_sol[idx_bnd] = e_bnd
    e_sol[idx_int] = e_sol_int
    
    error_dof_abs = np.max(np.abs(e_sol_int - e_exact[idx_int]))
    norm_dof_ref = np.max(np.abs(e_exact[idx_int]))
    error_dof_rel = error_dof_abs / max(norm_dof_ref, 1e-14)
    
    err_L2_E_sq = 0.0
    norm_L2_E_sq = 0.0
    err_L2_rot_sq = 0.0
    err_L2_div_sq = 0.0
    total_area = 0.0
    
    rot_exact = target_field["rot"]
    div_exact = target_field["div"]
    
    for g in gauss_info:
        Pg = g['Pg']
        wg = g['wg']
        supp_nodes = g['nodes']
        Phi_g = g['Phi_g']
        rot_Phi = g['rot_Phi']
        div_Phi = g['div_Phi']
        
        e_supp = e_sol[supp_nodes]
        E_h = Phi_g @ e_supp
        rot_h = np.dot(rot_Phi, e_supp)
        div_h = np.dot(div_Phi, e_supp)
        
        E_ex = E_func(Pg[0], Pg[1])
        
        err_L2_E_sq += wg * np.sum((E_h - E_ex)**2)
        norm_L2_E_sq += wg * np.sum(E_ex**2)
        err_L2_rot_sq += wg * (rot_h - rot_exact)**2
        err_L2_div_sq += wg * (div_h - div_exact)**2
        total_area += wg
        
    err_L2_E = np.sqrt(err_L2_E_sq / max(norm_L2_E_sq, 1e-14))
    err_L2_rot = np.sqrt(err_L2_rot_sq / total_area)
    err_L2_div = np.sqrt(err_L2_div_sq / total_area)
    
    return {
        'error_dof_abs': error_dof_abs,
        'error_dof_rel': error_dof_rel,
        'err_L2_E': err_L2_E,
        'err_L2_rot': err_L2_rot,
        'err_L2_div': err_L2_div,
        'num_internal_dofs': len(idx_int),
        'num_boundary_dofs': len(idx_bnd),
        # Backward compatibility aliases
        'erro_dof_abs': error_dof_abs,
        'erro_dof_rel': error_dof_rel,
        'n_dof_int': len(idx_int),
        'n_dof_bnd': len(idx_bnd)
    }


def run_patch_test_suite():
    """
    Executes full patch test suite across 7 affine states on two mesh topologies:
    1. Regular Cartesian grid (Nx=Ny=11, alternating directors 45°/135°)
    2. Irregular cloud with jitter (Nx=Ny=11, 25% jitter, random directors)
    """
    affine_modes = [
        {
            "name": "1. Translation X: E = [1, 0]^T",
            "func": lambda x, y: np.array([1.0, 0.0]),
            "rot": 0.0,
            "div": 0.0,
            "harmonic": True
        },
        {
            "name": "2. Translation Y: E = [0, 1]^T",
            "func": lambda x, y: np.array([0.0, 1.0]),
            "rot": 0.0,
            "div": 0.0,
            "harmonic": True
        },
        {
            "name": "3. Normal Strain (Harmonic): E = [x, -y]^T",
            "func": lambda x, y: np.array([x, -y]),
            "rot": 0.0,
            "div": 0.0,
            "harmonic": True
        },
        {
            "name": "4. Shear Strain (Harmonic): E = [y, x]^T",
            "func": lambda x, y: np.array([y, x]),
            "rot": 0.0,
            "div": 0.0,
            "harmonic": True
        },
        {
            "name": "5. Rigid Body Rotation (Pure Curl): E = [-y, x]^T",
            "func": lambda x, y: np.array([-y, x]),
            "rot": 2.0,
            "div": 0.0,
            "harmonic": False
        },
        {
            "name": "6. Volumetric Expansion (Pure Div): E = [x, y]^T",
            "func": lambda x, y: np.array([x, y]),
            "rot": 0.0,
            "div": 2.0,
            "harmonic": False
        },
        {
            "name": "7. Full Arbitrary Affine Field",
            "func": lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y]),
            "rot": 0.4 - (-0.3),
            "div": 0.8 + 1.2,
            "harmonic": False
        }
    ]
    
    topologies = [
        {
            "name": "REGULAR MESH (11x11, Alternating Directors 45°/135°)",
            "jitter": 0.0,
            "interior_type": "alternating"
        },
        {
            "name": "IRREGULAR CLOUD (11x11, 25% Jitter, Random Directors)",
            "jitter": 0.25,
            "interior_type": "random"
        }
    ]
    
    print("=" * 95)
    print("        GALERKIN PATCH TEST SUITE - VNMM 2D (COMPLETE P1 vs INCOMPLETE L1)")
    print("=" * 95)
    
    for topo in topologies:
        coords, vectors, is_bnd = generate_cavity_mesh(
            Nx=11,
            Ny=11,
            Lx=np.pi,
            Ly=np.pi,
            interior_type=topo["interior_type"],
            jitter_frac=topo["jitter"],
            seed=42
        )
        
        n_tot = len(coords)
        n_bnd = np.sum(is_bnd)
        n_int = n_tot - n_bnd
        
        print(f"\n>>> TOPOLOGY: {topo['name']}")
        print(f"    Total nodes: {n_tot} | Boundary nodes (Dirichlet): {n_bnd} | Internal nodes (free): {n_int}")
        print("-" * 95)
        print(f"{'Tested Affine State':<42} | {'Basis P1 (Rel DoF)':<17} | {'Basis L1 (Rel DoF)':<17} | {'Verdict':<10}")
        print("-" * 95)
        
        for mode in affine_modes:
            res_P1 = run_patch_test_mode(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                target_field=mode,
                basis="P1",
                s_div=6.0,
                Ncx=10,
                Ncy=10
            )
            
            res_L1 = run_patch_test_mode(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                target_field=mode,
                basis="L1",
                s_div=6.0,
                Ncx=10,
                Ncy=10
            )
            
            pass_P1 = res_P1['error_dof_rel'] < 1e-10
            str_P1 = f"{res_P1['error_dof_rel']:.2e}"
            str_L1 = f"{res_L1['error_dof_rel']:.2e}"
            
            verdict = "P1 PASS" if pass_P1 else "FAIL"
            if res_L1['error_dof_rel'] > 1e-3 and pass_P1:
                verdict += " (L1 FAIL)"
            elif res_L1['error_dof_rel'] < 1e-10 and pass_P1:
                verdict += " (L1 PASS)"
                
            print(f"{mode['name']:<42} | {str_P1:<17} | {str_L1:<17} | {verdict:<10}")


# Backward compatibility aliases
montar_sistema_patch_test = assemble_patch_test_system
executar_patch_test_modo = run_patch_test_mode
executar_bateria_patch_tests = run_patch_test_suite

if __name__ == "__main__":
    run_patch_test_suite()
