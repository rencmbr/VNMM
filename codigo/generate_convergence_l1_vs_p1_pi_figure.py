"""
Comparative convergence study of VNMM 2D interpolation: L1 Basis (3 nodes) vs Complete P1 Basis (6 nodes).
Domain: Omega = [0, pi] x [0, pi]
Meshes: 6 nodal distributions with 25% jitter (84 to 4192 nodes)
Evaluation Grid: Fixed 1600 internal Cartesian points (40 x 40)
Test Field: Analytical TE11 cavity mode
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CODE_DIR)
REPORTS_DIR = os.path.join(ROOT_DIR, "relatorios")
os.makedirs(REPORTS_DIR, exist_ok=True)

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from codigo.support_nodes_vnmm_2d_3_l1 import support_nodes_vnmm_2d_3_l1
from codigo.shape_functions_vnmm_2d_3_l1 import shape_functions_vnmm_2d_3_l1
from codigo.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from codigo.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1

# Global Publication Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.labelsize": 11,
    "axes.titlesize": 11.5,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.2,
    "figure.titlesize": 12.5,
    "axes.linewidth": 1.0,
    "grid.linewidth": 0.6,
    "grid.alpha": 0.45,
    "lines.markersize": 6.5,
    "lines.linewidth": 1.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

COLOR_NAVY = "#0f4c81"
COLOR_CRIMSON = "#c0392b"
COLOR_TEAL = "#16a085"
COLOR_GRAY = "#555555"


def exact_field_te11(point, lx=np.pi, ly=np.pi, P=None, Lx=None, Ly=None):
    """Analytical TE11 cavity mode."""
    if P is not None: point = P
    if Lx is not None: lx = Lx
    if Ly is not None: ly = Ly
    x, y = point[0], point[1]
    u = np.pi * x / lx
    v = np.pi * y / ly
    ex = np.cos(u) * np.sin(v)
    ey = -np.sin(u) * np.cos(v)
    curl_z = -(np.pi / lx + np.pi / ly) * np.cos(u) * np.cos(v)
    return np.array([ex, ey], dtype=float), float(curl_z)


def generate_nodal_mesh_pi(num_boundary, num_interior, seed=42, n_front=None, n_int=None):
    """
    Generate non-conforming nodal distribution over [0, pi] x [0, pi].
    Boundary: uniformly spaced nodes with tangential unit vectors.
    Interior: stratified grid with 25% jitter and random azimuthal vectors.
    """
    if n_front is not None: num_boundary = n_front
    if n_int is not None: num_interior = n_int
    if seed is not None:
        np.random.seed(seed)
        
    lx = np.pi
    ly = np.pi
    
    boundary_nodes_per_side = num_boundary // 4
    step = lx / boundary_nodes_per_side
    min_dist_boundary = 0.5 * step
    
    coords = []
    vectors = []
    
    # Bottom
    for i in range(boundary_nodes_per_side):
        coords.append([(i + 0.5) * step, 0.0])
        vectors.append([1.0, 0.0])
    # Right
    for i in range(boundary_nodes_per_side):
        coords.append([lx, (i + 0.5) * step])
        vectors.append([0.0, 1.0])
    # Top
    for i in range(boundary_nodes_per_side):
        coords.append([lx - (i + 0.5) * step, ly])
        vectors.append([-1.0, 0.0])
    # Left
    for i in range(boundary_nodes_per_side):
        coords.append([0.0, ly - (i + 0.5) * step])
        vectors.append([0.0, -1.0])
        
    lim_x = lx - 2.0 * min_dist_boundary
    lim_y = ly - 2.0 * min_dist_boundary
    side_count = int(np.ceil(np.sqrt(num_interior)))
    dx = lim_x / side_count
    dy = lim_y / side_count
    
    xs = np.linspace(min_dist_boundary + dx / 2.0, lx - min_dist_boundary - dx / 2.0, side_count)
    ys = np.linspace(min_dist_boundary + dy / 2.0, ly - min_dist_boundary - dy / 2.0, side_count)
    xx, yy = np.meshgrid(xs, ys)
    base_points = np.column_stack([xx.ravel(), yy.ravel()])
    
    jitter_x = np.random.uniform(-0.25 * dx, 0.25 * dx, size=len(base_points))
    jitter_y = np.random.uniform(-0.25 * dy, 0.25 * dy, size=len(base_points))
    interior_points = base_points + np.column_stack([jitter_x, jitter_y])
    
    if len(interior_points) > num_interior:
        selected_idx = np.random.choice(len(interior_points), num_interior, replace=False)
        interior_points = interior_points[selected_idx]
        
    for pt in interior_points:
        coords.append(pt.tolist())
        theta = np.random.uniform(0, 2 * np.pi)
        vectors.append([np.cos(theta), np.sin(theta)])
        
    h_avg = (4.0 * np.pi) / num_boundary
    return np.array(coords, dtype=float), np.array(vectors, dtype=float), h_avg


def evaluate_mesh_interpolation(coords, vectors, search_tree, test_points, h_mean, h_ref):
    """Evaluate L1 and P1 interpolation on fixed grid of test points."""
    e_nodes = np.zeros_like(coords)
    for i, pt in enumerate(coords):
        e_nodes[i], _ = exact_field_te11(pt)
    global_projections = np.sum(e_nodes * vectors, axis=1)
    
    tol_l1 = 0.25 * (h_mean / h_ref)
    tol_floor = 1.5e-5
    tol_p1 = max(0.08 * (h_mean / h_ref)**4, tol_floor)
    
    err_e_l1, err_rot_l1, dets_l1 = [], [], []
    err_e_p1, err_rot_p1, dets_p1, ks_p1 = [], [], [], []
    
    for pt in test_points:
        e_exact, rot_exact = exact_field_te11(pt)
        
        # L1 evaluation (3 nodes)
        sel_l1, det_a_l1, a_l1, _ = support_nodes_vnmm_2d_3_l1(
            point=pt, node_coords=coords, node_vectors=vectors, search_tree=search_tree,
            num_neighbors=8, det_tolerance=tol_l1, adaptive=True, step_k=4
        )
        phi_l1, rot_phi_l1, _ = shape_functions_vnmm_2d_3_l1(point=pt, node_coords=coords, node_vectors=vectors, selected_nodes=sel_l1, matrix_a=a_l1)
        e_l1 = global_projections[sel_l1]
        e_interp_l1 = phi_l1 @ e_l1
        rot_interp_l1 = float(np.dot(rot_phi_l1, e_l1))
        
        err_e_l1.append(np.linalg.norm(e_interp_l1 - e_exact))
        err_rot_l1.append(abs(rot_interp_l1 - rot_exact))
        dets_l1.append(det_a_l1)
        
        # P1 evaluation (6 nodes)
        sel_p1, det_a_p1, a_p1, k_ef_p1 = support_nodes_vnmm_2d_6_p1(
            point=pt, node_coords=coords, node_vectors=vectors, search_tree=search_tree,
            num_neighbors=12, det_tolerance=tol_p1, adaptive=True, step_k=4
        )
        phi_p1, rot_phi_p1, _ = shape_functions_vnmm_2d_6_p1(point=pt, node_coords=coords, node_vectors=vectors, selected_nodes=sel_p1, matrix_a=a_p1)
        e_p1 = global_projections[sel_p1]
        e_interp_p1 = phi_p1 @ e_p1
        rot_interp_p1 = float(np.dot(rot_phi_p1, e_p1))
        
        err_e_p1.append(np.linalg.norm(e_interp_p1 - e_exact))
        err_rot_p1.append(abs(rot_interp_p1 - rot_exact))
        dets_p1.append(det_a_p1)
        ks_p1.append(k_ef_p1)
        
    return {
        'err_e_l1_rms': float(np.sqrt(np.mean(np.array(err_e_l1)**2))),
        'err_e_l1_max': float(np.max(err_e_l1)),
        'err_rot_l1_rms': float(np.sqrt(np.mean(np.array(err_rot_l1)**2))),
        'err_rot_l1_max': float(np.max(err_rot_l1)),
        'det_l1_mean': float(np.mean(dets_l1)),
        
        'err_e_p1_rms': float(np.sqrt(np.mean(np.array(err_e_p1)**2))),
        'err_e_p1_max': float(np.max(err_e_p1)),
        'err_rot_p1_rms': float(np.sqrt(np.mean(np.array(err_rot_p1)**2))),
        'err_rot_p1_max': float(np.max(err_rot_p1)),
        'det_p1_mean': float(np.mean(dets_p1)),
        'k_ef_p1_mean': float(np.mean(ks_p1)),
        'tol_p1': tol_p1,
        # Backward compatibility
        'err_E_L1_rms': float(np.sqrt(np.mean(np.array(err_e_l1)**2))),
        'err_rot_L1_rms': float(np.sqrt(np.mean(np.array(err_rot_l1)**2))),
        'err_E_P1_rms': float(np.sqrt(np.mean(np.array(err_e_p1)**2))),
        'err_rot_P1_rms': float(np.sqrt(np.mean(np.array(err_rot_p1)**2))),
        'k_ef_P1_med': float(np.mean(ks_p1)),
        'det_L1_med': float(np.mean(dets_l1)),
        'det_P1_med': float(np.mean(dets_p1))
    }


def run_global_study():
    """Run full L1 vs P1 interpolation study across 6 meshes."""
    print("="*75)
    print("PARAMETRIC CONVERGENCE ANALYSIS: L1 BASIS vs. P1 BASIS IN [0, pi]^2")
    print("="*75)
    
    num_side_pts = 40
    xs = np.linspace(0.04 * np.pi, 0.96 * np.pi, num_side_pts)
    ys = np.linspace(0.04 * np.pi, 0.96 * np.pi, num_side_pts)
    xx, yy = np.meshgrid(xs, ys)
    test_points = np.column_stack([xx.ravel(), yy.ravel()])
    print(f"Fixed evaluation grid: {len(test_points)} Cartesian points in [0, pi]^2.\n")
    
    configurations = [
        (24, 60, "Coarse (N=84)"),
        (36, 150, "Medium-Coarse (N=186)"),
        (56, 360, "Medium (N=416)"),
        (84, 800, "Medium-Dense (N=884)"),
        (128, 1800, "Dense (N=1928)"),
        (192, 4000, "Very Dense (N=4192)")
    ]
    
    h_ref = (4.0 * np.pi) / 24.0
    results = []
    
    for n_front, n_int, label in configurations:
        t_start = time.time()
        coords, vectors, h_avg = generate_nodal_mesh_pi(n_front, n_int, seed=42)
        search_tree = KDTree(coords)
        n_total = len(coords)
        
        res = evaluate_mesh_interpolation(coords, vectors, search_tree, test_points, h_avg, h_ref)
        res['n_front'] = n_front
        res['n_int'] = n_int
        res['n_total'] = n_total
        res['h_avg'] = h_avg
        res['label'] = label
        t_elapsed = time.time() - t_start
        
        results.append(res)
        
        print(f"Mesh {label:22s} | N={n_total:5d} | h={h_avg:.4f} m | Time: {t_elapsed:.2f} s")
        print(f"   L1: RMS E Err = {res['err_e_l1_rms']:.4e} | RMS Rot Err = {res['err_rot_l1_rms']:.4e}")
        print(f"   P1: RMS E Err = {res['err_e_p1_rms']:.4e} | RMS Rot Err = {res['err_rot_p1_rms']:.4e} | K_mean = {res['k_ef_p1_mean']:.1f}")
        print()
        
    return results, test_points


def generate_publication_figure(results):
    """Generate publication figure comparing L1 vs P1 convergence."""
    h_vals = np.array([r['h_avg'] for r in results])
    
    e_rms_l1 = np.array([r['err_e_l1_rms'] for r in results])
    rot_rms_l1 = np.array([r['err_rot_l1_rms'] for r in results])
    
    e_rms_p1 = np.array([r['err_e_p1_rms'] for r in results])
    rot_rms_p1 = np.array([r['err_rot_p1_rms'] for r in results])
    
    p_e_l1 = np.polyfit(np.log(h_vals), np.log(e_rms_l1), 1)[0]
    p_e_p1 = np.polyfit(np.log(h_vals), np.log(e_rms_p1), 1)[0]
    p_rot_l1 = np.polyfit(np.log(h_vals), np.log(rot_rms_l1), 1)[0]
    p_rot_p1 = np.polyfit(np.log(h_vals), np.log(rot_rms_p1), 1)[0]
    
    fig, (ax_e, ax_rot) = plt.subplots(1, 2, figsize=(11.8, 5.0))
    
    # (a) Electric Field
    ax_e.loglog(h_vals, e_rms_l1, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                markeredgewidth=1.8, label=f"$\\mathcal{{L}}^1$ (3 nodes): $\\mathcal{{O}}(h^{{{p_e_l1:.2f}}})$")
    ax_e.loglog(h_vals, e_rms_p1, 'o-', color=COLOR_NAVY, markerfacecolor=COLOR_NAVY,
                label=f"$\\mathcal{{P}}^1$ (6 nodes): $\\mathcal{{O}}(h^{{{p_e_p1:.2f}}})$")
                
    href = np.linspace(h_vals.min() * 0.85, h_vals.max() * 1.15, 60)
    c_h2 = e_rms_p1[2] / (h_vals[2]**2)
    c_h1 = e_rms_l1[2] / (h_vals[2]**1)
    ax_e.loglog(href, c_h2 * (href**2), 'k:', alpha=0.55, label=r"Theoretical $\mathcal{O}(h^2)$")
    ax_e.loglog(href, c_h1 * (href**1), 'gray', linestyle='-.', alpha=0.45, label=r"Theoretical $\mathcal{O}(h^1)$")
    
    ax_e.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{nom}}$ [m]")
    ax_e.set_ylabel(r"Electric Field RMS Error $\|\vec{E} - \vec{E}^h\|_{\mathrm{RMS}}$")
    ax_e.set_title(r"(a) Electric Field Convergence ($\vec{E}$)")
    ax_e.set_ylim(5.0e-4, 0.35)
    ax_e.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_e.legend(loc="lower right", framealpha=0.92, fontsize=9.0)
    
    # (b) Curl
    ax_rot.loglog(h_vals, rot_rms_l1, 's--', color=COLOR_CRIMSON, markerfacecolor='white',
                  markeredgewidth=1.8, label=r"$\mathcal{L}^1$ (3 nodes): $\mathcal{O}(1)$ (stagnation)")
    ax_rot.loglog(h_vals, rot_rms_p1, 'd-', color=COLOR_TEAL, markerfacecolor=COLOR_TEAL,
                  label=f"$\\mathcal{{P}}^1$ (6 nodes): $\\mathcal{{O}}(h^{{{p_rot_p1:.2f}}})$")
                  
    c_rot1 = rot_rms_p1[2] / (h_vals[2]**1)
    ax_rot.loglog(href, c_rot1 * (href**1), 'k:', alpha=0.55, label=r"Theoretical $\mathcal{O}(h^1)$")
    
    ax_rot.set_xlabel(r"Characteristic Nodal Spacing $h_{\mathrm{nom}}$ [m]")
    ax_rot.set_ylabel(r"Curl RMS Error $\|(\nabla \times \vec{E})_z - (\nabla \times \vec{E}^h)_z\|_{\mathrm{RMS}}$")
    ax_rot.set_title(r"(b) Curl Operator Convergence ($(\nabla \times \vec{E})_z$)")
    ax_rot.set_ylim(1.5e-2, 1.8)
    ax_rot.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_rot.legend(loc="lower right", framealpha=0.92, fontsize=9.0)
    
    fig.tight_layout()
    
    pdf_path = os.path.join(REPORTS_DIR, "figura_convergencia_interpolacao_L1_vs_P1.pdf")
    png_path = os.path.join(REPORTS_DIR, "figura_convergencia_interpolacao_L1_vs_P1.png")
    artigo_pdf_path = os.path.join(ROOT_DIR, "artigo", "figura_convergencia_interpolacao_L1_vs_P1.pdf")
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    fig.savefig(artigo_pdf_path)
    plt.close(fig)
    print(f"Figures saved to: {pdf_path}")
    
    gain_e = e_rms_l1[-1] / e_rms_p1[-1]
    gain_rot = rot_rms_l1[-1] / rot_rms_p1[-1]

    return {
        'p_e_l1': p_e_l1,
        'p_e_p1': p_e_p1,
        'p_rot_l1': p_rot_l1,
        'p_rot_p1': p_rot_p1,
        'gain_e': gain_e,
        'gain_rot': gain_rot,
        # Backward compatibility
        'p_E_L1': p_e_l1,
        'p_E_P1': p_e_p1,
        'p_rot_L1': p_rot_l1,
        'p_rot_P1': p_rot_p1,
        'ganho_E': gain_e,
        'ganho_rot': gain_rot
    }


def main():
    results, test_points = run_global_study()
    metrics = generate_publication_figure(results)
    print("Done generating L1 vs P1 convergence figure!")


# Backward compatibility and alternative aliases
campo_exato_TE11 = exact_field_te11
gerar_malha_nodal_pi = generate_nodal_mesh_pi
generate_pi_nodal_mesh = generate_nodal_mesh_pi
avaliar_interpolacao_malha = evaluate_mesh_interpolation
executar_estudo_global = run_global_study
gerar_figura_publicacao = generate_publication_figure


if __name__ == "__main__":
    main()
