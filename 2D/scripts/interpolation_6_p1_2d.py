import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
MESHES_DIR = os.path.join(DIR_2D, "meshes")
for p in [DIR_2D, ROOT_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.load_mesh_2d import load_mesh
from scripts.generate_dense_mesh_2d import generate_dense_mesh
from scripts.plot_mesh_2d import plot_mesh
from scripts.build_search_tree_2d import build_search_tree
from scripts.evaluate_point_grid_6_p1_2d import evaluate_point_grid_6_p1
from scripts.print_point_details_2d import print_point_details


def main():
    print("=================================================================")
    print("      DENSE MESH VALIDATION: VNMM 2D FORMULATION (COMPLETE P1 BASIS)")
    print("=================================================================")
    
    mesh_path = os.path.join(MESHES_DIR, "dense_mesh_1928_2d.csv")
    
    if os.path.exists(mesh_path):
        coords, vectors = load_mesh(mesh_path)
    else:
        print("Generating dense mesh (128 boundary + 1800 interior = 1928 nodes)...")
        coords, vectors = generate_dense_mesh(
            filename=mesh_path,
            num_boundary_nodes=128,
            num_interior_nodes=1800,
            boundary_limit=10.0,
            seed=42
        )
        
    image_file = os.path.splitext(mesh_path)[0] + ".png"
    if not os.path.exists(image_file):
        plot_mesh(coords, vectors, output_path=image_file, show=False)
        
    tree = build_search_tree(coords)
    total_nodes = len(coords)
    
    # Characteristic spacing h
    h_char = (2.0 * 10.0) / (128 // 4)  # h = 20 / 32 = 0.625
    
    # Quartic adaptive determinant tolerance scaling Tol_det(h) ~ O(h^4)
    tol_ref = 1.0
    h_ref = 2.0
    tol_det = tol_ref * (h_char / h_ref)**4
    neighborhood_size = 12
    
    # Regular evaluation grid (15 x 15 = 225 interior points)
    nx, ny = 15, 15
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_vals = np.linspace(x_min + dx / 2.0, x_max - dx / 2.0, nx)
    y_vals = np.linspace(y_min + dy / 2.0, y_max - dy / 2.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    eval_points = np.column_stack([X.ravel(), Y.ravel()])
    total_eval_points = len(eval_points)
    
    print(f"\n--- Test Parameters ---")
    print(f"Evaluated mesh            : {mesh_path}")
    print(f"Total mesh nodes (N)      : {total_nodes}")
    print(f"Characteristic spacing    : h = {h_char:.4f}")
    print(f"Evaluation grid           : {nx} x {ny} = {total_eval_points} interior points")
    print(f"Tolerance Tol_det(h)      : {tol_det:.6e} (calibrated by Tol_ref * (h/h_ref)^4)")
    print(f"Initial neighborhood K    : {neighborhood_size}\n")
    
    # Evaluate grid with complete P1 basis
    res = evaluate_point_grid_6_p1(
        coords=coords,
        vectors=vectors,
        search_tree=tree,
        evaluation_points=eval_points,
        tolerance=tol_det,
        neighborhood_size=neighborhood_size,
        adaptive=True,
        step_K=4
    )
    
    print(f"=================================================================")
    print(f"               INTERPOLATION RESULTS (COMPLETE P1 BASIS)")
    print(f"=================================================================")
    print(f"Total points evaluated  : {res['total_points']}")
    print(f"Successful evaluations  : {res['successes']} ({res['success_rate']:.1f}%)")
    print(f"Failures                : {res['failures']}")
    
    if res['successes'] > 0:
        print(f"\n--- Determinant |det(A)| and Neighborhood K Statistics ---")
        print(f"|det(A)| Min    : {res['det_min']:.6e}")
        print(f"|det(A)| Max    : {res['det_max']:.6e}")
        print(f"|det(A)| Mean   : {res['det_mean']:.6e}")
        print(f"K neighbors Mean: {res['k_mean']:.2f}")
        print(f"K neighbors Max : {res['k_max']}")
        
        print(f"\n--- Field Interpolation Error ||E^h - E_exact|| Statistics ---")
        print(f"Min Error  : {res['vector_err_min']:.6e}")
        print(f"Max Error  : {res['vector_err_max']:.6e}")
        print(f"Mean Error : {res['vector_err_mean']:.6e}")
        print(f"RMS Error  : {res['vector_err_rms']:.6e}")
        
        print(f"\n--- Curl Error |curl(E^h) - curl(E)_exact| Statistics ---")
        print(f"Min Error  : {res['curl_err_min']:.6e}")
        print(f"Max Error  : {res['curl_err_max']:.6e}")
        print(f"Mean Error : {res['curl_err_mean']:.6e}")
        print(f"RMS Error  : {res['curl_err_rms']:.6e}")
        
        valid_results = [r for r in res['results'] if r['nodes'] is not None]
        pt_max_vec_err = max(valid_results, key=lambda r: r['vector_error'])
        pt_max_curl_err = max(valid_results, key=lambda r: r['curl_error'])
        
        print_point_details(
            pt_max_vec_err, 
            f"POINT WITH MAXIMUM VECTOR FIELD INTERPOLATION ERROR (Error = {pt_max_vec_err['vector_error']:.6e})",
            coords=coords,
            vectors=vectors
        )
        
        print_point_details(
            pt_max_curl_err, 
            f"POINT WITH MAXIMUM CURL APPROXIMATION ERROR (Error = {pt_max_curl_err['curl_error']:.6e})",
            coords=coords,
            vectors=vectors
        )
        
    print("=================================================================")
    print("Validation successfully completed!")
    print("=================================================================")
    return res


if __name__ == "__main__":
    main()
