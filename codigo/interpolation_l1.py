import os
import numpy as np

from codigo.load_mesh import load_mesh
from codigo.plot_mesh import plot_mesh
from codigo.build_search_tree import build_search_tree
from codigo.evaluate_point_grid import evaluate_point_grid
from codigo.print_point_details import print_point_details

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)


def main():
    filename = os.path.join(ROOT_DIR, "malhas", "malha_densa.csv")
    image_file = os.path.splitext(filename)[0] + ".png"
    
    # 1. Load data from disk
    coords, vectors = load_mesh(filename)
    
    # 2. Generate and save mesh visualization
    plot_mesh(coords, vectors, output_path=image_file, show=False)
    
    # 3. Construct spatial search tree
    tree = build_search_tree(coords)
    
    # 4. Define regular grid of strictly interior points in (-10, 10) x (-10, 10)
    nx = 10
    ny = 10
    x_min, x_max = -10.0, 10.0
    y_min, y_max = -10.0, 10.0
    
    tolerance = 1.0
    neighborhood_size = 8
    
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    x_vals = np.linspace(x_min + dx / 2.0, x_max - dx / 2.0, nx)
    y_vals = np.linspace(y_min + dy / 2.0, y_max - dy / 2.0, ny)
    X, Y = np.meshgrid(x_vals, y_vals)
    eval_points = np.column_stack([X.ravel(), Y.ravel()])
    total_points = len(eval_points)
    
    print(f"--- VNMM 2D Evaluation and Interpolation on Regular Grid (Interior Points) ---")
    print(f"Mesh file: {filename}")
    print(f"Total mesh nodes: {len(coords)}")
    print(f"Domain: [{x_min}, {x_max}] x [{y_min}, {y_max}]")
    print(f"Grid spacing: dx = {dx:.2f}, dy = {dy:.2f}")
    print(f"Grid density: {nx} x {ny} = {total_points} evaluation points")
    print(f"Neighborhood size K: {neighborhood_size}, Tolerance |det(A)|: {tolerance}\n")
    
    # 5. Evaluate point grid
    res = evaluate_point_grid(
        coords=coords,
        vectors=vectors,
        search_tree=tree,
        evaluation_points=eval_points,
        tolerance=tolerance,
        neighborhood_size=neighborhood_size,
        adaptive=True
    )
    
    # 6. Display statistics and summary
    print(f"--- Nodal Vector Projections ---")
    print(f"Sample nodal projections (first 5 nodes): {np.round(res['nodal_projections'][:5], 4)}\n")
    
    print(f"--- Execution and Interpolation Summary ---")
    print(f"Total points evaluated  : {res['total_points']}")
    print(f"Successful evaluations  : {res['successes']} ({res['success_rate']:.1f}%)")
    print(f"Failures                : {res['failures']}")
    
    if res['successes'] > 0:
        print(f"\n--- Determinant |det(A)| and Neighborhood K Statistics ---")
        print(f"|det(A)| Min  : {res['det_min']:.6f}")
        print(f"|det(A)| Max  : {res['det_max']:.6f}")
        print(f"|det(A)| Mean : {res['det_mean']:.6f}")
        print(f"K neighbors Mean: {res['k_mean']:.2f}")
        print(f"K neighbors Max : {res['k_max']}")
        
        print(f"\n--- Interpolation Error ||E^h - E_exact|| Statistics ---")
        print(f"Min Error  : {res['vector_err_min']:.6e}")
        print(f"Max Error  : {res['vector_err_max']:.6e}")
        print(f"Mean Error : {res['vector_err_mean']:.6e}")
        print(f"RMS Error  : {res['vector_err_rms']:.6e}")
        
        print(f"\n--- Curl Error |curl(E^h) - curl(E)_exact| Statistics ---")
        print(f"Min Error  : {res['curl_err_min']:.6e}")
        print(f"Max Error  : {res['curl_err_max']:.6e}")
        print(f"Mean Error : {res['curl_err_mean']:.6e}")
        print(f"RMS Error  : {res['curl_err_rms']:.6e}")
        
        # 7. Identify max error points
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


if __name__ == "__main__":
    main()
