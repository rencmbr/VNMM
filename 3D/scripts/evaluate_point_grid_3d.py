"""
3D Point Grid Evaluation for VNMM 3D.
Evaluates support node selection, placement matrix conditioning, and vector reconstruction over a regular 3D grid.
"""

import os
import sys
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_3D = os.path.dirname(CURRENT_DIR)
ROOT_DIR = os.path.dirname(DIR_3D)
for p in [DIR_3D, ROOT_DIR, CURRENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_3d_12_p1 import support_nodes_vnmm_3d_12_p1
from scripts.shape_functions_vnmm_3d_12_p1 import shape_functions_vnmm_3d_12_p1
from scripts.vector_field_resonant_3d import vector_field_te101


def evaluate_point_grid_3d(
    coords,
    vectors,
    search_tree,
    evaluation_points,
    tolerance=1e-6,
    neighborhood_size=24
):
    """
    Evaluates 3D VNMM shape functions and interpolation accuracy on a grid of points.
    """
    E_nodes, curl_E_nodes, _ = vector_field_te101(coords)
    nodal_projections = np.sum(E_nodes * vectors, axis=1)

    total_points = len(evaluation_points)
    successes = 0
    failures = 0
    errors_E = []
    errors_curl = []

    for idx, P in enumerate(evaluation_points):
        supp_nodes, det_val, A_mat, eff_k = support_nodes_vnmm_3d_12_p1(
            P=P,
            nodes_coords=coords,
            nodes_vectors=vectors,
            search_tree=search_tree,
            K=neighborhood_size,
            Tol_det=tolerance
        )

        if supp_nodes is None or len(supp_nodes) < 12:
            failures += 1
            continue

        successes += 1
        Phi_g, curl_Phi, div_Phi, beta = shape_functions_vnmm_3d_12_p1(
            P=P,
            nodes_coords=coords,
            nodes_vectors=vectors,
            selected_nodes=supp_nodes,
            matrix_a=A_mat
        )

        # Reconstructed field and curl at point P
        local_dofs = nodal_projections[supp_nodes]
        E_approx = Phi_g @ local_dofs
        curl_approx = curl_Phi @ local_dofs

        E_ex, curl_ex, _ = vector_field_te101(np.array([P]))
        E_ex = E_ex[0]
        curl_ex = curl_ex[0]

        err_E = np.linalg.norm(E_approx - E_ex)
        err_curl = np.linalg.norm(curl_approx - curl_ex)
        errors_E.append(err_E)
        errors_curl.append(err_curl)

    return {
        "total_points": total_points,
        "successes": successes,
        "failures": failures,
        "mean_error_E": float(np.mean(errors_E)) if errors_E else None,
        "mean_error_curl": float(np.mean(errors_curl)) if errors_curl else None
    }

