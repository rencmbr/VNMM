import numpy as np

from codigo.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from codigo.shape_functions_vnmm_2d_6_p1 import shape_functions_vnmm_2d_6_p1
from codigo.vector_field_te11 import vector_field_te11


def evaluate_point_grid_6_p1(
    coords, 
    vectors, 
    search_tree, 
    evaluation_points, 
    tolerance=1e-3, 
    neighborhood_size=12, 
    adaptive=True,
    step_K=4,
    K_max=None,
    # Compatibility aliases
    arvore=None,
    pontos_avaliacao=None,
    tolerancia=None,
    tamanho_vizinhanca=None,
    adaptativo=None,
    passo_K=None,
    k_neighbors=None,
    k=None,
    K=None,
    tol_det=None
):
    from scipy.spatial import KDTree
    if isinstance(evaluation_points, KDTree) and not isinstance(search_tree, KDTree):
        coords, vectors, search_tree, evaluation_points = vectors, search_tree, evaluation_points, coords

    if arvore is not None:
        search_tree = arvore
    if pontos_avaliacao is not None:
        evaluation_points = pontos_avaliacao
    if tolerancia is not None:
        tolerance = tolerancia
    elif tol_det is not None:
        tolerance = tol_det
    if tamanho_vizinhanca is not None:
        neighborhood_size = tamanho_vizinhanca
    elif k_neighbors is not None:
        neighborhood_size = k_neighbors
    elif k is not None:
        neighborhood_size = k
    elif K is not None:
        neighborhood_size = K
    if adaptativo is not None:
        adaptive = adaptativo
    if passo_K is not None:
        step_K = passo_K

    E_nodes, curl_E_nodes = vector_field_te11(coords)
    nodal_projections = np.sum(E_nodes * vectors, axis=1)
    
    total_points = len(evaluation_points)
    successes = 0
    failures = 0
    
    dets = []
    vector_errors = []
    curl_errors = []
    effective_ks = []
    results = []
    
    for pt_idx, P in enumerate(evaluation_points):
        selected_nodes, det_val, matrix_a, eff_k = support_nodes_vnmm_2d_6_p1(
            P=P,
            nodes_coords=coords,
            nodes_vectors=vectors,
            search_tree=search_tree,
            K=neighborhood_size,
            Tol_det=tolerance,
            adaptive=adaptive,
            step_K=step_K,
            K_max=K_max
        )
        
        exact_E, exact_curl_E = vector_field_te11(P)
        
        if selected_nodes is not None:
            Phi, curl_Phi, beta = shape_functions_vnmm_2d_6_p1(
                P=P,
                nodes_coords=coords,
                nodes_vectors=vectors,
                selected_nodes=selected_nodes,
                matrix_a=matrix_a
            )
            
            e_s = nodal_projections[selected_nodes]
            interp_E = Phi @ e_s
            interp_curl_E = float(np.dot(curl_Phi, e_s))
            
            vec_error = float(np.linalg.norm(interp_E - exact_E))
            curl_error = float(abs(interp_curl_E - exact_curl_E))
            
            successes += 1
            dets.append(det_val)
            vector_errors.append(vec_error)
            curl_errors.append(curl_error)
            effective_ks.append(eff_k)
            
            results.append({
                'id': pt_idx,
                'P': P,
                'nodes': selected_nodes,
                'det_A': det_val,
                'effective_k': eff_k,
                'e_s': e_s,
                'interpolated_E': interp_E,
                'interpolated_curl_E': interp_curl_E,
                'exact_E': exact_E,
                'exact_curl_E': exact_curl_E,
                'vector_error': vec_error,
                'curl_error': curl_error,
                'Phi': Phi,
                'curl_Phi': curl_Phi,
                'beta': beta,
                # Backward compatibility aliases
                'nos': selected_nodes,
                'k_efetivo': eff_k,
                'E_interpolado': interp_E,
                'rot_E_interpolado': interp_curl_E,
                'E_exato': exact_E,
                'rot_E_exato': exact_curl_E,
                'erro_vetorial': vec_error,
                'erro_rotacional': curl_error,
                'rot_Phi': curl_Phi
            })
        else:
            failures += 1
            results.append({
                'id': pt_idx,
                'P': P,
                'nodes': None,
                'det_A': 0.0,
                'effective_k': 0,
                'e_s': None,
                'interpolated_E': None,
                'interpolated_curl_E': None,
                'exact_E': exact_E,
                'exact_curl_E': exact_curl_E,
                'vector_error': None,
                'curl_error': None,
                'Phi': None,
                'curl_Phi': None,
                'beta': None,
                # Backward compatibility aliases
                'nos': None,
                'k_efetivo': 0,
                'E_interpolado': None,
                'rot_E_interpolado': None,
                'E_exato': exact_E,
                'rot_E_exato': exact_curl_E,
                'erro_vetorial': None,
                'erro_rotacional': None,
                'rot_Phi': None
            })
            
    vec_err_arr = np.array(vector_errors) if vector_errors else np.array([0.0])
    curl_err_arr = np.array(curl_errors) if curl_errors else np.array([0.0])
    dets_arr = np.array(dets) if dets else np.array([0.0])
    ks_arr = np.array(effective_ks) if effective_ks else np.array([0])
    
    return {
        'total_points': total_points,
        'successes': successes,
        'failures': failures,
        'success_rate': (successes / total_points) * 100.0 if total_points > 0 else 0.0,
        'det_min': float(np.min(dets_arr)),
        'det_max': float(np.max(dets_arr)),
        'det_mean': float(np.mean(dets_arr)),
        'k_mean': float(np.mean(ks_arr)),
        'k_max': int(np.max(ks_arr)),
        'vector_err_min': float(np.min(vec_err_arr)),
        'vector_err_max': float(np.max(vec_err_arr)),
        'vector_err_mean': float(np.mean(vec_err_arr)),
        'vector_err_rms': float(np.sqrt(np.mean(vec_err_arr**2))),
        'curl_err_min': float(np.min(curl_err_arr)),
        'curl_err_max': float(np.max(curl_err_arr)),
        'curl_err_mean': float(np.mean(curl_err_arr)),
        'curl_err_rms': float(np.sqrt(np.mean(curl_err_arr**2))),
        'results': results,
        'nodal_projections': nodal_projections,
        # Backward compatibility aliases
        'total_pontos': total_points,
        'sucessos': successes,
        'falhas': failures,
        'taxa_sucesso': (successes / total_points) * 100.0 if total_points > 0 else 0.0,
        'det_medio': float(np.mean(dets_arr)),
        'k_medio': float(np.mean(ks_arr)),
        'erro_vet_min': float(np.min(vec_err_arr)),
        'erro_vet_max': float(np.max(vec_err_arr)),
        'erro_vet_medio': float(np.mean(vec_err_arr)),
        'erro_vet_rms': float(np.sqrt(np.mean(vec_err_arr**2))),
        'erro_rot_min': float(np.min(curl_err_arr)),
        'erro_rot_max': float(np.max(curl_err_arr)),
        'erro_rot_medio': float(np.mean(curl_err_arr)),
        'erro_rot_rms': float(np.sqrt(np.mean(curl_err_arr**2))),
        'resultados': results,
        'projecoes_nos': nodal_projections
    }


# Backward compatibility alias
avaliar_grade_pontos_6_P1 = evaluate_point_grid_6_p1
