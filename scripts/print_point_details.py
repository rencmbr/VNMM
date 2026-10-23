import numpy as np


def print_point_details(r, title, coords=None, vectors=None, titulo=None):
    """
    Displays VNMM 2D interpolation and evaluation details for a point in the terminal.
    
    Parameters:
    - r: Dictionary containing evaluated point data.
    - title: Section title string.
    - coords: (Optional) Array of global nodal coordinates.
    - vectors: (Optional) Array of global director vectors.
    """
    if titulo is not None:
        title = titulo

    p_str = f"[{r['P'][0]:.2f}, {r['P'][1]:.2f}]"
    print(f"\n=======================================================")
    print(f"{title}")
    print(f"=======================================================")
    print(f"Point ID                     : {r.get('id')}")
    print(f"Coordinates P                : {p_str}")
    supp_nodes = r.get('nodes') or r.get('nos')
    print(f"Support nodes (indices)      : {supp_nodes}")
    if coords is not None and vectors is not None and supp_nodes is not None:
        print(f"Support node coordinates:")
        for idx_node in supp_nodes:
            print(f"  Node {idx_node:>3d}: Coords={coords[idx_node]}, Vector={vectors[idx_node]}")
    print(f"Determinant |det(A)|         : {r.get('det_A', 0.0):.6f}")
    if r.get('effective_k') is not None or r.get('k_efetivo') is not None:
        eff_k = r.get('effective_k') or r.get('k_efetivo')
        print(f"Effective K neighbors used   : {eff_k}")
    if r.get('e_s') is not None:
        print(f"Nodal projections e_s        : {np.round(r['e_s'], 4)}")
    if r.get('Phi') is not None:
        Phi = np.asarray(r['Phi'])
        print(f"Matrix Phi(P) ({Phi.shape[0]}x{Phi.shape[1]}):\n{np.round(Phi, 4)}")
    
    e_exact = r.get('exact_E') if r.get('exact_E') is not None else r.get('E_exato')
    if e_exact is not None:
        print(f"Exact vector E(P)            : [{e_exact[0]:.6f}, {e_exact[1]:.6f}]^T")
        
    e_interp = r.get('interpolated_E') if r.get('interpolated_E') is not None else r.get('E_interpolado')
    if e_interp is not None:
        print(f"Interpolated vector E^h(P)   : [{e_interp[0]:.6f}, {e_interp[1]:.6f}]^T")
        
    err_vec = r.get('vector_error') if r.get('vector_error') is not None else r.get('erro_vetorial')
    if err_vec is not None:
        print(f"Field error ||E^h - E||      : {err_vec:.6e}")
        
    rot_exact = r.get('exact_curl_E') if r.get('exact_curl_E') is not None else r.get('rot_E_exato')
    if rot_exact is not None:
        print(f"Exact curl curl(E)           : {rot_exact:.6f} z_hat")
        
    rot_interp = r.get('interpolated_curl_E') if r.get('interpolated_curl_E') is not None else r.get('rot_E_interpolado')
    if rot_interp is not None:
        print(f"Approx curl curl(E^h)        : {rot_interp:.6f} z_hat")
        
    err_curl = r.get('curl_error') if r.get('curl_error') is not None else r.get('erro_rotacional')
    if err_curl is not None:
        print(f"Curl error                   : {err_curl:.6e}")


# Backward compatibility alias
imprimir_detalhes_ponto = print_point_details
