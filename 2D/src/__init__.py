"""
VNMM 2D Package - 2D Electromagnetic Cavity Eigenvalue and Variational Solver.
"""

from .cavity_mesh_2d import generate_cavity_mesh, gerar_malha_cavidade
from .gauss_quadrature_2d import (
    get_gauss_points_weights_1d,
    generate_quadrature_cells,
    obter_pontos_pesos_gauss_1d,
    gerar_celulas_quadratura
)
from .vnmm_assembler_2d import (
    assemble_vnmm_matrices_2d,
    montar_matrizes_vnmm_2d
)
from .eigen_solver_cavity_2d import (
    apply_pec_boundary_conditions,
    solve_generalized_eigenvalue_problem,
    solve_cavity_eigenvalues,
    ANALYTICAL_REFERENCE_MODES,
    MODOS_ANALITICOS_REF,
    aplicar_condicao_pec,
    aplicar_condicoes_contorno_pec,
    resolver_problema_autovalores,
    resolver_problema_autovalor_generalizado,
    resolver_autovalores_cavidade
)
from .fem_edge_2d import (
    generate_triangular_cavity_mesh,
    assemble_edge_fem_matrices_2d,
    solve_edge_fem_cavity_eigenvalues
)
from .fem_vnmm_hybrid_2d import (
    generate_hybrid_cavity_mesh,
    assemble_hybrid_fem_vnmm_matrices,
    solve_hybrid_fem_vnmm_eigenvalues
)

__all__ = [
    "generate_cavity_mesh",
    "gerar_malha_cavidade",
    "get_gauss_points_weights_1d",
    "generate_quadrature_cells",
    "obter_pontos_pesos_gauss_1d",
    "gerar_celulas_quadratura",
    "assemble_vnmm_matrices_2d",
    "montar_matrizes_vnmm_2d",
    "apply_pec_boundary_conditions",
    "solve_generalized_eigenvalue_problem",
    "solve_cavity_eigenvalues",
    "ANALYTICAL_REFERENCE_MODES",
    "MODOS_ANALITICOS_REF",
    "aplicar_condicao_pec",
    "aplicar_condicoes_contorno_pec",
    "resolver_problema_autovalores",
    "resolver_problema_autovalor_generalizado",
    "resolver_autovalores_cavidade",
    "generate_triangular_cavity_mesh",
    "assemble_edge_fem_matrices_2d",
    "solve_edge_fem_cavity_eigenvalues",
    "generate_hybrid_cavity_mesh",
    "assemble_hybrid_fem_vnmm_matrices",
    "solve_hybrid_fem_vnmm_eigenvalues"
]
