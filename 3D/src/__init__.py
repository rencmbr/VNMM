"""
VNMM 3D Package - Three-Dimensional Vector Nodal Meshless Method.
"""

from .cavity_mesh_3d import (
    generate_cavity_mesh_3d,
    gerar_malha_cavidade_3d,
    generate_cavity_mesh,
    gerar_malha_cavidade
)
from .gauss_quadrature_3d import (
    get_gauss_points_weights_1d,
    generate_quadrature_cells_3d,
    obter_pontos_pesos_gauss_1d,
    gerar_celulas_quadratura_3d
)
from .vnmm_assembler_3d import (
    assemble_vnmm_matrices_3d,
    montar_matrizes_vnmm_3d,
    assemble_vnmm_matrices,
    montar_matrizes_vnmm
)
from .eigen_solver_cavity_3d import (
    apply_pec_boundary_conditions_3d,
    solve_generalized_eigenvalue_problem_3d,
    solve_cavity_eigenvalues_3d,
    ANALYTICAL_REFERENCE_MODES_3D,
    ANALYTICAL_REFERENCE_MODES,
    MODOS_ANALITICOS_REF,
    apply_pec_boundary_conditions,
    solve_generalized_eigenvalue_problem,
    solve_cavity_eigenvalues,
    resolver_autovalores_cavidade_3d
)

__all__ = [
    "generate_cavity_mesh_3d",
    "gerar_malha_cavidade_3d",
    "generate_cavity_mesh",
    "gerar_malha_cavidade",
    "get_gauss_points_weights_1d",
    "generate_quadrature_cells_3d",
    "obter_pontos_pesos_gauss_1d",
    "gerar_celulas_quadratura_3d",
    "assemble_vnmm_matrices_3d",
    "montar_matrizes_vnmm_3d",
    "assemble_vnmm_matrices",
    "montar_matrizes_vnmm",
    "apply_pec_boundary_conditions_3d",
    "solve_generalized_eigenvalue_problem_3d",
    "solve_cavity_eigenvalues_3d",
    "ANALYTICAL_REFERENCE_MODES_3D",
    "ANALYTICAL_REFERENCE_MODES",
    "MODOS_ANALITICOS_REF",
    "apply_pec_boundary_conditions",
    "solve_generalized_eigenvalue_problem",
    "solve_cavity_eigenvalues",
    "resolver_autovalores_cavidade_3d"
]
