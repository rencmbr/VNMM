# Vector Nodal Meshless Method in Two Dimensions (VNMM 2D)

This directory contains the complete computational implementation, numerical solvers, execution scripts, point clouds, and automated unit test suite for the **Vector Nodal Meshless Method in Two Dimensions (VNMM 2D)** applied to $H(\mathbf{curl})$ vector electromagnetic problems ($TE_z$ polarization).

---

## 1. Mathematical & Algorithmic Foundations

The 2D codebase implements the stabilized formulation described in the scientific manuscript submitted to *Engineering Analysis with Boundary Elements* (Elsevier):

1. **Complete Linear Vector Polynomial Basis ($\mathcal{P}^1$):**
   Employs $M_{\mathrm{supp}} = 6$ non-degenerate support nodes per stencil, eliminating the $\mathcal{O}(1)$ curl stagnation observed in incomplete Whitney-like meshless bases ($\mathcal{L}^1$) and restoring asymptotic $\mathcal{O}(h)$ curl convergence.
2. **Dimensionless Placement Matrix ($\widehat{\mathbf{A}}$):**
   Normalizes stencil coordinates by local neighbor spacing ($h_{\mathrm{loc}}$), eliminating the artificial $\mathcal{O}(h^4)$ determinant decay under mesh refinement and enabling scale-invariant candidate selection with $\widehat{\mathrm{Tol}} = 0.80$.
3. **Individual Gauss-Point Support:**
   Evaluates support stencils independently at each quadrature point (`ponto_gauss`), avoiding the artificial directional singularities associated with fixed cell-center schemes.
4. **Ritz-Galerkin Div-Curl Regularization:**
   Applies penalty parameter $s_{\mathrm{div}} = 6.0$ to purge the discrete gradient nullspace ($\nabla \times (\nabla \phi) \neq \mathbf{0}$), shifting non-physical modes beyond $\lambda > 50$ and ensuring 100% spurious-free operational bandwidth.
5. **Conforming Hybrid Edge-FEM + VNMM Coupling:**
   Direct master-slave formulation linking Whitney edge elements to VNMM points across an interface $\Gamma_{\mathrm{int}}$ via exact scale mapping ($c_k = e_k / \Delta y$) without Lagrange multipliers.

---

## 2. Directory Structure

```text
2D/
├── src/                    # Core mathematical and numerical library
│   ├── cavity_mesh_2d.py       # Point cloud generation, PEC boundary tagging, and directional assignment
│   ├── vnmm_assembler_2d.py    # Numerical quadrature, local matrix inversion, and global system assembly
│   ├── eigen_solver_cavity_2d.py # Sparse generalized eigensolver (Arpack / eigs) and modal sorting
│   ├── gauss_quadrature_2d.py  # 2D tensor-product Gauss-Legendre quadrature engine
│   ├── fem_edge_2d.py          # Pure triangular Nédélec 1st-kind edge finite element solver
│   └── fem_vnmm_hybrid_2d.py   # Direct coupled Edge-FEM + VNMM 2D solver
│
├── scripts/                # Execution scripts, benchmarks, parameter studies, and figure generators
│   ├── generate_*_2d.py        # Standalone scripts to reproduce Figures 1 to 9 of the manuscript
│   ├── galerkin_patch_test_2d.py # Variational patch test verifying machine-precision consistency
│   ├── galerkin_source_problem_2d.py # Method of Manufactured Solutions (MMS) source problem
│   ├── dimensionless_*_2d.py   # Dimensionless matrix formulation, scaling, and threshold sweeps
│   └── hybrid_*_2d.py          # Benchmark studies for the coupled Edge-FEM + VNMM formulation
│
├── meshes/                 # Benchmark point clouds and node coordinate datasets
│   ├── mesh_2d.csv             # Baseline irregular point cloud
│   ├── medium_mesh_416_2d.csv  # Intermediate 416-node cloud
│   ├── dense_mesh_2d.csv       # Standard dense benchmark cloud
│   └── dense_mesh_1928_2d.csv  # High-density 1,928-node cloud
│
└── tests/                  # Automated unit test suite (21 passing tests)
    ├── test_eigen_cavity_2d.py        # Mesh validity, PEC boundaries, and eigenvalue accuracy
    ├── test_patch_test_galerkin_2d.py # Consistency tests under affine vector modes
    ├── test_galerkin_source_problem_2d.py # MMS driven source problem convergence
    └── test_hybrid_fem_vnmm_2d.py     # Coupled hybrid solver symmetry and accuracy
```

---

## 3. Quick Start & Execution

All scripts and tests can be executed from the root of the repository:

### Running the Test Suite:
```bash
python3 -m unittest discover 2D/tests -v
```

### Reproducing Key Manuscript Results:
```bash
# 1. Variational Patch Test (Table 6 in paper):
python3 2D/scripts/galerkin_patch_test_2d.py

# 2. Method of Manufactured Solutions (Table 7 in paper):
python3 2D/scripts/galerkin_source_problem_2d.py

# 3. Manuscript figures generation:
python3 2D/scripts/generate_manuscript_results_figures_2d.py
```
