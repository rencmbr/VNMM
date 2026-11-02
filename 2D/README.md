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
6. **Truly Cell-Free VNMM-MLPG (Meshless Local Petrov--Galerkin):**
   Decouples spatial integration from background Cartesian grids by assigning autonomous circular subdomains $\Omega_s^I$ ($r_s^I = \alpha_s h_I$) to each node. Employs closed-form analytical integration over disks, eliminating numerical quadrature and intra-cell stencil switching. Resolves operator asymmetry via a novel **Symmetric Subdomain Galerkin Assembly**, guaranteeing strictly symmetric positive-semidefinite global operators. Solves the localized rank-2 deficiency of linear vector fields on circular disks through a **Dimensionless Orthogonal Energy Stabilization** ($\alpha_{\mathrm{stab}} = c_{\mathrm{stab}}(h/L)^2$) and superlinear divergence penalty scaling ($s_{\mathrm{div}} \propto h^{-1.5}$), eliminating mode softening and restoring near-quadratic convergence ($p = 2.15$, fine-mesh error down to $1.29\%$). Standardizes subdomain sizing with universal parameter $\alpha_s = 0.80$, bounded by geometric coverage and support containment limits.

---

## 2. Directory Structure

```text
2D/
├── src/                    # Core mathematical and numerical library
│   ├── cavity_mesh_2d.py       # Point cloud generation, PEC boundary tagging, and directional assignment
│   ├── vnmm_assembler_2d.py    # Numerical quadrature, local matrix inversion, and global system assembly
│   ├── mlpg_assembler_2d.py    # Truly cell-free MLPG assembler: closed-form disk integration, orthogonal stabilization, and CSR assembly
│   ├── mlpg_subdomains.py      # Subdomain geometry, active area/truncation calculations, and support search
│   ├── eigen_solver_cavity_2d.py # Sparse generalized eigensolver (Arpack / eigs) and modal sorting
│   ├── gauss_quadrature_2d.py  # 2D tensor-product Gauss-Legendre quadrature engine
│   ├── fem_edge_2d.py          # Pure triangular Nédélec 1st-kind edge finite element solver
│   └── fem_vnmm_hybrid_2d.py   # Direct coupled Edge-FEM + VNMM 2D solver
│
├── scripts/                # Execution scripts, benchmarks, parameter studies, and figure generators
│   ├── generate_*_2d.py        # Standalone scripts to reproduce Figures 1 to 9 of the stabilized paper
│   ├── galerkin_patch_test_2d.py # Variational patch test verifying machine-precision consistency
│   ├── galerkin_source_problem_2d.py # Method of Manufactured Solutions (MMS) source problem
│   ├── dimensionless_*_2d.py   # Dimensionless matrix formulation, scaling, and threshold sweeps
│   ├── hybrid_*_2d.py          # Benchmark studies for the coupled Edge-FEM + VNMM formulation
│   ├── mlpg_patch_test_2d.py   # MLPG Galerkin patch test across 7 affine vector modes (machine precision ~10^-15)
│   ├── mlpg_source_problem_2d.py # MLPG asymptotic convergence (MMS) under Dirichlet and mixed Dirichlet-Neumann
│   ├── mlpg_cavity_eigenvalues_2d.py # MLPG resonant cavity eigensolver, spectral purification, and benchmark vs. Gauss
│   └── parametric_analysis_alpha_s_mlpg_2d.py # Parametric sweep of subdomain factor alpha_s in [0.30, 2.00]
│
├── meshes/                 # Benchmark point clouds and node coordinate datasets
│   ├── mesh_2d.csv             # Baseline irregular point cloud
│   ├── medium_mesh_416_2d.csv  # Intermediate 416-node cloud
│   ├── dense_mesh_2d.csv       # Standard dense benchmark cloud
│   └── dense_mesh_1928_2d.csv  # High-density 1,928-node cloud
│
└── tests/                  # Automated unit test suite (41 passing tests across 7 modules)
    ├── test_mlpg_assembler_2d.py          # Unit tests for MLPG closed-form integration, stabilization, and assembly
    ├── test_mlpg_cavity_eigenvalues_2d.py # Unit tests for MLPG cavity eigensolver, s_div purification, and speedup
    ├── test_mlpg_patch_test_2d.py         # Unit tests for MLPG patch test across 7 modes on regular and jitter clouds
    ├── test_mlpg_source_problem_2d.py     # Unit tests for MLPG MMS monotonic convergence (Dirichlet & Mixed)
    ├── test_mlpg_subdomains_2d.py         # Unit tests for subdomain construction, coverage, and truncated geometry
    ├── test_eigen_cavity_2d.py            # Mesh validity, PEC boundaries, and eigenvalue accuracy
    ├── test_patch_test_galerkin_2d.py     # Consistency tests under affine vector modes
    ├── test_galerkin_source_problem_2d.py # MMS driven source problem convergence
    └── test_hybrid_fem_vnmm_2d.py         # Coupled hybrid solver symmetry and accuracy
```

---

## 3. Quick Start & Execution

All scripts and tests can be executed from the root of the repository:

### Running the Test Suite:
```bash
python3 -m unittest discover 2D/tests -v
```

### Reproducing Stabilized VNMM-Gauss Results:
```bash
# 1. Variational Patch Test (Table 6 in paper):
python3 2D/scripts/galerkin_patch_test_2d.py

# 2. Method of Manufactured Solutions (Table 7 in paper):
python3 2D/scripts/galerkin_source_problem_2d.py

# 3. Manuscript figures generation:
python3 2D/scripts/generate_manuscript_results_figures_2d.py
```

### Running Truly Cell-Free VNMM-MLPG Benchmarks:
```bash
# 1. MLPG Galerkin Patch Test across 7 affine modes:
python3 2D/scripts/mlpg_patch_test_2d.py

# 2. MLPG MMS Asymptotic Convergence (Dirichlet & Mixed Dirichlet-Neumann):
python3 2D/scripts/mlpg_source_problem_2d.py

# 3. MLPG Resonant Cavity Eigenvalues & Computational Benchmark:
python3 2D/scripts/mlpg_cavity_eigenvalues_2d.py

# 4. Parametric Sensitivity Study of alpha_s in [0.30, 2.00]:
python3 2D/scripts/parametric_analysis_alpha_s_mlpg_2d.py
```
