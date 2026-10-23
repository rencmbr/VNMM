# Vector Nodal Meshless Method (VNMM)
### *A Stabilized Computational Electromagnetics Research Platform in 2D and 3D*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Computational Scope](https://img.shields.io/badge/scope-2D%20%7C%203D-blue.svg)](#)
[![Unit Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](2D/tests/)
[![Paper EABE](https://img.shields.io/badge/paper-EABE%20(Elsevier)-orange.svg)](paper/VNMM2D-EABE/)
[![Paper Hybrid](https://img.shields.io/badge/paper-Hybrid%20FEM--VNMM-blueviolet.svg)](paper/HybridVNMM-EFEM/)
[![Repository](https://img.shields.io/badge/GitHub-VNMM-blue.svg)](https://github.com/rencmbr/VNMM)

---

## 1. Overview

This repository hosts the central scientific computing platform, numerical libraries, and research datasets for the **Vector Nodal Meshless Method (VNMM)**, developing advanced mesh-free and hybrid discretization techniques for vector electromagnetic field problems in $H(\mathbf{curl}; \Omega)$.

Rather than serving solely as a companion to a single publication, this repository functions as an **active, multi-dimensional research platform** dedicated to the continuous theoretical and computational advancement of the VNMM in both **two dimensions (2D)** and **three dimensions (3D)**:

- **Two-Dimensional Formulations ([`2D/`](2D/)):**
  - **Stabilized VNMM 2D:** Complete linear vector polynomial basis $\mathcal{P}^1$ ($M_{\text{supp}} = 6$), scale-invariant dimensionless placement criterion $\widehat{\mathbf{A}}$ ($\widehat{\mathrm{Tol}} = 0.80$), Gauss-point-based support selection, and Ritz-Galerkin div-curl regularization ($s_{\text{div}} = 6.0$).
  - **Conforming Hybrid Edge-FEM + VNMM 2D:** Direct master-slave domain decomposition coupling lowest-order Whitney edge finite elements to interior VNMM clouds via exact trace continuity without auxiliary Lagrange multipliers.
- **Three-Dimensional Formulations ([`3D/`](3D/)):**
  - **VNMM 3D Platform & Roadmap:** Development space for 3D vector meshless solvers, extending the stabilized framework to 3D high-frequency waveguiding and radiation problems with 12-dimensional polynomial spaces ($\mathcal{P}^1$ in 3D), 3D dimensionless stencil selection, and tetrahedral Whitney edge coupling.
- **Scientific Publications ([`paper/`](paper/)):**
  - **EABE Elsevier Submission ([`paper/VNMM2D-EABE/`](paper/VNMM2D-EABE/)):** *"A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds"*, under review in *Engineering Analysis with Boundary Elements*.
  - **Hybrid Coupling Manuscript ([`paper/HybridVNMM-EFEM/`](paper/HybridVNMM-EFEM/)):** *"Conforming Hybrid Edge Finite Element and Vector Nodal Meshless Formulation for 2D Electromagnetics"*.

---

## 2. Repository Architecture

```text
VNMM/
├── requirements.txt            # Scientific Python dependencies (numpy, scipy, matplotlib, pandas)
├── .gitignore                  # Git exclusion rules for Python caches, venvs, and LaTeX build logs
├── Readme.md                   # Main documentation, theoretical overview, and execution guide
│
├── 2D/                         # Two-Dimensional Vector Nodal Meshless Method (VNMM 2D)
│   ├── README.md               # Scope, foundations, and structure of the 2D formulation
│   ├── src/                    # Core 2D numerical libraries
│   │   ├── vnmm_assembler_2d.py       # Vector matrix assembler with Gauss point-based support (EFG-style)
│   │   ├── fem_edge_2d.py             # Lowest-order Nédélec edge finite element solver (Whitney 1-forms)
│   │   ├── fem_vnmm_hybrid_2d.py      # Coupled hybrid Edge-FEM + VNMM 2D solver with direct trace continuity
│   │   ├── eigen_solver_cavity_2d.py  # Generalized eigenvalue solver for resonant PEC cavities
│   │   ├── cavity_mesh_2d.py          # Nodal cloud generators with tangential boundary directors
│   │   └── gauss_quadrature_2d.py     # 1D and tensor-product 2D Gauss-Legendre quadrature schemes
│   │
│   ├── scripts/                # Executable research scripts, benchmarks, and figure generators (43 scripts)
│   │   ├── generate_*_2d.py           # Publication-quality figure generation (Figures 1 to 9 of the paper)
│   │   ├── galerkin_*_2d.py           # Variational patch test and Method of Manufactured Solutions (MMS)
│   │   ├── dimensionless_*_2d.py      # Dimensionless placement matrix A_hat and scale-invariant stencil search
│   │   ├── comparative_*_2d.py        # Systematic convergence benchmarks: VNMM 2D vs. Edge FEM
│   │   ├── hybrid_*_2d.py             # Coupled hybrid FEM-VNMM solver across regular and irregular meshes
│   │   └── parametric_*_2d.py         # Parametric tolerance sweeps and spatial density studies
│   │
│   ├── tests/                  # Automated unit test suite (21 passing tests)
│   │   ├── test_eigen_cavity_2d.py            # Tests for cavity mesh, PEC boundaries, and eigenvalue accuracy
│   │   ├── test_patch_test_galerkin_2d.py     # Unit tests for Galerkin patch test under affine vector modes
│   │   ├── test_galerkin_source_problem_2d.py # Unit tests for MMS driven source problem
│   │   └── test_hybrid_fem_vnmm_2d.py         # Unit tests for direct hybrid Edge-FEM + VNMM coupling
│   │
│   └── meshes/                 # Benchmark point cloud and mesh datasets (.csv)
│       ├── dense_mesh_2d.csv          # Standard dense test cloud
│       ├── dense_mesh_1928_2d.csv     # High-density 1,928-node benchmark cloud
│       ├── medium_mesh_416_2d.csv     # Intermediate 416-node cloud
│       └── mesh_2d.csv                # Baseline unstructured point cloud
│
├── 3D/                         # Three-Dimensional Vector Nodal Meshless Method (VNMM 3D - Future Extension)
│   ├── README.md               # Scope and architectural roadmap for the 3D formulation
│   ├── src/                    # Future 3D core modules (tetrahedral/cloud support, 3D Whitney FEM)
│   ├── scripts/                # Future 3D execution and validation scripts
│   ├── meshes/                 # Future 3D point clouds and tetrahedral benchmark meshes
│   └── tests/                  # Future automated test suite for 3D solvers
│
├── paper/                      # LaTeX publication sources and submissions
│   ├── README.md               # Overview of scientific manuscripts, compilation instructions, and policies
│   ├── VNMM2D-EABE/            # Elsevier EABE submission package (VNMM 2D Paper)
│   │   ├── stable-vnmm2d.tex   # Main manuscript source with continuous line numbering
│   │   ├── Cover_Letter_EABE.tex # Cover letter to Editor-in-Chief Prof. Alexander H.-D. Cheng
│   │   ├── Highlights.txt      # Research highlights (≤ 85 characters per bullet point)
│   │   ├── Suggested_Reviewers.txt # 5 independent international reviewers with institutional emails
│   │   └── references.bib      # BibTeX citation database
│   │
│   └── HybridVNMM-EFEM/        # Manuscript sources for hybrid Edge-FEM + VNMM coupling
│       ├── main-acoplamento-vnmm-efem.tex # Main manuscript source for hybrid formulation
│       └── references-acoplamento-vnmm-efem.bib # BibTeX citation database
│
└── reports/                    # In-depth technical reports, derivations, and historical notes (81 files)
    ├── plano_implementacao_base_P1.md # Conceptual and algebraic blueprint of the complete P1 basis
    └── *.md                    # Detailed Markdown reports on interpolation, coupling, and quadrature
```

---

## 3. Installation & Getting Started

### 3.1 Prerequisites
- **Python**: Version 3.10 or higher.
- **TeX Live / Tectonic** *(optional)*: Only required if recompiling the manuscript in [`paper/VNMM2D-EABE/`](paper/VNMM2D-EABE/).

### 3.2 Environment Setup
Clone the repository and set up a dedicated virtual environment:

```bash
# Clone the repository
git clone https://github.com/rencmbr/VNMM.git
cd VNMM

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3.3 Running Unit Tests
Validate the entire numerical suite (21 unit tests covering mesh generation, eigenvalue solvers, patch test, MMS, and hybrid coupling):

```bash
python3 -m unittest discover 2D/tests -v
# or from inside the 2D directory:
cd 2D && python3 -m unittest discover tests -v
```

---

## 4. Theoretical Formulations & Key Contributions

The VNMM 2D approximates $H(\mathbf{curl})$ vector fields by distributing $M$ nodes throughout a domain $\Omega$ and its boundary $\partial\Omega$. Each node $\mathbf{x}_i$ is assigned a single unit vector director $\vec{t}_i = [t_{ix}, t_{iy}]^T$. On PEC boundaries and material interfaces, $\vec{t}_i$ is aligned tangentially ($\vec{t}_i \parallel \partial\Omega$), enabling exact Dirichlet boundary condition enforcement by directly eliminating boundary degrees of freedom.

Vector shape functions $\vec{N}_i(\mathbf{x})$ satisfy the **vector Kronecker delta projection property**:
$$
\vec{N}_i(\mathbf{x}_k) \cdot \vec{t}_k = \delta_{ik} \implies \mathbf{A} \boldsymbol{\beta}_i = \mathbf{L}_i \quad \therefore \quad \vec{E}^h(\mathbf{x}) = \sum_{i=1}^{M_{\text{supp}}} \vec{N}_i(\mathbf{x}) e_i
$$

### 4.1 Resolution of Derivative Modal Aliasing: Incomplete $\mathcal{L}^1$ vs. Complete $\mathcal{P}^1$ Basis
- **The Incomplete $\mathcal{L}^1$ Bottleneck (3 Support Nodes):** The solenoidal 3-term basis $\mathcal{L}^1 = \langle [1,0]^T, [0,1]^T, [y,-x]^T \rangle$ artificially constrains longitudinal derivatives to zero ($\partial E_x/\partial x = \partial E_y/\partial y = 0$). On unconstrained point clouds—where closed-contour boundary circulation cancellation is absent—these unrepresented modes leak into the rotational degree of freedom, triggering an **$\mathcal{O}(1)$ curl error plateau** ($\approx 15\%$ error persisting under spatial refinement).
- **The Complete $\mathcal{P}^1$ Solution (6 Support Nodes):** Replacing $\mathcal{L}^1$ with the complete linear polynomial vector space $\mathcal{P}_1 \times \mathcal{P}_1$:
  $$
  \mathcal{P}^1 = \left\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \begin{bmatrix} x \\ 0 \end{bmatrix}, \begin{bmatrix} y \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ x \end{bmatrix}, \begin{bmatrix} 0 \\ y \end{bmatrix} \right\rangle
  $$
  captures all four components of the field Jacobian tensor $\mathbf{J} = \nabla\vec{E}$, eliminating derivative aliasing and delivering **observed first-order $\mathcal{O}(h)$ curl convergence** and **second-order $\mathcal{O}(h^2)$ electric field interpolation**.

### 4.2 Scale-Invariant Support Selection via the Dimensionless Matrix $\widehat{\mathbf{A}}$
In physical coordinates, the local moment matrix determinant decays as the fourth power of nodal spacing ($\det(\mathbf{A}) \sim \mathcal{O}(h^4)$). Under spatial refinement ($h \to 0$), fixed absolute tolerances trigger unphysical support dilation.  
By normalizing local coordinates by the local length scale $h_{\text{local}}$:
$$
\hat{x} = \frac{x - x_P}{h_{\text{local}}}, \quad \hat{y} = \frac{y - y_P}{h_{\text{local}}} \implies |\det(\widehat{\mathbf{A}})| = \frac{|\det(\mathbf{A})|}{h_{\text{local}}^4}
$$
the dimensionless placement matrix $\widehat{\mathbf{A}}$ becomes $\mathcal{O}(1)$. Coupled with $k$-d tree spatial queries, a transferable threshold $\widehat{\text{Tol}} = 0.80$ reliably rejects ill-conditioned and directional singularities (e.g., parallel directors, concurrent lines of action) while maintaining compact neighborhoods ($K_{\text{avg}} \approx 10\text{--}11$ nodes) even under $25\%$ stochastic disorder.

### 4.3 Spectral Purification: Non-Conformity and Variational Div-Curl Regularization
Although local VNMM shape functions exhibit exact constructive equivalence to lowest-order Whitney 1-forms on conforming simplices (Local Dual Equivalence Theorem), unconstrained clouds with dynamically shifting stencils produce a globally non-conforming approximation ($V_h \not\subset H(\mathbf{curl}; \Omega)$).  
This breaks the discrete de Rham complex exactness ($\operatorname{Im}(\nabla) \not\subset \ker(\nabla\times)$), allowing irrotational gradient nullspace modes to leak into the physical spectrum.  
We resolve this via a Ritz-Galerkin weak formulation regularized by a div-curl penalty:
$$
\int_{\Omega} (\nabla \times \vec{W})_z (\nabla \times \vec{E})_z \, d\Omega + s_{\text{div}} \int_{\Omega} (\nabla \cdot \vec{W}) (\nabla \cdot \vec{E}) \, d\Omega = \lambda \int_{\Omega} \vec{W} \cdot \vec{E} \, d\Omega
$$
Setting $s_{\text{div}} = 6.0$ selectively stiffens gradient modes ($\nabla \cdot \vec{E} \neq 0$), shifting them outside the operational frequency window ($\lambda_{\text{spur}} > 10$) while preserving solenoidal physical cavity modes ($\nabla \cdot \vec{E} \equiv 0$).

### 4.4 Conforming Hybrid Edge-FEM + VNMM 2D Coupling
For problems with geometrically complex boundaries or subdomains with established triangular edge finite elements, the repository provides direct master-slave hybrid coupling ([`2D/scripts/hybrid_fem_vnmm_convergence_comparison_2d.py`](2D/scripts/hybrid_fem_vnmm_convergence_comparison_2d.py)). By positioning VNMM interface nodes at the midpoints of adjacent Whitney edges and setting the exact dimensional scaling $c_k = e_k / \Delta y$, interface continuity is enforced directly without auxiliary Lagrange multipliers, preserving positive-definiteness and eliminating zero-mass saddle-point indefiniteness.

### 4.5 Architectural Roadmap for Three-Dimensional Extension (VNMM 3D)
Extending the stabilized vector nodal meshless methodology to three dimensions ([`3D/`](3D/)) addresses full 3D vector electromagnetic boundary-value and eigenvalue problems:
- **3D Complete Polynomial Space ($\mathcal{P}^1$ in 3D):** Requires $M_{\text{supp}} = 12$ support nodes per stencil to span the 12-dimensional vector polynomial space $\mathcal{P}_1 \times \mathcal{P}_1 \times \mathcal{P}_1$, completely capturing the 9-component 3D displacement gradient tensor $\mathbf{J} = \nabla\vec{E}$ and eliminating 3D curl stagnation.
- **3D Scale-Invariant Placement Matrix ($\widehat{\mathbf{A}}_{12 \times 12}$):** Extends dimensionless coordinate normalization to 3D spherical neighborhoods, establishing reliable conditioning thresholds against 3D coplanar, concurrent, and degenerated director stencils.
- **3D Variational Div-Curl Regularization:** Incorporates 3D divergence penalization ($\int_{\Omega} s_{\text{div}} (\nabla \cdot \vec{W})(\nabla \cdot \vec{E})\, d\Omega$) to stiffen the 3D discrete gradient nullspace across unconstrained volumetric point clouds.
- **Coupling with 3D Tetrahedral Whitney Elements:** Establishes master-slave trace coupling with tetrahedral Whitney 1-forms on boundary-fitted conformal interfaces.

---

## 5. Scripts Catalog & Execution Guide

All execution scripts for the 2D formulation are located in [`2D/scripts/`](2D/scripts/) and can be run directly from the repository root.

### 5.1 Reproduction of Manuscript Figures & Tables (EABE Submission)

| Script in [`2D/scripts/`](2D/scripts/) | Output in Manuscript | Scientific Topic / Contribution |
| :--- | :--- | :--- |
| [`generate_dual_whitney_vnmm_equivalence_figure_2d.py`](2D/scripts/generate_dual_whitney_vnmm_equivalence_figure_2d.py) | **Figure 1** | Geometric and numerical proof of exact dual equivalence between VNMM $\mathcal{L}^1$ and Whitney edge elements. |
| [`generate_matrix_a_singularities_figure_2d.py`](2D/scripts/generate_matrix_a_singularities_figure_2d.py) | **Figure 2** | Classification of the 4 pathological singularity configurations of matrix $\mathbf{A}$, geometric decoupling, and admissible support. |
| [`generate_conforming_mesh_vnmm_l1_figure_2d.py`](2D/scripts/generate_conforming_mesh_vnmm_l1_figure_2d.py) | **Figure 3** | Conforming simplicial triangulation topology used for the Whitney equivalence proof. |
| [`generate_efem_vs_vnmm_l1_figure_2d.py`](2D/scripts/generate_efem_vs_vnmm_l1_figure_2d.py) | **Figure 4** | Asymptotic convergence rates demonstrating numerical identity between Edge FEM and VNMM $\mathcal{L}^1$ in the conforming case. |
| [`generate_convergence_l1_vs_p1_pi_figure_2d.py`](2D/scripts/generate_convergence_l1_vs_p1_pi_figure_2d.py) | **Figure 5** | Vector interpolation benchmark: $\mathcal{O}(1)$ curl stagnation in $\mathcal{L}^1$ vs. restoration of $\mathcal{O}(h)$ curl convergence in $\mathcal{P}^1$. |
| [`generate_manuscript_results_figures_2d.py`](2D/scripts/generate_manuscript_results_figures_2d.py) | **Figures 6, 7, and 8** | Batch generation: stochastic robustness under disorder (Fig. 6), div-curl spectral purification (Fig. 7), and convergence vs. Edge FEM (Fig. 8). |
| [`generate_manuscript_random_cavity_figure_2d.py`](2D/scripts/generate_manuscript_random_cavity_figure_2d.py) | **Figure 9** | Spectral cavity validation on irregular point clouds with arbitrary vector directors. |
| [`galerkin_patch_test_2d.py`](2D/scripts/galerkin_patch_test_2d.py) | **Table 6** | Variational patch test across 7 affine vector modes: machine precision ($\le 7.4 \times 10^{-14}$) for complete basis $\mathcal{P}^1$. |
| [`galerkin_source_problem_2d.py`](2D/scripts/galerkin_source_problem_2d.py) | **Table 7** | Method of Manufactured Solutions (MMS) with source term: verified $L^2$ and $H(\mathbf{curl})$ error convergence. |
| [`s_div_sensitivity_study_2d.py`](2D/scripts/s_div_sensitivity_study_2d.py) | **Table 9** | Parametric sensitivity of cavity cutoff wavenumbers and spurious mode shift as a function of $s_{\text{div}}$. |
| [`base_case_integration_study_2d.py`](2D/scripts/base_case_integration_study_2d.py) | **Table 10** | Quadrature sensitivity and background integration cell density trade-off ($3\times 3$ Gauss rule). |
| [`comparative_convergence_vnmm_vs_fem_2d.py`](2D/scripts/comparative_convergence_vnmm_vs_fem_2d.py) | **Table 11** | 6-mesh systematic convergence comparison: pure VNMM 2D vs. Nédélec Edge FEM. |
| [`random_cavity_convergence_analysis_2d.py`](2D/scripts/random_cavity_convergence_analysis_2d.py) | **Table 12** | Spectral accuracy and modal statistics on randomly perturbed/jittered clouds. |

#### Reproduction Quick Commands:
```bash
# 1. Run the Galerkin Patch Test (Table 6):
python3 2D/scripts/galerkin_patch_test_2d.py

# 2. Run the Method of Manufactured Solutions (MMS) source problem (Table 7):
python3 2D/scripts/galerkin_source_problem_2d.py

# 3. Generate key manuscript result figures (Figures 6, 7, and 8):
python3 2D/scripts/generate_manuscript_results_figures_2d.py

# 4. Generate systematic convergence comparison VNMM vs Edge FEM (Table 11):
python3 2D/scripts/comparative_convergence_vnmm_vs_fem_2d.py
```

---

### 5.2 Dimensionless Formulation & Stability Analysis

| Script in [`2D/scripts/`](2D/scripts/) | Functionality & Role |
| :--- | :--- |
| [`dimensionless_vnmm_6_p1_2d.py`](2D/scripts/dimensionless_vnmm_6_p1_2d.py) | Module implementing the local dimensionless placement matrix $\widehat{\mathbf{A}}$ and scale-invariant stencil qualification. |
| [`dimensionless_matrix_sequence_test_2d.py`](2D/scripts/dimensionless_matrix_sequence_test_2d.py) | Test verifying elimination of the quartic $\mathcal{O}(h^4)$ determinant decay under spatial refinement. |
| [`run_dimensionless_sweep_2d.py`](2D/scripts/run_dimensionless_sweep_2d.py) | Automated parametric sweep across $\widehat{\text{Tol}} \in [0.05, 6.0]$ to identify the stable operating window. |
| [`generate_pi_floor_robustness_study_2d.py`](2D/scripts/generate_pi_floor_robustness_study_2d.py) | Stability study evaluating conditioning and error behavior under $25\%$ directional and positional perturbations. |
| [`dimensionless_cavity_test_2d.py`](2D/scripts/dimensionless_cavity_test_2d.py) | End-to-end cavity eigenvalue benchmark using pure dimensionless support selection. |

---

### 5.3 Comparative Studies & Alternative Formulations

| Script in [`2D/scripts/`](2D/scripts/) | Functionality & Role |
| :--- | :--- |
| [`compare_vnmm_vs_edge_fem_2d.py`](2D/scripts/compare_vnmm_vs_edge_fem_2d.py) | Modewise direct comparison of physical eigenvalues computed by VNMM and Edge FEM in the baseline cavity. |
| [`cavity_convergence_analysis_2d.py`](2D/scripts/cavity_convergence_analysis_2d.py) | Modal convergence analysis across uniform and semi-structured clouds. |
| [`l1_basis_study_2d.py`](2D/scripts/l1_basis_study_2d.py) | Investigates why $\mathcal{L}^1$ cannot be regularized (divergence identically zero: $\mathbf{K}_{\text{div}} \equiv 0$). |
| [`unregularized_vnmm_study_2d.py`](2D/scripts/unregularized_vnmm_study_2d.py) | Demonstrates gradient nullspace contamination when div-curl penalization is inactive ($s_{\text{div}} = 0.0$). |
| [`mimetic_curl_stokes_study_2d.py`](2D/scripts/mimetic_curl_stokes_study_2d.py) | Investigates a mimetic discrete Stokes circulation weak form based on closed boundary contours. |

---

### 5.4 Advanced Research: Coupled Hybrid Edge-FEM + VNMM 2D Solver

| Script in [`2D/scripts/`](2D/scripts/) | Functionality & Role |
| :--- | :--- |
| [`test_fem_vnmm_coupling_2d.py`](2D/scripts/test_fem_vnmm_coupling_2d.py) | Demonstrates the coupled hybrid Edge-FEM + VNMM 2D solver enforcing direct tangential continuity across $\Gamma_{\text{int}}$. |
| [`hybrid_fem_vnmm_convergence_comparison_2d.py`](2D/scripts/hybrid_fem_vnmm_convergence_comparison_2d.py) | Compares accuracy, DoF efficiency, and convergence between hybrid coupling, pure VNMM, and pure Edge FEM. |
| [`hybrid_random_mesh_convergence_analysis_2d.py`](2D/scripts/hybrid_random_mesh_convergence_analysis_2d.py) | Evaluates hybrid coupling under severe stochastic mesh deformation. |

---

## 6. Key Benchmark Results

### 6.1 Baseline PEC Resonant Cavity Benchmark ($TE_z$)
For the standard square cavity $\Omega = [0, \pi]^2$ with $N_x=21, N_y=21$ (361 active internal DoFs), comparison against analytical eigenvalues ($\lambda = n^2 + m^2$) confirms excellent spectral fidelity:

| Mode ($TE_{nm}$) | $\lambda_{\text{analytical}}$ | $k_{c, \text{analytical}}$ | $\lambda_{\text{VNMM } \mathcal{P}^1}$ | $k_{c, \text{VNMM}}$ | Error $k_c$ (%) | Diagnosis |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$TE_{10}$** | 1.00 | 1.000 | 0.9785 | 0.989 | **1.08%** | Fundamental Physical Mode |
| **$TE_{01}$** | 1.00 | 1.000 | 0.9904 | 0.995 | **0.48%** | Fundamental Physical Mode |
| **$TE_{11}$** | 2.00 | 1.414 | 1.9723 | 1.404 | **0.70%** | Physical Resonance |
| **$TE_{20}$** | 4.00 | 2.000 | 3.9634 | 1.991 | **0.46%** | Physical Resonance |
| **$TE_{02}$** | 4.00 | 2.000 | 4.0152 | 2.004 | **0.19%** | Physical Resonance |
| **$TE_{21}$** | 5.00 | 2.236 | 4.9650 | 2.228 | **0.35%** | Physical Resonance |
| **$TE_{12}$** | 5.00 | 2.236 | 5.0341 | 2.244 | **0.34%** | Physical Resonance |
| **$TE_{22}$** | 8.00 | 2.828 | 7.9734 | 2.824 | **0.17%** | Physical Resonance |
| **$TE_{30}$** | 9.00 | 3.000 | 9.3872 | 3.064 | **2.13%** | Physical Resonance |
| **$TE_{03}$** | 9.00 | 3.000 | 9.4751 | 3.078 | **2.61%** | Physical Resonance |

- **Global mean $k_c$ error:** **$0.85\%$** (Maximum modal error: **$2.61\%$**)
- **Assembly and solution time:** **$\approx 0.13\text{ s}$**

---

### 6.2 Systematic Convergence Comparison

| Grid Refinement | $h$ (m) | DoFs VNMM | Mean $k_c$ Err VNMM | DoFs Hybrid | Mean $k_c$ Err Hybrid | DoFs Edge FEM | Mean $k_c$ Err FEM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **N1 (Coarse)** | 0.3927 | 49 | **32.42%** | 113 | **3.74%** | 40 | **3.19%** |
| **N2** | 0.2618 | 121 | **10.82%** | 265 | **2.40%** | 133 | **1.05%** |
| **N3** | 0.1963 | 225 | **5.01%** | 481 | **0.71%** | 225 | **0.64%** |
| **N4 (Baseline)** | 0.1571 | **361** | **2.53%** | **761** | **0.53%** | **341** | **0.44%** |
| **N5** | 0.1309 | 529 | **1.31%** | 1105 | **0.22%** | 560 | **0.27%** |
| **N6** | 0.1122 | 729 | **0.79%** | 1513 | **0.36%** | 736 | **0.21%** |
| **N7 (Fine)** | 0.0982 | **961** | **0.50%** | **1985** | **0.26%** | **936** | **0.16%** |

---

## 7. Technical Reports & Historical Research Notes

For in-depth mathematical derivations, failure mode analyses, and comprehensive parameter studies, refer to the technical reports in [`reports/`](reports/):

- [Implementation Blueprint for the Complete $\mathcal{P}^1$ Basis](reports/plano_implementacao_base_P1.md)
- [Local Dual Equivalence Theorem: Whitney 1-Forms and VNMM](reports/dual_equivalence_theorem_whitney_vnmm.md)
- [Theoretical Error Analysis: Comparing VNMM and Edge FEM](reports/analise_origem_erros_vnmm_vs_fem.md)
- [Coupled Hybrid FEM-VNMM Formulation Report](reports/relatorio_acoplamento_hibrido_fem_vnmm.md)
- [Theoretical Strategies for Hybrid CEM Coupling](reports/estrategias_acoplamento_fem_vnmm.md)
- [Convergence Analysis of Hybrid Solver on Random Clouds](reports/relatorio_acoplamento_hibrido_malhas_aleatorias.md)
- [Numerical Integration and Gauss Quadrature in the Baseline Cavity](reports/relatorio_estudo_integracao_caso_base.md)
- [Under-Integration Diagnosis and Solutions for Dense Clouds](reports/diagnostico_caso_N25_vnmm.md)
- [Investigation of the Unregularized Formulation ($s_{\text{div}}=0$)](reports/relatorio_sem_regularizacao_divergente.md)
- [Spectral Study of the Incomplete $\mathcal{L}^1$ Basis](reports/relatorio_estudo_base_L1.md)
- [Mimetic Stokes Discrete Curl Formulation Study](reports/relatorio_rotacional_mimetico_stokes.md)

---

## 8. Citation

If this codebase, the stabilized 2D formulation, or the VNMM research platform contributes to your research, please cite:

```bibtex
@article{mesquita2026stabilized,
  title={A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds},
  author={Mesquita, Renato C. and da Silva, Elson J. and Gon{\c{c}}alves, B{\'a}rbara M. F. and Ortiz, Luilly A. G.},
  journal={Engineering Analysis with Boundary Elements},
  year={2026},
  note={Under review. Public repository at \url{https://github.com/rencmbr/VNMM}}
}
```

---

## 9. Contact & Support

- **Prof. Renato C. Mesquita** (Corresponding Author): `renato@ufmg.br`  
  *Department of Electrical Engineering, Universidade Federal de Minas Gerais (UFMG)*  
  *Belo Horizonte, MG, Brazil*
