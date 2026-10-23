# Vector Nodal Meshless Method in Two Dimensions (VNMM 2D)
### *A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Unit Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](tests/)
[![Paper Status](https://img.shields.io/badge/manuscript-EABE%20(Elsevier)-orange.svg)](paper/)
[![Code Availability](https://img.shields.io/badge/GitHub-vnmm2d--eabe-blue.svg)](https://github.com/rencmbr/vnmm2d-eabe)

---

## 1. Overview

This repository provides the complete scientific computing infrastructure, numerical libraries, and benchmark reproduction scripts for the **Two-Dimensional Vector Nodal Meshless Method (VNMM 2D)** applied to electromagnetic boundary-value and eigenvalue problems in $H(\mathbf{curl}; \Omega)$.

The codebase directly supports and accompanies the research paper submitted to ***Engineering Analysis with Boundary Elements*** (Elsevier):
> **"A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds"**  
> *Authors:* Renato C. Mesquita, Elson J. da Silva, Bárbara M. F. Gonçalves, and Luilly A. G. Ortiz.  
> *Manuscript sources & compilation:* [`paper/`](paper/) | *Public Repository:* [github.com/rencmbr/vnmm2d-eabe](https://github.com/rencmbr/vnmm2d-eabe)

In addition to serving as a fully reproducible artifact for the submitted paper, this repository functions as an **active research platform** for ongoing developments in computational electromagnetics, including **direct conforming hybrid Edge-FEM + VNMM 2D coupling** on structured, unstructured, and perturbed point clouds.

---

## 2. Repository Architecture

```text
teste-cod/
├── requirements.txt            # Scientific Python dependencies (numpy, scipy, matplotlib, pandas)
├── .gitignore                  # Git exclusion rules for Python caches, venvs, and LaTeX build logs
├── Readme.md                   # Main documentation, theoretical overview, and execution guide
│
├── src/                        # Core reusable Python package
│   ├── vnmm_assembler.py       # Vector matrix assembler with Gauss point-based support (EFG-style)
│   ├── fem_edge_2d.py          # Lowest-order Nédélec edge finite element solver (Whitney 1-forms)
│   ├── fem_vnmm_hybrid_2d.py   # Coupled hybrid Edge-FEM + VNMM 2D solver with direct trace continuity
│   ├── eigen_solver_cavity.py  # Generalized eigenvalue solver for resonant PEC cavities
│   ├── cavity_mesh.py          # Nodal cloud generators with tangential boundary directors
│   └── gauss_quadrature.py     # 1D and tensor-product 2D Gauss-Legendre quadrature schemes
│
├── scripts/                    # Executable research scripts, benchmarks, and figure generators (43 scripts)
│   ├── generate_*.py           # Publication-quality figure generation (Figures 1 to 9 of the paper)
│   ├── galerkin_*.py           # Variational patch test and Method of Manufactured Solutions (MMS)
│   ├── dimensionless_*.py      # Dimensionless placement matrix A_hat and scale-invariant stencil search
│   ├── comparative_*.py        # Systematic convergence benchmarks: VNMM 2D vs. Edge FEM
│   ├── hybrid_*.py             # Coupled hybrid FEM-VNMM solver across regular and irregular meshes
│   └── parametric_*.py         # Parametric tolerance sweeps and spatial density studies
│
├── paper/                      # LaTeX publication sources and submission package for Elsevier EABE
│   ├── stable-vnmm2d.tex       # Main manuscript source with continuous line numbering
│   ├── stable-vnmm2d.pdf       # Compiled submission PDF
│   ├── Cover_Letter_EABE.pdf   # 1-page cover letter to Editor-in-Chief Prof. Alexander H.-D. Cheng
│   ├── Highlights.txt          # Research highlights (≤ 85 characters per bullet point)
│   ├── Suggested_Reviewers.txt # 5 independent international reviewers with institutional emails
│   ├── references.bib          # BibTeX citation database
│   └── figura_*.pdf            # High-resolution vector figures included in the manuscript
│
├── reports/                    # In-depth technical reports, derivations, and historical notes (81 files)
│   ├── plano_implementacao_base_P1.md # Conceptual and algebraic blueprint of the complete P1 basis
│   ├── *.md                    # Detailed Markdown reports on interpolation, coupling, and quadrature
│   └── *.png                   # Supporting high-resolution convergence and spectral plots
│
├── tests/                      # Automated unit test suite (21 passing tests)
│   ├── test_eigen_cavity.py    # Tests for cavity mesh, PEC boundaries, and eigenvalue accuracy
│   ├── test_patch_test_galerkin.py # Unit tests for Galerkin patch test under affine vector modes
│   ├── test_galerkin_source_problem.py # Unit tests for MMS driven source problem
│   └── test_hybrid_fem_vnmm.py # Unit tests for direct hybrid Edge-FEM + VNMM coupling
│
└── meshes/                     # Benchmark point cloud and mesh datasets (.csv, .png)
    ├── dense_mesh.csv          # Standard dense test cloud
    ├── dense_mesh_1928.csv     # High-density 1,928-node benchmark cloud
    └── medium_mesh_416.csv     # Intermediate 416-node cloud
```

---

## 3. Installation & Getting Started

### 3.1 Prerequisites
- **Python**: Version 3.10 or higher.
- **TeX Live / Tectonic** *(optional)*: Only required if recompiling the manuscript in [`paper/`](paper/).

### 3.2 Environment Setup
Clone the repository and set up a dedicated virtual environment:

```bash
# Clone the repository
git clone https://github.com/rencmbr/vnmm2d-eabe.git
cd vnmm2d-eabe

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3.3 Running Unit Tests
Validate the entire numerical suite (21 unit tests covering mesh generation, eigenvalue solvers, patch test, MMS, and hybrid coupling):

```bash
python3 -m unittest discover tests -v
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

---

## 5. Scripts Catalog & Execution Guide

All execution scripts are located in [`scripts/`](scripts/) and can be run directly from the repository root.

### 5.1 Reproduction of Manuscript Figures & Tables (EABE Submission)

| Script in [`scripts/`](scripts/) | Output in Manuscript | Scientific Topic / Contribution |
| :--- | :--- | :--- |
| [`generate_dual_whitney_vnmm_equivalence_figure.py`](scripts/generate_dual_whitney_vnmm_equivalence_figure.py) | **Figure 1** (`figura_equivalencia_dual_whitney_vnmm.pdf`) | Geometric and numerical proof of exact dual equivalence between VNMM $\mathcal{L}^1$ and Whitney edge elements. |
| [`generate_matrix_a_singularities_figure.py`](scripts/generate_matrix_a_singularities_figure.py) | **Figure 2** (`figura_singularidades_matriz_A.pdf`) | Classification of the 4 pathological singularity configurations of matrix $\mathbf{A}$, geometric decoupling, and admissible support. |
| [`generate_conforming_mesh_vnmm_l1_figure.py`](scripts/generate_conforming_mesh_vnmm_l1_figure.py) | **Figure 3** (`figura_malha_conforme_vnmm_L1.pdf`) | Conforming simplicial triangulation topology used for the Whitney equivalence proof. |
| [`generate_efem_vs_vnmm_l1_figure.py`](scripts/generate_efem_vs_vnmm_l1_figure.py) | **Figure 4** (`figura_convergencia_efem_vs_vnmm_L1.pdf`) | Asymptotic convergence rates demonstrating numerical identity between Edge FEM and VNMM $\mathcal{L}^1$ in the conforming case. |
| [`generate_convergence_l1_vs_p1_pi_figure.py`](scripts/generate_convergence_l1_vs_p1_pi_figure.py) | **Figure 5** (`figura_convergencia_interpolacao_L1_vs_P1.pdf`) | Vector interpolation benchmark: $\mathcal{O}(1)$ curl stagnation in $\mathcal{L}^1$ vs. restoration of $\mathcal{O}(h)$ curl convergence in $\mathcal{P}^1$. |
| [`generate_manuscript_results_figures.py`](scripts/generate_manuscript_results_figures.py) | **Figures 6, 7, and 8** | Batch generation: stochastic robustness under disorder (Fig. 6), div-curl spectral purification (Fig. 7), and convergence vs. Edge FEM (Fig. 8). |
| [`generate_manuscript_random_cavity_figure.py`](scripts/generate_manuscript_random_cavity_figure.py) | **Figure 9** (`figura_convergencia_cavidade_aleatoria.pdf`) | Spectral cavity validation on irregular point clouds with arbitrary vector directors. |
| [`galerkin_patch_test.py`](scripts/galerkin_patch_test.py) | **Table 6** | Variational patch test across 7 affine vector modes: machine precision ($\le 7.4 \times 10^{-14}$) for complete basis $\mathcal{P}^1$. |
| [`galerkin_source_problem.py`](scripts/galerkin_source_problem.py) | **Table 7** | Method of Manufactured Solutions (MMS) with source term: verified $L^2$ and $H(\mathbf{curl})$ error convergence. |
| [`s_div_sensitivity_study.py`](scripts/s_div_sensitivity_study.py) | **Table 9** | Parametric sensitivity of cavity cutoff wavenumbers and spurious mode shift as a function of $s_{\text{div}}$. |
| [`base_case_integration_study.py`](scripts/base_case_integration_study.py) | **Table 10** | Quadrature sensitivity and background integration cell density trade-off ($3\times 3$ Gauss rule). |
| [`comparative_convergence_vnmm_vs_fem.py`](scripts/comparative_convergence_vnmm_vs_fem.py) | **Table 11** | 6-mesh systematic convergence comparison: pure VNMM 2D vs. Nédélec Edge FEM. |
| [`random_cavity_convergence_analysis.py`](scripts/random_cavity_convergence_analysis.py) | **Table 12** | Spectral accuracy and modal statistics on randomly perturbed/jittered clouds. |

#### Reproduction Quick Commands:
```bash
# 1. Run the Galerkin Patch Test (Table 6):
python3 scripts/galerkin_patch_test.py

# 2. Run the Method of Manufactured Solutions (MMS) source problem (Table 7):
python3 scripts/galerkin_source_problem.py

# 3. Generate key manuscript result figures (Figures 6, 7, and 8):
python3 scripts/generate_manuscript_results_figures.py

# 4. Generate systematic convergence comparison VNMM vs Edge FEM (Table 11):
python3 scripts/comparative_convergence_vnmm_vs_fem.py
```

---

### 5.2 Dimensionless Formulation & Stability Analysis

| Script in [`scripts/`](scripts/) | Functionality & Role |
| :--- | :--- |
| [`dimensionless_vnmm_6_p1.py`](scripts/dimensionless_vnmm_6_p1.py) | Module implementing the local dimensionless placement matrix $\widehat{\mathbf{A}}$ and scale-invariant stencil qualification. |
| [`dimensionless_matrix_sequence_test.py`](scripts/dimensionless_matrix_sequence_test.py) | Test verifying elimination of the quartic $\mathcal{O}(h^4)$ determinant decay under spatial refinement. |
| [`run_dimensionless_sweep.py`](scripts/run_dimensionless_sweep.py) | Automated parametric sweep across $\widehat{\text{Tol}} \in [0.05, 6.0]$ to identify the stable operating window. |
| [`generate_pi_floor_robustness_study.py`](scripts/generate_pi_floor_robustness_study.py) | Stability study evaluating conditioning and error behavior under $25\%$ directional and positional perturbations. |
| [`dimensionless_cavity_test.py`](scripts/dimensionless_cavity_test.py) | End-to-end cavity eigenvalue benchmark using pure dimensionless support selection. |

---

### 5.3 Comparative Studies & Alternative Formulations

| Script in [`scripts/`](scripts/) | Functionality & Role |
| :--- | :--- |
| [`compare_vnmm_vs_edge_fem.py`](scripts/compare_vnmm_vs_edge_fem.py) | Modewise direct comparison of physical eigenvalues computed by VNMM and Edge FEM in the baseline cavity. |
| [`cavity_convergence_analysis.py`](scripts/cavity_convergence_analysis.py) | Modal convergence analysis across uniform and semi-structured clouds. |
| [`l1_basis_study.py`](scripts/l1_basis_study.py) | Investigates why $\mathcal{L}^1$ cannot be regularized (divergence identically zero: $\mathbf{K}_{\text{div}} \equiv 0$). |
| [`unregularized_vnmm_study.py`](scripts/unregularized_vnmm_study.py) | Demonstrates gradient nullspace contamination when div-curl penalization is inactive ($s_{\text{div}} = 0.0$). |
| [`mimetic_curl_stokes_study.py`](scripts/mimetic_curl_stokes_study.py) | Investigates a mimetic discrete Stokes circulation weak form based on closed boundary contours. |

---

### 5.4 Advanced Research: Coupled Hybrid Edge-FEM + VNMM 2D Solver

| Script in [`scripts/`](scripts/) | Functionality & Role |
| :--- | :--- |
| [`test_fem_vnmm_coupling.py`](scripts/test_fem_vnmm_coupling.py) | Demonstrates the coupled hybrid Edge-FEM + VNMM 2D solver enforcing direct tangential continuity across $\Gamma_{\text{int}}$. |
| [`hybrid_fem_vnmm_convergence_comparison.py`](scripts/hybrid_fem_vnmm_convergence_comparison.py) | Compares accuracy, DoF efficiency, and convergence between hybrid coupling, pure VNMM, and pure Edge FEM. |
| [`hybrid_random_mesh_convergence_analysis.py`](scripts/hybrid_random_mesh_convergence_analysis.py) | Evaluates hybrid coupling under severe stochastic mesh deformation. |

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

If this codebase or methodology contributes to your research, please cite:

```bibtex
@article{mesquita2026stabilized,
  title={A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds},
  author={Mesquita, Renato C. and da Silva, Elson J. and Gon{\c{c}}alves, B{\'a}rbara M. F. and Ortiz, Luilly A. G.},
  journal={Engineering Analysis with Boundary Elements},
  year={2026},
  note={Under review. Public repository at \url{https://github.com/rencmbr/vnmm2d-eabe}}
}
```

---

## 9. Contact & Support

- **Prof. Renato C. Mesquita** (Corresponding Author): `renato@ufmg.br`  
  *Department of Electrical Engineering, Universidade Federal de Minas Gerais (UFMG)*  
  *Belo Horizonte, MG, Brazil*
