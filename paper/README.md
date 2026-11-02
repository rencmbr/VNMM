# Publications & Manuscripts (`paper/`)

This directory contains the LaTeX source documents, citation bibliographies, cover letters, and submission artifacts for scientific journal publications resulting from the Vector Nodal Meshless Method (VNMM) research project.

---

## 1. Directory Structure

```text
paper/
├── README.md                   # This document
│
├── VNMM2D-EABE/                # Submission package for Engineering Analysis with Boundary Elements (Elsevier)
│   ├── stable-vnmm2d.tex       # Main manuscript source with continuous line numbering
│   ├── Cover_Letter_EABE.tex   # Formal cover letter to Editor-in-Chief Prof. Alexander H.-D. Cheng
│   ├── Highlights.txt          # Mandatory Elsevier highlights (≤ 85 characters per bullet)
│   ├── Suggested_Reviewers.txt # 5 independent international reviewers with institutional emails
│   ├── references.bib          # Primary BibTeX citation database
│   ├── elsarticle.cls          # Elsevier LaTeX document class
│   └── elsarticle-num.bst      # Elsevier numerical citation style
│
├── mlpg-vnmm/                  # Truly Cell-Free VNMM-MLPG 2D Manuscript (EABE Elsevier)
│   ├── main-vnmm-mlpg.tex      # Main manuscript source for the truly cell-free MLPG formulation
│   ├── generate_mlpg_figures.py # Dedicated script generating all 6 publication figures (PNG)
│   ├── references.bib          # BibTeX citation database
│   ├── elsarticle.cls          # Elsevier LaTeX document class
│   ├── fig_mlpg_*.png          # 6 high-resolution publication figures
│   └── main-vnmm-mlpg.pdf      # Compiled manuscript
│
└── HybridVNMM-EFEM/            # Manuscript on conforming Hybrid Edge-FEM + VNMM coupling
    ├── main-acoplamento-vnmm-efem.tex # Article source detailing domain decomposition and interface coupling
    └── references-acoplamento-vnmm-efem.bib # Specialized bibliography on hybrid numerical methods
```

---

## 2. Manuscript Summaries

### 2.1 Pure VNMM 2D Paper ([`VNMM2D-EABE/`](VNMM2D-EABE/))
- **Title:** *A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds*
- **Journal Target:** *Engineering Analysis with Boundary Elements* (Elsevier)
- **Key Focus:** Establishes the stabilized 2D meshless formulation using the complete polynomial space $\mathcal{P}^1$, individual Gauss-point support selection, the scale-invariant dimensionless placement criterion $\widehat{\mathrm{Tol}} = 0.80$, and Ritz-Galerkin div-curl regularization.

### 2.2 Truly Cell-Free VNMM-MLPG Paper ([`mlpg-vnmm/`](mlpg-vnmm/))
- **Title:** *A Truly Cell-Free Vector Nodal Meshless Method via Meshless Local Petrov--Galerkin (VNMM-MLPG) for 2D Electromagnetic Problems*
- **Journal Target:** *Engineering Analysis with Boundary Elements* (Elsevier)
- **Key Focus:** Develops the first truly cell-free VNMM formulation by completely eliminating background integration cells. Constructs directional shape functions via Vector Point Interpolation (PIM) with exact Kronecker-delta projection, employs closed-form analytical integration over autonomous circular subdomains $\Omega_s^I$, introduces a **Symmetric Subdomain Galerkin Assembly** ensuring strictly symmetric positive-semidefinite operators, resolves localized rank-2 stiffness deficiency via a **Dimensionless Orthogonal Energy Stabilization** ($\alpha_{\mathrm{stab}} \propto h^2$) with superlinear penalty scaling ($s_{\mathrm{div}} \propto h^{-1.5}$) restoring quadratic convergence ($p = 2.15$, $1.29\%$ fine-grid error), establishes a universal subdomain radius scaling standard $\alpha_s = 0.80$ bounded by geometric domain coverage and support containment limits, and outperforms cell-based VNMM-Gauss across all regular resolutions while delivering $2.0\times\text{--}3.7\times$ faster assembly and $20\%\text{--}24\%$ sparser matrices.

### 2.3 Hybrid Coupling Paper ([`HybridVNMM-EFEM/`](HybridVNMM-EFEM/))
- **Title:** *Conforming Hybrid Edge Finite Element and Vector Nodal Meshless Formulation for 2D Electromagnetics*
- **Key Focus:** Formulates direct master-slave coupling between Whitney edge finite elements (Edge-FEM) and interior VNMM nodal clouds across an arbitrary interface $\Gamma_{\mathrm{int}}$, achieving seamless tangential continuity without zero-mass saddle-point indefiniteness or Lagrange multipliers.

---

## 3. Compilation Guide

To compile any manuscript locally using `tectonic` (recommended) or `pdflatex`:

```bash
# Compile Truly Cell-Free VNMM-MLPG manuscript:
cd paper/mlpg-vnmm
tectonic --keep-intermediates main-vnmm-mlpg.tex
# or using pdflatex / bibtex:
# pdflatex main-vnmm-mlpg.tex && bibtex main-vnmm-mlpg && pdflatex main-vnmm-mlpg.tex && pdflatex main-vnmm-mlpg.tex

# Compile VNMM 2D EABE manuscript:
cd ../VNMM2D-EABE
pdflatex stable-vnmm2d.tex
bibtex stable-vnmm2d
pdflatex stable-vnmm2d.tex
pdflatex stable-vnmm2d.tex

# Compile Cover Letter:
pdflatex Cover_Letter_EABE.tex

# Compile Hybrid Coupling manuscript:
cd ../HybridVNMM-EFEM
pdflatex main-acoplamento-vnmm-efem.tex
bibtex main-acoplamento-vnmm-efem
pdflatex main-acoplamento-vnmm-efem.tex
pdflatex main-acoplamento-vnmm-efem.tex
```

---

## 4. Policy on Binary Artifacts

In accordance with repository configuration:
- Compiled binaries (`.pdf`, `.png`) are excluded from Git tracking via [`.gitignore`](../.gitignore).
- All vector figures included in the papers can be regenerated on demand by executing the corresponding scripts in [`2D/scripts/`](../2D/scripts/).
