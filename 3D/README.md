# Vector Nodal Meshless Method in Three Dimensions (VNMM 3D)

This directory is designated for the 3D formulation and numerical implementation of the Vector Nodal Meshless Method (VNMM 3D), extending both the cell-based and truly cell-free MLPG frameworks to three-dimensional vector electromagnetic problems in $H(\mathbf{curl}; \Omega)$.

## Core Theoretical Foundations for 3D:
- **Complete Linear Polynomial Basis ($\mathcal{P}_1^3$):** Spans the 12-dimensional vector polynomial space ($M_{\mathrm{supp}} = 12$ nodes per stencil in 3D), completely capturing the 9 components of $\nabla\vec{E}$ and eliminating 3D curl stagnation.
- **Scale-Invariant Support Selection ($\widehat{\mathbf{A}}_{12 \times 12}$):** Dimensionless coordinate normalization in 3D spherical neighborhoods.
- **Truly Cell-Free 3D MLPG:** Closed-form analytical integration over autonomous spherical subdomains $\Omega_s^I$ of radius $r_s^I$ ($V_0 = \frac{4}{3}\pi (r_s^I)^3$ and second moment of volume $I_2 = \frac{4}{15}\pi (r_s^I)^5$), totally eliminating numerical quadrature in 3D.
- **Conforming Interface Coupling:** Master-slave continuity coupling with tetrahedral Whitney edge finite elements (Whitney 1-forms).

## Planned Directory Structure:
- `src/`: Core 3D modules (spherical/tetrahedral support selection, 3D shape functions, closed-form spherical integration, and 3D Whitney edge FEM).
- `scripts/`: Executable benchmarks and validation scripts in 3D (waveguides, resonant cavities, scattering).
- `meshes/`: 3D nodal point clouds and tetrahedral benchmark meshes.
- `tests/`: Automated unit tests for 3D solvers and assembly routines.
