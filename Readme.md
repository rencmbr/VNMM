# Método Sem Malha Nodal Vetorial em 2 Dimensões (VNMM 2D)
### *A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Unit Tests](https://img.shields.io/badge/tests-21%20passed-brightgreen.svg)](tests/)
[![Paper Status](https://img.shields.io/badge/manuscript-EABE%20(Elsevier)-orange.svg)](artigo/)
[![Code Availability](https://img.shields.io/badge/GitHub-vnmm2d--eabe-blue.svg)](https://github.com/rencmbr/vnmm2d-eabe)

---

## 1. Visão Geral do Repositório

Este repositório contém a infraestrutura computacional completa, módulos numéricos e suíte de experimentos para o desenvolvimento, validação e aplicação do **Método Sem Malha Nodal Vetorial Bidimensional (VNMM 2D)** em problemas de eletromagnetismo governados pelo espaço funcional $H(\mathbf{curl}; \Omega)$.

O código apoia diretamente o artigo científico submetido ao periódico internacional ***Engineering Analysis with Boundary Elements*** (Elsevier):
> **"A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds"**  
> *Autores:* Renato C. Mesquita, Elson J. da Silva, Bárbara M. F. Gonçalves e Luilly A. G. Ortiz.  
> *Manuscrito e fontes disponíveis em:* [`artigo/`](artigo/) | *Repositório público:* [github.com/rencmbr/vnmm2d-eabe](https://github.com/rencmbr/vnmm2d-eabe)

Além de replicar rigorosamente todos os resultados, figuras e tabelas do artigo, este repositório é uma **plataforma ativa de pesquisa** voltada à modelagem eletromagnética avançada, incluindo o **acoplamento híbrido conforme entre Elementos Finitos de Aresta (Edge FEM) e VNMM 2D**.

---

## 2. Estrutura do Repositório

```text
teste-cod/
├── requirements.txt            # Dependências científicas do Python (numpy, scipy, matplotlib, pandas)
├── .gitignore                  # Regras de exclusão para compilações LaTeX e Python
├── Readme.md                   # Documentação central do projeto e guia de uso dos scripts
│
├── src/                        # Biblioteca central de classes e rotinas de montagem/resolução
│   ├── vnmm_assembler.py       # Montador vetorial VNMM 2D com suporte local por ponto de Gauss (EFG-style)
│   ├── fem_edge_2d.py          # Solver de Elementos Finitos de Aresta de Nédélec (Whitney 1-forms)
│   ├── fem_vnmm_hybrid_2d.py   # Solver Híbrido acoplado Edge-FEM + VNMM 2D com interface conforme
│   ├── eigen_solver_cavity.py  # Solver generalizado de autovalores para cavidades PEC
│   ├── cavity_mesh.py          # Gerador e manipulador de malhas/nuvens com vetores diretores unitários
│   └── gauss_quadrature.py     # Regras de quadratura de Gauss-Legendre (1D e produtos tensoriais 2D)
│
├── codigo/                     # Scripts executáveis de experimentos, benchmarks e geração de figuras (43 scripts)
│   ├── generate_*.py           # Geradores de figuras prontas para publicação (Figuras 1 a 9 do artigo)
│   ├── galerkin_*.py           # Patch test e Método de Soluções Manufaturadas (MMS)
│   ├── dimensionless_*.py      # Formulação adimensional da matriz A e seleção de suporte por escala
│   ├── comparative_*.py        # Comparações sistemáticas de convergência VNMM 2D vs Edge FEM
│   ├── hybrid_*.py             # Acoplamento híbrido FEM de aresta + VNMM em malhas regulares e aleatórias
│   └── parametric_*.py         # Estudos paramétricos de tolerância e densidade nodal
│
├── artigo/                     # Manuscrito em LaTeX e arquivos formais de submissão ao EABE
│   ├── stable-vnmm2d.tex       # Fonte LaTeX principal do manuscrito (com numeração contínua e referências)
│   ├── stable-vnmm2d.pdf       # Versão compilada de submissão do artigo
│   ├── Cover_Letter_EABE.pdf   # Carta de apresentação endereçada ao Prof. Alexander H.-D. Cheng
│   ├── Highlights.txt          # Destaques científicos estruturados (≤ 85 caracteres por tópico)
│   ├── Suggested_Reviewers.txt # Lista de 5 revisores internacionais independentes
│   ├── references.bib          # Base de dados bibliográfica BibTeX
│   └── figura_*.pdf            # Figuras vetoriais de alta resolução incluídas no artigo
│
├── relatorios/                 # Relatórios técnicos aprofundados e gráficos de análises preliminares (80 arquivos)
│   ├── *.md                    # Relatórios técnicos em Markdown de cada etapa do desenvolvimento
│   └── *.png                   # Painéis de convergência, diagramas e mapas espectrais
│
├── tests/                      # Suíte de testes unitários automatizados (21 testes unitários)
│   ├── test_eigen_cavity.py
│   ├── test_patch_test_galerkin.py
│   ├── test_galerkin_source_problem.py
│   └── test_hybrid_fem_vnmm.py
│
└── malhas/                     # Conjuntos de dados de nuvens de pontos e malhas de teste (.csv e .png)
```

---

## 3. Instalação e Configuração

### 3.1 Requisitos de Sistema
- **Python**: Versão 3.10 ou superior.
- **TeX Live / Tectonic**: Opcional, apenas caso deseje recompilar o artigo em [`artigo/`](artigo/).

### 3.2 Instalação das Dependências
Recomenda-se criar um ambiente virtual isolado:

```bash
# Criação do ambiente virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalação dos pacotes científicos fundamentais
pip install -r requirements.txt
```

### 3.3 Execução dos Testes Automatizados
O repositório possui uma suíte com 21 testes unitários cobrindo o solver de autovalores, o Patch Test de Galerkin, o problema MMS com fonte e o solver acoplado híbrido FEM-VNMM:

```bash
python3 -m unittest discover tests -v
```

---

## 4. Fundamentos Teóricos e Soluções Metodológicas

O VNMM 2D aproxima campos vetoriais $H(\mathbf{curl})$ associando a cada nó $\mathbf{x}_i$ um vetor diretor unitário $\vec{t}_i = [t_{ix}, t_{iy}]^T$. Em nós de contorno e interfaces materiais, $\vec{t}_i$ é orientado tangencialmente ($\vec{t}_i \parallel \partial\Omega$), permitindo a imposição exata de condições de contorno de Dirichlet (PEC) por eliminação direta de graus de liberdade.

As funções de forma vetoriais $\vec{N}_i(\mathbf{x})$ satisfazem a **condição de projeção nodal (delta de Kronecker vetorial)**:
$$
\vec{N}_i(\mathbf{x}_k) \cdot \vec{t}_k = \delta_{ik} \implies \mathbf{A} \boldsymbol{\beta}_i = \mathbf{L}_i \quad \therefore \quad \vec{E}^h(\mathbf{x}) = \sum_{i=1}^{M_{\text{supp}}} \vec{N}_i(\mathbf{x}) e_i
$$

### 4.1 Resolução do Vazamento Modal: Base Incompleta $\mathcal{L}^1$ vs. Base Completa $\mathcal{P}^1$
- **O Gargalo da Base $\mathcal{L}^1$ (3 nós):** A base solenoidal de 3 termos $\mathcal{L}^1 = \langle [1,0]^T, [0,1]^T, [y,-x]^T \rangle$ força artificialmente nulas as derivadas normais ($\partial E_x/\partial x = \partial E_y/\partial y = 0$). Em nuvens arbitrárias de pontos (onde não há o cancelamento de circulação fechada característico de malhas triangulares de Whitney), essas componentes não representadas vazam para o grau de liberdade rotacional, gerando um **platô de estagnação de erro $\mathcal{O}(1)$ no rotacional** ($\approx 15\%$ de erro residual mesmo sob refinamento espacial estrito).
- **A Solução via Base Completa $\mathcal{P}^1$ (6 nós):** Adota-se o espaço linear vetorial completo $\mathcal{P}_1 \times \mathcal{P}_1$ com 6 nós de suporte:
  $$
  \mathcal{P}^1 = \left\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \begin{bmatrix} x \\ 0 \end{bmatrix}, \begin{bmatrix} y \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ x \end{bmatrix}, \begin{bmatrix} 0 \\ y \end{bmatrix} \right\rangle
  $$
  A base $\mathcal{P}^1$ captura todas as 4 componentes do Jacobiano de campo $\mathbf{J} = \nabla \vec{E}$, elimina o vazamento modal e restaura a **convergência de 1ª ordem $\mathcal{O}(h)$ no rotacional** e **superconvergência $\mathcal{O}(h^2)$ na interpolação do campo**.

### 4.2 Seleção de Suporte Invariante de Escala (Matriz Adimensional $\widehat{\mathbf{A}}$)
Em coordenadas físicas, o determinante da matriz de colocação $6 \times 6$ escala com a quarta potência do espaçamento nodal ($\det(\mathbf{A}) \sim \mathcal{O}(h^4)$). Sob refinamento espacial ($h \to 0$), qualquer limiar fixo causava dilatação artificial do raio de suporte.  
Ao normalizar as coordenadas locais pela escala de comprimento local $h_{\text{local}}$:
$$
\hat{x} = \frac{x - x_P}{h_{\text{local}}}, \quad \hat{y} = \frac{y - y_P}{h_{\text{local}}} \implies |\det(\widehat{\mathbf{A}})| = \frac{|\det(\mathbf{A})|}{h_{\text{local}}^4}
$$
A matriz $\widehat{\mathbf{A}}$ torna-se adimensional ($\mathcal{O}(1)$). O algoritmo de busca via `scipy.spatial.KDTree` associado ao limiar universal $\widehat{\text{Tol}} = 0.80$ rejeita singularidades direcionais e geométricas (paralelismo, linhas de ação concorrentes) de forma estável, mantendo vizinhanças compactas ($K_{\text{avg}} \approx 10\text{--}11$ nós) mesmo sob $25\%$ de desordem estocástica.

### 4.3 Estabilização Espectral: Não Conformidade Global e Regularização Div-Curl
Embora o VNMM apresente equivalência dual construtiva exata a elementos de Whitney de menor ordem em triângulos conformes (Teorema da Equivalência Dual Local), em nuvens desestruturadas os suportes nodais mudam dinamicamente ponto a ponto, quebrando a conformidade global de traço ($V_h \not\subset H(\mathbf{curl}; \Omega)$).  
Essa não conformidade viola o complexo de de Rham discreto, fazendo com que modos espúrios de gradiente invadam a faixa espectral física. Para contornar essa patologia, formula-se o problema variacional com termo de penalidade div-curl:
$$
\int_{\Omega} (\nabla \times \vec{W})_z (\nabla \times \vec{E})_z \, d\Omega + s_{\text{div}} \int_{\Omega} (\nabla \cdot \vec{W}) (\nabla \cdot \vec{E}) \, d\Omega = \lambda \int_{\Omega} \vec{W} \cdot \vec{E} \, d\Omega
$$
Com $s_{\text{div}} = 6.0$, os modos de gradiente ($\nabla \cdot \vec{E} \neq 0$) são seletivamente enrijecidos e **deslocados para fora da janela de frequências de interesse operacional**, deixando os modos físicos da cavidade ressonante ($\nabla \cdot \vec{E} \equiv 0$) imperturbados.

---

## 5. Catálogo Completo de Scripts e Guia de Execução

Todos os scripts encontram-se no diretório [`codigo/`](codigo/) e podem ser executados diretamente a partir da raiz do repositório.

### 5.1 Reprodução Direta das Figuras e Tabelas do Artigo EABE

| Script em [`codigo/`](codigo/) | Saída / Figura Gerada no Artigo | Descrição Científica |
| :--- | :--- | :--- |
| [`generate_dual_whitney_vnmm_equivalence_figure.py`](codigo/generate_dual_whitney_vnmm_equivalence_figure.py) | **Figura 1** (`figura_equivalencia_dual_whitney_vnmm.pdf`) | Demonstração geométrica e numérica da equivalência dual exata entre VNMM $\mathcal{L}^1$ e formas de Whitney em triângulos conformes. |
| [`generate_matrix_a_singularities_figure.py`](codigo/generate_matrix_a_singularities_figure.py) | **Figura 2** (`figura_singularidades_matriz_A.pdf`) | Painel com as 4 configurações singulares patológicas da matriz $\mathbf{A}$, desacoplamento geométrico-direcional e suporte admissível. |
| [`generate_conforming_mesh_vnmm_l1_figure.py`](codigo/generate_conforming_mesh_vnmm_l1_figure.py) | **Figura 3** (`figura_malha_conforme_vnmm_L1.pdf`) | Malha simplicial de teste utilizada na validação conforme do VNMM $\mathcal{L}^1$. |
| [`generate_efem_vs_vnmm_l1_figure.py`](codigo/generate_efem_vs_vnmm_l1_figure.py) | **Figura 4** (`figura_convergencia_efem_vs_vnmm_L1.pdf`) | Curvas assintóticas de convergência comprovando identidade numérica entre Edge FEM e VNMM $\mathcal{L}^1$ no caso conforme. |
| [`generate_convergence_l1_vs_p1_pi_figure.py`](codigo/generate_convergence_l1_vs_p1_pi_figure.py) | **Figura 5** (`figura_convergencia_interpolacao_L1_vs_P1.pdf`) | Comparação de interpolação: estagnação $\mathcal{O}(1)$ da base $\mathcal{L}^1$ vs restauração de taxa $\mathcal{O}(h)$ e $\mathcal{O}(h^2)$ na base $\mathcal{P}^1$. |
| [`generate_manuscript_results_figures.py`](codigo/generate_manuscript_results_figures.py) | **Figuras 6, 7 e 8** | Gera em lote: robustez estocástica (Fig. 6), purificação espectral div-curl (Fig. 7) e convergência comparativa VNMM vs FEM (Fig. 8). |
| [`generate_manuscript_random_cavity_figure.py`](codigo/generate_manuscript_random_cavity_figure.py) | **Figura 9** (`figura_convergencia_cavidade_aleatoria.pdf`) | Validação espectral da cavidade em nuvens de pontos irregulares com diretores arbitrários. |
| [`galerkin_patch_test.py`](codigo/galerkin_patch_test.py) | **Tabela 6** | Patch test variacional para campos afins: erro em precisão de máquina ($\le 7.4 \times 10^{-14}$) na base $\mathcal{P}^1$. |
| [`galerkin_source_problem.py`](codigo/galerkin_source_problem.py) | **Tabela 7** | Método das Soluções Manufaturadas (MMS) com termo fonte: convergência em $L^2$ e $H(\mathbf{curl})$. |
| [`s_div_sensitivity_study.py`](codigo/s_div_sensitivity_study.py) | **Tabela 9** | Análise de sensibilidade ao parâmetro de penalização $s_{\text{div}}$ na cavidade ressonante. |
| [`base_case_integration_study.py`](codigo/base_case_integration_study.py) | **Tabela 10** | Estudo de convergência e trade-off quanto à ordem de quadratura e densidade de células de fundo. |
| [`comparative_convergence_vnmm_vs_fem.py`](codigo/comparative_convergence_vnmm_vs_fem.py) | **Tabela 11** | Estudo comparativo de convergência em 6 refinamentos: VNMM 2D puro vs Edge FEM. |
| [`random_cavity_convergence_analysis.py`](codigo/random_cavity_convergence_analysis.py) | **Tabela 12** | Convergência espectral e estatísticas modais em nuvens de pontos aleatórias/perturbadas. |

#### Como executar as reproduções:
```bash
# Para reproduzir o Patch Test de Galerkin (Tabela 6):
python3 codigo/galerkin_patch_test.py

# Para reproduzir a análise MMS de convergência com termo fonte (Tabela 7):
python3 codigo/galerkin_source_problem.py

# Para reproduzir as figuras principais de resultados (Figuras 6, 7 e 8):
python3 codigo/generate_manuscript_results_figures.py

# Para gerar o estudo comparativo VNMM vs FEM (Tabela 11):
python3 codigo/comparative_convergence_vnmm_vs_fem.py
```

---

### 5.2 Formulação e Estabilidade Adimensional (Matriz $\widehat{\mathbf{A}}$)

| Script em [`codigo/`](codigo/) | Função e Conteúdo |
| :--- | :--- |
| [`dimensionless_vnmm_6_p1.py`](codigo/dimensionless_vnmm_6_p1.py) | Módulo com funções para cálculo da matriz adimensional $\widehat{\mathbf{A}}$ e qualificação geométrica local. |
| [`dimensionless_matrix_sequence_test.py`](codigo/dimensionless_matrix_sequence_test.py) | Teste numérico verificando a eliminação da dependência quártica $\mathcal{O}(h^4)$ do determinante sob refinamento. |
| [`run_dimensionless_sweep.py`](codigo/run_dimensionless_sweep.py) | Varredura paramétrica do limiar adimensional $\widehat{\text{Tol}} \in [0.05, 6.0]$ para determinação da janela ótima de operação. |
| [`generate_pi_floor_robustness_study.py`](codigo/generate_pi_floor_robustness_study.py) | Avaliação detalhada de estabilidade sob perturbações estocásticas severas ($25\%$ de deslocamento e rotação). |
| [`dimensionless_cavity_test.py`](codigo/dimensionless_cavity_test.py) | Teste integrado da cavidade ressonante utilizando o seletor de suporte puramente adimensional. |

---

### 5.3 Análise Comparativa e Formulações Alternativas

| Script em [`codigo/`](codigo/) | Função e Conteúdo |
| :--- | :--- |
| [`compare_vnmm_vs_edge_fem.py`](codigo/compare_vnmm_vs_edge_fem.py) | Comparação direta modo a modo dos autovalores calculados pelo VNMM e pelo FEM de aresta no caso base. |
| [`cavity_convergence_analysis.py`](codigo/cavity_convergence_analysis.py) | Análise de convergência espectral em malhas uniformes e semi-estruturadas. |
| [`l1_basis_study.py`](codigo/l1_basis_study.py) | Demonstração da incapacidade de regularização da base $\mathcal{L}^1$ devido à divergência identicamente nula ($K_{\text{div}} \equiv 0$). |
| [`unregularized_vnmm_study.py`](codigo/unregularized_vnmm_study.py) | Diagnóstico da invasão de modos espúrios de gradiente quando a penalização é desligada ($s_{\text{div}} = 0.0$). |
| [`mimetic_curl_stokes_study.py`](codigo/mimetic_curl_stokes_study.py) | Implementação e teste da formulação mimética de Stokes baseada em circulação de contorno fechado. |

---

### 5.4 Pesquisa em Andamento: Acoplamento Híbrido FEM de Aresta + VNMM 2D

| Script em [`codigo/`](codigo/) | Função e Conteúdo |
| :--- | :--- |
| [`test_fem_vnmm_coupling.py`](codigo/test_fem_vnmm_coupling.py) | Demonstração do solver híbrido acoplado com continuidade tangencial direta na interface $\Gamma_{\text{int}}$. |
| [`hybrid_fem_vnmm_convergence_comparison.py`](codigo/hybrid_fem_vnmm_convergence_comparison.py) | Comparação sistemática de acurácia e DoFs entre a formulação híbrida, o VNMM puro e o FEM de aresta puro. |
| [`hybrid_random_mesh_convergence_analysis.py`](codigo/hybrid_random_mesh_convergence_analysis.py) | Estudo de convergência do solver híbrido sob malhas deformadas e perturbadas aleatoriamente. |

---

## 6. Resultados Principais

### 6.1 Desempenho no Problema de Autovalores da Cavidade PEC ($TE_z$)
No caso base ($N_x=21, N_y=21$, 361 graus de liberdade ativos), a comparação com os autovalores analíticos exatos ($\lambda = n^2 + m^2$) apresenta excelente precisão:

| Modo ($TE_{nm}$) | $\lambda_{\text{analítico}}$ | $k_{c, \text{analítico}}$ | $\lambda_{\text{VNMM } \mathcal{P}^1}$ | $k_{c, \text{VNMM}}$ | Erro $k_c$ (%) | Diagnóstico |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$TE_{10}$** | 1.00 | 1.000 | 0.9785 | 0.989 | **1.08%** | Modo Fundamental |
| **$TE_{01}$** | 1.00 | 1.000 | 0.9904 | 0.995 | **0.48%** | Modo Fundamental |
| **$TE_{11}$** | 2.00 | 1.414 | 1.9723 | 1.404 | **0.70%** | Físico |
| **$TE_{20}$** | 4.00 | 2.000 | 3.9634 | 1.991 | **0.46%** | Físico |
| **$TE_{02}$** | 4.00 | 2.000 | 4.0152 | 2.004 | **0.19%** | Físico |
| **$TE_{21}$** | 5.00 | 2.236 | 4.9650 | 2.228 | **0.35%** | Físico |
| **$TE_{12}$** | 5.00 | 2.236 | 5.0341 | 2.244 | **0.34%** | Físico |
| **$TE_{22}$** | 8.00 | 2.828 | 7.9734 | 2.824 | **0.17%** | Físico |
| **$TE_{30}$** | 9.00 | 3.000 | 9.3872 | 3.064 | **2.13%** | Físico |
| **$TE_{03}$** | 9.00 | 3.000 | 9.4751 | 3.078 | **2.61%** | Físico |

- **Erro médio global de $k_c$:** **$0.85\%$**
- **Tempo de montagem e solução:** **$0.13\text{s}$**

---

### 6.2 Comparação de Convergência Sistemática

| Nível ($N$) | $h$ (m) | DoFs VNMM | Erro Médio $k_c$ VNMM | DoFs Híbrido | Erro Médio $k_c$ Híbrido | DoFs FEM de Aresta | Erro Médio $k_c$ FEM |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **N1** | 0.3927 | 49 | **32.42%** | 113 | **3.74%** | 40 | **3.19%** |
| **N2** | 0.2618 | 121 | **10.82%** | 265 | **2.40%** | 133 | **1.05%** |
| **N3** | 0.1963 | 225 | **5.01%** | 481 | **0.71%** | 225 | **0.64%** |
| **N4 (Base)** | 0.1571 | **361** | **2.53%** | **761** | **0.53%** | **341** | **0.44%** |
| **N5** | 0.1309 | 529 | **1.31%** | 1105 | **0.22%** | 560 | **0.27%** |
| **N6** | 0.1122 | 729 | **0.79%** | 1513 | **0.36%** | 736 | **0.21%** |
| **N7** | 0.0982 | **961** | **0.50%** | **1985** | **0.26%** | **936** | **0.16%** |

---

## 7. Relatórios Técnicos e Documentos Complementares

Para aprofundamento teórico, diagnósticos matemáticos e histórico das formulações testadas, consulte os relatórios em [`relatorios/`](relatorios/):

- [Plano de Implementação da Base Completa $\mathcal{P}^1$](relatorios/plano_implementacao_base_P1.md)
- [Equivalência Dual entre Formas de Whitney e VNMM](relatorios/dual_equivalence_theorem_whitney_vnmm.md)
- [Análise Teórica das Origens dos Erros entre VNMM e FEM](relatorios/analise_origem_erros_vnmm_vs_fem.md)
- [Relatório de Acoplamento Híbrido FEM-VNMM](relatorios/relatorio_acoplamento_hibrido_fem_vnmm.md)
- [Estratégias Teóricas de Acoplamento Híbrido](relatorios/estrategias_acoplamento_fem_vnmm.md)
- [Convergência do Acoplamento Híbrido em Malhas Aleatórias](relatorios/relatorio_acoplamento_hibrido_malhas_aleatorias.md)
- [Estudo de Integração Numérica e Quadratura no Caso Base](relatorios/relatorio_estudo_integracao_caso_base.md)
- [Diagnóstico de Sub-integração e Soluções para Malhas Densas](relatorios/diagnostico_caso_N25_vnmm.md)
- [Estudo do Problema Sem Regularização do Divergente ($s_{\text{div}}=0$)](relatorios/relatorio_sem_regularizacao_divergente.md)
- [Estudo do Problema com a Base $\mathcal{L}^1$](relatorios/relatorio_estudo_base_L1.md)
- [Estudo da Formulação Mimética de Stokes](relatorios/relatorio_rotacional_mimetico_stokes.md)

---

## 8. Como Citar

Se este código ou trabalho for útil em sua pesquisa, por favor cite:

```bibtex
@article{mesquita2026stabilized,
  title={A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds},
  author={Mesquita, Renato C. and da Silva, Elson J. and Gon{\c{c}}alves, B{\'a}rbara M. F. and Ortiz, Luilly A. G.},
  journal={Engineering Analysis with Boundary Elements},
  year={2026},
  note={Under review. Source code available at \url{https://github.com/rencmbr/vnmm2d-eabe}}
}
```

---

## 9. Suporte e Contato

- **Prof. Renato C. Mesquita** (Autor Correspondente): `renato@ufmg.br`  
  *Departamento de Engenharia Elétrica, Universidade Federal de Minas Gerais (UFMG)*  
  *Belo Horizonte, MG, Brasil*
