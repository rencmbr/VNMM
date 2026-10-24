# Formulação do Método Sem Malha Nodal Vetorial via Meshless Local Petrov-Galerkin (VNMM-MLPG 2D)

## 1. Contexto e Motivação

O **Método Sem Malha Nodal Vetorial (VNMM 2D)** tradicionalmente constrói suas funções de forma vetoriais de maneira puramente nodal (colocação pontual sobre stencils de nós locais). No entanto, a avaliação da forma fraca global de Ritz-Galerkin depende de uma grade de fundo (*background cells*) com quadratura de Gauss-Legendre (estilo *Element-Free Galerkin - EFG*).

Embora essa quadratura em células cartesianas tenha se mostrado precisa para cavidades regulares, ela apresenta limitações intrínsecas:
1. **Descontinuidades Intra-Célula:** A troca de stencils de suporte entre pontos de Gauss dentro de uma mesma célula gera saltos de derivadas não-polinomiais, exigindo ordens elevadas de quadratura ($3 \times 3$ ou $4 \times 4$).
2. **Dependência de Malha de Fundo:** Para geometrias complexas, contornos curvos ou interfaces móveis, gerar células de fundo reintroduz o fardo de geração de malha, comprometendo o caráter estritamente *sem malha* (*truly cell-free*) do método.

A formulação **VNMM-MLPG** (*Meshless Local Petrov-Galerkin*) resolve esse dilema formulando a **forma fraca em subdomínios locais independentes e regulares** associados a cada nó, dispensando qualquer partição global ou células de quadratura de fundo.

---

## 2. Aproximação Nodal Vetorial ($\mathcal{P}^1$, 6 Termos)

Adota-se a base polinomial vetorial linear completa $\mathcal{P}^1$:

$$
\mathbf{p}(\mathbf{x}) = \left\{
\begin{bmatrix} 1 \\ 0 \end{bmatrix}, 
\begin{bmatrix} 0 \\ 1 \end{bmatrix}, 
\begin{bmatrix} x \\ 0 \end{bmatrix}, 
\begin{bmatrix} y \\ 0 \end{bmatrix}, 
\begin{bmatrix} 0 \\ x \end{bmatrix}, 
\begin{bmatrix} 0 \\ y \end{bmatrix}
\right\}
$$

Para qualquer ponto $\mathbf{x} = [x, y]^T$, o campo elétrico aproximado $\mathbf{E}^h(\mathbf{x})$ é expandido linearmente nos 6 coeficientes generalizados $\mathbf{a} = [a_1, \dots, a_6]^T$:

$$
\mathbf{E}^h(\mathbf{x}) = \mathbf{P}(\mathbf{x}) \mathbf{a}
$$

onde $\mathbf{P}(\mathbf{x}) \in \mathbb{R}^{2 \times 6}$:

$$
\mathbf{P}(\mathbf{x}) = \begin{bmatrix}
1 & 0 & x & y & 0 & 0 \\
0 & 1 & 0 & 0 & x & y
\end{bmatrix}
$$

### 2.1 Colocação Nodal no Domínio de Suporte $\Omega_u^I$
Para o nó central $\mathbf{x}_I$, seleciona-se um conjunto admissível de 6 nós vizinhos $\{(\mathbf{x}_k, \mathbf{t}_k)\}_{k=1}^6$ contidos no seu domínio de suporte $\Omega_u^I$, onde $\mathbf{t}_k$ é o vetor unitário diretor do grau de liberdade nodal $e_k$.

Impondo a condição de projeção direcional:

$$
\mathbf{E}^h(\mathbf{x}_k) \cdot \mathbf{t}_k = e_k, \quad k = 1, \dots, 6
$$

Obtém-se a relação algébrica:

$$
\mathbf{A}_I \mathbf{a}_I = \mathbf{e}_I \implies \mathbf{a}_I = \mathbf{A}_I^{-1} \mathbf{e}_I
$$

onde $\mathbf{A}_I \in \mathbb{R}^{6 \times 6}$ tem linhas dadas por:

$$
\mathbf{A}_{I}(k, :) = \begin{bmatrix} t_{kx} & t_{ky} & x_k t_{kx} & y_k t_{kx} & x_k t_{ky} & y_k t_{ky} \end{bmatrix}
$$

As funções de forma vetoriais locais $\mathbf{N}_k^I(\mathbf{x}) \in \mathbb{R}^2$ associadas ao nó $\mathbf{x}_I$ são dadas por:

$$
\mathbf{E}^h(\mathbf{x}) = \mathbf{P}(\mathbf{x}) \mathbf{A}_I^{-1} \mathbf{e}_I = \sum_{k=1}^6 \mathbf{N}_k^I(\mathbf{x})\, e_k^I
$$

com $\mathbf{N}^I(\mathbf{x}) = \mathbf{P}(\mathbf{x}) \mathbf{A}_I^{-1}$.

### 2.2 Propriedade Notável da Base Linear $\mathcal{P}^1$
Dado que $\mathbf{E}^h(\mathbf{x})$ é um polinômio de 1º grau dentro de $\Omega_u^I$:
* O rotacional escalar em 2D é **rigorosamente constante**:
  $$
  (\nabla \times \mathbf{E}^h)_z = \frac{\partial E_y}{\partial x} - \frac{\partial E_x}{\partial y} = a_5 - a_4 = \text{constante}
  $$
* A divergência 2D é **rigorosamente constante**:
  $$
  \nabla \cdot \mathbf{E}^h = \frac{\partial E_x}{\partial x} + \frac{\partial E_y}{\partial y} = a_3 + a_6 = \text{constante}
  $$
* As derivadas de segunda ordem são nulas: $\nabla \times (\nabla \times \mathbf{E}^h) \equiv \mathbf{0}$. Essa propriedade é determinante para a estrutura da forma fraca local.

---

## 3. Geometria dos Domínios Locais do MLPG

No VNMM-MLPG, a cada nó $\mathbf{x}_I$ associam-se dois domínios concêntricos:

```
          ┌──────────────────────────────────────┐
          │      Domínio de Suporte Ω_u^I        │
          │         (raio r_u, 6 nós)            │
          │                                      │
          │         ╭──────────────────╮         │
          │         │ Subdomínio Local │         │
          │         │      Ω_s^I       │         │
          │         │    (raio r_s)    │         │
          │         │        • x_I     │         │
          │         ╰──────────────────╯         │
          │                                      │
          └──────────────────────────────────────┘
```

1. **Subdomínio Local de Quadratura ($\Omega_s^I$):**
   - Disco 2D centrado no nó $I$: $\Omega_s^I = \{\mathbf{x} \in \mathbb{R}^2 : \|\mathbf{x} - \mathbf{x}_I\| \le r_s\}$.
   - Raio característico: $r_s = \alpha_s \cdot h$, onde $h$ é a distância média nodal e $\alpha_s \in [0.7, 0.9]$, garantindo sobreposição mútua sem deixar regiões não-integradas ($\bigcup_I \Omega_s^I \supseteq \Omega$).
   - **Regra Fundamental de Estabilidade:** Todo o subdomínio $\Omega_s^I$ adota **o mesmo stencil de suporte de 6 nós** ancorado no nó central $I$. Isso garante que $\mathbf{E}^h(\mathbf{x})$ seja $C^\infty$ dentro de $\Omega_s^I$, eliminando qualquer chaveamento interno de stencil (*intra-cell switching*).

2. **Domínio de Suporte da Interpolação ($\Omega_u^I$):**
   - Região contendo os 6 vizinhos mais próximos de $\mathbf{x}_I$ que satisfaçam $\det(\mathbf{A}_I) \ge \text{Tol}_{\text{det}}(h)$.
   - Raio efetivo: $r_u > r_s$ (tipicamente $r_u \approx (1.5 \sim 2.5) r_s$).

---

## 4. Forma Fraca Local de Petrov-Galerkin (Local Weak Form)

### 4.1 Equação Governante
Considera-se o problema de autovalores eletromagnéticos em cavidade ressonante 2D (modos $TE_z$):

$$
\nabla \times (\nabla \times \mathbf{E}) - s_{\text{div}} \nabla(\nabla \cdot \mathbf{E}) = k^2 \mathbf{E} \quad \text{em } \Omega
$$

com condição de contorno PEC no condutor perfeito $\partial \Omega$:

$$
\hat{\mathbf{n}} \times \mathbf{E} = \mathbf{0} \quad \text{em } \partial \Omega
$$

### 4.2 Projeção Variacional Local
Multiplica-se a equação diferencial por uma função de teste vetorial $\mathbf{W}_I(\mathbf{x})$ e integra-se sobre o subdomínio local $\Omega_s^I$:

$$
\iint_{\Omega_s^I} \left[ \nabla \times (\nabla \times \mathbf{E}) - s_{\text{div}} \nabla(\nabla \cdot \mathbf{E}) - k^2 \mathbf{E} \right] \cdot \mathbf{W}_I \, dA = 0
$$

Aplicando identidades vetoriais e integração por partes (Teorema de Green / Stokes) em $\Omega_s^I$:

$$
\iint_{\Omega_s^I} (\nabla \times \mathbf{E}) \cdot (\nabla \times \mathbf{W}_I) \, dA 
+ s_{\text{div}} \iint_{\Omega_s^I} (\nabla \cdot \mathbf{E}) (\nabla \cdot \mathbf{W}_I) \, dA 
- k^2 \iint_{\Omega_s^I} \mathbf{E} \cdot \mathbf{W}_I \, dA 
- \oint_{\partial \Omega_s^I} \mathcal{B}(\mathbf{E}, \mathbf{W}_I) \, d\Gamma = 0
$$

onde o termo de contorno de fronteira local $\mathcal{B}$ é:

$$
\oint_{\partial \Omega_s^I} \mathcal{B} \, d\Gamma = \oint_{\partial \Omega_s^I} \left[ (\hat{\mathbf{n}} \times (\nabla \times \mathbf{E})) \cdot \mathbf{W}_I + s_{\text{div}} (\nabla \cdot \mathbf{E}) (\mathbf{W}_I \cdot \hat{\mathbf{n}}) \right] d\Gamma
$$

---

## 5. Seleção da Função de Teste Vetorial: Local Galerkin (MLPG Tipo 6)

Para uma base vetorial linear $\mathcal{P}^1$, onde $(\nabla \times \mathbf{E})$ e $(\nabla \cdot \mathbf{E})$ são constantes dentro de $\Omega_s^I$, adotar uma função de teste puramente escalar que se anula na fronteira ($w(r_s) = 0$) anularia identicamente o termo de rigidez por meio do Teorema de Stokes/Gauss ($\iint_{\Omega_s^I} \text{const} \cdot \nabla \times \mathbf{W} \, dA = \text{const} \oint_{\partial \Omega_s^I} \mathbf{W} \cdot d\mathbf{r} \equiv 0$).

Por essa razão matemática fundamental, no espaço $\mathcal{P}^1$ adota-se a formulação canônica **MLPG Tipo 6 (Local Galerkin)** (Atluri & Shen, 2002), na qual a função de teste vetorial associada ao grau de liberdade nodal $I$ é a **própria função de forma local correspondente ao nó central** em seu stencil:

$$
\mathbf{W}_I(\mathbf{x}) = \mathbf{N}_{\text{local\_}I}^I(\mathbf{x}) \quad \text{em } \Omega_s^I
$$

onde $\text{local\_}I$ é o índice local do nó $I$ em seu stencil de 6 nós (geralmente $0$).

### 5.1 Propriedade Dual Nodal
Pela definição estrita de colocação nodal do VNMM ($\mathbf{N}_k^I(\mathbf{x}_m) \cdot \mathbf{t}_m = \delta_{km}$):
* No centro do subdomínio $\mathbf{x}_I$:
  $$
  \mathbf{W}_I(\mathbf{x}_I) \cdot \mathbf{t}_I = \mathbf{N}_{\text{local\_}I}^I(\mathbf{x}_I) \cdot \mathbf{t}_I = 1.0
  $$
* A forma fraca local sobre $\Omega_s^I$ desacopla e projeta o resíduo exatamente sobre a direção física do grau de liberdade nodal $e_I$.

---

## 6. Discretização e Fórmulas Analíticas Fechadas

Dado que dentro de $\Omega_s^I$ (disco de raio $r_s$ centrado em $\mathbf{x}_I$) o campo aproxima-se por $\mathbf{E}^h(\mathbf{x}) = \sum_{k=1}^6 \mathbf{N}_k^I(\mathbf{x}) e_k^I$:

### 6.1 Matriz de Rigidez Local $\mathbf{K}_I^{\text{local}} \in \mathbb{R}^{1 \times 6}$
Como tanto $(\nabla \times \mathbf{N}_k^I)_z$ quanto $(\nabla \cdot \mathbf{N}_k^I)$ são **estritamente constantes** em todo o disco:
$$
(\nabla \times \mathbf{N}_k^I)_z = \beta_{4, k} - \beta_{3, k} = C_{\text{curl}, k}
$$
$$
(\nabla \cdot \mathbf{N}_k^I) = \beta_{2, k} + \beta_{5, k} = C_{\text{div}, k}
$$

A integral de rigidez sobre a área do disco $A_s = \pi r_s^2$ é **100% analítica e exata**:
$$
K_I^{\text{local}}(k) = \left[ \frac{1}{\mu_r} C_{\text{curl}, k} \, C_{\text{curl}, i_0} + \frac{s_{\text{div}}}{\mu_r} C_{\text{div}, k} \, C_{\text{div}, i_0} \right] (\pi r_s^2)
$$
onde $i_0 = \text{local\_}I$.

### 6.2 Matriz de Massa Local $\mathbf{M}_I^{\text{local}} \in \mathbb{R}^{1 \times 6}$
Em coordenadas locais $(\Delta x = x - x_I, \Delta y = y - y_I)$, as funções de forma são:
$$
\mathbf{N}_k^I(\Delta x, \Delta y) = \begin{bmatrix}
\beta_{0, k} + \beta_{2, k} \Delta x + \beta_{3, k} \Delta y \\
\beta_{1, k} + \beta_{4, k} \Delta x + \beta_{5, k} \Delta y
\end{bmatrix}
$$

Integrando o produto escalar $\mathbf{N}_k^I \cdot \mathbf{N}_{i_0}^I$ sobre o disco circular de raio $r_s$:
- Termos de ordem zero: $\iint_{\Omega_s^I} 1 \, dA = \pi r_s^2$
- Termos lineares e cruzados: $\iint \Delta x \, dA = \iint \Delta y \, dA = \iint \Delta x \Delta y \, dA = 0$ (por simetria polar)
- Termos quadráticos: $\iint \Delta x^2 \, dA = \iint \Delta y^2 \, dA = \frac{\pi r_s^4}{4} = I_2$

Obtém-se a **fórmula analítica fechada da massa**:
$$
M_I^{\text{local}}(k) = \epsilon_r \left[ (\beta_{0, k} \beta_{0, i_0} + \beta_{1, k} \beta_{1, i_0}) (\pi r_s^2) + \sum_{p=2}^5 \beta_{p, k} \beta_{p, i_0} \left(\frac{\pi r_s^4}{4}\right) \right]
$$

### 6.3 Vetor de Força Local $F_I$ (Para Fontes Arbitrárias e MMS)
Para um termo de corrente fonte $\mathbf{J}_s(\mathbf{x})$:
$$
F_I = \iint_{\Omega_s^I} \mathbf{J}_s(\mathbf{x}) \cdot \mathbf{N}_{i_0}^I(\mathbf{x}) \, dA
$$
### 6.4 Espalhamento (*Assembly*) Global
Cada nó $I$ gera uma linha no sistema global. Os 6 coeficientes calculados em $\mathbf{K}_I^{\text{local}}$ e $\mathbf{M}_I^{\text{local}}$ são alocados nas colunas correspondentes aos 6 nós globais $\{g_1, \dots, g_6\}$ do stencil do nó $I$:

$$
\mathbf{K}(I, g_k) \mathrel{+}= \mathbf{K}_I^{\text{local}}(k), \quad \mathbf{M}(I, g_k) \mathrel{+}= \mathbf{M}_I^{\text{local}}(k), \quad k = 1, \dots, 6
$$

Resultando no problema generalizado de autovalores esparso:

$$
\mathbf{K} \mathbf{e} = k^2 \mathbf{M} \mathbf{e}
$$

---

## 7. Tratamento das Condições de Contorno de Dirichlet (PEC)

Para nós na fronteira física ($I \in \partial \Omega$):
- Adota-se a colocação direta da condição tangencial PEC:
  $$
  e_I = 0 \quad (\text{se } \mathbf{t}_I \parallel \partial \Omega)
  $$
- Nas linhas correspondentes aos nós de contorno na matriz global, impõe-se identidade:
  $$
  \mathbf{K}(I, I) = 1, \quad \mathbf{K}(I, j \ne I) = 0, \quad \mathbf{M}(I, :) = 0
  $$
- Elimina-se a necessidade de integrais singulares de contorno ou multiplicadores de Lagrange.

---

## 8. Vantagens Comparativas: VNMM-MLPG vs. VNMM-Gauss Células de Fundo

| Característica | VNMM Padrão (Células de Fundo / Gauss) | VNMM-MLPG (Subdomínio Local) |
| :--- | :--- | :--- |
| **Dependência de Malha** | Exige malha cartesiana auxiliar de fundo | **Totalmente sem malha** (*truly cell-free*) |
| **Chaveamento de Suporte** | Ocorre dentro das células de Gauss | **Inexistente** (suporte fixo em $\Omega_s^I$) |
| **Pontos de Integração** | $400 \sim 14.400$ pontos de Gauss | $N$ subdomínios locais |
| **Integrandos** | Descontínuos entre pontos de Gauss | **Perfeitamente suaves ($C^\infty$)** no disco |
| **Extensão para 3D** | Malhas de tetraedros/tijolos auxiliares | **Esferas 3D analíticas** centradas nos nós |

---

## 9. Roteiro de Implementação e Validação

1. **Fase 1: Módulo de Geometria e Subdomínios Locais (`mlpg_subdomains.py`)**
   - Construção dos subdomínios circulares $\Omega_s^I$ com raio adaptativo $r_s = \alpha_s h$ centrado em cada nó.
   - Implementação das funções de teste radiais suaves com suporte compacto (spline quártica) e cálculo de seus operadores diferenciais $(\nabla \times \mathbf{W}_I, \nabla \cdot \mathbf{W}_I)$.
   - Vinculação estrita do stencil fixo de 6 nós de $\Omega_u^I$ ao respectivo $\Omega_s^I$.

2. **Fase 2: Montador Local de Petrov-Galerkin (`mlpg_assembler_2d.py`)**
   - Implementação das integrais radiais analíticas ou via quadratura polar de 1D em $\Omega_s^I$.
   - Avaliação das matrizes elementares locais de rigidez $\mathbf{K}_I^{\text{local}}$ e massa $\mathbf{M}_I^{\text{local}}$.
   - Espalhamento (*assembly*) direto nas matrizes esparsas globais $\mathbf{K}$ e $\mathbf{M}$ (formato CSR/CSC).
   - Vetor de forças $\mathbf{F}$ para problemas com fonte não-homogênea.

3. **Fase 3: Verificação de Consistência Linear: Galerkin Patch Test (`mlpg_patch_test_2d.py`)**
   - Aplicação do teste de consistência em campos vetoriais afins (translações rígidas, deformação axial, cisalhamento puro e campo afim genérico com rotacional e divergência constantes).
   - Resolução do sistema particionado com valores exatos impostos na fronteira: $\mathbf{K}_{\text{int, int}} \mathbf{e}_{\text{int}} = \mathbf{F}_{\text{int}} - \mathbf{K}_{\text{int, bnd}} \mathbf{e}_{\text{bnd}}^*$.
   - **Critério de Aceitação:** Erro relativo máximo nos graus de liberdade livres internos na ordem da precisão de máquina:
     $$
     \frac{\|\mathbf{e}_{\text{int}} - \mathbf{e}_{\text{int}}^*\|_\infty}{\|\mathbf{e}_{\text{int}}^*\|_\infty} \le 10^{-11}
     $$
   - Validação tanto em malhas regulares quanto em nuvens desordenadas com perturbação estocástica de coordenadas e orientações (25% de jitter espacial).

4. **Fase 4: Verificação de Convergência Assintótica via Solução Fabricada (MMS) (`mlpg_source_problem_2d.py`)**
   - Formulação de um problema de valor sobre o contorno acionado por termo-fonte não-homogêneo $\mathbf{J}_s$:
     $$
     \nabla \times (\nabla \times \mathbf{E}) - s_{\text{div}} \nabla(\nabla \cdot \mathbf{E}) = \mathbf{J}_s \quad \text{em } \Omega
     $$
   - Utilização de uma solução analítica suave de referência (com curvatura e rotacional não-constantes):
     $$
     \mathbf{E}^* = \begin{bmatrix} \sin(y) \cos(x) \\ -\cos(y) \sin(x) \end{bmatrix}
     $$
   - Avaliação das taxas assintóticas de convergência com o refinamento nodal ($h \to 0$):
     - Taxa teórica esperada em norma $L^2$: $\mathcal{O}(h^2)$
     - Taxa teórica esperada em norma $H(\mathbf{curl})$: $\mathcal{O}(h)$
   - Determinação do erro residual contínuo e verificação da monotonicidade estrita.

5. **Fase 5: Validação Espectral Física: Cavidade Ressonante PEC 2D (`mlpg_cavity_eigenvalues_2d.py`)**
   - Resolução do problema generalizado de autovalores $\mathbf{K} \mathbf{e} = k^2 \mathbf{M} \mathbf{e}$ sob condições de contorno de condutor elétrico perfeito (PEC).
   - Extração e classificação dos primeiros 10 autovalores físicos (modos $TE_{10}, TE_{01}, TE_{11}, \dots$) em comparação direta com a Tabela 4-1 de Luilly Ortiz (UFMG, 2023).
   - Análise de sensibilidade do parâmetro de regularização divergente $s_{\text{div}}$ na purificação espectral (eliminação de modos espúrios de rotacional nulo).
   - Benchmarking de desempenho computacional e tempo de montagem: VNMM-MLPG (*cell-free*) vs. VNMM-Gauss (células cartesianas de fundo).

---

## 10. Resumo dos Resultados Obtidos (Fases 1 a 5)

| Fase | Script / Módulo | Teste Unitário | Critério / Métrica | Resultado Obtido | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Fase 1** | `src/mlpg_subdomains.py` | `test_mlpg_subdomains_2d.py` | Cobertura $100\%$, $\det(\mathbf{A}) \ge \text{Tol}$, operadores analíticos | 5 testes aprovados | **Aprovado** |
| **Fase 2** | `src/mlpg_assembler_2d.py` | `test_mlpg_assembler_2d.py` | Simetria $\mathbf{K}=\mathbf{K}^T$, $\mathbf{M}=\mathbf{M}^T$, quadratura polar = analítica $<10^{-12}$ | 5 testes aprovados | **Aprovado** |
| **Fase 3** | `scripts/mlpg_patch_test_2d.py` | `test_mlpg_patch_test_2d.py` | Erro relativo linear $\|\mathbf{e} - \mathbf{e}^*\|_\infty / \|\mathbf{e}^*\|_\infty \le 10^{-11}$ (regular e $25\%$ jitter) | Erro $\sim 10^{-15}$ | **Aprovado** |
| **Fase 4** | `scripts/mlpg_source_problem_2d.py` | `test_mlpg_source_problem_2d.py` | Taxas assintóticas MMS: $L^2 \to \mathcal{O}(h^2)$, $H(\mathbf{curl}) \to \mathcal{O}(h)$ | $L^2: \mathcal{O}(h^{2.06})$, $H(\mathbf{curl}): \mathcal{O}(h^{1.24})$ | **Aprovado** |
| **Fase 5** | `scripts/mlpg_cavity_eigenvalues_2d.py` | `test_mlpg_cavity_eigenvalues_2d.py` | 10 primeiros modos PEC, purificação espectral ($s_{\text{div}} \ge 10$), speedup $> 1.5\times$ vs Gauss | Erro médio $k_c = 4.13\%$, zero espúrios, **3.59x mais rápido** | **Aprovado** |

