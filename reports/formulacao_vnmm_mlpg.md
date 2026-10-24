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

## 5. Seleção da Função de Teste Vetorial $\mathbf{W}_I(\mathbf{x})$

Uma das maiores vantagens do MLPG reside na liberdade de escolha da função de teste no espaço de Petrov-Galerkin:

### Opção 1: Função de Teste com Suporte Compacto que se Anula em $\partial \Omega_s^I$ (MLPG Tipo 1 / 4)
Define-se uma função de peso radial suave $w_I(r)$ com raio $r_s$, tal que $w_I(r_s) = 0$ e $\frac{\partial w_I}{\partial r}(r_s) = 0$ (ex.: spline quártica):

$$
w_I(r) = \begin{cases} 
1 - 6\left(\frac{r}{r_s}\right)^2 + 8\left(\frac{r}{r_s}\right)^3 - 3\left(\frac{r}{r_s}\right)^4, & r \le r_s \\
0, & r > r_s 
\end{cases}
$$

A função de teste vetorial associada ao grau de liberdade nodal $I$ com orientação $\mathbf{t}_I$ é:

$$
\mathbf{W}_I(\mathbf{x}) = w_I(\|\mathbf{x} - \mathbf{x}_I\|)\, \mathbf{t}_I
$$

#### Vantagem Fundamental:
Para nós internos ($\Omega_s^I \cap \partial \Omega = \emptyset$), como $\mathbf{W}_I = \mathbf{0}$ em $\partial \Omega_s^I$, **a integral de contorno se anula identicamente**:

$$
\oint_{\partial \Omega_s^I} \mathcal{B}(\mathbf{E}, \mathbf{W}_I)\, d\Gamma \equiv 0
$$

A forma fraca local simplifica-se para integrações puramente volumétricas/superficiais sobre o disco:

$$
\iint_{\Omega_s^I} (\nabla \times \mathbf{E}) \cdot (\nabla \times \mathbf{W}_I) \, dA 
+ s_{\text{div}} \iint_{\Omega_s^I} (\nabla \cdot \mathbf{E}) (\nabla \cdot \mathbf{W}_I) \, dA 
= k^2 \iint_{\Omega_s^I} \mathbf{E} \cdot \mathbf{W}_I \, dA
$$

---

## 6. Discretização e Montagem Algébrica

Substituindo a aproximação nodal $\mathbf{E}^h(\mathbf{x}) = \mathbf{N}^I(\mathbf{x}) \mathbf{e}_I = \mathbf{P}(\mathbf{x}) \mathbf{A}_I^{-1} \mathbf{e}_I$:

Para a equação da linha associada ao nó $I$, as matrizes locais de rigidez $\mathbf{K}_I^{\text{local}} \in \mathbb{R}^{1 \times 6}$ e massa $\mathbf{M}_I^{\text{local}} \in \mathbb{R}^{1 \times 6}$ são calculadas integrando sobre o disco $\Omega_s^I$:

$$
\mathbf{K}_I^{\text{local}} = \left[ \iint_{\Omega_s^I} (\nabla \times \mathbf{P}) \cdot (\nabla \times \mathbf{W}_I) \, dA + s_{\text{div}} \iint_{\Omega_s^I} (\nabla \cdot \mathbf{P}) (\nabla \cdot \mathbf{W}_I) \, dA \right] \mathbf{A}_I^{-1}
$$

$$
\mathbf{M}_I^{\text{local}} = \left[ \iint_{\Omega_s^I} \mathbf{P}(\mathbf{x}) \cdot \mathbf{W}_I(\mathbf{x}) \, dA \right] \mathbf{A}_I^{-1}
$$

### 6.1 Eficiência Extrema com $\mathcal{P}^1$
Como $(\nabla \times \mathbf{P})$ e $(\nabla \cdot \mathbf{P})$ são **vetores constantes**:
- As derivadas do campo saem da integral!
- Apenas a função de teste $\mathbf{W}_I$ precisa ser integrada sobre o disco:
  $$
  \iint_{\Omega_s^I} \nabla \times \mathbf{W}_I \, dA = \oint_{\partial \Omega_s^I} \mathbf{W}_I \cdot d\mathbf{r} = 0 \quad (\text{pois } \mathbf{W}_I|_{\partial \Omega_s^I} = 0)
  $$
- Isso permite pré-computar as integrais polares $(r, \theta)$ de forma analítica ou com regras radiais de Gauss unidimensionais de 3 a 5 pontos.

### 6.2 Espalhamento (*Assembly*) Global
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

1. **Fase 1: Módulo de Geometria Local (`mlpg_subdomains.py`)**
   - Criação dos subdomínios $\Omega_s^I$ com raio adaptativo $r_s = \alpha_s h$.
   - Avaliação e cache da função de teste radial e seus operadores diferenciais.

2. **Fase 2: Montador Local de Petrov-Galerkin (`mlpg_assembler_2d.py`)**
   - Implementação das integrais radiais analíticas/quadratura polar em $\Omega_s^I$.
   - Espalhamento direto nas matrizes esparsas $\mathbf{K}$ e $\mathbf{M}$.

3. **Fase 3: Validação da Cavidade Ressonante PEC 2D**
   - Reprodução dos 10 primeiros modos da cavidade quadrada $[0, \pi]^2$ (Tabela 4-1 de Luilly Ortiz).
   - Verificação de convergência assintótica e comparação direta de tempo de montagem vs. o montador de Gauss.
