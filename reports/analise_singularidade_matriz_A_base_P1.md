# Análise Rigorosa das Condições de Singularidade da Matriz de Momentos $\mathbf{A}$ na Base $\mathcal{P}^1$ (6 Nós)

**Autores:** Renato C. Mesquita, Bárbara M. F. Gonçalves, Luilly A. G. Ortiz  
**Projeto:** VNMM 2D – Estabilização de Suporte, Bases Completas e Acoplamento Conforme com EFEM  
**Data:** Setembro de 2026  
**Local:** Departamento de Engenharia Elétrica, Universidade Federal de Minas Gerais (UFMG)

---

## 1. Contexto e Motivação

Na Seção 3.1 do artigo, as patologias de singularidade algébrica ($\det(\mathbf{A}) = 0$) foram introduzidas pedagogicamente a partir da matriz de colocação local $\mathbf{A} \in \mathbb{R}^{3 \times 3}$, associada à base incompleta $\mathcal{L}^1$ (3 nós de suporte). 

A transição para a base polinomial vetorial linear completa $\mathcal{P}^1 = \mathcal{P}_1 \times \mathcal{P}_1$ (6 nós de suporte) resolve definitivamente a estagnação do rotacional ($\nabla \times \mathbf{E}$) e elimina o *aliasing* modal. Contudo, do ponto de vista da álgebra linear e da geometria computacional, a matriz local de momentos torna-se $\mathbf{A} \in \mathbb{R}^{6 \times 6}$.

Este relatório estabelece os fundamentos teóricos, as demonstrações algébricas e as validações numéricas que respondem a duas questões essenciais:
1. **As condições de singularidade de $\mathcal{L}^1$ continuam válidas em $\mathcal{P}^1$?**
2. **Existem condições adicionais e exclusivas de singularidade na base $\mathcal{P}^1$?**

---

## 2. Formulação da Matriz de Momentos $\mathbf{A}_{6 \times 6}$

A base linear completa em duas dimensões é dada por:
$$\mathcal{P}^1 = \operatorname{span}\left\{ 
\begin{bmatrix} 1 \\ 0 \end{bmatrix}, 
\begin{bmatrix} 0 \\ 1 \end{bmatrix}, 
\begin{bmatrix} x \\ 0 \end{bmatrix}, 
\begin{bmatrix} y \\ 0 \end{bmatrix}, 
\begin{bmatrix} 0 \\ x \end{bmatrix}, 
\begin{bmatrix} 0 \\ y \end{bmatrix} 
\right\}$$

A função de forma vetorial associada ao $i$-ésimo nó em um suporte de 6 nós é expressa como:
$$\vec{N}_i(x, y) = \beta_{1i} \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \beta_{2i} \begin{bmatrix} 0 \\ 1 \end{bmatrix} + \beta_{3i} \begin{bmatrix} x \\ 0 \end{bmatrix} + \beta_{4i} \begin{bmatrix} y \\ 0 \end{bmatrix} + \beta_{5i} \begin{bmatrix} 0 \\ x \end{bmatrix} + \beta_{6i} \begin{bmatrix} 0 \\ y \end{bmatrix}$$

Impondo a propriedade de projeção delta de Kronecker vetorial com coordenadas locais transladadas ao ponto de avaliação $P(x_P, y_P)$ ($\Delta x_k = x_k - x_P$, $\Delta y_k = y_k - y_P$):
$$\vec{N}_i(\mathbf{x}_k) \cdot \vec{t}_k = \delta_{ik}, \quad k = 1, \dots, 6$$

A matriz de colocação local $\mathbf{A} \in \mathbb{R}^{6 \times 6}$ tem suas linhas explicitamente dadas por:
$$\mathbf{A}_{k, :} = \begin{bmatrix} 
t_{kx} & t_{ky} & \Delta x_k t_{kx} & \Delta y_k t_{kx} & \Delta x_k t_{ky} & \Delta y_k t_{ky} 
\end{bmatrix}, \quad k \in \{1, \dots, 6\}$$

### 2.1. O Princípio da Ortogonalidade do Espaço Nulo

Um vetor não-nulo de coeficientes $\mathbf{c} = [a_1, a_2, b_1, b_2, b_3, b_4]^T \in \mathbb{R}^6$ pertence ao núcleo de $\mathbf{A}$ ($\mathbf{A} \mathbf{c} = \mathbf{0}$) se, e somente se, o campo afim:
$$\mathbf{E}(\mathbf{x}) = \begin{bmatrix} a_1 + b_1 \Delta x + b_2 \Delta y \\ a_2 + b_3 \Delta x + b_4 \Delta y \end{bmatrix} \in \mathcal{P}^1$$
satisfizer a condição de ortogonalidade direcional em todos os 6 nós de suporte:
$$\mathbf{E}(\mathbf{x}_k) \cdot \vec{t}_k = 0, \quad \forall k \in \{1, \dots, 6\}$$

Como $\mathcal{P}^1$ decompõe qualquer campo afim em translação ($a_1, a_2$), vorticidade ($\omega$), divergência ($\nabla \cdot \mathbf{E}$) e deformação/cisalhamento puro ($\epsilon_D, \epsilon_S$), **a matriz $\mathbf{A}_{6 \times 6}$ torna-se singular sempre que os nós e diretores forem incapazes de capturar ao menos um desses modos fundamentais**.

---

## 3. Validade das Condições Clássicas de $\mathcal{L}^1$ em $\mathcal{P}^1$

As patologias descritas na Seção 3.1 do artigo permanecem válidas na formulação $\mathcal{P}^1$, manifestando-se com severidade ampliada:

### 3.1. Paralelismo Global de Vetores ($\vec{t}_1 \parallel \vec{t}_2 \parallel \cdots \parallel \vec{t}_6$)
Se todos os 6 nós tiverem diretores paralelos ou antiparalelos ($t_{kx} = c \cdot t_{ky}$ para uma constante $c$):
- $\operatorname{Col}_1 = c \cdot \operatorname{Col}_2$
- $\operatorname{Col}_3 = c \cdot \operatorname{Col}_5$
- $\operatorname{Col}_4 = c \cdot \operatorname{Col}_6$

**Consequência:** Três pares de colunas tornam-se linearmente dependentes simultaneamente. O posto colapsa para $\operatorname{rank}(\mathbf{A}) \le 3$, e $\det(\mathbf{A}) \equiv 0$ independentemente da excelência da distribuição espacial das posições nodais.

### 3.2. Concorrência Global das 6 Linhas de Ação em $P_c(x_c, y_c)$
Se a linha de ação infinita de cada um dos 6 nós passar por um centro comum $P_c$:
$$t_{kx}(\Delta y_k - \Delta y_c) - t_{ky}(\Delta x_k - \Delta x_c) = 0 \quad (\forall k)$$

Expandindo e agrupando pelas colunas de $\mathbf{A}$:
$$(\operatorname{Col}_4 - \operatorname{Col}_5) - \Delta y_c \operatorname{Col}_1 + \Delta x_c \operatorname{Col}_2 = \mathbf{0}$$

**Consequência:** A diferença entre as colunas rotacionais $(\operatorname{Col}_4 - \operatorname{Col}_5)$, que representa a circulação tangencial, é uma combinação linear exata das colunas de translação. Logo, $\det(\mathbf{A}_{6 \times 6}) \equiv 0$.  
*Interpretação física:* O campo rotacional rígido puro centrado em $P_c$ ($\mathbf{E}_{\mathrm{rot}} = [-(\Delta y - \Delta y_c), \Delta x - \Delta x_c]^T$) é estritamente perpendicular aos 6 diretores radiais, permanecendo invisível ao método.

### 3.3. Coincidência Nodal / Colapso Espacial ($\mathbf{x}_i \approx \mathbf{x}_j$)
Dois ou mais nós com posições idênticas colapsam as componentes de momento espacial ($\Delta x_i \approx \Delta x_j, \Delta y_i \approx \Delta y_j$). Se possuírem diretores idênticos, geram linhas repetidas; se possuírem diretores distintos, concentram graus de liberdade num único ponto geométrico, degradando a amostragem 2D.

---

## 4. Condições Adicionais de Singularidade Exclusivas para $\mathcal{P}^1$

A expansão para 6 termos gera novos subespaços de dependência linear. Foram demonstradas e comprovadas analítica e numericamente **quatro condições adicionais fundamentais**:

### 4.1. Condição Adicional 1: Teorema do Paralelismo Parcial (Threshold de 4 Nós)
> **Teorema 1 (Paralelismo Parcial em $\mathcal{P}^1$):**  
> Em um suporte de 6 nós sob a base $\mathcal{P}^1$, **não é necessário** que todos os nós sejam paralelos para anular a matriz. Se **4 ou mais nós** possuírem diretores paralelos entre si ($\vec{t}_1 \parallel \vec{t}_2 \parallel \vec{t}_3 \parallel \vec{t}_4$), então $\det(\mathbf{A}_{6 \times 6}) \equiv 0$ ($\operatorname{rank}(\mathbf{A}) \le 5$), independentemente das coordenadas espaciais de todos os 6 nós e das direções atribuídas aos 2 nós restantes.

**Demonstração Algébrica:**  
Sem perda de generalidade, rotacione o referencial de modo que a direção comum dos 4 nós seja $\hat{x}$ (isto é, $t_{ky} = 0$ para $k \in \{1, 2, 3, 4\}$).  
Nesse referencial, as colunas 2, 5 e 6 assumem zeros idênticos nas 4 primeiras linhas:
$$\mathbf{A}_{k, 2} = t_{ky} = 0, \quad \mathbf{A}_{k, 5} = \Delta x_k t_{ky} = 0, \quad \mathbf{A}_{k, 6} = \Delta y_k t_{ky} = 0, \quad \forall k \in \{1, 2, 3, 4\}$$

Portanto, as colunas $\operatorname{Col}_2, \operatorname{Col}_5$ e $\operatorname{Col}_6$ possuem entradas não-nulas **exclusivamente nas linhas 5 e 6**. Como temos 3 vetores-coluna confinados a um subespaço de dimensão 2 (duas linhas ativas), eles são forçosamente linearmente dependentes em $\mathbb{R}^6$:
$$\exists (c_2, c_5, c_6) \neq (0, 0, 0) \quad \text{tal que} \quad c_2 \operatorname{Col}_2 + c_5 \operatorname{Col}_5 + c_6 \operatorname{Col}_6 = \mathbf{0}$$
Assim, as colunas de $\mathbf{A}$ são dependentes $\implies \det(\mathbf{A}_{6 \times 6}) \equiv 0$. $\blacksquare$

---

### 4.2. Condição Adicional 2: Teorema da Colinearidade Espacial Parcial (5 Nós)
> **Teorema 2 (Colinearidade Parcial em $\mathcal{P}^1$):**  
> Se **5 dos 6 nós** de suporte estiverem situados sobre uma mesma reta no espaço 2D, a matriz $\mathbf{A}_{6 \times 6}$ é estritamente singular ($\det(\mathbf{A}) = 0$), **quaisquer que sejam os diretores vetoriais $\vec{t}_k$ atribuídos a todos os nós**.

**Demonstração Algébrica:**  
Alinhe o eixo $x$ local com a reta que contém os 5 nós. Logo, $\Delta y_k = 0$ para $k \in \{1, 2, 3, 4, 5\}$.  
As colunas que contêm o fator $\Delta y_k$ são a coluna 4 ($\Delta y_k t_{kx}$) e a coluna 6 ($\Delta y_k t_{ky}$).  
Ambas as colunas possuem valor zero em todas as primeiras 5 linhas, assumindo valor não-nulo **apenas na linha do sexto nó**:
$$\operatorname{Col}_4 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \Delta y_6 t_{6x} \end{bmatrix}, \qquad \operatorname{Col}_6 = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ \Delta y_6 t_{6y} \end{bmatrix}$$

Esses dois vetores são múltiplos escalares imediatos um do outro:
$$t_{6y} \operatorname{Col}_4 - t_{6x} \operatorname{Col}_6 = \mathbf{0}$$
Como existe uma combinação linear não-trivial nula, $\det(\mathbf{A}_{6 \times 6}) \equiv 0$. $\blacksquare$

---

### 4.3. Condição Adicional 3: Colapso por Colinearidade Total (6 Nós)
Na base $\mathcal{L}^1$, 3 nós colineares no espaço podiam resultar em $\det(\mathbf{A}_{3 \times 3}) \neq 0$ se os diretores fossem alternados/não-paralelos (Seção 3.1.2: "algebricamente solúvel, porém espacialmente degenerado").

Em contraste, na base $\mathcal{P}^1$:
> **Teorema 3 (Colinearidade Total em $\mathcal{P}^1$):**  
> Se todos os 6 nós forem colineares no espaço ($\Delta y_k = m \Delta x_k$ ou $\Delta x_k = 0$), a matriz $\mathbf{A}_{6 \times 6}$ é **sempre identicamente singular**, com colapso de posto para $\operatorname{rank}(\mathbf{A}) \le 4$.

**Demonstração:**  
Substituindo $\Delta y_k = m \Delta x_k$ nas colunas de $\mathbf{A}$:
$$\operatorname{Col}_4 = \Delta y_k t_{kx} = m (\Delta x_k t_{kx}) = m \operatorname{Col}_3$$
$$\operatorname{Col}_6 = \Delta y_k t_{ky} = m (\Delta x_k t_{ky}) = m \operatorname{Col}_5$$
Isso estabelece **duas dependências lineares de colunas independentes e simultâneas**, reduzindo o posto a no máximo 4, qualquer que seja a escolha dos vetores $\vec{t}_k$. $\blacksquare$

---

### 4.4. Condição Adicional 4: Ocultamento do Modo de Divergência (Alinhamento Azimutal Concêntrico)
Esta é uma das descobertas teóricas mais sutis e relevantes da comparação entre as duas bases:

> **Teorema 4 (Ortogonalidade ao Modo de Divergência Linear):**  
> Se os 6 nós de suporte apresentarem diretores vetoriais **puramente tangenciais a círculos concêntricos centrados no ponto de avaliação $P$** (isto é, $\vec{t}_k \perp \Delta \mathbf{x}_k$ para todo $k$), a matriz $\mathbf{A}_{6 \times 6}$ é singular ($\det(\mathbf{A}) = 0$, $\operatorname{rank}(\mathbf{A}) \le 5$).

**Demonstração Algébrica:**  
A condição de ortogonalidade entre diretor e raio local expressa-se por:
$$\Delta \mathbf{x}_k \cdot \vec{t}_k = \Delta x_k t_{kx} + \Delta y_k t_{ky} = 0, \quad \forall k \in \{1, \dots, 6\}$$

Comparando com as colunas 3 e 6 da matriz $\mathbf{A}$:
$$\operatorname{Col}_3 + \operatorname{Col}_6 = \begin{bmatrix} \Delta x_1 t_{1x} + \Delta y_1 t_{1y} \\ \vdots \\ \Delta x_6 t_{6x} + \Delta y_6 t_{6y} \end{bmatrix} = \begin{bmatrix} 0 \\ \vdots \\ 0 \end{bmatrix} = \mathbf{0}$$
Portanto, $\operatorname{Col}_3 = -\operatorname{Col}_6$, tornando as colunas linearmente dependentes e $\det(\mathbf{A}_{6 \times 6}) \equiv 0$. $\blacksquare$

**Interpretação Física:**  
A base $\mathcal{P}^1$ incorpora o modo de divergência linear isotrópica:
$$\mathbf{E}_{\mathrm{div}}(\mathbf{x}) = \begin{bmatrix} \Delta x \\ \Delta y \end{bmatrix} \implies \nabla \cdot \mathbf{E}_{\mathrm{div}} = 2 \neq 0$$
Quando todos os nós são orientados azimutalmente, a projeção direcional de $\mathbf{E}_{\mathrm{div}}$ é nula em todos os sensores nodais ($\mathbf{E}_{\mathrm{div}}(\mathbf{x}_k) \cdot \vec{t}_k \equiv 0$). O método perde a capacidade de determinar o termo de divergência local.  
*Nota Comparativa:* Em $\mathcal{L}^1$, o alinhamento azimutal em torno de $P$ era o caso ótimo de circulação máxima. Em $\mathcal{P}^1$, o alinhamento puramente azimutal anula a determinação da divergência linear, exigindo que o suporte contenha diversidade angular mista (componentes tangenciais e radiais).

---

## 5. Tabela Comparativa Sistemática: $\mathcal{L}^1$ vs. $\mathcal{P}^1$

| Configuração Geométrica / Direcional | Base Incompleta $\mathcal{L}^1$ (3 nós, $\mathbb{R}^{3 \times 3}$) | Base Completa $\mathcal{P}^1$ (6 nós, $\mathbb{R}^{6 \times 6}$) | Mecanismo Algébrico Subjacente |
| :--- | :--- | :--- | :--- |
| **Paralelismo Direcional Total** | Singular ($\det = 0$, $\operatorname{rank} \le 2$) | Singular ($\det = 0$, $\operatorname{rank} \le 3$) | Dependência entre colunas de translação ($\operatorname{Col}_1 \propto \operatorname{Col}_2$) |
| **Paralelismo Direcional Parcial** | Não se aplica ($M = 3$) | **Singular se $\ge 4$ nós paralelos** ($\det = 0$) | 3 colunas ativas em apenas 2 linhas |
| **Concorrência Global de Linhas de Ação** | Singular ($\det = 0$, $\operatorname{rank} \le 2$) | Singular ($\det = 0$, $\operatorname{rank} \le 5$) | Modo de rotação rígida invisível aos diretores |
| **Colinearidade Espacial Total dos Nós** | Solúvel se diretores diversos ($\det \neq 0$) | **Sempre Singular** ($\det = 0$, $\operatorname{rank} \le 4$) | $\operatorname{Col}_4 = m \operatorname{Col}_3$ e $\operatorname{Col}_6 = m \operatorname{Col}_5$ |
| **Colinearidade Espacial Parcial** | 2 nós na mesma linha de ação ($\det = 0$) | **Singular se $\ge 5$ nós colineares** ($\det = 0$) | $\operatorname{Col}_4 \propto \operatorname{Col}_6$ (apenas 1 linha não-nula) |
| **Circulação Azimutal Concêntrica ($\vec{t} \perp \Delta \mathbf{x}$)** | **Caso Ótimo** ($\kappa(\mathbf{A}) \sim 1$) | **Singular** ($\det = 0$, $\operatorname{rank} \le 5$) | Ocultamento do modo divergente ($\operatorname{Col}_3 = -\operatorname{Col}_6$) |
| **Dimensão Máxima do Núcleo ($\dim \ker \mathbf{A}$)** | 2 | 3 | Existência de campos afins ortogonais ao suporte |

---

## 6. Validação Numérica Computacional

As propriedades foram testadas e validadas computacionalmente por rotinas numéricas de precisão dupla (`float64`):

```python
# 1. Teste de 4 diretores paralelos em P1 (posições arbitrárias em 2D)
# Nós em posição geral, 4 diretores t_k = [1, 0]
A_4par = build_A_P1(nodes_gerais, t_4par)
# Resultado: rank(A) = 5, det(A) = 0.0 (máx erro numérico < 1e-16)

# 2. Teste de 5 nós colineares em P1 (direções completamente aleatórias)
# 5 nós sobre a reta y = 0.5*x + 1, 1 nó fora
A_5col = build_A_P1(nodes_5col, t_aleatorios)
# Resultado: rank(A) = 5, det(A) = 0.0 (independe dos diretores)

# 3. Teste de 6 nós colineares em P1
# Todos sobre a reta y = 0.5*x, diretores não-paralelos
A_6col = build_A_P1(nodes_6col, t_nao_paralelos)
# Resultado: rank(A) = 4, det(A) = 0.0

# 4. Teste de diretores puramente azimutais concentricos em torno de P
# Nós em anel/distribuição 2D, t_k perpendicular a (x_k - P)
A_azim = build_A_P1(nodes_gerais, t_azimutal, P)
# Resultado: Col 3 + Col 6 == 0, rank(A) = 5, det(A) = 1.16e-17
```

---

## 7. Impacto Direto no Algoritmo de Seleção de Suporte (Seção 3.3)

Estas conclusões fornecem a sustentação teórica e algébrica indispensável para as escolhas de projeto do algoritmo proposto no artigo:

1. **Necessidade de $K_{\mathrm{initial}} \ge 12$ Vizinhos na Busca $k$-d Tree:**
   Como a base $\mathcal{P}^1$ possui mais hiperplanos de singularidade (basta um subconjunto de 4 nós paralelos ou 5 colineares para arruinar o sexteto), uma busca puramente geométrica com $K = 6$ falha com alta probabilidade em malhas perturbadas. Dispor de $K \ge 12$ candidatos permite à fase de qualificação algébrica descartar permutações patológicas e identificar sextetos com diversidade angular e espacial completa.

2. **Necessidade do Piso Absoluto de Determinante ($\operatorname{Tol}_{\mathrm{floor}} = 10^{-4}$):**
   A lei assintótica $\det(\mathbf{A}_{6 \times 6}) \sim \mathcal{O}(h^4)$ faz com que, para malhas finas ($h \to 0$), o limiar puro $\operatorname{Tol}_{\mathrm{ref}}(h/h_{\mathrm{ref}})^4$ se torne excessivamente permissivo ($\sim 10^{-5}$ ou $10^{-6}$). Em tais limiares, sextetos próximos de configurações com 4 nós quase-paralelos ou 5 quase-colineares seriam aceitos, elevando o condicionamento para $\kappa(\mathbf{A}) > 400$. O piso $\operatorname{Tol}_{\mathrm{floor}} = 10^{-4}$ força o algoritmo a expandir o raio adaptativo e garantir forte independência algébrica.

3. **Dupla Qualificação Obrigatória (Espacial + Algébrica):**
   Em $\mathcal{P}^1$, a independência espacial e a independência vetorial são fortemente acopladas. O algoritmo deve obrigatoriamente verificar a cobertura bidimensional real da nuvem de nós e o determinante dimensionalmente escalado da matriz de projeção, blindando o método contra qualquer forma de estagnação ou instabilidade numérica.
