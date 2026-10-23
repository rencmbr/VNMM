# Plano de Implementação: Base Completa $\mathcal{P}^1$ (6 Termos) no VNMM 2D

Este documento detalha o plano de desenvolvimento e validação da formulação do Método Sem Malha Nodal Vetorial (VNMM 2D) utilizando a **base polinomial vetorial linear completa $\mathcal{P}^1$ (6 termos)** com **6 nós de suporte por domínio de colocação**.

O objetivo principal desta implementação é **eliminar o vazamento modal (*aliasing*)** diagnosticado na base $\mathcal{L}^1$ (3 nós), alcançando **convergência de 2ª ordem $O(h^2)$ no campo vetorial $\vec{E}$** e **convergência de 1ª ordem $O(h)$ no rotacional $\nabla \times \vec{E}$**.

---

## 1. Fundamentação Matemática e Escala Dimensional

### 1.1 Base Vetorial e Condição de Projeção
A base polinomial linear completa no espaço $\mathcal{P}_1 \times \mathcal{P}_1$ é definida por:

$$
\mathcal{P}^1 = \left\langle \begin{bmatrix} 1 \\\\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\\\ 1 \end{bmatrix}, \begin{bmatrix} x \\\\ 0 \end{bmatrix}, \begin{bmatrix} y \\\\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\\\ x \end{bmatrix}, \begin{bmatrix} 0 \\\\ y \end{bmatrix} \right\rangle
$$

A função de forma associada ao $i$-ésimo nó de suporte em coordenadas locais $(\Delta x = x - x_P, \Delta y = y - y_P)$ em torno do ponto de avaliação $P$ é:

$$
\vec{N}_i(x, y) = \beta_{1i} \begin{bmatrix} 1 \\\\ 0 \end{bmatrix} + \beta_{2i} \begin{bmatrix} 0 \\\\ 1 \end{bmatrix} + \beta_{3i} \begin{bmatrix} x \\\\ 0 \end{bmatrix} + \beta_{4i} \begin{bmatrix} y \\\\ 0 \end{bmatrix} + \beta_{5i} \begin{bmatrix} 0 \\\\ x \end{bmatrix} + \beta_{6i} \begin{bmatrix} 0 \\\\ y \end{bmatrix}
$$

Impondo a condição de colocação nodal nos 6 nós de suporte:

$$
\vec{N}_i(\mathbf{x}_k) \cdot \vec{t}_k = \delta_{ik}, \quad \text{para } i, k \in \{1, 2, \dots, 6\}
$$

obtém-se o sistema linear:

$$
A \beta = I_6 \implies \beta = A^{-1}
$$

onde a matriz de colocação $A \in \mathbb{R}^{6 \times 6}$ é expressa explicitamente por:

$$
A = \begin{bmatrix}
t_{1x} & t_{1y} & \Delta x_1 t_{1x} & \Delta y_1 t_{1x} & \Delta x_1 t_{1y} & \Delta y_1 t_{1y} \\\\
t_{2x} & t_{2y} & \Delta x_2 t_{2x} & \Delta y_2 t_{2x} & \Delta x_2 t_{2y} & \Delta y_2 t_{2y} \\\\
t_{3x} & t_{3y} & \Delta x_3 t_{3x} & \Delta y_3 t_{3x} & \Delta x_3 t_{3y} & \Delta y_3 t_{3y} \\\\
t_{4x} & t_{4y} & \Delta x_4 t_{4x} & \Delta y_4 t_{4x} & \Delta x_4 t_{4y} & \Delta y_4 t_{4y} \\\\
t_{5x} & t_{5y} & \Delta x_5 t_{5x} & \Delta y_5 t_{5x} & \Delta x_5 t_{5y} & \Delta y_5 t_{5y} \\\\
t_{6x} & t_{6y} & \Delta x_6 t_{6x} & \Delta y_6 t_{6x} & \Delta x_6 t_{6y} & \Delta y_6 t_{6y}
\end{bmatrix}
$$

### 1.2 Análise Dimensional e Lei de Escala do Determinante $Tol_{det}(h) \propto h^4$
- As **colunas 1 e 2** de $A$ contêm apenas $(t_{kx}, t_{ky})$, sendo adimensionais ($O(1)$).
- As **colunas 3, 4, 5 e 6** contêm produtos $\Delta x_k t_{kx}$, $\Delta y_k t_{kx}$, etc., possuindo dimensão de comprimento ($O(h)$).
- Portanto, o determinante escala como:
  $$
  \det(A_{6 \times 6}) \sim O(1)^2 \cdot O(h)^4 = \mathbf{O(h^4)}
  $$

Consequentemente, para manter o número de vizinhos efetivos $K_{méd}$ invariante com o adensamento da malha, a tolerância adaptativa deve seguir a **lei quártica**:

$$
Tol_{det}(h) = Tol_{ref} \cdot \left(\frac{h}{h_{ref}}\right)^4
$$

### 1.3 Avaliação no Ponto $P$ ($\Delta x = 0, \Delta y = 0$)

#### 1. Funções de forma no ponto $P$:

$$
\vec{N}_i(P) = \begin{bmatrix} \beta_{1i} \\\\ \beta_{2i} \end{bmatrix}, \quad \Phi(P) = \beta[0:2, :]
$$

#### 2. Rotacional das funções de forma:

$$
\nabla \times \vec{N}_i(P) = \begin{bmatrix} 0 \\\\ 0 \\\\ \beta_{5i} - \beta_{4i} \end{bmatrix}, \quad \nabla \times \Phi(P) = \beta[4, :] - \beta[3, :]
$$

#### 3. Campo vetorial interpolado:

$$
\vec{E}^h(P) = \Phi(P) e_s, \quad \text{com } e_s = \begin{bmatrix} e_1 \\\\ \vdots \\\\ e_6 \end{bmatrix}
$$

#### 4. Rotacional interpolado:

$$
\nabla \times \vec{E}^h(P) = \begin{bmatrix} 0 \\\\ 0 \\\\ \sum_{i=1}^6 (\beta_{5i} - \beta_{4i}) e_i \end{bmatrix}
$$

---

## 2. Componentes e Arquitetura de Módulos

Seguindo o padrão de nomenclatura estrita do repositório, criaremos os seguintes módulos dentro do diretório `codigo/`:

```
teste-cod/
├── codigo/
│   ├── nos_suporte_vnmm_2d_6_P1.py      # Seleção adaptativa de 6 nós de suporte
│   ├── funcoes_forma_vnmm_2d_6_P1.py    # Cálculo de Phi(P), rot_Phi(P) e beta (6 nós)
│   ├── avaliar_grade_pontos_6_P1.py     # Avaliador de grade para a formulação 6-P1
│   ├── interpolacao_6_P1.py             # Script de validação em malha única densa
│   └── analise_parametrica_6_P1.py      # Análise paramétrica completa (Tol e Densidade)
├── malhas/                              # Arquivos de dados de malha (.csv) e imagens (.png)
└── relatorios/                          # Gráficos de convergência e relatórios técnicos (.md)
    └── relatorio_analise_parametrica_P1.md
```

---

## 3. Detalhamento das Etapas de Implementação

### Etapa 1: Módulo de Seleção de 6 Nós de Suporte (`codigo/nos_suporte_vnmm_2d_6_P1.py`)
- **Entradas:** Ponto $P$, coordenadas globais dos nós, vetores unitários, árvore KDTree, $K_{ini}=12$, $Tol_{det}$, `adaptativo=True`, `passo_K=4`, `K_max`.
- **Estratégia de Busca:**
  1. Consulta a KDTree para obter os $K$ nós mais próximos de $P$.
  2. Itera sobre combinações de 6 nós (iniciando com o nó mais próximo como âncora e variando os 5 nós subsequentes).
  3. Monta a matriz $A \in \mathbb{R}^{6 \times 6}$ de forma vetorizada via NumPy.
  4. Calcula $|\det(A)|$.
  5. Se $|\det(A)| \ge Tol_{det}$, aceita imediatamente o sexteto (*early stopping*).
  6. Caso contrário, registra o melhor sexteto encontrado e, se `adaptativo=True`, expande a vizinhança $K \gets K + \text{passo}$ (com `passo_K=4`) até encontrar um sexteto válido ou atingir `K_max`.
- **Retorno:** `(sexteto_indices, det_A, matriz_A, k_efetivo)`.

### Etapa 2: Módulo de Funções de Forma (`codigo/funcoes_forma_vnmm_2d_6_P1.py`)
- **Entradas:** Ponto $P$, coordenadas, vetores, lista dos 6 nós selecionados, matriz $A$ pré-calculada (opcional).
- **Cálculo:**
  1. Monta a matriz $A_{6 \times 6}$ em coordenadas locais $\Delta x_k = x_k - x_P, \Delta y_k = y_k - y_P$.
  2. Inverte o sistema: $\beta = A^{-1}$.
  3. Extrai $\Phi(P) = \beta[0:2, :]$ (dimensão $2 \times 6$).
  4. Calcula o rotacional $\nabla \times \Phi(P) = \beta[4, :] - \beta[3, :]$ (dimensão 6).
- **Retorno:** `(Phi, rot_Phi, beta)`.

### Etapa 3: Avaliação em Grade e Teste na Malha Densa (`codigo/interpolacao_6_P1.py`)
- **Cenário de Teste:** Cavidade PEC bidimensional de $20 \times 20\text{ m}$ com modo analítico $\text{TE}_{11}$.
- **Malha Densa:** $N=1928$ nós ($128$ nós de contorno e $1800$ nós internos), espaçamento característico $h \approx 0.625\text{ m}$.
- **Grade de Avaliação:** $15 \times 15 = 225$ pontos uniformemente distribuídos em $[-9.0, 9.0] \times [-9.0, 9.0]\text{ m}$.
- **Métricas:** Erro RMS e Máximo de $\vec{E}^h$, Erro RMS e Máximo de $\nabla \times \vec{E}^h$, taxa de sucesso, $\vert\det(A)\vert_{méd}$ e $K_{méd}$.
- **Comparação direta:** Comparar os resultados na mesma malha densa com a formulação 3-L1.

### Etapa 4: Análise Paramétrica Completa (`codigo/analise_parametrica_6_P1.py`)
Execução de dois estudos comparativos sistemáticos:

#### Estudo 1: Varredura de Tolerância do Determinante ($Tol_{det}$)
- Fixar a malha intermediária ($N=416$) e variar $Tol_{det}$ em uma faixa representativa de valores calibrados para $O(h^4)$ (ex.: $10^{-5}$ a $1.0$).
- Avaliar o impacto na taxa de sucesso (100%), qualidade do condicionamento e $K_{efetivo}$.

#### Estudo 2: Varredura de Densidade de Nós com Escala Quártica $Tol_{det}(h) \propto h^4$
- Avaliar a convergência nas 7 configurações padrão de malha:
  1. Esparsa: $N=84$ nós ($h \approx 3.33$)
  2. Média-Esparsa: $N=186$ nós ($h \approx 2.22$)
  3. Média: $N=416$ nós ($h \approx 1.43$)
  4. Média-Densa: $N=884$ nós ($h \approx 0.95$)
  5. Densa: $N=1928$ nós ($h \approx 0.625$)
  6. Muito Densa: $N=4192$ nós ($h \approx 0.417$)
  7. Ultra Densa: $N=8408$ nós ($h \approx 0.217$)
- **Calibração da Tolerância:** $Tol_{det}(h) = Tol_{ref} \cdot (h / h_{ref})^4$.
- **Geração de Gráficos e Relatório:**
  - Gráfico log-log de convergência do campo vetorial $\vec{E}$ (ordem esperada $O(h^2)$).
  - Gráfico log-log de convergência do rotacional $\nabla \times \vec{E}$ (ordem esperada $O(h)$).
  - Gráfico comparativo de convergência: Base $\mathcal{L}^1$ vs. Base $\mathcal{P}^1$.
  - Relatório técnico estruturado em Markdown (`relatorios/relatorio_analise_parametrica_P1.md`).

---

## 4. Plano de Verificação

### Testes Automatizados e Execuções de Validação
1. **Teste Unitário dos Módulos 6-P1:**
   - Executar `interpolacao_6_P1.py` na malha densa ($N=1928$).
   - Verificar: 100% de sucesso nos 225 pontos de avaliação, $|\det(A)| > 0$, condicionamento estável.
2. **Teste da Lei de Escala Quártica:**
   - Verificar que $K_{méd}$ se mantém estável na faixa de $8 - 12$ nós para todas as 7 malhas (de $N=84$ a $N=8408$).
3. **Validação Assintótica de Convergência:**
   - Calcular as inclinações das curvas log-log:
     - Inclinação do erro de $\vec{E}$: deve ser próxima de **2.0** ($O(h^2)$).
     - Inclinação do erro de $\nabla \times \vec{E}$: deve ser próxima de **1.0** ($O(h)$), superando a estagnação de ordem $0.0$ observada na base $\mathcal{L}^1$.
