# Análise do Comportamento Assintótico em Malhas Densas: Formulação VNMM 2D (Base $\mathcal{P}^1$)

Este documento apresenta uma análise detalhada dos mecanismos físico-matemáticos e numéricos que explicam o comportamento das curvas de erro do campo elétrico $\vec{E}$ e de seu rotacional $\nabla \times \vec{E}$ observados na formulação com base completa $\mathcal{P}^1$ (6 nós), em especial a transição entre o regime assintótico ideal ($N \le 1928$) e a desaceleração observada nas malhas muito densa ($N = 4192$) e ultra densa ($N = 8408$).

---

## 1. Contextualização do Fenômeno

Nos estudos de densidade com a base linear completa $\mathcal{P}^1$, observa-se:
1. **Até a Malha Densa ($N = 84 \dots 1928$, com $h = 3.33 \to 0.625\text{ m}$):**
   - O erro RMS do campo $\vec{E}^h$ decresce estritamente com taxa de 2ª ordem $O(h^2)$.
   - O erro RMS do rotacional $\nabla \times \vec{E}^h$ decresce monotonicamente com taxa linear de 1ª ordem $O(h)$.
2. **Nas Malhas Muito Densa e Ultra Densa ($N = 4192$ e $N = 8408$, com $h = 0.417\text{ m}$ e $h = 0.217\text{ m}$):**
   - As taxas de redução do erro sofrem uma atenuação, tendendo a um patamar residual (*plateau*).

A investigação numérica e teórica revelou que esse fenômeno é governado por **três fatores conjugados**.

---

## 2. Mecanismos Físico-Matemáticos e Numéricos

### 2.1 Efeito da Lei de Tolerância Quártica em Baixos Limiares e Subcondicionamento Local de $A$

A matriz de colocação $A \in \mathbb{R}^{6 \times 6}$ em torno do ponto de avaliação $P$ é dada por:

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

Como as colunas 1 e 2 são adimensionais ($O(1)$) e as colunas 3 a 6 possuem dimensão de comprimento ($O(h)$), o determinante escala como:

$$
\det(A_{6 \times 6}) \sim O(h^4)
$$

Adotando a lei de calibração clássica para manter o suporte compacto invariante:

$$
Tol_{det}(h) = Tol_{ref} \cdot \left(\frac{h}{h_{ref}}\right)^4
$$

Para a malha ultra densa ($h = 0.2174\text{ m}$ e $h_{ref} = 2.0\text{ m}$), a tolerância assume o valor:

$$
Tol_{det}(0.2174) = 1.0 \cdot \left(\frac{0.2174}{2.0}\right)^4 \approx 1.40 \times 10^{-4}
$$

**Mecanismo de Degradação:**
- Com um limiar tão baixo ($1.40 \times 10^{-4}$), o critério de parada antecipada aceita o primeiríssimo sexteto candidato formado pelos 6 vizinhos geométricos imediatos ($K_{méd} = 6.1$).
- Em distribuições nodais não-estruturadas, esse sexteto inicial frequentemente apresenta **baixa diversidade angular** (vetores unitários quase paralelos) ou **distribuição espacial quase degenerada**.
- Embora o volume dimensional seja de ordem $h^4$, a matriz $A$ torna-se mal-condicionada, elevando o número de condicionamento:

$$
\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)} > 400
$$

A matriz inversa $\beta = A^{-1}$ passa então a amplificar o erro de truncamento da expansão em série.

---

### 2.2 Amplificação Dimensional $O(1/h)$ nas Derivadas Espaciais

Considere a expansão em série de Taylor do campo exato em torno do ponto $P$:

$$
\vec{E}(\mathbf{x}_k) = \vec{E}(P) + \mathbf{J}(P) \Delta\mathbf{x}_k + \frac{1}{2} \mathbf{H}(P)(\Delta\mathbf{x}_k, \Delta\mathbf{x}_k) + O(h^3)
$$

A projeção nodal no nó $k$ é expressa por:

$$
e_k = \vec{E}(\mathbf{x}_k) \cdot \vec{t}_k = \mathbf{p}(\mathbf{x}_k) \cdot \vec{t}_k + r_k
$$

onde $\mathbf{p}$ é a parte linear e $r_k \sim O(h^2)$ é o resíduo quadrático de truncamento. A solução do sistema linear de colocação resulta em:

$$
\boldsymbol{\beta} = A^{-1} \mathbf{e} = \boldsymbol{\beta}_{exato} + A^{-1} \mathbf{r}
$$

Analisando a estrutura dimensional dos blocos da inversa $A^{-1}$:

#### 1. Linhas 1 e 2 (Campo $\vec{E}(P)$):
Possuem dimensão $O(1)$. Logo:

$$
\|\vec{E}^h(P) - \vec{E}(P)\| \sim O(1) \cdot \|\mathbf{r}\| \sim O(1) \cdot O(h^2) = \mathbf{O(h^2)}
$$

#### 2. Linhas 3 a 6 (Derivadas da Jacobiana e Rotacional):
Como multiplicam termos com dimensão de metros, possuem escala dimensional $O(h^{-1})$. Logo:

$$
\|\nabla \times \vec{E}^h(P) - \nabla \times \vec{E}(P)\| \sim O(h^{-1}) \cdot \|\mathbf{r}\| \sim O(h^{-1}) \cdot O(h^2) = \mathbf{O(h^1)}
$$

Conforme $h \to 0.21\text{ m}$, o operador de derivada $O(h^{-1})$ torna-se **15 vezes mais sensível** a qualquer assimetria ou subcondicionamento local de $A$ do que a interpolação direta do campo.

---

### 2.3 Disparidade de Escala entre a Malha e a Grade Fixa de Avaliação

- A grade regular de avaliação possui espaçamento fixo $\Delta x_{grade} = 2.0\text{ m}$ (100 pontos em $[-9, 9] \times [-9, 9]$).
- Nas malhas esparsas e intermediárias ($h \approx 1.4 \dots 3.3\text{ m}$), o espaçamento característico da malha está na mesma escala da grade de teste.
- Na malha ultra densa ($h = 0.217\text{ m}$), a malha é **10 vezes mais refinada do que a distância entre pontos de teste consecutivos**. Os pontos de teste caem em offsets locais aleatórios em relação aos nós vizinhos, e as flutuações estatísticas locais da orientação vetorial em suportes minúsculos ($r \approx 0.2\text{ m}$) criam uma variância residual que mascara a taxa assintótica pura se o critério de seleção for puramente de vizinhança mínima.

---

## 3. Comprovação Numérica Experimental

Para validar esta análise, a malha ultra densa ($N = 8408$, $h = 0.2174\text{ m}$) foi avaliada variando-se o nível de tolerância $Tol_{det}$:

| Tolerância $Tol_{det}$ | Taxa de Sucesso | $K_{méd}$ Efetivo | Erro RMS $\vec{E}$ | Erro RMS $\nabla\times\vec{E}$ | Diagnóstico |
|:---:|:---:|:---:|:---:|:---:|:---|
| **$1.4 \times 10^{-4}$** (Lei $h^4$) | 100.0% | 6.1 | $1.50 \times 10^{-3}$ | $9.58 \times 10^{-3}$ | Vizinhança mínima com subcondicionamento |
| **$5.0 \times 10^{-4}$** | 100.0% | 6.3 | $9.45 \times 10^{-4}$ | $6.13 \times 10^{-3}$ | Eliminação de nós quase-colineares |
| **$1.0 \times 10^{-3}$** | 100.0% | 6.8 | $7.31 \times 10^{-4}$ | $4.44 \times 10^{-3}$ | Melhora na diversidade angular |
| **$5.0 \times 10^{-3}$** | 100.0% | 9.8 | **$5.99 \times 10^{-4}$** | **$3.46 \times 10^{-3}$** | **Ótimo global (Redução de quase 3x no erro)** |
| **$1.0 \times 10^{-2}$** | 100.0% | 10.9 | $6.51 \times 10^{-4}$ | $3.64 \times 10^{-3}$ | Condicionamento excelente |

### Conclusão dos Testes:
Ao ajustar a tolerância para $Tol_{det} = 5.0 \times 10^{-3}$ (expandindo $K_{méd}$ de $6.1$ para apenas $9.8$ vizinhos), o algoritmo seleciona sextetos com excelente distribuição angular:
- O **erro RMS do campo $\vec{E}$** reduziu de $1.50 \times 10^{-3}$ para **$5.99 \times 10^{-4}$**;
- O **erro RMS do rotacional $\nabla \times \vec{E}$** reduziu de $9.58 \times 10^{-3}$ para **$3.46 \times 10^{-3}$**, restabelecendo a tendência assintótica estrita.

---

## 4. Recomendações e Estratégias de Otimização

#### 1. Adoção de Piso Mínimo de Tolerância (*Tolerance Floor*):
Definir uma lei de tolerância truncada inferiormente:

$$
Tol_{det}(h) = \max\left(Tol_{ref} \cdot \left(\frac{h}{h_{ref}}\right)^4, \, Tol_{piso}\right)
$$

com $Tol_{piso} \approx 10^{-3}$, impedindo que malhas altamente refinadas aceitem sextetos com determinantes excessivamente pequenos.

#### 2. Critério de Qualidade Angular:
Priorizar sextetos cuja matriz de direções:

$$
T_{6 \times 2} = \begin{bmatrix}
t_{1x} & t_{1y} \\\\
t_{2x} & t_{2y} \\\\
t_{3x} & t_{3y} \\\\
t_{4x} & t_{4y} \\\\
t_{5x} & t_{5y} \\\\
t_{6x} & t_{6y}
\end{bmatrix}
$$

apresente cobertura angular abrangente (autovalores balanceados no tensor de orientação local).
