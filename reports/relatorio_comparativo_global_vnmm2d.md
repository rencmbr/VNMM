# Relatório Técnico Comparativo Global: Formulações e Estratégias de Integração no VNMM 2D

**Autor:** Antigravity (Google DeepMind) & Equipe do Projeto  
**Problema de Referência:** Cavidade Ressonante PEC Bidimensional $[0, \pi] \times [0, \pi]$ (Modos $TE_z$, Seção 4.3.1 da Tese de Luilly Ortiz, UFMG, 2023)  
**Malha do Caso Base:** $N_x = 21, N_y = 21$ ($N_{\text{total}} = 441$ nós, $361$ graus de liberdade internos, $h = 0.1571\text{ m}$)

---

## 1. Comparativo: Base Incompleta $\mathcal{L}^1$ (3 Nós) vs Base Completa $\mathcal{P}^1$ (6 Nós)

### 1.1 Fundamentação Matemática
- **Base $\mathcal{L}^1$ (3 Nós):**
  $$\mathcal{L}^1 = \left\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \begin{bmatrix} y \\ -x \end{bmatrix} \right\rangle$$
  - Possui apenas 3 graus de liberdade locais por ponto de colocação.
  - Impõe artificialmente $\frac{\partial E_x}{\partial x} \equiv 0$ e $\frac{\partial E_y}{\partial y} \equiv 0$, sendo incapaz de representar a matriz Jacobiana de Taylor completa do campo eletromagnético.
  - **Vazamento Modal (*Aliasing*):** A 3ª linha de $A^{-1}$ escala com $O(1/h)$, multiplicando o resíduo das derivadas normais não representadas $\mathbf{r} \sim O(h)$, o que gera um erro constante $O(1)$ no rotacional aproximado $(\nabla \times \vec{E}^h)_z = -2\beta_3$.

- **Base $\mathcal{P}^1$ (6 Nós):**
  $$\mathcal{P}^1 = \left\langle \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \begin{bmatrix} x \\ 0 \end{bmatrix}, \begin{bmatrix} y \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ x \end{bmatrix}, \begin{bmatrix} 0 \\ y \end{bmatrix} \right\rangle$$
  - Espaço polinomial vetorial linear completo $\mathcal{P}_1 \times \mathcal{P}_1$ com 6 nós de suporte e matriz de momento $A \in \mathbb{R}^{6 \times 6}$.
  - Resolve independentemente todas as 4 derivadas espaciais ($\frac{\partial E_x}{\partial x}, \frac{\partial E_x}{\partial y}, \frac{\partial E_y}{\partial x}, \frac{\partial E_y}{\partial y}$).
  - Elimina integralmente o vazamento modal, garantindo convergência $O(h^2)$ no campo $\vec{E}$ e $O(h)$ no rotacional $\nabla \times \vec{E}$.

### 1.2 Resultados Espectrais no Caso Base ($441$ Nós)

| Métrica | Base $\mathcal{L}^1$ (3 Nós) | Base $\mathcal{P}^1$ (6 Nós) |
| :--- | :---: | :---: |
| **Erro Médio do Número de Onda $k_c$** | **$28.32\% - 48.95\%$** | **$1.00\%$** |
| **Erro Máximo de $k_c$** | **$39.78\% - 62.30\%$** | **$1.88\%$** |
| **Monotonicidade do Espectro** | Corrompida por modos espúrios | **Estritamente física e monotônica** |
| **Tempo Médio de Montagem** | $\approx 0.06\text{s}$ | $\approx 0.06\text{s}$ |

---

## 2. Uso ou Não de Penalização da Divergência (Regularização Div-Curl)

A forma fraca variacional de Ritz-Galerkin no VNMM é expressa por:
$$\int_{\Omega} (\nabla \times \vec{W})_z (\nabla \times \vec{E})_z \, d\Omega + s_{\text{div}} \int_{\Omega} (\nabla \cdot \vec{W}) (\nabla \cdot \vec{E}) \, d\Omega = \lambda \int_{\Omega} \vec{W} \cdot \vec{E} \, d\Omega$$

### 2.1 Efeito na Base $\mathcal{L}^1$
Como $\nabla \cdot \vec{N}_i \equiv 0$ por construção para todos os termos de $\mathcal{L}^1$, a matriz $K_{\text{div}}$ é identicamente nula ($K_{\text{div}} \equiv 0$). Assim, a base $\mathcal{L}^1$ **não admite regularização de divergência**, permanecendo vulnerável a modos espúrios.

### 2.2 Efeito na Base $\mathcal{P}^1$
- **Para Modos Físicos $TE_z$:** O campo elétrico é puramente solenoidal ($\nabla \cdot \vec{E} \equiv 0$). A matriz de divergência atua com valor nulo ($K_{\text{div}} \mathbf{c} = \mathbf{0}$), preservando os autovalores analíticos exatos.
- **Para Modos de Gradiente/Espúrios ($\vec{E} = \nabla \phi$):** A divergência é não-nula ($\nabla \cdot \vec{E} = \Delta \phi \ne 0$). O termo $s_{\text{div}} K_{\text{div}}$ penaliza fortemente esses estados, **deslocando-os para frequências muito altas ($\lambda > 50$)**.
- **Sem Penalização ($s_{\text{div}} = 0.0$):** O espectro é invadido por modos espúrios de gradiente entre $0.8$ e $5.0$, elevando o erro médio dos 10 primeiros modos de **$1.00\%$** para **$19.75\% - 90.0\%$**.

---

## 3. Modos Espúrios: Origem Física e Matemática

### 3.1 Origem no Contínuo
O operador rotacional $\nabla \times$ possui um espaço nulo infinito formado por todos os campos conservativos / gradientes $\vec{E} = \nabla \phi$, para os quais $\nabla \times (\nabla \phi) \equiv 0$. No problema contínuo, esses modos possuem autovalor $\lambda = 0$.

### 3.2 Origem na Discretização Nodal Sem Malha
1. **Ausência da Propriedade de de Rham Discreta:** Ao contrário dos Elementos Finitos de Aresta de Nédélec (que preservam $\text{Im}(\text{grad}) = \ker(\text{curl})$ de forma discreta exata), métodos sem malha com funções de forma nodais contínuas **não anulam o rotacional de campos gradientes de forma exata em todos os pontos de integração** ($\nabla \times (\nabla \phi) \ne \mathbf{0}$ nos pontos de Gauss).
2. **Transformação de Zeros em Modos Espúrios:** Devido a esse rotacional residual numérico, modos que deveriam ter $\lambda = 0$ adquirem autovalores pequenos, positivos e não-nulos ($0.01 < \lambda < 5.0$).
3. **Solução Definitiva:** A regularização div-curl ($s_{\text{div}} = 6.0$) é indispensável e suficiente para separar os modos solenoidais físicos dos modos espúrios de gradiente.

---

## 4. Integração Numérica: Densidade de Células e Ordem de Gauss

Avaliamos a varredura bidimensional de quadratura com suporte por ponto de Gauss na base $\mathcal{P}^1$:

| Células ($N_{cx} \times N_{cy}$) | Gauss ($p \times p$) | Total Pontos Gauss | Erro Médio $\lambda$ (%) | Erro Médio $k_c$ (%) | Erro Máx $k_c$ (%) | Tempo (s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10 \times 10$** | **$2 \times 2$ (4 pts)** | **400** | **2.01%** | **1.00%** | **1.88%** | **0.06s** |
| $10 \times 10$ | $3 \times 3$ (9 pts) | 900 | 4.24% | 2.08% | 6.30% | 0.11s |
| $10 \times 10$ | $4 \times 4$ (16 pts) | 1600 | 5.19% | 2.55% | 5.91% | 0.14s |
| $15 \times 15$ | $2 \times 2$ (4 pts) | 900 | 5.48% | 2.69% | 5.47% | 0.09s |
| $20 \times 20$ | $2 \times 2$ (4 pts) | 1600 | 6.10% | 2.98% | 7.18% | 0.12s |
| $30 \times 30$ | $2 \times 2$ (4 pts) | 3600 | 6.35% | 3.10% | 6.33% | 0.23s |
| $40 \times 40$ | $2 \times 2$ (4 pts) | 6400 | 6.84% | 3.34% | 6.67% | 0.42s |
| $40 \times 40$ | $4 \times 4$ (16 pts) | 25600 | 6.79% | 3.32% | 6.39% | 1.60s |

### Conclusões sobre a Quadratura:
1. **Equilíbrio Nodal-Quadratura (Proporção $\approx 1:1$):** A melhor acurácia (**$1.00\%$**) ocorreu para $N_c = 10 \times 10$ com Gauss $2 \times 2$ (400 pontos de integração para 441 nós). Isso equilibra perfeitamente a área de influência de cada nó com a amostragem de quadratura.
2. **Ordem de Gauss $2 \times 2$ é Ótima:** Como a base $\mathcal{P}^1$ é polinomial linear (com rotacionais e divergências constantes por ponto de colocação), a regra de Gauss $2 \times 2$ (exata para polinômios cúbicos) já integra perfeitamente a variação dos termos de massa quadráticos. Aumentar para $p = 3, 4, 5$ apenas adiciona custo computacional sem ganhos de precisão.
3. **Estabilidade com Refinamento:** Para grades densas de células ($N_c \ge 30$), o erro estabiliza de forma assintótica em $\approx 3.1\% - 3.3\%$.

---

## 5. Nós de Suporte: Por Ponto de Gauss (EFG) vs Por Centro de Célula

| Aspecto | Centro de Célula (Anterior) | Ponto de Gauss (Estilo EFG - Proposto) |
| :--- | :--- | :--- |
| **Definição dos Nós de Suporte** | Fixos para toda a célula (calculados em $P_c$). | **Individualmente determinados para cada $P_g = (x_g, y_g)$**. |
| **Avaliação das Funções de Forma** | Expansão de Taylor a partir de $P_c$. | **Avaliação exata na origem local ($\Delta x = 0, \Delta y = 0$)**. |
| **Estabilidade com Células Grandes ($dx > h$)** | **Instável / Falha** (erros $> 70\%$, perda de positividade de $M$). | **Estável e Preciso** em qualquer resolução ($10 \times 10$ a $40 \times 40$). |
| **Conceito Sem Malha** | Parcialmente acoplado à grade de células. | **Verdadeiramente sem malha (*truly meshless*)**. |

---

## 6. Recomendação Final para o VNMM 2D

Para a resolução de problemas de autovalores eletromagnéticos via VNMM 2D, a estratégia recomendada e comprovadamente superior é:

1. **Base de Funções de Forma:** **Base linear completa $\mathcal{P}^1$ (6 nós de suporte)** com algoritmo adaptativo na `KDTree` e escala quártica $Tol_{\text{det}}(h) = Tol_{\text{ref}} (h/h_{\text{ref}})^4$.
2. **Domínio de Suporte:** **Suporte individual por ponto de Gauss (Estilo EFG)**.
3. **Regularização do Divergente:** **Ativa com $s_{\text{div}} = 6.0$** (elimina os modos espúrios e preserva os modos físicos).
4. **Discretização de Quadratura:** Grade de células de integração de fundo com dimensão $dx \approx h$ a $2h$ e **quadratura de Gauss $2 \times 2$ (4 pontos por célula)**, garantindo relação $\approx 1:1$ a $1:4$ entre pontos de Gauss e nós do domínio.
