# Estratégias de Acoplamento Híbrido: VNMM 2D com Elementos Finitos de Aresta (FEM 2D)

**Autor:** Antigravity (Google DeepMind) & Equipe do Projeto VNMM  
**Contexto:** Formulação Híbrida Sem Malha / Elementos Finitos para Problemas Eletromagnéticos de Autovalores 2D ($TE_z$)  
**Subdomínios:** $\Omega = \Omega_{\text{FEM}} \cup \Omega_{\text{VNMM}}$, com interface de acoplamento $\Gamma_{\text{int}} = \overline{\Omega}_{\text{FEM}} \cap \overline{\Omega}_{\text{VNMM}}$.

---

## 1. Visão Geral e Motivação do Acoplamento Híbrido

O acoplamento entre o **Método Sem Malha Nodal Vetorial (VNMM 2D)** e o **Método dos Elementos Finitos de Aresta de Nédélec (FEM 2D)** combina as vantagens complementares de cada método:

```
        +-----------------------------------+-----------------------------------+
        |                                   |  •       •       •       •        |
        |         Subdomínio FEM            |      •       •       •       •    |
        |         $\Omega_{\text{FEM}}$     |          Subdomínio VNMM          |
        |       (Malha Triangular)          |       $\Omega_{\text{VNMM}}$          |
        |         /\  /\  /\  /\            |        (Nuvem de Nós)             |
        |        /__\/__\/__\/__\           |  •       •       •       •        |
        +-----------------------------------+-----------------------------------+
                                      Interface $\Gamma_{\text{int}}$
                          (Arestas de FEM $\leftrightarrow$ Nós de VNMM)
```

1. **No Subdomínio FEM ($\Omega_{\text{FEM}}$):** Utiliza-se a conformidade estrita em $H(\text{curl})$ proporcionada pelas 1-formas de Whitney em triângulos, ideal para regiões com geometrias intrincadas, cantos vivos ou fortes descontinuidades de permissividade/permeabilidade.
2. **No Subdomínio VNMM ($\Omega_{\text{VNMM}}$):** Utiliza-se a flexibilidade sem malha (*meshless*) com funções de base polinomiais completas $\mathcal{P}^1$ (6 nós de suporte), ideal para regiões homogêneas volumosas, espaço livre, partes móveis ou refinamentos adaptativos nodais sem custo de remalhamento.

---

## 2. Compatibilidade Física e Dimensional na Interface $\Gamma_{\text{int}}$

### 2.1 Natureza dos Graus de Liberdade
- **Grau de Liberdade do FEM de Aresta ($e_k$):**  
  Representa a **circulação tangencial do campo elétrico ao longo da aresta $E_k$**:
  $$e_k = \int_{E_k} \vec{E} \cdot d\vec{\ell} \quad [\text{Unidade: Volts (V)}]$$
  Para um elemento linear de 1ª ordem, o campo elétrico tangencial ao longo de uma aresta de comprimento $\ell_k$ é constante e dado por:
  $$E_{\text{FEM}, t} = \frac{e_k}{\ell_k} \quad [\text{Unidade: V/m}]$$

- **Grau de Liberdade do VNMM 2D ($c_i$):**  
  Representa a **projeção pontual do vetor campo elétrico na direção do vetor unitário $\vec{t}_i$**:
  $$c_i = \vec{E}(\mathbf{x}_i) \cdot \vec{t}_i \quad [\text{Unidade: Volts/metro (V/m)}]$$

### 2.2 Relação de Conversão e Compatibilidade Vetorial
Na interface $\Gamma_{\text{int}}$, para que a continuidade tangencial $\vec{E}_{\text{FEM}} \cdot \vec{t} = \vec{E}_{\text{VNMM}} \cdot \vec{t}$ seja satisfeita de forma dimensional e fisicamente correta:
1. **Sentido Vetorial:** O vetor unitário do nó VNMM $\vec{t}_i$ deve possuir **exatamente o mesmo sentido e direção** do vetor diretor da aresta orientada no FEM ($\vec{t}_i = \vec{t}_{\text{aresta}, k}$).
2. **Fator de Escala Dimensional:** O grau de liberdade nodal do VNMM no ponto médio da aresta deve ser igual ao grau de liberdade de circulação do FEM **dividido pelo comprimento da aresta $\ell_k$**:
   $$c_i = \frac{e_k}{\ell_k} \iff e_k = \ell_k \cdot c_i$$

---

## 3. Detalhamento das Quatro Estratégias de Acoplamento

---

### Estratégia 1: Acoplamento Direto Conforme (Master-Slave / Nós de Aresta) — *(Recomendada)*

```
       Triângulo FEM                  Interface \Gamma_{int}                Nó VNMM
            / \                                 |
           /   \                                |
          /  e  \                               |
         +-------+ ----------------------- [Nó i] (t_i // aresta) ---- •  •  • (VNMM)
       Vértice 1  Vértice 2                 Aresta E_k
```

#### Princípio:
- Os nós de contorno do VNMM na interface $\Gamma_{\text{int}}$ são posicionados exatamente nos **pontos médios das arestas de contorno da malha FEM**.
- O vetor diretor do nó é definido como $\vec{t}_i = \frac{\mathbf{x}_2 - \mathbf{x}_1}{\ell_k}$ (alinhado à aresta).
- Define-se a matriz de transformação diagonal de interface $\mathbf{T} \in \mathbb{R}^{N_\Gamma \times N_\Gamma}$:
  $$\mathbf{T}_{kk} = \frac{1}{\ell_k}$$
  tal que $\mathbf{c}_\Gamma = \mathbf{T} \mathbf{e}_\Gamma$.

#### Sistema Global Acoplado:
As incógnitas mestras globais são $\mathbf{u} = [\mathbf{e}_{\text{FEM, int}}^T, \mathbf{e}_\Gamma^T, \mathbf{c}_{\text{VNMM, int}}^T]^T$, e as matrizes globais de rigidez e massa assumem a estrutura simétrica:

$$
K_{\text{híbrido}} = \begin{bmatrix}
K_{\text{FEM}, ii} & K_{\text{FEM}, i\Gamma} & 0 \\
K_{\text{FEM}, \Gamma i} & K_{\text{FEM}, \Gamma\Gamma} + \mathbf{T}^T K_{\text{VNMM}, \Gamma\Gamma} \mathbf{T} & \mathbf{T}^T K_{\text{VNMM}, \Gamma v} \\
0 & K_{\text{VNMM}, v\Gamma} \mathbf{T} & K_{\text{VNMM}, vv}
\end{bmatrix}
$$

$$
M_{\text{híbrido}} = \begin{bmatrix}
M_{\text{FEM}, ii} & M_{\text{FEM}, i\Gamma} & 0 \\
M_{\text{FEM}, \Gamma i} & M_{\text{FEM}, \Gamma\Gamma} + \mathbf{T}^T M_{\text{VNMM}, \Gamma\Gamma} \mathbf{T} & \mathbf{T}^T M_{\text{VNMM}, \Gamma v} \\
0 & M_{\text{VNMM}, v\Gamma} \mathbf{T} & M_{\text{VNMM}, vv}
\end{bmatrix}
$$

#### Vantagens:
- **Simetria e Positividade Definida:** Preserva integralmente a estrutura definida positiva do problema generalizado de autovalores.
- **Eficiência Computacional Máxima:** Sem variáveis extras nem multiplicadores de Lagrange.

---

### Estratégia 2: Método de Nitsche (Penalização Variacional Consistente)

Indicada para situações onde a malha de elementos finitos e os nós do VNMM são **totalmente não-conformes** na interface $\Gamma_{\text{int}}$.

#### Forma Fraca Variacional:
$$
a_{\text{híbrido}}(\vec{W}, \vec{E}) = a_{\text{FEM}}(\vec{W}, \vec{E}) + a_{\text{VNMM}}(\vec{W}, \vec{E})
- \int_{\Gamma_{\text{int}}} \left( \{\text{fluxo}(\vec{W})\} [[\vec{E}_t]] + \{\text{fluxo}(\vec{E})\} [[\vec{W}_t]] \right) d\Gamma
+ \frac{\gamma}{h_{\text{int}}} \int_{\Gamma_{\text{int}}} [[\vec{W}_t]] [[\vec{E}_t]] \, d\Gamma
$$
onde $[[\vec{E}_t]] = \vec{E}_{\text{FEM}, t} - \vec{E}_{\text{VNMM}, t}$ é o salto tangencial e $\gamma > 0$ é o parâmetro de Nitsche.

#### Vantagens e Desvantagens:
- **Vantagem:** Malhas 100% independentes em cada subdomínio.
- **Desvantagem:** Exige calibrar o parâmetro $\gamma$ para garantir estabilidade coerciva.

---

### Estratégia 3: Método dos Multiplicadores de Lagrange (Método Mortar)

Introduz um campo de multiplicadores de Lagrange $\boldsymbol{\lambda}(s)$ ao longo de $\Gamma_{\text{int}}$ representando a corrente magnética de superfície:

$$
\begin{bmatrix}
K_{\text{FEM}} & 0 & B_{\text{FEM}}^T \\
0 & K_{\text{VNMM}} & B_{\text{VNMM}}^T \\
B_{\text{FEM}} & B_{\text{VNMM}} & 0
\end{bmatrix}
\begin{bmatrix} \mathbf{e} \\ \mathbf{c} \\ \boldsymbol{\lambda} \end{bmatrix}
= \lambda
\begin{bmatrix}
M_{\text{FEM}} & 0 & 0 \\
0 & M_{\text{VNMM}} & 0 \\
0 & 0 & 0
\end{bmatrix}
\begin{bmatrix} \mathbf{e} \\ \mathbf{c} \\ \boldsymbol{\lambda} \end{bmatrix}
$$

#### Vantagens e Desvantagens:
- **Vantagem:** Imposição da continuidade no sentido $L^2$ exato.
- **Desvantagem:** Gera um sistema de ponto de sela indefinido com autovalores infinitos espúrios.

---

### Estratégia 4: Zona de Transição com Partição da Unidade (*Blending*)

Define-se uma região de sobreposição $\Omega_{\text{trans}}$ de espessura $\delta$ onde os campos são combinados por uma função suave de rampa $\alpha(x) \in [0, 1]$:
$$\vec{E}(x, y) = \alpha(x) \vec{E}_{\text{FEM}}(x, y) + (1 - \alpha(x)) \vec{E}_{\text{VNMM}}(x, y)$$

---

## 4. Quadro Comparativo das Estratégias

| Estratégia | Conformidade de Malha | Tipo de Matriz Global | Simetria | Espectro Livre de Infinitos | Complexidade |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Acoplamento Direto Conforme** | **Conforme na interface** | **Definida Positiva** | **Sim** | **Sim (100% limpo)** | **Baixa** |
| **2. Método de Nitsche** | Não-conforme | Definida Positiva | Sim | Sim | Média |
| **3. Multiplicadores de Lagrange** | Não-conforme | Ponto de Sela | Sim (Indefinida) | Não (gera $\lambda=\infty$) | Alta |
| **4. Zona de Transição (*Blending*)** | Sobreposição | Definida Positiva | Sim | Sim | Média |
