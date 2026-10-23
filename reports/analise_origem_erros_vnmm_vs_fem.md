# Análise Teórico-Numérica: Origens dos Erros e Diferenças entre o VNMM 2D e o FEM de Aresta

**Autor:** Antigravity (Google DeepMind) & Equipe do Projeto VNMM  
**Contexto:** Estudo Eletromagnético Bidimensional ($TE_z$) em Cavidade Ressonante PEC $[0, \pi]^2$  
**Referência Comparativa:** VNMM 2D (Base Linear Completa $\mathcal{P}^1$) vs. Elementos Finitos de Aresta Triangulares de Nédélec (1ª Ordem / 1-formas de Whitney)

---

## 1. Visão Geral e Motivação

Nos testes comparativos com número equivalente de graus de liberdade (~350 incógnitas ativas), observou-se que:
- O **FEM de Aresta de Nédélec** atinge um erro relativo médio de $k_c$ de **$0.44\%$** (erro máximo de **$1.10\%$**).
- O **VNMM 2D ($\mathcal{P}^1$)** atinge um erro relativo médio de $k_c$ de **$1.00\%$** (erro máximo de **$1.88\%$**), convergindo para **$0.52\%$** com o refinamento nodal.

Esta análise documenta detalhadamente as **razões físico-matemáticas e computacionais** que explicam essa diferença de erro entre as duas formulações.

```
       [FEM de Aresta de Nédélec]                    [VNMM 2D Sem Malha Nodal]
            Aresta contínua                              Nós pontuais
        +-----------------------+                  •                   •
        |  E_t contínuo ao      |                     \ (t_i)         / (t_j)
        |  longo de toda aresta |                       • Ponto de Gauss (x_g, y_g)
        +-----------------------+                  •                   •
```

---

## 2. Os Cinco Fatores Determinantes da Diferença de Erro

### 2.1 Conformidade Estrita em $H(\text{curl})$ vs. Colocação Nodal Pontual

1. **Conformidade no FEM de Aresta:**
   - O grau de liberdade de Nédélec é definido pela circulação ao longo da aresta física $E_k$:
     $$e_k = \int_{E_k} \vec{E} \cdot d\vec{\ell}$$
   - A base de Whitney garante que a componente tangencial do campo elétrico seja **estritamente contínua ao longo de toda a extensão de qualquer interface entre triângulos vizinhos**.
   - O espaço de aproximação é estritamente conforme no espaço de Sobolev $H(\text{curl}; \Omega)$.

2. **Aproximação Pontual no VNMM 2D:**
   - No VNMM, a imposição é feita pontualmente nos nós através da condição de projeção:
     $$\vec{N}_i(\mathbf{x}_k) \cdot \vec{t}_k = \delta_{ik}$$
   - Através do domínio de suporte, a continuidade tangencial entre diferentes pontos de colocação é satisfeita no sentido de aproximação local (via partição de vizinhança da `KDTree`), mas não é identicamente conforme em cada ponto do espaço.
   - Pequenos saltos residuais na componente tangencial ocorrem na transição entre domínios de suporte locais de diferentes pontos de integração de Gauss.

---

### 2.2 Resíduo Numérico da Penalização de Divergência (*Mild Divergence Locking*)

A forma fraca variacional do VNMM é dada por:
$$K = K_{\text{curl}} + s_{\text{div}} K_{\text{div}}$$

1. **No FEM de Aresta:**
   - A discretização preserva a sequência exata de de Rham no nível discreto:
     $$\nabla \times (\nabla \phi_h) \equiv \mathbf{0} \quad \forall \phi_h \in \mathcal{S}_h$$
   - Todos os campos de gradiente $\vec{E} = \nabla \phi$ produzem rotacional identicamente nulo e caem com precisão de máquina no autovalor zero analítico.
   - O método **não necessita de termo de penalização de divergência** ($s_{\text{div}} = 0$). A matriz de rigidez é pura ($K = K_{\text{curl}}$) e não sofre perturbação.

2. **No VNMM 2D:**
   - Para os modos físicos $TE_z$, a divergência analítica é nula ($\nabla \cdot \vec{E} \equiv 0$).
   - No entanto, a base numérica discreta $\mathcal{P}^1$ possui um pequeno resíduo de divergência nos pontos de quadratura ($\nabla \cdot \vec{E}^h \sim O(h)$).
   - Ao multiplicar esse resíduo pelo fator de regularização $s_{\text{div}} = 6.0$, adiciona-se uma **rigidez artificial adicional ao sistema (*mild locking*)**, deslocando levemente os autovalores em relação à solução analítica.

---

### 2.3 Inconsistência da Grade de Integração de Fundo (*Background Cell Integration*)

1. **No FEM de Aresta:**
   - O domínio de definição da função de base coincide perfeitamente com a geometria do triângulo $T_e$.
   - A integração das matrizes elementares $K^e$ e $M^e$ é **exata analiticamente** via fórmulas de coordenadas baricêntricas.

2. **No VNMM 2D:**
   - Como método sem malha de Galerkin, a integração é calculada sobre uma grade de células quadriláteras de fundo cobrindo o domínio $[0, \pi]^2$.
   - Os domínios de suporte dos nós (definidos pelos 6 vizinhos mais próximos na `KDTree`) têm fronteiras poligonais/esféricas que cortam transversalmente as células de fundo.
   - Em cada ponto de Gauss, um conjunto diferente de 6 nós pode ser selecionado, tornando o integrando seccionalmente não-polinomial dentro da célula e gerando um **erro intrínseco de quadratura de fundo**.

---

### 2.4 Representação Escalar Nodal (1 GDL por Nó com Vetor Diretor Arbitrário)

1. **No FEM de Aresta:**
   - Cada aresta possui um vetor tangente naturalmente alinhado com a topologia da malha triangular, capturando as componentes ortogonais do campo com simetria espacial.

2. **No VNMM 2D:**
   - Cada nó interno possui apenas **1 único grau de liberdade escalar** na direção de seu vetor diretor arbitrário $\vec{t}_i$ (ex.: alternando $45^\circ$ e $135^\circ$).
   - A reconstrução das duas componentes do campo vetorial $\vec{E} = [E_x, E_y]^T$ em qualquer ponto depende da inversão da matriz de momento $A_{6 \times 6}$ combinando 6 nós vizinhos.
   - Flutuações locais na distribuição espacial ou angular dos vizinhos afetam o condicionamento de $A$, introduzindo pequenas variações na qualidade da interpolação local.

---

### 2.5 Imposição Contínua vs. Pontual da Condição PEC nas Paredes

1. **No FEM de Aresta:**
   - As arestas de contorno alinham-se exatamente com as paredes físicas da cavidade ($x = 0, \pi$ e $y = 0, \pi$).
   - A condição de Dirichlet homogênea anula o grau de liberdade da aresta, garantindo que a componente tangencial seja identicamente zero ($E_t \equiv 0$) ao longo de **toda a linha contínua da fronteira**.

2. **No VNMM 2D:**
   - A condição de contorno $E_t = 0$ é imposta diretamente nos nós de contorno ($c_k = 0$).
   - Entre dois nós consecutivos de fronteira, o campo tangencial é interpolado pelas funções de forma dos nós vizinhos, permitindo pequenos resíduos de $E_t \ne 0$ nas regiões intermodais da parede.

---

## 3. Síntese Comparativa das Propriedades Numéricas

| Propriedade / Mecanismo | FEM de Aresta de Nédélec | VNMM 2D (Base Linear Completa $\mathcal{P}^1$) |
| :--- | :--- | :--- |
| **Tipo de Conformidade** | Estrita em $H(\text{curl})$ (contínua nas arestas) | Colocação nodal pontual / Aproximada entre suportes |
| **Integração das Matrizes** | Exata analiticamente no triângulo | Quadratura numérica em células de fundo |
| **Eliminação de Modos Espúrios** | Topológica exata via sequência de de Rham | Penalização variacional da divergência ($s_{\text{div}} = 6.0$) |
| **Impacto da Regularização** | Inexistente ($s_{\text{div}} = 0$) | Leve rigidez artificial adicional (*mild locking*) |
| **Condição de Contorno PEC** | $E_t \equiv 0$ contínuo na aresta | $E_t = 0$ pontual nos nós de fronteira |
| **Geração de Malha** | Triangulação conforme obrigatória | **Nuvens de nós arbitrárias sem malha (*meshless*)** |
| **Erro Médio $k_c$ (Caso Base)** | **$0.44\%$** | **$1.00\%$** |
| **Erro Médio $k_c$ (Refinado)** | **$0.16\%$** | **$0.52\%$** |

---

## 4. Conclusão Final: O Trade-off do Método Sem Malha

O erro ligeiramente superior do VNMM 2D em relação ao FEM de aresta reflete o **custo matemático natural da ausência de malha (*meshless trade-off*)**:
1. O FEM de aresta alcança precisão máxima porque explora a estrutura topológica rígida de triângulos e arestas conformes.
2. O VNMM 2D troca essa rigidez topológica por **total flexibilidade geométrica**, operando diretamente sobre pontos dispersos sem gerar nem manter malhas triangulares.
3. Atingir um erro médio de **$1.00\%$** no caso base e **$0.52\%$** na malha refinada demonstra que o VNMM 2D com base $\mathcal{P}^1$ e suporte por ponto de Gauss é um método eletromagnético altamente acurado, competitivo e robusto.
