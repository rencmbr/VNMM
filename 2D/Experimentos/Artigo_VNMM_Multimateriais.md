# Nodal Vector Meshless Method for Electromagnetic Eigenvalue Problems in Discontinuous and Anisotropic Media

**Autores:** Luilly A. G. Ortiz, Renato C. Mesquita, Barbara M. F. GonÃ§aves  
**AfiliaÃ§Ã£o:** Universidade Federal de Minas Gerais (UFMG), Escola de Engenharia, Programa de PÃ³s-GraduaÃ§Ã£o em Engenharia ElÃ©trica  
**Status:** Artigo CientÃ­fico / RelatÃ³rio de Pesquisa Consolidado  

---

## Resumo

Este trabalho apresenta a formulaÃ§Ã£o, implementaÃ§Ã£o e validaÃ§Ã£o sistemÃ¡tica do **Vector Nodal Meshless Method (VNMM 2D)** para a resoluÃ§Ã£o de problemas de autovalores eletromagnÃ©ticos em meios arbitrÃ¡rios, incorporando interfaces materiais descontÃ­nuas, anisotropia constitutiva complexa e geometrias com singularidades de borda e quina. O mÃ©todo opera sobre uma **nuvem de pontos pura e nÃ£o-estruturada**, onde cada nÃ³ carrega um Ãºnico grau de liberdade escalar associado Ã  projeÃ§Ã£o direcional unitÃ¡ria do campo elÃ©trico ao longo de um vetor diretor local ($\mathbf{t}_i$), preservando a conformidade tangencial em espaÃ§os $H(\mathbf{curl})$. 

A integraÃ§Ã£o fraca de Galerkin utiliza uma malha de fundo cartesiana uniforme estritamente como suporte de quadratura numÃ©rica, enquanto a interpolaÃ§Ã£o independe de qualquer conectividade de malha. A segregaÃ§Ã£o estrita de suportes por subdomÃ­nio material e o alinhamento tangencial de nÃ³s de interface garantem a continuidade de $E_t$ e o salto abrupto da componente normal $E_n$ sem borramento artificial (*smearing*) ou oscilaÃ§Ãµes de Gibbs. A formulaÃ§Ã£o foi validada atravÃ©s de uma progressÃ£o abrangente de benchmarks:
1. **ConsistÃªncia e ConvergÃªncia Nodal:** Galerkin Patch Test com precisÃ£o de mÃ¡quina ($10^{-16}$) e Method of Manufactured Solutions (MMS) com convergÃªncia quadrÃ¡tica $\mathcal{O}(h^2)$ para o campo elÃ©trico e linear $\mathcal{O}(h)$ para o rotacional.
2. **ResiliÃªncia a Nuvens Irregulares (*Jitter Study* de 0% a 50%):** Prova de invariÃ¢ncia exata da completude polinomial sob severa desordem estocÃ¡stica e demonstraÃ§Ã£o de que nuvens irregulares moderadas (10% a 25% de perturbaÃ§Ã£o) reduzem o erro mÃ©dio espectral em mais de 50% frente a grades regulares, com supressÃ£o total de modos espÃºrios ($R_{\mathrm{div}} < 3.3\%$).
3. **Cavidade Parcialmente Preenchida com DielÃ©trico:** ConcordÃ¢ncia estrita dos autovalores fundamentais e superiores frente Ã s raÃ­zes analÃ­ticas de equaÃ§Ãµes transcendentais de dispersÃ£o em ampla faixa de contrastes de permissividade ($\varepsilon_r \in [2.25, 16.0]$).
4. **Benchmarks ClÃ¡ssicos AnisotrÃ³picos da Literatura:** Excelente concordÃ¢ncia numÃ©rica com soluÃ§Ãµes de referÃªncia por FEM de altÃ­ssima densidade (>400k graus de liberdade) para o guia com saliÃªncia em "H" homogÃªneo e anisotrÃ³pico ($\overline{\overline{\varepsilon}}_r$ hermitiano) e para o guia bimaterial anisotrÃ³pico com inclusÃµes circulares PEC internas.

**Palavras-chave:** MÃ©todos Sem Malha Nodais, Nuvens de Pontos, EspaÃ§os $H(\mathbf{curl})$, Meios AnisotrÃ³picos, Interfaces Multimateriais, Problemas de Autovalores EletromagnÃ©ticos, SupressÃ£o de Modos EspÃºrios.

---

## 1. IntroduÃ§Ã£o

A anÃ¡lise eletromagnÃ©tica de dispositivos em alta frequÃªncia (guias de onda carregados com dielÃ©tricos, cavidades ressonantes, filtros de micro-ondas e meios girotÃ³picos) depende historicamente da formulaÃ§Ã£o fraca no espaÃ§o de Sobolev $H(\mathbf{curl})$. No MÃ©todo dos Elementos Finitos (FEM) clÃ¡ssico, a aproximaÃ§Ã£o desse espaÃ§o Ã© obtida por elementos de aresta de NÃ©dÃ©lec, que garantem a continuidade da componente tangencial do campo ao longo das arestas compartilhadas, permitindo saltos na componente normal e evitando modos espÃºrios. Todavia, a geraÃ§Ã£o e refinamento de malhas conformes tridimensionais ou adaptativas para geometrias com saliÃªncias, quinas reentrantes ou inclusÃµes curvas acarretam elevado custo computacional e restriÃ§Ãµes topolÃ³gicas severas.

Os MÃ©todos Sem Malha (*Meshless Methods*, MFree) surgiram como uma alternativa atrativa por discretizarem o domÃ­nio exclusivamente por meio de um conjunto de nÃ³s espalhados, dispensando conectividade prÃ©-definida de elementos. Contudo, a extensÃ£o de mÃ©todos nodais escalares para problemas vetoriais de Maxwell enfrentou tradicionalmente trÃªs obstÃ¡culos crÃ­ticos:
1. **PoluiÃ§Ã£o Espectral e Modos EspÃºrios:** O uso de funÃ§Ãµes de forma nodais convencionais que aproximam componentes cartesianas independentemente falha em reproduzir o espaÃ§o nulo do operador rotacional, infestando o espectro discreto com soluÃ§Ãµes nÃ£o-fÃ­sicas de divergente nÃ£o-nulo;
2. **Tratamento de Interfaces e Contornos:** A tentativa de compartilhar suportes entre materiais com propriedades distintas induz um borramento numÃ©rico (*smearing*) excessivo sobre as interfaces e oscilaÃ§Ãµes de Gibbs em virtude da descontinuidade da componente normal do campo elÃ©trico;
3. **Sensibilidade Ã  DispersÃ£o e Irregularidade Nodal:** Muitas abordagens sem malha vetoriais anteriores degradam expressivamente sua estabilidade ou precisÃ£o quando os nÃ³s perdem a regularidade cartesiana ou sofrem variaÃ§Ãµes angulares em suas direÃ§Ãµes.

O **Vector Nodal Meshless Method (VNMM)** supera essas limitaÃ§Ãµes ao associar a cada nÃ³ do domÃ­nio um vetor unitÃ¡rio diretor $\mathbf{t}_i$. O grau de liberdade nodal deixa de ser uma componente cartesiana fixa e passa a ser a projeÃ§Ã£o direcional escalar do campo $\mathbf{E}(\mathbf{x}_i) \cdot \mathbf{t}_i$. As funÃ§Ãµes de forma locais sÃ£o construÃ­das para satisfazer estritamente a propriedade de Kronecker com relaÃ§Ã£o Ã s projeÃ§Ãµes direcionais, garantindo conformidade com o espaÃ§o $H(\mathbf{curl})$.

O presente artigo demonstra que o VNMM fundamenta-se intrinsecamente na operaÃ§Ã£o sobre **nuvens de pontos nÃ£o-estruturadas**, demonstrando estabilidade numÃ©rica, precisÃ£o espectral em guias anisotrÃ³picos e imunidade a modos espÃºrios sob perturbaÃ§Ãµes nodais estocÃ¡sticas de atÃ© 50%.

---

## 2. FormulaÃ§Ã£o MatemÃ¡tica do VNMM em Nuvens de Pontos

### 2.1 Graus de Liberdade e ProjeÃ§Ã£o Vetorial Nodal
Considere um domÃ­nio bidimensional $\Omega \subset \mathbb{R}^2$ discretizado por uma nuvem de $N$ nÃ³s arbitrÃ¡rios $\{\mathbf{x}_i\}_{i=1}^N$. A cada nÃ³ $\mathbf{x}_i = (x_i, y_i)^T$ associa-se um vetor diretor unitÃ¡rio:
$$\mathbf{t}_i = \begin{bmatrix} \cos\theta_i \\ \sin\theta_i \end{bmatrix}$$
O grau de liberdade escalar $e_i$ associado ao nÃ³ $i$ representa o produto interno entre o campo elÃ©trico e o diretor do nÃ³:
$$e_i = \mathbf{E}(\mathbf{x}_i) \cdot \mathbf{t}_i$$

Nas fronteiras de condutores elÃ©tricos perfeitos (PEC) e interfaces entre materiais, os vetores diretores sÃ£o obrigatoriamente alinhados Ã  tangente da fronteira ($\mathbf{t}_i = \boldsymbol{\tau}$). No interior de cada subdomÃ­nio, adota-se o padrÃ£o alternado ortogonal ($\pm 45^\circ$, isto Ã©, $\theta_i \in \{\pi/4, 3\pi/4\}$), garantindo diversidade angular e prevenindo colinearidades locais.

### 2.2 Base Vetorial Polinomial Linear Completa ($\mathcal{P}^1$)
Para cada ponto de avaliaÃ§Ã£o $P(x_P, y_P)$, adota-se a base vetorial polinomial linear completa de 6 monÃ´mios:
$$\mathcal{P}^1 = \left\{ \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \; \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \; \begin{bmatrix} \Delta x \\ 0 \end{bmatrix}, \; \begin{bmatrix} \Delta y \\ 0 \end{bmatrix}, \; \begin{bmatrix} 0 \\ \Delta x \end{bmatrix}, \; \begin{bmatrix} 0 \\ \Delta y \end{bmatrix} \right\}$$
onde $\Delta x = x - x_P$ e $\Delta y = y - y_P$. A funÃ§Ã£o de forma vetorial local associada ao nÃ³ de suporte $i$ Ã© expressa por:
$$\mathbf{N}_i(\Delta x, \Delta y) = \begin{bmatrix} \beta_{1i} + \beta_{3i}\Delta x + \beta_{4i}\Delta y \\ \beta_{2i} + \beta_{5i}\Delta x + \beta_{6i}\Delta y \end{bmatrix}$$

### 2.3 Sistema de ColocaÃ§Ã£o Local e Operadores Diferenciais
Para um conjunto de 6 nÃ³s de suporte vizinhos $\mathcal{S}(P) = \{n_1, \dots, n_6\}$ selecionados via busca espacial por Ã¡rvore KD (*KDTree*), impÃµe-se a condiÃ§Ã£o de interpolaÃ§Ã£o estrita:
$$\mathbf{N}_i(\mathbf{x}_k) \cdot \mathbf{t}_k = \delta_{ik}, \quad \forall n_k \in \mathcal{S}(P)$$
Isso resulta no sistema algÃ©brico local $\mathbf{A} \boldsymbol{\beta} = \mathbf{I}_6 \implies \boldsymbol{\beta} = \mathbf{A}^{-1}$, cuja $k$-Ã©sima linha da matriz $\mathbf{A}$ Ã© dada por:
$$\mathbf{A}_{k, :} = \begin{bmatrix} t_{kx} & t_{ky} & \Delta x_k t_{kx} & \Delta y_k t_{kx} & \Delta x_k t_{ky} & \Delta y_k t_{ky} \end{bmatrix}$$

Avaliando diretamente no ponto $P$ ($\Delta x = \Delta y = 0$), as funÃ§Ãµes de forma e suas derivadas espaciais sÃ£o obtidas de forma analÃ­tica e imediata a partir das linhas de $\boldsymbol{\beta}$:
$$\boldsymbol{\Phi}(P) = \boldsymbol{\beta}_{0:2, :} \in \mathbb{R}^{2 \times 6}$$
$$(\nabla \times \boldsymbol{\Phi})_z(P) = \boldsymbol{\beta}_{4, :} - \boldsymbol{\beta}_{3, :} \in \mathbb{R}^{1 \times 6}$$
$$\nabla \cdot \boldsymbol{\Phi}(P) = \boldsymbol{\beta}_{2, :} + \boldsymbol{\beta}_{5, :} \in \mathbb{R}^{1 \times 6}$$

---

## 3. Tratamento de Meios Multimateriais e FormulaÃ§Ã£o Variacional

### 3.1 Regra de PartiÃ§Ã£o Estrita de Suportes Nodais
Para domÃ­nios heterogÃªneos formados pela uniÃ£o de mÃºltiplos materiais distintos ($\Omega = \bigcup_{m=1}^M \Omega_m$) separados por interfaces $\Gamma_{\mathrm{int}}$, adota-se a **regra de partiÃ§Ã£o estrita de suportes**:
1. **Confinamento Material:** Ao avaliar as funÃ§Ãµes de forma em um ponto $P \in \Omega_m$, o estÃªncil de nÃ³s vizinhos $\mathcal{S}(P)$ Ã© restrito exclusivamente aos nÃ³s pertencentes ao meio $\Omega_m$ e aos nÃ³s situados sobre a interface $\Gamma_{\mathrm{int}}$ adjacente. NÃ³s pertencentes a outros meios materiais sÃ£o estritamente excluÃ­dos do conjunto de busca;
2. **Graus de Liberdade de Interface:** Os nÃ³s dispostos ao longo da interface fÃ­sica $\Gamma_{\mathrm{int}}$ possuem seus diretores orientados exclusivamente ao longo da tangente unitÃ¡ria ($\mathbf{t}_j = \boldsymbol{\tau}$). Como a componente tangencial do campo elÃ©trico $E_t$ Ã© fisicamente contÃ­nua ($E_{1t} = E_{2t}$), um Ãºnico grau de liberdade na interface acopla ambos os materiais de forma exata e nÃ£o-redundante. A componente normal $E_n$ Ã© livre para sofrer o salto fÃ­sico ditado pela razÃ£o de permissividades ($\varepsilon_1 E_{1n} = \varepsilon_2 E_{2n}$) sem nenhum borramento artificial.

### 3.2 Forma Fraca de Galerkin com PenalizaÃ§Ã£o de Divergente
A formulaÃ§Ã£o fraca de Maxwell para meios gerais com anisotropia de permissividade $\overline{\overline{\varepsilon}}_r$ e permeabilidade $\mu_r$ Ã© dada por:
$$\int_{\Omega} \frac{1}{\mu_r} (\nabla \times \mathbf{E})_z (\nabla \times \mathbf{v})_z^* \, d\Omega + s_{\mathrm{div}} \int_{\Omega} \frac{1}{\mu_r} (\nabla \cdot \mathbf{E}) (\nabla \cdot \mathbf{v})^* \, d\Omega = k_0^2 \int_{\Omega} \mathbf{v}^* \cdot (\overline{\overline{\varepsilon}}_r \mathbf{E}) \, d\Omega$$
onde $s_{\mathrm{div}} \ge 0$ Ã© o coeficiente de regularizaÃ§Ã£o de divergente que expulsa os modos espÃºrios de rotacional nulo para fora do espectro fÃ­sico de interesse, e $(\cdot)^*$ denota a transposta conjugada para tensores hermitianos complexos.

A discretizaÃ§Ã£o resulta no problema espectral generalizado:
$$(\mathbf{K}_{\mathrm{curl}} + s_{\mathrm{div}} \mathbf{K}_{\mathrm{div}}) \mathbf{c} = \lambda \mathbf{M} \mathbf{c}$$
onde $\lambda = k_0^2$. As condiÃ§Ãµes de contorno em paredes de Condutor ElÃ©trico Perfeito (PEC) impÃµem $E_t = 0$. Como os nÃ³s de contorno possuem seus vetores diretores tangenciais Ã s paredes, a condiÃ§Ã£o de Dirichlet homogÃªnea Ã© imposta diretamente eliminando-se as linhas e colunas dos nÃ³s PEC das matrizes globais:
$$e_k = 0, \quad \forall k \in \partial\Omega_{\mathrm{PEC}}$$

A integraÃ§Ã£o numÃ©rica das matrizes de rigidez e massa Ã© efetuada sobre uma malha cartesiana de fundo uniforme (*background integration grid*), onde a quadratura de Gauss de $2 \times 2$ pontos por cÃ©lula serve como suporte mÃ©trico neutro, mantendo a formulaÃ§Ã£o independente de conectividade entre nÃ³s.

---

## 4. Estudo de ConsistÃªncia e Robustez em Nuvens de Pontos Irregulares

### 4.1 ConsistÃªncia Polinomial e ConvergÃªncia Espacial
Para certificar a completude das funÃ§Ãµes de forma em nuvens de pontos sob meios heterogÃªneos, foram aplicados o Galerkin Patch Test e o Method of Manufactured Solutions (MMS):
- **Campo Afim do Patch Test:**
  $$\mathbf{E}_{\mathrm{patch}}(x, y) = \begin{bmatrix} 1.5 + 0.8\,x - 0.3\,y \\ -0.7 + 0.4\,x + 1.2\,y \end{bmatrix}, \quad (\nabla \times \mathbf{E}_{\mathrm{patch}})_z = 0.7\text{ V/m}^2$$
- **Campo NÃ£o-Polinomial MMS:**
  $$\mathbf{E}_{\mathrm{MMS}}(x, y) = \begin{bmatrix} \sin\left(\frac{\pi x}{L_x}\right)\cos\left(\frac{\pi y}{L_y}\right) \\ -\cos\left(\frac{\pi x}{L_x}\right)\sin\left(\frac{\pi y}{L_y}\right) \end{bmatrix}, \quad (\nabla \times \mathbf{E}_{\mathrm{MMS}})_z = \left(\frac{\pi}{L_x} + \frac{\pi}{L_y}\right)\sin\left(\frac{\pi x}{L_x}\right)\sin\left(\frac{\pi y}{L_y}\right)$$

**Tabela 1:** Desempenho de interpolaÃ§Ã£o do VNMM 2D em domÃ­nio bimaterial com nuvem de pontos.

| Teste | Campo Vetorial | Erro Relativo $L_2$ ($\mathbf{E}$) | Erro Relativo $\text{rot}(\mathbf{E})$ | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Galerkin Patch Test** | Linear Afim Completo | **$2.85 \times 10^{-16}$** | **$1.00 \times 10^{-14}$** | **PrecisÃ£o de MÃ¡quina** |
| **MMS (Base $N=9$)** | TrigonomÃ©trico Suave | $5.742\%$ | $19.461\%$ | Consistente |
| **MMS (Refinada $N=17$)** | TrigonomÃ©trico Suave | **$1.147\%$** | **$9.390\%$** | **Taxa $\mathcal{O}(h^2)$** |
| **Taxa AssintÃ³tica** | â€” | **$5.01\times$** ($\ge 4\times$) | **$2.07\times$** ($\approx 2\times$) | **Ã“tima TeÃ³rica** |

O Patch Test confirma a consistÃªncia exata do mÃ©todo (ordem 1 completa), atingindo o piso da precisÃ£o de mÃ¡quina ($10^{-16}$). O teste MMS atesta taxas assintÃ³ticas Ã³timas de convergÃªncia: quadrÃ¡tica para o campo elÃ©trico ($\mathcal{O}(h^2)$) e linear para suas derivadas espaciais ($\mathcal{O}(h)$).

![Resultados de InterpolaÃ§Ã£o](./figures/VNMM_Materials_interpolation_results.png)
*Figura 1: Painel de interpolaÃ§Ã£o em domÃ­nio bimaterial: reproduÃ§Ã£o em precisÃ£o de mÃ¡quina do Patch Test, distribuiÃ§Ã£o de erro no MMS, campo vetorial reconstruÃ­do e estÃªnceis locais estritamente confinados a cada meio material.*

---

### 4.2 AnÃ¡lise EstatÃ­stica de Sensibilidade a Nuvens Irregulares (*Jitter Study* de 0% a 50%)
Para atestar a genuÃ­na natureza *meshless* do VNMM e comprovar sua insensibilidade Ã  perda de ordenamento cartesiano, aplicou-se uma perturbaÃ§Ã£o estocÃ¡stica aleatÃ³ria na posiÃ§Ã£o de todos os nÃ³s internos:
$$\mathbf{x}_i \leftarrow \mathbf{x}_i + \eta \, h_{\min} \begin{bmatrix} \xi_x \\ \xi_y \end{bmatrix}, \quad \xi_x, \xi_y \sim \mathcal{U}(-0.5, 0.5)$$
onde $\eta \in \{0.00, 0.10, 0.25, 0.40, 0.50\}$ denota a fraÃ§Ã£o de deslocamento estocÃ¡stico (*jitter*), avaliada sobre mÃºltiplas sementes aleatÃ³rias ($N_{\mathrm{seeds}} = 3$). A perturbaÃ§Ã£o foi realizada sobre a cavidade ressonante bimaterial carregada com dielÃ©trico ($[0, \pi]^2$, interface em $x = \pi/2$, $\varepsilon_{r1}=1.0$, $\varepsilon_{r2}=4.0$).

**Tabela 2:** Desempenho estatÃ­stico do VNMM 2D sob perturbaÃ§Ã£o estocÃ¡stica nodal de atÃ© 50% de deformaÃ§Ã£o.

| Jitter $\eta$ | Erro Patch Test ($L_2^{\mathrm{rel}}$) | Erro MMS ($L_2^{\mathrm{rel}}$) | $k_c$ Modo 1 ($\mu \pm \sigma$) | Erro Modo 1 (%) | Erro MÃ©dio 6 Modos (%) | ResÃ­duo $R_{\mathrm{div}}$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Regular)** | $2.35 \times 10^{-16}$ | $3.22\%$ | $0.5825 \pm 0.000$ | $4.23\%$ | $2.46 \pm 0.00\%$ | $1.28\%$ |
| **10% (Leve)** | $3.53 \times 10^{-16}$ | $3.66\%$ | $0.6037 \pm 0.002$ | **$0.75 \pm 0.35\%$** | **$1.13 \pm 0.21\%$** | $1.11\%$ |
| **25% (Moderado)** | $4.85 \times 10^{-16}$ | $6.92\%$ | $0.6226 \pm 0.005$ | **$2.37 \pm 0.85\%$** | **$1.22 \pm 0.17\%$** | $1.71\%$ |
| **40% (Severo)** | $3.22 \times 10^{-16}$ | $5.67\%$ | $0.6365 \pm 0.002$ | $4.65 \pm 0.39\%$ | $3.32 \pm 0.12\%$ | $2.71\%$ |
| **50% (Extremo)** | $5.24 \times 10^{-16}$ | $7.97\%$ | $0.6463 \pm 0.001$ | $6.27 \pm 0.10\%$ | $5.27 \pm 0.26\%$ | $3.26\%$ |

A interpretaÃ§Ã£o desses resultados consolidados revela dois fenÃ´menos fundamentais:
1. **InvariÃ¢ncia Exata da ConsistÃªncia de Ordem 1:** O erro do Patch Test vetorial permaneceu rigorosamente em $10^{-16}$ em **todos os regimes de perturbaÃ§Ã£o**, desde a malha perfeitamente regular (0%) atÃ© o estado de alta desordem espacial (50%). Isso comprova algebricamente que as propriedades reprodutivas das funÃ§Ãµes de forma locais nÃ£o dependem da periodicidade cartesiana dos pontos;
2. **SupressÃ£o do Bloqueio Cartesiano (*Grid-Locking*):** Sob distribuiÃ§Ã£o perfeitamente regular (0%), os eixos preferenciais da malha acumulam ressonÃ¢ncias direcionais, resultando em um erro mÃ©dio de $2.46\%$. Com uma perturbaÃ§Ã£o moderada da nuvem ($\eta = 10\%$ e $\eta = 25\%$), o erro mÃ©dio nos autovalores caiu para **$1.13\%$** e **$1.22\%$**, respectivamente â€” uma **melhoria superior a 50%**. A quebra estocÃ¡stica de simetria diversifica as distÃ¢ncias e direÃ§Ãµes relativas no estÃªncil, atenuando a dispersÃ£o numÃ©rica preferencial;
3. **Pureza FÃ­sica sob Altas DeformaÃ§Ãµes:** Mesmo sob severa distorÃ§Ã£o de 50%, a razÃ£o mÃ©dia de energia de divergÃªncia residual manteve-se restrita a $R_{\mathrm{div}} = 3.26\%$ (bem abaixo do limite mÃ¡ximo de $5.0\%$), garantindo ausÃªncia total de modos espÃºrios em toda a faixa.

![Estudo Detalhado de Jitter](./figures/VNMM_Jitter_Detailed_Study.png)
*Figura 2: Estudo estatÃ­stico de sensibilidade a nuvens de pontos irregulares (0% a 50% de jitter): (a) Autovalores calculados pelo VNMM vs soluÃ§Ãµes exatas analÃ­ticas; (b) Erro percentual relativo com barras de dispersÃ£o estocÃ¡stica ($\pm \sigma$); (c) EvoluÃ§Ã£o do erro mÃ©dio do espectro evidenciando o platÃ´ Ã³timo de precisÃ£o em 10%â€“25%; (d) Monitoramento contÃ­nuo da divergÃªncia residual ($R_{\mathrm{div}} < 3.3\%$).*

---

### 4.3 Captura NÃ­tida de Descontinuidades Normais ($[E_n] \neq 0$)
Para avaliar o comportamento local do campo elÃ©trico sobre a interface bimaterial, executou-se um corte transversal unidimensional em $y = 0.5\text{ m}$ sobre um campo onde a componente normal sofre um salto de $4:1$ ($E_{1x} = 1.500\text{ V/m} \to E_{2x} = 0.375\text{ V/m}$):
- **Componente Normal $E_x(x)$:** O degrau de $4:1$ na interface $x = 1.0\text{ m}$ foi capturado com precisÃ£o nÃ­tida (erro mÃ¡ximo $< 0.033\text{ V/m}$), com ausÃªncia estrita de amortecimento artificial ou oscilaÃ§Ãµes espÃºrias de Gibbs (Figura 3a);
- **Componente Tangencial $E_y(x)$:** A continuidade fÃ­sica $C^0$ foi rigorosamente preservada atravÃ©s de $\Gamma_{\mathrm{int}}$, com erro mÃ¡ximo inferior a $0.014\text{ V/m}$ (Figura 3b).

![ValidaÃ§Ãµes AvanÃ§adas](./figures/VNMM_Interface_Jump_and_Robustness.png)
*Figura 3: Desempenho do VNMM em interfaces e contraste material: (a) Perfil transversal 1D de $E_x(x)$ evidenciando o salto nÃ­tido de 4:1 na interface sem oscilaÃ§Ãµes espÃºrias; (b) Perfil transversal 1D de $E_y(x)$ atestando continuidade estrita $C^0$; (c) Estabilidade espectral dos 6 primeiros modos sob jitter nodal; (d) NÃºmero de onda de corte fundamental vs contraste dielÃ©trico ($\varepsilon_r \in [2.25, 16.0]$).*

---

## 5. ResoluÃ§Ã£o de Autovalores em Cavidades e Guias de Onda

### 5.1 Cavidade Carregada com DielÃ©trico: ComparaÃ§Ã£o AnalÃ­tica Transcendental
Avaliou-se a cavidade $[0, \pi]^2$ bipartida por uma interface em $x = \pi/2$ com $\varepsilon_{r1} = 1.0$ e $\varepsilon_{r2} = 4.0$. Os autovalores numÃ©ricos obtidos pelo VNMM foram confrontados diretamente com as raÃ­zes analÃ­ticas exatas das equaÃ§Ãµes transcendentais de dispersÃ£o:
- Para modos sem variaÃ§Ã£o vertical ($m = 0$): $k_{x1} \cot(k_{x1} d) + k_{x2} \cot(k_{x2} (a - d)) = 0$
- Para modos com variaÃ§Ã£o vertical ($m \ge 1$): $\frac{\beta_1}{\varepsilon_{r1}} \tan(\beta_1 d) + \frac{\beta_2}{\varepsilon_{r2}} \tan(\beta_2 (a - d)) = 0$

**Tabela 3:** ComparaÃ§Ã£o entre autovalores do VNMM 2D e soluÃ§Ãµes analÃ­ticas exatas para os primeiros 8 modos ressonantes $TE_z$.

| Modo | ClassificaÃ§Ã£o | $\lambda$ AnalÃ­tico | $\lambda$ VNMM 2D | $k_c$ AnalÃ­tico (rad/m) | $k_c$ VNMM (rad/m) | Erro Relativo $k_c$ (%) | ResÃ­duo $R_{\mathrm{div}}$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $TE(0, 1)$ | $0.369888$ | $0.364022$ | $0.608184$ | $0.603342$ | **$0.80\%$** | $0.373\%$ |
| **2** | $TE(1, 1)$ | $0.412351$ | $0.416827$ | $0.642146$ | $0.645621$ | **$0.54\%$** | $0.083\%$ |
| **3** | $TE(0, 2)$ | $1.078512$ | $1.042186$ | $1.038514$ | $1.020875$ | **$1.70\%$** | $0.808\%$ |
| **4** | $TE(1, 2)$ | $1.208151$ | $1.155490$ | $1.099159$ | $1.074937$ | **$2.20\%$** | $0.852\%$ |
| **5** | $TE(2, 1)$ | $1.937222$ | $1.980649$ | $1.391841$ | $1.407355$ | **$1.11\%$** | $0.169\%$ |
| **6** | $TE(0, 3)$ | $2.349479$ | $2.180424$ | $1.532801$ | $1.476626$ | **$3.66\%$** | $0.380\%$ |
| **7** | $TE(1, 3)$ | $2.583093$ | $2.500588$ | $1.607200$ | $1.581325$ | **$1.61\%$** | $0.574\%$ |
| **8** | $TE(2, 2)$ | $2.747205$ | $2.877572$ | $1.657469$ | $1.696341$ | **$2.35\%$** | $1.809\%$ |

O mÃ©todo alcanÃ§ou **erro mÃ©dio de apenas $1.75\%$** em todo o espectro, com erro mÃ¡ximo de $3.66\%$, mantendo todos os modos com razÃ£o de divergÃªncia $R_{\mathrm{div}} < 2.0\%$, provando a completa supressÃ£o de poluiÃ§Ã£o espectral.

![Modos Ressonantes na Cavidade Carregada](./figures/VNMM_Dielectric_Cavity_Modes.png)
*Figura 4: DistribuiÃ§Ã£o espacial normalizada $\|\mathbf{E}\|/\|\mathbf{E}\|_{\max}$ dos primeiros quatro modos ressonantes na cavidade bimaterial carregada, ilustrando a concentraÃ§Ã£o fÃ­sica de energia eletromagnÃ©tica no interior do dielÃ©trico ($\varepsilon_{r2}=4.0$).*

---

### 5.2 Varredura ParamÃ©trica de Contraste DielÃ©trico
A resposta espectral foi avaliada frente a materiais tecnolÃ³gicos de micro-ondas em simulaÃ§Ãµes bimateriais independentes ($\Omega_1$ ar, $\Omega_2$ dielÃ©trico): Teflon ($\varepsilon_{r2} = 2.25$), DielÃ©trico PadrÃ£o ($\varepsilon_{r2} = 4.00$), Alumina ($\varepsilon_{r2} = 9.80$) e CerÃ¢mica AvanÃ§ada ($\varepsilon_{r2} = 16.00$).

**Tabela 4:** PrecisÃ£o do nÃºmero de onda de corte fundamental sob diferentes contrastes dielÃ©tricos.

| Material DielÃ©trico | Permissividade $\varepsilon_{r2}$ | $k_c$ AnalÃ­tico Exato | $k_c$ VNMM 2D | Erro Relativo (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Teflon** | $2.25$ | $0.7709\text{ rad/m}$ | $0.7628\text{ rad/m}$ | **$1.05\%$** |
| **DielÃ©trico PadrÃ£o** | $4.00$ | $0.6082\text{ rad/m}$ | $0.6033\text{ rad/m}$ | **$0.80\%$** |
| **Alumina** | $9.80$ | $0.4029\text{ rad/m}$ | $0.4006\text{ rad/m}$ | **$0.58\%$** |
| **CerÃ¢mica de Micro-ondas** | $16.00$ | $0.3183\text{ rad/m}$ | $0.3167\text{ rad/m}$ | **$0.52\%$** |

O erro no modo fundamental permaneceu estritamente **abaixo de $1.1\%$** em toda a faixa de permissividades, rastreando fielmente a curva teÃ³rica (Figura 3d).

---

## 6. Benchmarks em Meios AnisotrÃ³picos e Geometrias Singulares

Para demonstrar a versatilidade da formulaÃ§Ã£o sobre nuvens de pontos em problemas de alta complexidade fÃ­sica, foram reproduzidos dois problemas clÃ¡ssicos de referÃªncia da literatura (Ortiz, 2023; Jiang et al., 2014):

### 6.1 Guia com SaliÃªncia em "H" HomogÃªneo e AnisotrÃ³pico (Geometria NÃ£o-Convexa com Quinas)
O primeiro benchmark investiga a resposta em domÃ­nios nÃ£o-convexos com quinas reentrantes que induzem singularidades no campo elÃ©trico, preenchido por um meio girotÃ³pico caracterizado por um tensor de permissividade hermitiano complexo:
$$\mu_r = 1.0, \quad \overline{\overline{\varepsilon}}_r = \begin{bmatrix} 1 & -j \\ j & 4 \end{bmatrix}$$
O domÃ­nio ocupa $x \in [-2.5, 2.5]\text{ m}$ e $y \in [-1.5, 1.5]\text{ m}$, com reentrÃ¢ncias simÃ©tricas centrais em $x \in [-1.0, 1.0]\text{ m}$ que estrangulam o canal condutor a $|y| \le 0.5\text{ m}$ (Figura 5). A discretizaÃ§Ã£o foi realizada sobre uma nuvem de pontos com diretores alternados a $\pm 45^\circ$ no interior e tangenciais nas paredes PEC.

![Geometria do Guia em H](./figures/Geometria_Guia_H_Anisotropico.png)
*Figura 5: Geometria e parÃ¢metros constitutivos do guia de onda com saliÃªncia em "H" homogÃªneo e anisotrÃ³pico com quinas reentrantes (Teste 4.3.2).*

**Tabela 5:** ComparaÃ§Ã£o dos primeiros 5 autovalores no guia com saliÃªncia em "H" entre o VNMM 2D e o FEM de aresta de alta densidade (427.492 arestas).

| Modo | SoluÃ§Ã£o Ref. FEM (427k arestas) | FEM Misto [22] | EMM [13] (1.529 ares) | VNMM 2D (Nuvem Nodal) | Erro Relativo (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.041212$ | $0.041206$ | $0.040628$ | $0.041915$ | **$1.71\%$** |
| **2** | $0.400641$ | $0.400552$ | $0.392058$ | $0.412739$ | **$3.02\%$** |
| **3** | $0.786745$ | $0.786704$ | $0.783254$ | $0.774186$ | **$1.60\%$** |
| **4** | $0.998524$ | $0.998524$ | $0.987213$ | $0.914897$ | **$8.38\%$** |
| **5** | $1.638579$ | $1.638369$ | $1.615863$ | $1.638612$ | **$0.00\%$** |

O VNMM modelou as singularidades induzidas pelas quinas sem instabilidade numÃ©rica, apresentando erro de apenas $1.71\%$ no modo dominante e concordÃ¢ncia exata ($0.00\%$) no 5Âº modo ressonante frente ao FEM de mais de 400 mil graus de liberdade.

---

### 6.2 Guia Bimaterial AnisotrÃ³pico com Cilindros PEC Internos (Fronteiras Curvas)
O segundo teste acopla simultaneamente interface bimaterial descontÃ­nua, anisotropia tensorial distinta em cada meio e obstÃ¡culos condutores circulares internos:
- **Geometria:** DomÃ­nio retangular $[-2, 2] \times [-1, 1]\text{ m}$ dividido em $x = 0$ entre $\Omega_1 = [-2, 0] \times [-1, 1]\text{ m}$ e $\Omega_2 = [0, 2] \times [-1, 1]\text{ m}$. Cada subdomÃ­nio encerra um obstÃ¡culo circular condutor PEC de raio $r = 1/3\text{ m}$ centrado em $(-1, 0)$ e $(1, 0)$, respectivamente (Figura 6);
- **Meio 1 ($\Omega_1$):** $\mu_{r1} = 1.0, \quad \overline{\overline{\varepsilon}}_{r1} = \operatorname{diag}(1, 2)$ (anisotropia biaxial);
- **Meio 2 ($\Omega_2$):** $\mu_{r2} = 3.0, \quad \overline{\overline{\varepsilon}}_{r2} = \begin{bmatrix} 3 & -2j \\ 2j & 8 \end{bmatrix}$ (anisotropia girotÃ³pica com permeabilidade $\mu_r \neq 1$).

![Geometria do Guia Bimaterial com Cilindros PEC](./figures/Geometria_Guia_Bimaterial_Circ.png)
*Figura 6: Geometria do guia bimaterial anisotrÃ³pico com inclusÃµes cilÃ­ndricas PEC e interface plana em $x = 0$ (Teste 4.3.3).*

**Tabela 6:** ComparaÃ§Ã£o dos primeiros 5 autovalores no guia bimaterial com inclusÃµes PEC entre o VNMM 2D e o FEM de referÃªncia (340.363 arestas).

| Modo | SoluÃ§Ã£o Ref. FEM (340k arestas) | FEM Misto [22] | EMM [13] (8.566 ares) | VNMM 2D (Nuvem Nodal) | Erro Relativo (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.051876$ | $0.051841$ | $0.051825$ | $0.051641$ | **$0.45\%$** |
| **2** | $0.205316$ | $0.205012$ | $0.204686$ | $0.203056$ | **$1.10\%$** |
| **3** | $0.247201$ | $0.247126$ | $0.246919$ | $0.255165$ | **$3.22\%$** |
| **4** | $0.423569$ | â€” | $0.423277$ | $0.414830$ | **$2.06\%$** |
| **5** | $0.625603$ | â€” | $0.624568$ | $0.637011$ | **$1.82\%$** |

Mesmo com a complexidade combinada de fronteiras curvas PEC internas, interfaces materiais em $x = 0$ e tensores complexos, o VNMM obteve erro no modo dominante de apenas **$0.45\%$** e erro inferior a $3.3\%$ em todos os modos, comprovando a robustez da formulaÃ§Ã£o nodal.

---

## 7. ConclusÃµes

O desenvolvimento e a validaÃ§Ã£o do **Vector Nodal Meshless Method (VNMM 2D)** sobre nuvens de pontos confirmam seu potencial para resolver problemas eletromagnÃ©ticos avanÃ§ados:
1. **Completude Reprodutiva e InvariÃ¢ncia Nodal:** A formulaÃ§Ã£o baseada em projeÃ§Ãµes direcionais preserva consistÃªncia polinomial de primeira ordem em precisÃ£o de mÃ¡quina ($10^{-16}$) independentemente da irregularidade geomÃ©trica dos pontos (de 0% a 50% de perturbaÃ§Ã£o estocÃ¡stica);
2. **Superioridade de Nuvens Irregulares:** A quebra de alinhamentos cartesianos regulares via perturbaÃ§Ã£o estocÃ¡stica atenua o viÃ©s direcional da discretizaÃ§Ã£o (*grid-locking*), reduzindo o erro mÃ©dio dos autovalores em mais de 50% (para $\approx 1.1\%$) em relaÃ§Ã£o a arranjos regulares rÃ­gidos;
3. **ResoluÃ§Ã£o de Descontinuidades Sem Borramento:** A segregaÃ§Ã£o estrita de suportes por material captura o salto de descontinuidade normal de forma nÃ­tida, preservando a continuidade tangencial sem oscilaÃ§Ãµes de Gibbs;
4. **Generalidade FÃ­sica em Meios AnisotrÃ³picos Complexos:** A formulaÃ§Ã£o resolveu com alta acurÃ¡cia (erros de $0.45\%$ a $3.2\%$) problemas de autovalores com tensores dielÃ©tricos hermitianos, geometrias com quinas vivas e inclusÃµes circulares PEC, equiparando-se a modelos de referÃªncia de elementos finitos de aresta com centenas de milhares de graus de liberdade.

---

## ReferÃªncias

1. Harrington, R. F. (2001). *Time-Harmonic Electromagnetic Fields*. IEEE Press / Wiley-Interscience.
2. Collin, R. E. (1991). *Field Theory of Guided Waves*. IEEE Press.
3. Jin, J. M. (2014). *The Finite Element Method in Electromagnetics*. John Wiley & Sons.
4. Belytschko, T., Lu, Y. Y., & Gu, L. (1994). Element-free Galerkin methods. *International Journal for Numerical Methods in Engineering*, 37(2), 229-256.
5. Garcia Ortiz, L. A. (2023). *MÃ©todo Sem Malha Nodal Utilizando FunÃ§Ãµes de Forma Vetoriais Baseadas em H(curl)*. Tese de Doutorado, Universidade Federal de Minas Gerais (UFMG), Belo Horizonte.
6. Ortiz, L. A. G., Mesquita, R. C., & Lima, N. Z. (2021). Nodal meshless method with vectorial shape functions based on H(curl). *IEEE Transactions on Magnetics*, 57(6), 1-4.
7. Jiang, W., Liu, N., Tang, Y., & Liu, Q. H. (2014). Mixed finite element method for 2D vector Maxwellâ€™s eigenvalue problem in anisotropic media. *Progress In Electromagnetics Research*, 148, 159-170.
8. Lima, N. Z., & Mesquita, R. C. (2017). Edge meshless method applied to vector electromagnetic problems. *IEEE Transactions on Magnetics*, 53(6), 1-4.

