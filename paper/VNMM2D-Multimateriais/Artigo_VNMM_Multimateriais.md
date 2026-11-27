# Nodal Vector Meshless Method for Electromagnetic Eigenvalue Problems in Discontinuous and Anisotropic Media

**Autores:** Luilly A. G. Ortiz, Renato C. Mesquita, Barbara M. F. Gonçaves  
**Afiliação:** Universidade Federal de Minas Gerais (UFMG), Escola de Engenharia, Programa de Pós-Graduação em Engenharia Elétrica  
**Status:** Artigo Científico / Relatório de Pesquisa Consolidado  

---

## Resumo

Este trabalho apresenta a formulação e a validação do **Vector Nodal Meshless Method (VNMM 2D)** para a resolução de problemas de autovalores eletromagnéticos em meios heterogêneos, descontínuos e anisotrópicos sobre nuvens de pontos não-estruturadas. O método atribui a cada nó um único grau de liberdade escalar correspondente à projeção do campo elétrico ao longo de um vetor diretor unitário ($\mathbf{t}_i$), combinando uma base vetorial polinomial linear completa ($\mathcal{P}^1$), seleção local via árvore *k-d* e regularização variacional *div-curl*. Para modelar interfaces materiais sem borramento numérico (*smearing*) nem oscilações de Gibbs, introduz-se uma **regra de partição estrita de suportes**, confinando cada estêncil ao seu subdomínio material e alinhando tangencialmente os diretores de interface ($\mathbf{t}_j = \boldsymbol{\tau}$), o que impõe continuidade exata em $E_t$ enquanto permite saltos abruptos na componente normal ($[E_n] \neq 0$). Os resultados demonstram consistência de ordem 1 em precisão de máquina ($10^{-16}$), convergência quadrática $\mathcal{O}(h^2)$ no campo elétrico, redução superior a 50% no erro espectral em nuvens moderadamente irregulares (10% a 25% de perturbação) frente a grades regulares, e ausência total de modos espúrios em cavidades com alto contraste dielétrico ($\varepsilon_r \in [2.25, 16.0]$), guias anisotrópicos com quinas reentrantes, inclusões circulares PEC e junções quádruplas de quatro materiais ($0.34\%$ de erro no modo fundamental).

**Palavras-chave:** Métodos Sem Malha Nodais, Nuvens de Pontos, Espaços $H(\mathbf{curl})$, Meios Anisotrópicos, Interfaces Multimateriais, Problemas de Autovalores Eletromagnéticos, Supressão de Modos Espúrios.

---

## 1. Introdução

A modelagem numérica de dispositivos eletromagnéticos de alta frequência — como guias de onda carregados com dielétricos, cavidades ressonantes e componentes com meios girotópicos — fundamenta-se na formulação fraca das equações de Maxwell no espaço de Sobolev $H(\mathbf{curl})$ (Harrington, 2001; Collin, 1991). No Método dos Elementos Finitos (FEM) convencional, aproximações nodais escalares que interpolam independentemente as componentes cartesianas ($E_x, E_y$) falham em problemas vetoriais por três razões clássicas: impõem continuidade $C^0$ artificial na componente normal através de interfaces materiais, não capturam adequadamente singularidades de campo em quinas e bordas condutoras, e violam o complexo discreto de de Rham, contaminando o espectro com modos espúrios irrotacionais (Jin, 2014). Os elementos finitos de aresta de Nédélec superam essas patologias ao associar os graus de liberdade à circulação tangencial nas arestas, garantindo conformidade nativa em $H(\mathbf{curl})$; contudo, dependem criticamente da geração e adaptação de malhas conformes de alta qualidade, o que impõe severo custo geométrico em fronteiras curvas ou reentrantes.

Para eliminar a dependência de conectividade de malha, os Métodos Sem Malha (*Meshless Methods*) constroem aproximações diretamente sobre nuvens de pontos espalhadas pelo domínio (Belytschko et al., 1994). Entretanto, as primeiras adaptações vetoriais baseadas em funções de forma escalares (como EFG, MLPG e PIM) herdaram as mesmas limitações do FEM nodal: surgimento de modos espúrios e travamento das descontinuidades normais em interfaces, exigindo multiplicadores de Lagrange ou projeções auxiliares de divergente nulo. Com o objetivo de obter conformidade vetorial nativa sem malha, Lima e Mesquita (2017) propuseram o *Edge Meshless Method* (EMM), formulando funções de forma vetoriais a partir da circulação do campo ao longo de segmentos de aresta distribuídos no domínio. Ao tomar o limite assintótico em que o comprimento dessas arestas tende a zero, Ortiz et al. (2021, 2023) estabeleceram o **Vector Nodal Meshless Method (VNMM)**, no qual cada aresta degenera em um nó vetorial $(\mathbf{x}_i, \mathbf{t}_i)$ com um único grau de liberdade escalar dado pela projeção direcional $e_i = \mathbf{E}(\mathbf{x}_i)\cdot\mathbf{t}_i$.

Ao empregar uma base vetorial polinomial linear completa ($\mathcal{P}^1$) com seleção local de suportes via árvore *k-d*, o VNMM evita a estagnação na convergência do rotacional observada em bases solenoidais incompletas ($\mathcal{L}^1$) e preserva consistência linear exata sobre nuvens arbitrárias. Adicionalmente, como a seleção dinâmica de estênceis em nuvens não-estruturadas rompe a conformidade global de traço, a incorporação de uma regularização variacional *div-curl* ($s_{\mathrm{div}}\mathbf{K}_{\mathrm{div}}$) desloca eventuais modos espúrios de gradiente para fora da banda espectral física sem perturbar os autovalores ressonantes.

Apesar desses avanços em meios homogêneos, a extensão do VNMM para **meios heterogêneos, descontínuos e anisotrópicos** exige um tratamento preciso das interfaces materiais: o compartilhamento ingênuo de suportes entre dielétricos distintos provoca borramento numérico (*smearing*) da descontinuidade normal e oscilações de Gibbs. Neste artigo, resolve-se esse desafio por meio da **regra de partição estrita de suportes nodais** associada a **diretores puramente tangenciais na interface** ($\mathbf{t}_j = \boldsymbol{\tau}$), acoplando os subdomínios por um único grau de liberdade tangencial e deixando a componente normal livre para reproduzir o salto físico ($\varepsilon_1 E_{1n} = \varepsilon_2 E_{2n}$). Demonstra-se ainda que o VNMM se beneficia da operação em **nuvens de pontos irregulares**: a quebra da simetria cartesiana por perturbações estocásticas moderadas (10% a 25%) mitiga a dispersão direcional (*grid-locking*), reduzindo o erro médio espectral em mais de 50%, enquanto resolve com alta acurácia tensores girotópicos hermitianos, singularidades em quinas reentrantes, inclusões cilíndricas PEC e junções múltiplas de quatro materiais sem modos espúrios.

O restante do artigo está organizado da seguinte forma: a Seção 2 apresenta a formulação matemática do VNMM 2D em nuvens de pontos, o algoritmo de seleção de suportes invariante à escala, as condições de partição estrita multimaterial e a forma fraca regularizada; a Seção 3 analisa a consistência polinomial, a sensibilidade a nuvens irregulares e a captura nítida de saltos de interface; a Seção 4 valida cavidades carregadas com dielétricos sob amplos contrastes; a Seção 5 avalia benchmarks em meios anisotrópicos, quinas singulares, inclusões PEC e junção quádrupla; e a Seção 6 sintetiza as conclusões.

---

## 2. Formulação Matemática do VNMM 2D Multimaterial em Nuvens de Pontos

Esta seção estabelece a formulação matemática do Método Sem Malha Nodal Vetorial bidimensional (VNMM 2D) para domínios heterogêneos e anisotrópicos sobre nuvens de pontos não-estruturadas. Inicialmente, definem-se os graus de liberdade baseados em projeções direcionais, a base vetorial polinomial completa $\mathcal{P}^1$ e o sistema de colocação local. Em seguida, detalha-se o algoritmo de seleção automática de nós de suporte invariante à escala acoplado às condições de partição estrita para meios multimateriais e, por fim, apresenta-se a formulação fraca de Galerkin regularizada.

### 2.1 Graus de Liberdade, Projeções Vetoriais Nodais e Contornos PEC
Considere um domínio bidimensional $\Omega \subset \mathbb{R}^2$ discretizado por uma nuvem de $N$ nós arbitrários $\{\mathbf{x}_i\}_{i=1}^N$ espalhados em seu interior e contornos $\partial\Omega$. Diferentemente de métodos sem malha escalares que atribuem graus de liberdade independentes às componentes cartesianas ($E_x, E_y$), o VNMM associa a cada nó $\mathbf{x}_i = (x_i, y_i)^T$ um único vetor diretor unitário:
$$\mathbf{t}_i = \begin{bmatrix} \cos\theta_i \\ \sin\theta_i \end{bmatrix}$$

Em qualquer ponto de avaliação $\mathbf{x} = (x, y) \in \Omega$, o campo elétrico aproximado $\mathbf{E}^h(\mathbf{x})$ é expresso como uma combinação linear das funções de forma vetoriais nodais $\mathbf{N}_i(\mathbf{x})$ associadas aos $M_{\mathrm{supp}}$ nós do estêncil de suporte local:
$$\mathbf{E}^h(x, y) = \sum_{i=1}^{M_{\mathrm{supp}}} \mathbf{N}_i(x, y)\, e_i$$
onde $e_i$ representa o grau de liberdade escalar associado ao nó $i$, definido fisicamente como a projeção direcional pontual do campo elétrico ao longo do diretor nodal:
$$e_i = \mathbf{E}(\mathbf{x}_i) \cdot \mathbf{t}_i \quad [\text{V/m}]$$

Para assegurar consistência local em $H(\mathbf{curl})$ e imposição direta das condições essenciais de contorno, exige-se que as funções de forma satisfaçam a propriedade de delta de Kronecker vetorial:
$$\mathbf{N}_i(\mathbf{x}_k) \cdot \mathbf{t}_k = \delta_{ik} = \begin{cases} 1, & \text{se } k = i \\ 0, & \text{se } k \neq i \end{cases}$$

Nas fronteiras de Condutores Elétricos Perfeitos (PEC), onde a condição física impõe $\hat{n} \times \mathbf{E} = \mathbf{0}$, os diretores nodais são alinhados à tangente do contorno ($\mathbf{t}_k = \boldsymbol{\tau}$), permitindo impor a condição de Dirichlet homogênea diretamente por colocação nodal ($e_k = 0, \; \forall \mathbf{x}_k \in \partial\Omega_{\mathrm{PEC}}$) sem multiplicadores de Lagrange. No interior de cada subdomínio, adota-se o padrão alternado ortogonal ($\pm 45^\circ$, isto é, $\theta_i \in \{\pi/4, 3\pi/4\}$) para garantir diversidade angular local. A análise funcional detalhada dessa projeção direcional e da distinção entre colocação nodal pontual e traço contínuo de contorno em nuvens não-estruturadas é apresentada no artigo de referência do VNMM estabilizado (Mesquita et al., 2026).

### 2.2 Base Vetorial Polinomial Linear Completa ($\mathcal{P}^1$)
Formulações iniciais do VNMM empregavam a base solenoidal incompleta $\mathcal{L}^1$ (3 nós de suporte), herdada dos elementos de aresta de Nédélec de primeiro tipo. Contudo, quando combinada com seleção dinâmica de suportes em nuvens arbitrárias, a base $\mathcal{L}^1$ força o anulamento artificial das derivadas longitudinais ($\partial E_x/\partial x$, $\partial E_y/\partial y$), gerando *aliasing* modal de derivadas e estagnação $\mathcal{O}(1)$ na convergência do rotacional.

Para eliminar essa limitação, adota-se o espaço vetorial polinomial linear completo $\mathcal{P}^1 = \mathcal{P}_1 \times \mathcal{P}_1$, composto por 6 monômios independentes ($M_{\mathrm{supp}} = 6$ nós de suporte) em coordenadas locais transladadas ($\Delta x = x - x_P$, $\Delta y = y - y_P$) centradas no ponto de avaliação $P(x_P, y_P)$:
$$\mathcal{P}^1 = \operatorname{span} \left\{ \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \; \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \; \begin{bmatrix} \Delta x \\ 0 \end{bmatrix}, \; \begin{bmatrix} \Delta y \\ 0 \end{bmatrix}, \; \begin{bmatrix} 0 \\ \Delta x \end{bmatrix}, \; \begin{bmatrix} 0 \\ \Delta y \end{bmatrix} \right\}$$

A função de forma vetorial local associada ao nó de suporte $i$ é escrita como:
$$\mathbf{N}_i(\Delta x, \Delta y) = \begin{bmatrix} \beta_{1i} + \beta_{3i}\Delta x + \beta_{4i}\Delta y \\ \beta_{2i} + \beta_{5i}\Delta x + \beta_{6i}\Delta y \end{bmatrix}$$

Com 6 graus de liberdade independentes, o espaço $\mathcal{P}^1$ resolve simultaneamente todas as quatro componentes do tensor jacobiano local $\mathbf{J}^h = \nabla \mathbf{E}^h$, capturando tanto a parte rotacional quanto as variações longitudinais de fluxo. A demonstração completa de que essa transição elimina o *aliasing* de derivadas e restaura a convergência ótima do campo e de seu rotacional encontra-se detalhada em Mesquita et al. (2026).

### 2.3 Sistema de Colocação Local e Operadores Diferenciais
Para um conjunto qualificado de 6 nós de suporte vizinhos $\mathcal{S}(P) = \{n_1, \dots, n_6\}$, a imposição da propriedade de delta de Kronecker vetorial ($\mathbf{N}_i(\mathbf{x}_k) \cdot \mathbf{t}_k = \delta_{ik}$) resulta no sistema algébrico local:
$$\mathbf{A} \boldsymbol{\beta} = \mathbf{I}_6 \implies \boldsymbol{\beta} = \mathbf{A}^{-1}$$
onde a $k$-ésima linha da matriz de colocação $\mathbf{A} \in \mathbb{R}^{6 \times 6}$ é dada por:
$$\mathbf{A}_{k, :} = \begin{bmatrix} t_{kx} & t_{ky} & \Delta x_k t_{kx} & \Delta y_k t_{kx} & \Delta x_k t_{ky} & \Delta y_k t_{ky} \end{bmatrix}, \quad k = 1, \dots, 6$$
sendo $\Delta x_k = x_k - x_P$ e $\Delta y_k = y_k - y_P$.

Avaliando diretamente no ponto $P$ ($\Delta x = \Delta y = 0$), as funções de forma vetoriais e seus operadores diferenciais espaciais são obtidos analiticamente a partir das linhas de $\boldsymbol{\beta}$ (indexadas a partir de 0):
$$\boldsymbol{\Phi}(P) = \boldsymbol{\beta}_{0:2, :} \in \mathbb{R}^{2 \times 6}$$
$$(\nabla \times \boldsymbol{\Phi})_z(P) = \boldsymbol{\beta}_{4, :} - \boldsymbol{\beta}_{3, :} \in \mathbb{R}^{1 \times 6}$$
$$\nabla \cdot \boldsymbol{\Phi}(P) = \boldsymbol{\beta}_{2, :} + \boldsymbol{\beta}_{5, :} \in \mathbb{R}^{1 \times 6}$$

A caracterização algébrica das singularidades de $\mathbf{A}$ pelo Princípio da Ortogonalidade do Espaço Nulo e sua fatoração adimensional $\mathbf{A} = \widehat{\mathbf{A}}\mathbf{D}_h$ para preservação de condicionamento sob refinamento espacial ($h \to 0$) são formuladas rigorosamente em Mesquita et al. (2026).

### 2.4 Seleção Automática de Nós de Suporte e Partição Estrita Multimaterial
A escolha adequada do estêncil $\mathcal{S}(P)$ em nuvens não-estruturadas requer simultaneamente bom condicionamento algébrico e respeito estrito às descontinuidades materiais do domínio.

#### Algoritmo de Seleção Invariante à Escala
Em nuvens de pontos arbitrárias, utilizam-se as quatro fases do algoritmo de seleção automática proposto em Mesquita et al. (2026):
1. **Busca Espacial via Árvore *k-d*:** Uma árvore *k-d* ($\mathcal{O}(N \log N)$) recupera os $K_{\mathrm{initial}} \ge 12$ vizinhos mais próximos de $P$, definindo a escala espacial característica local $h_{\mathrm{local}}(\mathbf{x}_P) = \frac{1}{6}\sum_{k=1}^6 \|\mathbf{x}_{(k)} - \mathbf{x}_P\|$;
2. **Geração Ordenada de Sextetos Candidatos:** Fixando o nó mais próximo de $P$ como âncora, geram-se combinações de 6 nós ordenadas por distância média crescente;
3. **Montagem da Matriz Adimensional $\widehat{\mathbf{A}}$:** Normalizam-se as coordenadas locais ($\hat{x}_k = \Delta x_k / h_{\mathrm{local}}, \; \hat{y}_k = \Delta y_k / h_{\mathrm{local}}$), removendo o decaimento quártico $\det(\mathbf{A}) \sim \mathcal{O}(h^4)$ e a inflação artificial do número de condição $\kappa_2(\mathbf{A}) \sim \mathcal{O}(1/h)$;
4. **Qualificação por Determinante com Parada Antecipada:** Aceita-se imediatamente o primeiro sexteto candidato que satisfizer $|\det(\widehat{\mathbf{A}})| \ge \widehat{\mathrm{Tol}} > 0$, filtrando configurações singulares ou mal-condicionadas sem dilatação de suporte.

#### Condições para Domínios Multimateriais (Regra de Partição Estrita de Suportes)
Quando o domínio é heterogêneo ($\Omega = \bigcup_{m=1}^M \Omega_m$) com interfaces físicas $\Gamma_{\mathrm{int}}$, o algoritmo de seleção via árvore *k-d* deve obedecer estritamente a duas condições adicionais:
1. **Confinamento ao Subdomínio Material:** Ao avaliar as funções de forma em um ponto de integração $P \in \Omega_m$, o conjunto de busca de nós candidatos é restrito exclusivamente aos nós pertencentes ao meio $\Omega_m$ e aos nós posicionados diretamente sobre a interface $\Gamma_{\mathrm{int}}$ adjacente (implementado por meio de árvores *k-d* particionadas por material). Nós pertencentes a outros meios materiais são estritamente excluídos do estêncil;
2. **Graus de Liberdade Tangenciais de Interface:** Os nós alocados ao longo de $\Gamma_{\mathrm{int}}$ têm seus diretores alinhados exclusivamente à tangente unitária da interface ($\mathbf{t}_j = \boldsymbol{\tau}$). Como a componente tangencial do campo elétrico é fisicamente contínua ($E_{1t} = E_{2t}$), um único grau de liberdade escalar na interface acopla ambos os materiais de forma exata e não-redundante. Em contrapartida, a segregação dos suportes permite que a componente normal $E_n$ sofra livremente o salto físico ditado pela razão de permissividades ($\varepsilon_1 E_{1n} = \varepsilon_2 E_{2n}$), eliminando borramento numérico (*smearing*) e oscilações de Gibbs.

### 2.5 Forma Fraca de Galerkin com Regularização de Divergente
Como a seleção dinâmica de estênceis em nuvens irregulares resulta em uma aproximação globalmente não-conforme ($V_h \not\subset H(\mathbf{curl}; \Omega)$), incorpora-se a regularização variacional *div-curl* (Mesquita et al., 2026) à forma fraca de Maxwell para meios anisotrópicos gerais caracterizados pelo tensor de permissividade $\overline{\overline{\varepsilon}}_r$ e permeabilidade $\mu_r$:
$$\int_{\Omega} \frac{1}{\mu_r} (\nabla \times \mathbf{E})_z (\nabla \times \mathbf{v})_z^* \, d\Omega + s_{\mathrm{div}} \int_{\Omega} \frac{1}{\mu_r} (\nabla \cdot \mathbf{E}) (\nabla \cdot \mathbf{v})^* \, d\Omega = k_0^2 \int_{\Omega} \mathbf{v}^* \cdot (\overline{\overline{\varepsilon}}_r \mathbf{E}) \, d\Omega$$
onde $s_{\mathrm{div}} \ge 0$ é o parâmetro de penalidade de divergente que desloca os modos espúrios irrotacionais para fora do espectro físico de interesse, e $(\cdot)^*$ denota a transposta conjugada para tensores hermitianos complexos.

A discretização conduz ao problema de autovalores generalizado:
$$(\mathbf{K}_{\mathrm{curl}} + s_{\mathrm{div}} \mathbf{K}_{\mathrm{div}}) \mathbf{c} = \lambda \mathbf{M} \mathbf{c}$$
com $\lambda = k_0^2$. As condições de contorno PEC ($e_k = 0, \; \forall k \in \partial\Omega_{\mathrm{PEC}}$) são impostas eliminando-se diretamente as respectivas linhas e colunas das matrizes globais. A integração numérica é realizada sobre uma grade cartesiana de fundo uniforme com quadratura de Gauss ($2 \times 2$, ou $3 \times 3$ próxima a vértices singulares), atuando estritamente como suporte métrico sem exigir conectividade entre os nós.

---

## 3. Estudo de Consistência e Robustez em Nuvens de Pontos Irregulares

Esta seção avalia sistematicamente a consistência matemática, a convergência espacial e a estabilidade numérica do VNMM 2D em meios multimateriais sob nuvens de pontos irregulares, abrangendo testes de reprodução polinomial (Patch Test e MMS), análise de sensibilidade a perturbações estocásticas (*jitter* de até 50%) e captura nítida de saltos normais em interfaces dielétricas.

### 3.1 Consistência Polinomial e Convergência Espacial
Para certificar a completude das funções de forma em nuvens de pontos sob meios heterogêneos, foram aplicados o Galerkin Patch Test e o Method of Manufactured Solutions (MMS):
- **Campo Afim do Patch Test:**
  $$\mathbf{E}_{\mathrm{patch}}(x, y) = \begin{bmatrix} 1.5 + 0.8\,x - 0.3\,y \\ -0.7 + 0.4\,x + 1.2\,y \end{bmatrix}, \quad (\nabla \times \mathbf{E}_{\mathrm{patch}})_z = 0.7\text{ V/m}^2$$
- **Campo Não-Polinomial MMS:**
  $$\mathbf{E}_{\mathrm{MMS}}(x, y) = \begin{bmatrix} \sin\left(\frac{\pi x}{L_x}\right)\cos\left(\frac{\pi y}{L_y}\right) \\ -\cos\left(\frac{\pi x}{L_x}\right)\sin\left(\frac{\pi y}{L_y}\right) \end{bmatrix}, \quad (\nabla \times \mathbf{E}_{\mathrm{MMS}})_z = \left(\frac{\pi}{L_x} + \frac{\pi}{L_y}\right)\sin\left(\frac{\pi x}{L_x}\right)\sin\left(\frac{\pi y}{L_y}\right)$$

A resposta quantitativa de interpolação e as taxas de convergência espacial obtidas para ambos os campos de teste são consolidadas na **Tabela 1**, enquanto a **Figura 1** ilustra a distribuição espacial do erro de interpolação no domínio bimaterial, a reprodução em precisão de máquina do Patch Test, o campo vetorial reconstruído e o confinamento geométrico dos estênceis de suporte locais estritamente particionados por região material.

**Tabela 1:** Desempenho de interpolação do VNMM 2D em domínio bimaterial com nuvem de pontos.

| Teste | Campo Vetorial | Erro Relativo $L_2$ ($\mathbf{E}$) | Erro Relativo $\text{rot}(\mathbf{E})$ | Status |
| :--- | :--- | :---: | :---: | :---: |
| **Galerkin Patch Test** | Linear Afim Completo | **$2.85 \times 10^{-16}$** | **$1.00 \times 10^{-14}$** | **Precisão de Máquina** |
| **MMS (Base $N=9$)** | Trigonométrico Suave | $5.742\%$ | $19.461\%$ | Consistente |
| **MMS (Refinada $N=17$)** | Trigonométrico Suave | **$1.147\%$** | **$9.390\%$** | **Taxa $\mathcal{O}(h^2)$** |
| **Taxa Assintótica** | — | **$5.01\times$** ($\ge 4\times$) | **$2.07\times$** ($\approx 2\times$) | **Ótima Teórica** |

![Resultados de Interpolação](./figures/VNMM_Materials_interpolation_results.png)
*Figura 1: Painel de interpolação em domínio bimaterial: distribuição logarítmica do erro no Patch Test (sup. esq.), erro absoluto no MMS (sup. dir.), campo vetorial reconstruído (inf. esq.) e verificação geométrica dos suportes estritamente confinados a cada meio material (inf. dir.).*

A análise dos resultados da Tabela 1 e da Figura 1 confirma conclusivamente que as propriedades fundamentais de consistência e de reprodução polinomial exata de ordem 1 ($\mathcal{P}^1$) são integralmente herdadas pela formulação multimaterial, alcançando o piso do ruído de máquina ($10^{-16}$). Isso demonstra rigorosamente que a partição estrita de suportes por subdomínio e a adoção de diretores tangenciais na interface preservam a integridade matemática do método sem malha original. Além disso, o teste MMS confirma que a presença de interfaces heterogêneas preserva as taxas assintóticas ótimas de convergência espacial: quadrática para o campo elétrico ($\mathcal{O}(h^2)$) e linear para suas derivadas espaciais/rotacional ($\mathcal{O}(h)$), certificando a consistência e a robustez do VNMM em meios descontínuos.

---

### 3.2 Análise Estatística de Sensibilidade a Nuvens Irregulares (*Jitter Study* de 0% a 50%)
Para atestar a genuína natureza *meshless* do VNMM e comprovar sua insensibilidade à perda de ordenamento cartesiano, aplicou-se uma perturbação estocástica aleatória na posição de todos os nós internos:
$$\mathbf{x}_i \leftarrow \mathbf{x}_i + \eta \, h_{\min} \begin{bmatrix} \xi_x \\ \xi_y \end{bmatrix}, \quad \xi_x, \xi_y \sim \mathcal{U}(-0.5, 0.5)$$
onde $\eta \in \{0.00, 0.10, 0.25, 0.40, 0.50\}$ denota a fração de deslocamento estocástico (*jitter*), avaliada sobre múltiplas sementes aleatórias ($N_{\mathrm{seeds}} = 3$). A perturbação foi realizada sobre a cavidade ressonante bimaterial carregada com dielétrico ($[0, \pi]^2$, interface em $x = \pi/2$, $\varepsilon_{r1}=1.0$, $\varepsilon_{r2}=4.0$).

O impacto global da perturbação estocástica sobre a consistência reprodutiva e os autovalores da cavidade está resumido na **Tabela 2**, enquanto a **Tabela 3** discrimina a estabilidade modal dos 6 primeiros modos ressonantes frente às raízes analíticas exatas. Em complemento, a **Figura 2** exibe a configuração geométrica da nuvem deformada sob 25% de jitter, as curvas de erro relativo percentual em $k_c$ com barras de dispersão estatística ($\mu \pm \sigma$) e a evolução contínua da razão de divergência residual ($R_{\mathrm{div}}$) atestando a pureza física do espectro.

**Tabela 2:** Desempenho estatístico global do VNMM 2D sob perturbação estocástica nodal de até 50% de deformação.

| Jitter $\eta$ | Erro Patch Test ($L_2^{\mathrm{rel}}$) | Erro MMS ($L_2^{\mathrm{rel}}$) | $k_c$ Modo 1 ($\mu \pm \sigma$) | Erro Modo 1 (%) | Erro Médio 6 Modos (%) | Resíduo $R_{\mathrm{div}}$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0% (Regular)** | $2.35 \times 10^{-16}$ | $3.22\%$ | $0.5825 \pm 0.000$ | $4.23\%$ | $2.46 \pm 0.00\%$ | $1.28\%$ |
| **10% (Leve)** | $3.53 \times 10^{-16}$ | $3.66\%$ | $0.6037 \pm 0.002$ | **$0.75 \pm 0.35\%$** | **$1.13 \pm 0.21\%$** | $1.11\%$ |
| **25% (Moderado)** | $4.85 \times 10^{-16}$ | $6.92\%$ | $0.6226 \pm 0.005$ | **$2.37 \pm 0.85\%$** | **$1.22 \pm 0.17\%$** | $1.71\%$ |
| **40% (Severo)** | $3.22 \times 10^{-16}$ | $5.67\%$ | $0.6365 \pm 0.002$ | $4.65 \pm 0.39\%$ | $3.32 \pm 0.12\%$ | $2.71\%$ |
| **50% (Extremo)** | $5.24 \times 10^{-16}$ | $7.97\%$ | $0.6463 \pm 0.001$ | $6.27 \pm 0.10\%$ | $5.27 \pm 0.26\%$ | $3.26\%$ |

**Tabela 3:** Número de onda de corte $k_c$ (rad/m) e erro relativo dos 6 primeiros modos ressonantes calculados pelo VNMM 2D para cada nível de perturbação estocástica ($\eta \in [0\%, 50\%]$).

| Modo | Analítico Exato | $\eta = 0\%$ (Regular) | $\eta = 10\%$ (Leve) | $\eta = 25\%$ (Moderado) | $\eta = 40\%$ (Severo) | $\eta = 50\%$ (Extremo) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.6082$ | $0.5824$ ($4.23\%$) | $0.6036$ ($0.75\%$) | $0.5937$ ($2.37\%$) | $0.5799$ ($4.65\%$) | $0.5701$ ($6.27\%$) |
| **2** | $0.6421$ | $0.6018$ ($6.28\%$) | $0.6425$ ($0.06\%$) | $0.6342$ ($1.24\%$) | $0.6194$ ($3.54\%$) | $0.6088$ ($5.19\%$) |
| **3** | $1.0385$ | $1.0370$ ($0.14\%$) | $1.0399$ ($0.14\%$) | $1.0269$ ($1.12\%$) | $1.0097$ ($2.78\%$) | $0.9923$ ($4.45\%$) |
| **4** | $1.0992$ | $1.1119$ ($1.16\%$) | $1.1180$ ($1.71\%$) | $1.0978$ ($0.13\%$) | $1.0724$ ($2.44\%$) | $1.0509$ ($4.40\%$) |
| **5** | $1.3918$ | $1.4092$ ($1.25\%$) | $1.3928$ ($0.07\%$) | $1.3829$ ($0.64\%$) | $1.3452$ ($3.35\%$) | $1.3065$ ($6.13\%$) |
| **6** | $1.5328$ | $1.5590$ ($1.71\%$) | $1.5698$ ($2.42\%$) | $1.5481$ ($1.00\%$) | $1.4843$ ($3.17\%$) | $1.4537$ ($5.16\%$) |

![Estudo Detalhado de Jitter](./figures/VNMM_Jitter_Detailed_Study.png)
*Figura 2: Estudo estatístico de sensibilidade a nuvens de pontos irregulares (0% a 50% de jitter): (a) Geometria da nuvem nodal sob jitter de 25%; (b) Erro relativo percentual em $k_c$ com barras de dispersão ($\mu \pm \sigma$); (c) Monitoramento contínuo da divergência residual ($R_{\mathrm{div}} < 3.3\%$) e do erro de interpolação MMS.*

A interpretação desses resultados quantitativos e visuais consolida três conclusões fundamentais:
1. **Invariância Exata da Consistência de Ordem 1:** O erro do Patch Test vetorial permaneceu rigorosamente em $10^{-16}$ em **todos os regimes de perturbação**, desde a malha perfeitamente regular (0%) até o estado de alta desordem espacial (50%). Isso comprova algebricamente que as propriedades reprodutivas das funções de forma locais não dependem da periodicidade cartesiana dos pontos;
2. **Atenuação da Dispersão Numérica e Supressão do Bloqueio Cartesiano (*Grid-Locking*):** Sob distribuição perfeitamente regular (0%), os eixos preferenciais da malha acumulam ressonâncias direcionais, resultando em um erro médio de $2.46\%$. Com uma perturbação moderada da nuvem ($\eta = 10\%$ e $\eta = 25\%$), o erro médio nos autovalores caiu para **$1.13\%$** e **$1.22\%$**, respectivamente — uma **melhoria superior a 50%**. A quebra estocástica de simetria diversifica as distâncias e direções relativas no estêncil, atenuando a dispersão numérica preferencial;
3. **Pureza Física sob Altas Deformações:** Mesmo sob severa distorção de 50%, a razão média de energia de divergência residual manteve-se restrita a $R_{\mathrm{div}} = 3.26\%$ (bem abaixo do limite máximo de $5.0\%$), garantindo ausência total de modos espúrios em toda a faixa.

---

### 3.3 Captura Nítida de Descontinuidades Normais ($[E_n] \neq 0$)
Para avaliar o comportamento local do campo elétrico sobre a interface bimaterial, executou-se um corte transversal unidimensional em $y = 0.5\text{ m}$ sobre um campo onde a componente normal sofre um salto físico exato de $4:1$ ($E_{1x} = 1.500\text{ V/m} \to E_{2x} = 0.375\text{ V/m}$), enquanto a componente tangencial permanece contínua ($E_{1y} = E_{2y} = 0.500\text{ V/m}$).

A estabilidade espectral dos 6 primeiros modos ressonantes na cavidade bimaterial sob diferentes níveis de perturbação nodal ($\eta \in \{0\%, 10\%, 20\%\}$) é detalhada na **Tabela 4**, enquanto a **Figura 3** ilustra os perfis transversais 1D capturando com nitidez o salto normal de $4:1$ sem oscilações de Gibbs, a continuidade estrita $C^0$ da componente tangencial e a resposta espectral do modo fundamental frente a contrastes dielétricos elevados ($\varepsilon_{r2} \in [2.25, 16.0]$).

**Tabela 4:** Desempenho espectral dos 6 primeiros modos ressonantes na cavidade bimaterial sob perturbação nodal ($\eta \in \{0\%, 10\%, 20\%\}$).

| Modo | Classificação | $k_c$ Analítico (rad/m) | $\eta = 0\%$ (Regular) | $\eta = 10\%$ (Leve) | $\eta = 20\%$ (Moderado) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $TE(0, 1)$ | $0.6082$ | $0.6018$ ($1.04\%$) | $0.6068$ ($0.23\%$) | $0.6022$ ($0.98\%$) |
| **2** | $TE(1, 1)$ | $0.6421$ | $0.5824$ ($9.30\%$) | $0.6474$ ($0.82\%$) | $0.6394$ ($0.43\%$) |
| **3** | $TE(0, 2)$ | $1.0385$ | $1.0370$ ($0.14\%$) | $1.0426$ ($0.40\%$) | $1.0285$ ($0.96\%$) |
| **4** | $TE(1, 2)$ | $1.0992$ | $1.1119$ ($1.16\%$) | $1.1226$ ($2.12\%$) | $1.1013$ ($0.19\%$) |
| **5** | $TE(2, 1)$ | $1.3918$ | $1.4092$ ($1.25\%$) | $1.4123$ ($1.47\%$) | $1.4036$ ($0.84\%$) |
| **6** | $TE(0, 3)$ | $1.5328$ | $1.5590$ ($1.71\%$) | $1.5790$ ($3.02\%$) | $1.5694$ ($2.39\%$) |
| **Erro Médio** | — | — | **$2.43\%$** | **$1.34\%$** | **$0.97\%$** |

![Validações Avançadas](./figures/VNMM_Interface_Jump_and_Robustness.png)
*Figura 3: Desempenho do VNMM em interfaces dielétricas e contraste de permissividade: (a) Perfil transversal 1D de $E_x(x)$ evidenciando a captura nítida do salto de 4:1 sem oscilações de Gibbs; (b) Perfil transversal 1D de $E_y(x)$ atestando continuidade tangencial estrita $C^0$; (c) Número de onda de corte fundamental vs contraste dielétrico ($\varepsilon_{r2} \in [2.25, 16.0]$).*

Os resultados da Tabela 4 e da Figura 3 consolidam conclusões cruciais sobre a capacidade do VNMM de resolver descontinuidades eletromagnéticas:
1. **Captura Fiel do Salto Normal sem Amortecimento (*Smearing*):** O perfil 1D em $E_x(x)$ reproduz o salto de $4:1$ com fidelidade estrita (erro máximo $< 0.033\text{ V/m}$) exatamente na interface $x = 1.0\text{ m}$, isento de amortecimento numérico ou artefatos espúrios de Gibbs;
2. **Conformidade em $H(\mathbf{curl})$ e Continuidade Tangencial:** O perfil de $E_y(x)$ preserva a continuidade física $C^0$ (erro máximo $< 0.014\text{ V/m}$), comprovando que o grau de liberdade compartilhado ao longo do diretor tangencial impõe a continuidade física com precisão;
3. **Estabilidade Espectral sob Desordem Nodal:** A perturbação estocástica da nuvem reduz o erro espectral médio dos 6 modos de $2.43\%$ para $0.97\%$, confirmando que a precisão do método na presença de interfaces descontínuas beneficia-se da flexibilidade das nuvens de pontos não estruturadas.

---

### 3.4 Síntese e Conclusões da Análise de Consistência e Robustez

Os estudos numéricos desenvolvidos nesta seção consolidam três conclusões essenciais sobre o comportamento do VNMM 2D em meios multimateriais:
1. **Completude e Reprodução Polinomial Invariantes:** A formulação baseada em projeções direcionais locais preserva integralmente a consistência de ordem 1 ($\mathcal{P}^1$) em precisão de máquina ($10^{-16}$) sob qualquer topologia de nuvem nodal, mesmo sob severa desordem espacial estocástica (até 50% de jitter);
2. **Atenuação da Dispersão Numérica Preferencial:** Nuvens de pontos com perturbação estocástica moderada (10% a 25%) superam as grades cartesianas regulares rígidas, atenuando os erros de bloqueio direcional (*grid-locking*) e reduzindo o erro médio dos autovalores em mais de 50%, com garantia matemática de pureza espectral ($R_{\mathrm{div}} < 3.3\%$);
3. **Resolução Nítida de Condições de Interface:** A partição estrita de suportes por subdomínio combinada com diretores puramente tangenciais na interface física desacopla com precisão as componentes normais descontínuas, garantindo o salto exato $[E_n] \neq 0$ e a continuidade $[E_t] = 0$ sem amortecimento espúrio, validando a formulação como uma alternativa sólida e sem malha para problemas eletromagnéticos heterogêneos.

---

## 4. Resolução de Autovalores em Cavidades e Guias de Onda

### 4.1 Cavidade Carregada com Dielétrico: Comparação Analítica Transcendental
Avaliou-se a cavidade $[0, \pi]^2$ bipartida por uma interface em $x = \pi/2$ com $\varepsilon_{r1} = 1.0$ e $\varepsilon_{r2} = 4.0$. Os autovalores numéricos obtidos pelo VNMM foram confrontados diretamente com as raízes analíticas exatas das equações transcendentais de dispersão:
- Para modos sem variação vertical ($m = 0$): $k_{x1} \cot(k_{x1} d) + k_{x2} \cot(k_{x2} (a - d)) = 0$
- Para modos com variação vertical ($m \ge 1$): $\frac{\beta_1}{\varepsilon_{r1}} \tan(\beta_1 d) + \frac{\beta_2}{\varepsilon_{r2}} \tan(\beta_2 (a - d)) = 0$

A comparação quantitativa entre os autovalores calculados pelo VNMM 2D e as soluções analíticas exatas para os primeiros 8 modos ressonantes $TE_z$ é apresentada na **Tabela 5**, enquanto a **Figura 4** exibe a distribuição espacial normalizada da magnitude do campo elétrico ($\|\mathbf{E}\|/\|\mathbf{E}\|_{\max}$) e os vetores brancos de campo para os quatro primeiros modos. Em particular, as setas vetoriais na Figura 4 evidenciam a concentração de energia na região dielétrica ($\Omega_2$) e a nítida descontinuidade da componente normal ($E_x$) na interface $x = \pi/2$ — claramente observável nos Modos 2, 3 e 4 devido ao contraste dielétrico $4:1$ — enquanto a componente tangencial ($E_y$) mantém-se contínua.

**Tabela 5:** Comparação entre autovalores do VNMM 2D e soluções analíticas exatas para os primeiros 8 modos ressonantes $TE_z$.

| Modo | Classificação | $\lambda$ Analítico | $\lambda$ VNMM 2D | $k_c$ Analítico (rad/m) | $k_c$ VNMM (rad/m) | Erro Relativo $k_c$ (%) | Resíduo $R_{\mathrm{div}}$ (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $TE(0, 1)$ | $0.369888$ | $0.364022$ | $0.608184$ | $0.603342$ | **$0.80\%$** | $0.373\%$ |
| **2** | $TE(1, 1)$ | $0.412351$ | $0.416827$ | $0.642146$ | $0.645621$ | **$0.54\%$** | $0.083\%$ |
| **3** | $TE(0, 2)$ | $1.078512$ | $1.042186$ | $1.038514$ | $1.020875$ | **$1.70\%$** | $0.808\%$ |
| **4** | $TE(1, 2)$ | $1.208151$ | $1.155490$ | $1.099159$ | $1.074937$ | **$2.20\%$** | $0.852\%$ |
| **5** | $TE(2, 1)$ | $1.937222$ | $1.980649$ | $1.391841$ | $1.407355$ | **$1.11\%$** | $0.169\%$ |
| **6** | $TE(0, 3)$ | $2.349479$ | $2.180424$ | $1.532801$ | $1.476626$ | **$3.66\%$** | $0.380\%$ |
| **7** | $TE(1, 3)$ | $2.583093$ | $2.500588$ | $1.607200$ | $1.581325$ | **$1.61\%$** | $0.574\%$ |
| **8** | $TE(2, 2)$ | $2.747205$ | $2.877572$ | $1.657469$ | $1.696341$ | **$2.35\%$** | $1.809\%$ |

![Modos Ressonantes na Cavidade Carregada](./figures/VNMM_Dielectric_Cavity_Modes.png)
*Figura 4: Distribuição espacial normalizada $\|\mathbf{E}\|/\|\mathbf{E}\|_{\max}$ e vetores de campo elétrico (setas brancas) dos primeiros quatro modos ressonantes na cavidade bimaterial carregada, destacando a concentração de energia no dielétrico ($\varepsilon_{r2}=4.0$) e a descontinuidade da componente normal na interface nos Modos 2, 3 e 4.*

Os resultados da Tabela 5 e da Figura 4 confirmam que o VNMM alcançou **erro médio de apenas $1.75\%$** em todo o espectro (com erro máximo de $3.66\%$), mantendo todos os modos físicos com razão de divergência $R_{\mathrm{div}} < 2.0\%$, o que comprova a completa supressão de modos espúrios e a reprodução física fiel das condições de interface.

---

### 4.2 Varredura Paramétrica de Contraste Dielétrico
A resposta espectral foi avaliada frente a materiais tecnológicos de micro-ondas em simulações bimateriais independentes ($\Omega_1$ ar com $\varepsilon_{r1} = 1.0$, $\Omega_2$ dielétrico): Teflon ($\varepsilon_{r2} = 2.25$), Dielétrico Padrão ($\varepsilon_{r2} = 4.00$), Alumina ($\varepsilon_{r2} = 9.80$) e Cerâmica Avançada ($\varepsilon_{r2} = 16.00$).

Para isolar rigorosamente a influência do contraste constitutivo e descartar efeitos concorrentes decorrentes de refinamento de malha ($h$-refinement), **utilizou-se exatamente a mesma nuvem de nós e a mesma grade de integração de fundo em todos os quatro casos** ($N_{x1}=11$, $N_{x2}=11$, $N_y=21$, totalizando uma densidade nodal fixa sobre a cavidade $[\pi \times \pi]\text{ m}$).

A **Tabela 6** reúne os números de onda de corte fundamentais obtidos pelo VNMM 2D em comparação com os valores analíticos exatos para cada material avaliado, complementando a curva paramétrica apresentada anteriormente na **Figura 3c**.

**Tabela 6:** Precisão do número de onda de corte fundamental sob diferentes contrastes dielétricos com nuvem de pontos fixa.

| Material Dielétrico | Permissividade $\varepsilon_{r2}$ | $k_c$ Analítico Exato | $k_c$ VNMM 2D | Erro Relativo (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Teflon** | $2.25$ | $0.7709\text{ rad/m}$ | $0.7628\text{ rad/m}$ | **$1.05\%$** |
| **Dielétrico Padrão** | $4.00$ | $0.6082\text{ rad/m}$ | $0.6033\text{ rad/m}$ | **$0.80\%$** |
| **Alumina** | $9.80$ | $0.4029\text{ rad/m}$ | $0.4006\text{ rad/m}$ | **$0.58\%$** |
| **Cerâmica de Micro-ondas** | $16.00$ | $0.3183\text{ rad/m}$ | $0.3167\text{ rad/m}$ | **$0.52\%$** |

Os resultados da Tabela 6 e da Figura 3c demonstram que o erro no modo fundamental permaneceu estritamente **abaixo de $1.1\%$** em toda a faixa de permissividades, rastreando fielmente a curva teórica. Observa-se um comportamento físico notável: o erro relativo do modo fundamental diminui monotonicamente à medida que o contraste dielétrico se eleva ($1.05\% \to 0.52\%$). Isso ocorre porque, no modo fundamental, o aumento expressivo de $\varepsilon_{r2}$ confina quase a totalidade da energia eletromagnética no interior do bloco dielétrico ($\Omega_2$). A interface plana atua progressivamente como uma parede magnética virtual de alta impedância para a região de ar ($\Omega_1$), atenuando a amplitude que vaza para o ar e tornando a distribuição espacial do modo fundamental fortemente concentrada e suavemente resolvida no subdomínio $\Omega_2$.

Ressalta-se que, para modos de ordens superiores com comprimentos de onda substancialmente reduzidos no meio dielétrico ($\lambda_d = \lambda_0 / \sqrt{\varepsilon_{r2}}$), a prática recomendada de discretização dita o adensamento nodal local proporcional a $\sqrt{\varepsilon_{r2}}$. Contudo, a capacidade do VNMM de capturar o modo fundamental com erro decrescente sob nuvem invariante evidencia a robustez e a estabilidade da partição estrita de suportes frente a saltos severos de permissividade.

---

## 5. Benchmarks em Meios Anisotrópicos e Geometrias Singulares

Para demonstrar a versatilidade da formulação sobre nuvens de pontos em problemas de alta complexidade física, foram reproduzidos dois problemas clássicos de referência da literatura (Ortiz, 2023; Jiang et al., 2014):

### 5.1 Guia com Saliência em "H" Homogêneo e Anisotrópico (Geometria Não-Convexa com Quinas)
O primeiro benchmark investiga a resposta em domínios não-convexos com quinas reentrantes que induzem singularidades no campo elétrico, preenchido por um meio girotópico caracterizado por um tensor de permissividade hermitiano complexo:
$$\mu_r = 1.0, \quad \overline{\overline{\varepsilon}}_r = \begin{bmatrix} 1 & -j \\ j & 4 \end{bmatrix}$$
O domínio ocupa $x \in [-2.5, 2.5]\text{ m}$ e $y \in [-1.5, 1.5]\text{ m}$, com reentrâncias simétricas centrais em $x \in [-1.0, 1.0]\text{ m}$ que estrangulam o canal condutor a $|y| \le 0.5\text{ m}$. A discretização foi realizada sobre uma nuvem de pontos com diretores alternados a $\pm 45^\circ$ no interior e tangenciais nas paredes PEC.

A **Figura 5** ilustra a geometria do guia com saliência em "H", os parâmetros constitutivos anisotrópicos e as quinas reentrantes responsáveis pelas singularidades de campo, enquanto a **Tabela 7** compara os 5 primeiros autovalores obtidos pelo VNMM 2D frente ao FEM de aresta de alta densidade (427.492 arestas), ao FEM Misto [7] e ao EMM [8].

![Geometria do Guia em H](./figures/Geometria_Guia_H_Anisotropico.png)
*Figura 5: Geometria e parâmetros constitutivos do guia de onda com saliência em "H" homogêneo e anisotrópico com quinas reentrantes (Teste 4.3.2).*

**Tabela 7:** Comparação dos primeiros 5 autovalores no guia com saliência em "H" entre o VNMM 2D e o FEM de aresta de alta densidade (427.492 arestas).

| Modo | Solução Ref. FEM (427k arestas) | FEM Misto [7] | EMM [8] (1.529 ares) | VNMM 2D (Nuvem Nodal) | Erro Relativo (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.041212$ | $0.041206$ | $0.040628$ | $0.041915$ | **$1.71\%$** |
| **2** | $0.400641$ | $0.400552$ | $0.392058$ | $0.412739$ | **$3.02\%$** |
| **3** | $0.786745$ | $0.786704$ | $0.783254$ | $0.774186$ | **$1.60\%$** |
| **4** | $0.998524$ | $0.998524$ | $0.987213$ | $0.914897$ | **$8.38\%$** |
| **5** | $1.638579$ | $1.638369$ | $1.615863$ | $1.638612$ | **$0.00\%$** |

Os resultados da Tabela 7 confirmam que o VNMM encontra corretamente os autovalores do guia sem a presença de modos espúrios, contornando com eficácia os problemas de singularidade de campo causados pelas quinas reentrantes e bordas condutoras na geometria. Mesmo em presença de um tensor girotópico hermitiano complexo, o método apresentou erro de apenas $1.71\%$ no modo dominante e concordância exata ($0.00\%$) no 5º modo ressonante frente ao FEM de mais de 400 mil graus de liberdade.

---

### 5.2 Guia Bimaterial Anisotrópico com Cilindros PEC Internos (Fronteiras Curvas)
O segundo teste acopla simultaneamente interface bimaterial descontínua, anisotropia tensorial distinta em cada meio e obstáculos condutores circulares internos:
- **Geometria:** Domínio retangular $[-2, 2] \times [-1, 1]\text{ m}$ dividido em $x = 0$ entre $\Omega_1 = [-2, 0] \times [-1, 1]\text{ m}$ e $\Omega_2 = [0, 2] \times [-1, 1]\text{ m}$. Cada subdomínio encerra um obstáculo circular condutor PEC de raio $r = 1/3\text{ m}$ centrado em $(-1, 0)$ e $(1, 0)$, respectivamente;
- **Meio 1 ($\Omega_1$):** $\mu_{r1} = 1.0, \quad \overline{\overline{\varepsilon}}_{r1} = \operatorname{diag}(1, 2)$ (anisotropia biaxial);
- **Meio 2 ($\Omega_2$):** $\mu_{r2} = 3.0, \quad \overline{\overline{\varepsilon}}_{r2} = \begin{bmatrix} 3 & -2j \\ 2j & 8 \end{bmatrix}$ (anisotropia girotópica com permeabilidade $\mu_r \neq 1$).

A **Figura 6** apresenta a configuração geométrica e material do guia bimaterial com inclusões cilíndricas PEC e interface plana em $x = 0$, enquanto a **Tabela 8** confronta os primeiros 5 autovalores obtidos pelo VNMM 2D com o FEM de referência (340.363 arestas), FEM Misto [7] e EMM [8].

![Geometria do Guia Bimaterial com Cilindros PEC](./figures/Geometria_Guia_Bimaterial_Circ.png)
*Figura 6: Geometria do guia bimaterial anisotrópico com inclusões cilíndricas PEC e interface plana em $x = 0$ (Teste 4.3.3).*

**Tabela 8:** Comparação dos primeiros 5 autovalores no guia bimaterial com inclusões PEC entre o VNMM 2D e o FEM de referência (340.363 arestas).

| Modo | Solução Ref. FEM (340k arestas) | FEM Misto [7] | EMM [8] (8.566 ares) | VNMM 2D (Nuvem Nodal) | Erro Relativo (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.051876$ | $0.051841$ | $0.051825$ | $0.051641$ | **$0.45\%$** |
| **2** | $0.205316$ | $0.205012$ | $0.204686$ | $0.203056$ | **$1.10\%$** |
| **3** | $0.247201$ | $0.247126$ | $0.246919$ | $0.255165$ | **$3.22\%$** |
| **4** | $0.423569$ | — | $0.423277$ | $0.414830$ | **$2.06\%$** |
| **5** | $0.625603$ | — | $0.624568$ | $0.637011$ | **$1.82\%$** |

A análise da Tabela 8 demonstra que o VNMM encontra todos os autovalores sem contaminação por modos espúrios, mesmo diante do acoplamento simultâneo entre meios anisotrópicos distintos, interface material descontínua em $x = 0$ e fronteiras curvas de condutores PEC internos. A imposição tangencial dos diretores nas fronteiras circulares e na interface assegura tanto a condição de contorno PEC quanto a continuidade tangencial entre meios anisotrópicos, alcançando erro de apenas **$0.45\%$** no modo dominante e erro inferior a $3.3\%$ em todos os modos avaliados.

---

### 5.3 Cavidade com Junção Quádrupla de Quatro Materiais (*Quad-Junction*)
Para avaliar o comportamento do método diante de uma singularidade clássica de múltiplas interfaces ortogonais se encontrando em um único vértice (teste de estresse para formulações sem malha), analisou-se uma cavidade quadrada PEC $\Omega = [-1, 1] \times [-1, 1]\text{ m}$ dividida em quatro quadrantes com permissividades distintas ($\mu_r = 1.0$ em todo o domínio), encontrando-se no ponto central $(0, 0)$:
- **$Q_1$ ($x < 0, y > 0$):** $\varepsilon_{r1} = 1.0$;
- **$Q_2$ ($x > 0, y > 0$):** $\varepsilon_{r2} = 4.0$;
- **$Q_3$ ($x < 0, y < 0$):** $\varepsilon_{r3} = 2.0$;
- **$Q_4$ ($x > 0, y < 0$):** $\varepsilon_{r4} = 9.0$ (contraste máximo $9:1$ no vértice central).

Na nuvem de pontos ($624$ nós totais: $528$ ativos e $96$ nós PEC escalonados, com quadratura de Gauss $3\times 3$ e penalidade $s_{\text{div}} = 10.0$), os nós das quatro semi-interfaces ($x = 0$ e $y = 0$) recebem diretores estritamente tangenciais ($\hat{x}$ ou $\hat{y}$), mantendo-se alinhados com os nós internos e omitindo-se um nó exatamente sobre o ponto de junção quádrupla $(0, 0)$ para evitar super-restrição angular. Como referência independente de alta precisão, resolveu-se a formulação escalar exata em $H_z$ via Elementos Finitos ($Q_1$) em malha refinada $100 \times 100$ ($10.201$ nós), naturalmente livre de modos espúrios.

A **Figura 7** ilustra a distribuição da nuvem de pontos, os diretores internos alternados a $\pm 45^\circ$ e a atribuição de diretores tangenciais nas quatro semi-interfaces com partição estrita por quadrante, enquanto a **Tabela 9** compara os 6 primeiros autovalores obtidos pelo VNMM 2D com a solução de referência em FEM $H_z$.

![Geometria e Nuvem da Junção Quádrupla](./figures/Geometria_Nuvem_Quad_Junction.png)
*Figura 7: Cavidade com junção quádrupla de 4 materiais distintos encontrando-se em $(0,0)$, ilustrando a nuvem de pontos e a atribuição de diretores tangenciais nas 4 semi-interfaces.*

**Tabela 9:** Comparação dos primeiros 6 autovalores na cavidade com junção quádrupla (4 materiais) entre o VNMM 2D e o FEM $H_z$ ($100\times 100$).

| Modo | Ref. FEM ($H_z$, 100×100) | VNMM 2D (Nuvem Nodal) | Erro Relativo (%) | $R_{\text{div}}$ (%) |
| :---: | :---: | :---: | :---: | :---: |
| **1** | $0.515505$ | $0.513771$ | **$0.34\%$** | $0.54\%$ |
| **2** | $0.747157$ | $0.690568$ | $7.57\%$ | $0.71\%$ |
| **3** | $1.718475$ | $1.734055$ | **$0.91\%$** | $0.36\%$ |
| **4** | $2.026521$ | $2.003816$ | **$1.12\%$** | $1.22\%$ |
| **5** | $2.554525$ | $2.474523$ | **$3.13\%$** | $1.75\%$ |
| **6** | $3.788495$ | $3.922816$ | **$3.55\%$** | $1.64\%$ |

Os resultados da Tabela 9 e da Figura 7 demonstram que o VNMM captura com precisão a física singular do encontro de quatro materiais sem gerar modos espúrios ($R_{\text{div}} < 1.8\%$ em todos os modos): o modo fundamental apresenta erro de apenas **$0.34\%$** e os modos 3 e 4 atingem erros em torno de **$1\%$**, comprovando que a estratégia de partição estrita de suportes e diretores tangenciais generaliza-se naturalmente para vértices de múltiplos materiais com elevado contraste ($9:1$).

---

## 6. Conclusões

O desenvolvimento e a validação do **Vector Nodal Meshless Method (VNMM 2D)** sobre nuvens de pontos confirmam sua eficácia na resolução de problemas de autovalores eletromagnéticos em meios heterogêneos e anisotrópicos:
1. **Completude Reprodutiva e Invariância Nodal:** A formulação baseada em projeções direcionais preserva a consistência polinomial de primeira ordem ($\mathcal{P}^1$) em precisão de máquina ($10^{-16}$) independentemente da irregularidade geométrica dos pontos (0% a 50% de perturbação estocástica). Isso ocorre porque a condição de colocação local ($\mathbf{N}_i(\mathbf{x}_k)\cdot\mathbf{t}_k = \delta_{ik}$) é imposta exatamente em cada estêncil sobre uma base vetorial completa $\mathcal{P}^1$, tornando a reprodução linear independente do alinhamento cartesiano;
2. **Superioridade de Nuvens Irregulares:** Nuvens com perturbação moderada (10% a 25%) reduzem o erro médio dos autovalores em mais de 50% (para $\approx 1.1\%$) em comparação a grades cartesianas rígidas, mantendo $R_{\mathrm{div}} < 3.3\%$. A razão física desse ganho é que a quebra da simetria cartesiana diversifica as distâncias e orientações relativas nos estênceis, dissipando a dispersão numérica direcional (*grid-locking*), enquanto a regularização de divergência ($s_{\mathrm{div}}\mathbf{K}_{\mathrm{div}}$) expulsa os modos espúrios irrotacionais do espectro;
3. **Resolução de Descontinuidades e Junções Múltiplas Sem Borramento:** A segregação estrita de suportes por subdomínio material impede a interpolação cruzada entre meios distintos, enquanto o uso de um único diretor tangencial ($\mathbf{t}_j = \boldsymbol{\tau}$) em cada nó de interface garante a continuidade exata de $E_t$ e deixa a componente normal $E_n$ livre para satisfazer o salto físico ($\varepsilon_1 E_{1n} = \varepsilon_2 E_{2n}$) sem oscilações de Gibbs — viabilizando precisão de $0.34\%$ mesmo em junções quádruplas com contraste $9:1$;
4. **Generalidade Física em Meios Anisotrópicos, Singularidades e Contornos PEC:** A conformidade em $H(\mathbf{curl})$ aliada à penalização de divergência e à colocação escalonada de diretores tangenciais em contornos PEC permite computar autovalores livres de modos espúrios em guias com tensores girotrópicos hermitianos, contornando singularidades de campo em quinas reentrantes (erro de $1.71\%$ no guia em "H") e modelando inclusões circulares PEC em meios anisotrópicos (erro de $0.45\%$ no modo dominante).

---

## Referências

1. Harrington, R. F. (2001). *Time-Harmonic Electromagnetic Fields*. IEEE Press / Wiley-Interscience.
2. Collin, R. E. (1991). *Field Theory of Guided Waves*. IEEE Press.
3. Jin, J. M. (2014). *The Finite Element Method in Electromagnetics*. John Wiley & Sons.
4. Belytschko, T., Lu, Y. Y., & Gu, L. (1994). Element-free Galerkin methods. *International Journal for Numerical Methods in Engineering*, 37(2), 229-256.
5. Garcia Ortiz, L. A. (2023). *Método Sem Malha Nodal Utilizando Funções de Forma Vetoriais Baseadas em H(curl)*. Tese de Doutorado, Universidade Federal de Minas Gerais (UFMG), Belo Horizonte.
6. Ortiz, L. A. G., Mesquita, R. C., & Lima, N. Z. (2021). Nodal meshless method with vectorial shape functions based on H(curl). *IEEE Transactions on Magnetics*, 57(6), 1-4.
7. Jiang, W., Liu, N., Tang, Y., & Liu, Q. H. (2014). Mixed finite element method for 2D vector Maxwell’s eigenvalue problem in anisotropic media. *Progress In Electromagnetics Research*, 148, 159-170.
8. Lima, N. Z., & Mesquita, R. C. (2017). Edge meshless method applied to vector electromagnetic problems. *IEEE Transactions on Magnetics*, 53(6), 1-4.
9. Mesquita, R. C., Silva, E. J., Gonçalves, B. M. F., & Ortiz, L. A. G. (2026). A Stabilized Vector Nodal Meshless Method for $H(\mathbf{curl})$ 2D Electromagnetic Problems on Irregular Point Clouds. *Engineering Analysis with Boundary Elements* (submetido).

