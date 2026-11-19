# Relatório Técnico: Consistência Global, Multiplicidade Espacial $M(\mathbf{x})$ e MMS Volumétrico Convencional no VNMM-MLPG 2D

**Data:** 2026-11-18  
**Referências principais:** [main-vnmm-mlpg.tex](file:///home/renato/Insync/rencmbr@gmail.com/Google%20Drive/doc/CNPq/Projeto2026/VNMM2D-Estudo/VNMM/paper/mlpg-vnmm/main-vnmm-mlpg.tex), [mlpg_source_problem_2d.py](file:///home/renato/Insync/rencmbr@gmail.com/Google%20Drive/doc/CNPq/Projeto2026/VNMM2D-Estudo/VNMM/2D/scripts/mlpg_source_problem_2d.py)

---

## 1. Objetivo

Este relatório consolida as demonstrações analíticas, os experimentos numéricos diagnósticos e as modificações aplicadas ao código e ao manuscrito do VNMM-MLPG 2D para resolver de forma transparente três questões matemáticas críticas da formulação *Symmetric Subdomain Galerkin* (SSG):

1. **Questão 2.1 — Consistência da montagem sobre subdomínios sobrepostos:** Decomposição estrutural e quantificação exata do resíduo nas fronteiras circulares artificiais $\mathcal{R}_h(\vec{w}_h)$.
2. **Questão 2.2 — Interpretação da multiplicidade espacial $M(\mathbf{x})$ no quociente de Rayleigh:** Demonstração por homogeneização assintótica em malhas regulares ($\mathcal{O}(h_{\mathrm{nom}}^2)$) e quantificação direta do desvio espectral sobre os 10 primeiros modos $\mathrm{TE}_{nm}$ em malhas regulares e perturbadas ($25\%$ de *jitter*).
3. **Questão 2.3 — Benchmark MMS convencional com fonte volumétrica pura (sem fluxos artificiais em $L_{sI}$):** Avaliação sob refinamento ($15\times 15$ a $49\times 49$) com e sem estabilização ortogonal (incluindo termo fonte volumétrico consistente $\vec{J}_s^{\mathrm{stab}}$) e isolamento experimental das duas parcelas do Lema de Strang.

---

## 2. Questão 2.2: Multiplicidade Espacial $M(\mathbf{x})$ no Quociente de Rayleigh

### 2.1 Formulação Exata do Quociente Ponderado
Para um campo contínuo solenoidal $\vec{E}$ ($\vec{E}_{I,h} = \vec{E}|_{D_I}$, $\nabla \cdot \vec{E} = 0$), a soma sobre os subdomínios sobrepostos $\{D_I\}_{I=1}^N$ conduz ao quociente contínuo ponderado pela multiplicidade espacial $M(\mathbf{x}) = \sum_{I=1}^N \chi_{D_I}(\mathbf{x}) = \bar{M} + \widetilde{M}(\mathbf{x})$:

$$
\lambda^{(M)}(\vec{E}) = \frac{\iint_{\Omega} M(\mathbf{x})\,\mu_r^{-1}\,|\nabla \times \vec{E}|^2\,d\Omega}{\iint_{\Omega} M(\mathbf{x})\,\epsilon_r\,|\vec{E}|^2\,d\Omega}.
$$

Enquanto a média global $\bar{M}$ se cancela algebricamente entre numerador e denominador, a razão $M(\mathbf{x})/\bar{M} = 1 + \widetilde{M}(\mathbf{x})/\bar{M} \ne 1$ introduz uma ponderação espacial não uniforme, mesmo em malhas cartesianas regulares com $r_s^I = 0.80 h_I$.

### 2.2 Homogeneização Assintótica e Cancelamento na Fronteira
Em malhas cartesianas regulares de espaçamento $h_{\mathrm{nom}}$ com nós de contorno escalonados (*staggered*) nos pontos médios dos subintervalos:
- Em cada célula periódica interna $Q_{ij}$, a flutuação $\widetilde{M}(\mathbf{x}) = M(\mathbf{x}) - \bar{M}$ possui momento de ordem zero $\mathcal{O}(h_{\mathrm{nom}}^4)$ e momento de primeira ordem identicamente nulo por simetria de reflexão ($\iint_{Q_{ij}} (\mathbf{x}-\mathbf{x}_{ij}^c)\widetilde{M}\,d\Omega = \mathbf{0}$).
- Ao longo do contorno $\partial\Omega$, os meio-discos escalonados compensam a meia faixa de borda das linhas internas adjacentes (restando apenas sobreposições nos 4 cantos de área $\mathcal{O}(h_{\mathrm{nom}}^2)$, com $|\bar{M} - \pi\alpha_s^2| = \mathcal{O}(h_{\mathrm{nom}}^2)$).
- Consequentemente, para densidades modais suaves $f \in W^{2,\infty}(\Omega)$:

$$
\left|\iint_{\Omega} \frac{\widetilde{M}(\mathbf{x})}{\bar{M}}\,f(\mathbf{x})\,d\Omega\right| \le C_{\mathrm{hom}}\,h_{\mathrm{nom}}^2 \|f\|_{W^{2,\infty}(\Omega)} \implies \left|\lambda^{(M)}(\vec{E}) - \lambda(\vec{E})\right| = \mathcal{O}(h_{\mathrm{nom}}^2).
$$

Alternativamente, a ponderação pontual $w_I(\mathbf{x}) = \chi_{D_I}(\mathbf{x})/M(\mathbf{x})$ restaura peso unitário sobre $\Omega_{\mathrm{cov}}$, porém requer quadratura por sub-regiões nas lentes de interseção.

### 2.3 Quantificação Espectral nos 10 Primeiros Modos $\mathrm{TE}_{nm}$
Os desvios relativos $k_{c,nm}^{(M)} = \sqrt{\lambda^{(M)}(\vec{E}_{nm}^*)}$ médios e máximos sobre os 10 primeiros modos exatos da cavidade PEC ($[0, \pi]^2$, $\alpha_s = 0.80$, `seed=42`, calculados por [`quantify_multiplicity_rayleigh_perturbation`](file:///home/renato/Insync/rencmbr@gmail.com/Google%20Drive/doc/CNPq/Projeto2026/VNMM2D-Estudo/VNMM/2D/scripts/mlpg_source_problem_2d.py#L591-L656)) são:

| Malha | $h_{\mathrm{nom}}$ [m] | $0\%$ Jitter — Média $\|k_c^{(M)} - k_c\|/k_c$ | $0\%$ Jitter — Máximo | $25\%$ Jitter — Média $\|k_c^{(M)} - k_c\|/k_c$ | $25\%$ Jitter — Máximo |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $15\times 15$ | 0.2244 | $0.6349\%$ | $1.0508\%$ | $2.0265\%$ | $3.9211\%$ |
| $22\times 22$ | 0.1496 | $0.3192\%$ | $0.4865\%$ | $1.4914\%$ | $2.5917\%$ |
| $29\times 29$ | 0.1122 | $0.1873\%$ | $0.2776\%$ | $1.4265\%$ | $2.3104\%$ |
| $36\times 36$ | 0.0898 | $0.1223\%$ | $0.1788\%$ | $1.2064\%$ | $1.7843\%$ |
| $43\times 43$ | 0.0748 | $0.0858\%$ | $0.1246\%$ | $0.9625\%$ | $1.2219\%$ |
| $49\times 49$ | 0.0654 | $0.0661\%$ | $0.0956\%$ | $1.3300\%$ | $2.0905\%$ |
| **Taxa Empírica $p$** | --- | **$1.84$** | --- | **$0.46$** | --- |

- **Em malhas regulares ($0\%$):** O erro geométrico induzido por $M(\mathbf{x})$ decai com taxa $p = 1.84 \approx 2$ (caindo para $0.066\%$ em $49\times 49$), tornando-se desprezível frente ao erro de discretização ($1.29\%$).
- **Em nuvens perturbadas ($25\%$):** A quebra da periodicidade e os $5.6\%$ de vazios intersticiais ($M(\mathbf{x}) = 0$) mantêm um piso geométrico de $\sim 1.0\%\text{--}1.3\%$, explicando tanto o patamar residual quanto a leve oscilação de $1.77\%$ ($43^2$) para $1.97\%$ ($49^2$) observada nos autovalores da cavidade com *jitter* (Tabela 11 do manuscrito).

---

## 3. Questão 2.1: Decomposição Estrutural e Norma Dual do Resíduo $\mathcal{R}_h$

Para qualquer campo admissível globalmente contínuo $\vec{w} \in \boldsymbol{H}_0(\mathbf{curl}, \Omega)$ que interpole os graus de liberdade nodais de $\vec{w}_h \in \mathcal{V}_{h,0}$ e vetor de fluxo $\vec{\Phi}(\vec{E}, \vec{w}) \equiv \mu_r^{-1}(\nabla\times\vec{E})\times\vec{w} + s_{\mathrm{div}}\mu_r^{-1}(\nabla\cdot\vec{E})\vec{w}$, o teorema da divergência decompõe o funcional de consistência exatamente em duas parcelas:

$$
\mathcal{R}_h(\vec{w}_h) = \sum_{I=1}^N \int_{L_{sI}} \vec{\tau}(\vec{E})\cdot\vec{w}_{I,h}\,ds =
\underbrace{\iint_{\Omega} \widetilde{M}(\mathbf{x})\nabla\cdot\vec{\Phi}(\vec{E},\vec{w})\,d\Omega - \int_{\partial\Omega}\widetilde{M}(\mathbf{x})\vec{\tau}(\vec{E})\cdot\vec{w}\,ds}_{\mathcal{R}_h^{\mathrm{mult}}(\vec{w})}
+ \underbrace{\sum_{I=1}^N \int_{L_{sI}} \vec{\tau}(\vec{E})\cdot(\vec{w}_{I,h} - \vec{w})\,ds}_{\mathcal{R}_h^{\mathrm{disc}}(\vec{w}_h)}.
$$

No problema elíptico deslocado $(\mathbf{K} + \kappa\mathbf{M})\mathbf{e} = \mathbf{F}$ ($\kappa > 0$), tem-se $\mathcal{R}_h(\vec{w}_h) = \mathbf{w}_{\mathrm{int}}^T \mathbf{R}_{h,\mathrm{int}}$ com $\mathbf{R}_{h,\mathrm{int}} = (\mathbf{F}_{\mathrm{eq}} - \mathbf{F}_{\mathrm{vol}})_{\mathrm{int}}$. Pelo Teorema da Representação de Riesz na norma de energia discreta $\|\vec{w}_h\|_{h,a,\kappa} = \sqrt{\mathbf{w}_{\mathrm{int}}^T \mathbf{A}_{\mathrm{int},\mathrm{int}} \mathbf{w}_{\mathrm{int}}}$, o supremo do Lema de Strang possui forma fechada exata:

$$
\|\mathcal{R}_h\|_{\mathcal{V}_{h,0}^*} \equiv \sup_{\vec{w}_h \in \mathcal{V}_{h,0}\setminus\{\mathbf{0}\}} \frac{|\mathcal{R}_h(\vec{w}_h)|}{\|\vec{w}_h\|_{h,a,\kappa}} = \sqrt{\mathbf{R}_{h,\mathrm{int}}^T \mathbf{A}_{\mathrm{int},\mathrm{int}}^{-1} \mathbf{R}_{h,\mathrm{int}}} = \|\vec{E}_h^{\mathrm{eq}} - \vec{E}_h^{\mathrm{vol}}\|_{h,a,\kappa}.
$$

Eliminar inteiramente $\mathcal{R}_h$ exigiria acoplamento/penalização de saltos sobre lentes de interseção $\Omega_s^I \cap \Omega_s^J$, ponderação por partição da unidade $1/M(\mathbf{x})$ acoplada à estabilização de gradientes ou correções de Consistência Variacional (VCI). Na formulação atual em subdomínios autônomos em forma fechada, $\|\mathcal{R}_h\|_{\mathcal{V}_{h,0}^*}/\|\mathbf{e}^*\|_{\mathbf{A}}$ permanece $\mathcal{O}(1)$ ($\approx 23.5\%\text{--}29.6\%$), delimitando a convergência teórica de problemas de valor de contorno dirigidos por fonte.

---

## 4. Questão 2.3: Benchmark MMS Convencional com Fonte Volumétrica Pura

Em [mlpg_source_problem_2d.py](file:///home/renato/Insync/rencmbr@gmail.com/Google%20Drive/doc/CNPq/Projeto2026/VNMM2D-Estudo/VNMM/2D/scripts/mlpg_source_problem_2d.py) (`load_mode="volumetric"`), resolveu-se o problema MMS deslocado ($\kappa = 1.0$, $\vec{E}^*(x,y) = [\sin x \cos y, -\cos x \sin y]^T$, diretores aleatórios com `seed=42`, $\alpha_s = 0.80$, Dirichlet puro em $\partial\Omega$) usando **exclusivamente** a integral volumétrica $f_{I,a}^{\mathrm{vol}} = \iint_{D_I} \vec{J}_s \cdot \vec{N}_a^I\,d\Omega$ e **nenhum fluxo analítico em $L_{sI}$**.

Para o caso estabilizado, avaliou-se tanto a fonte Maxwell padrão $\vec{J}_s = (2+\kappa)\vec{E}^*$ quanto a fonte volumétrica consistente do operador estabilizado $\vec{J}_s^{\mathrm{stab}} = (2 + 2\alpha_{\mathrm{stab}} + \kappa)\vec{E}^*$ (derivada de $-\nabla\cdot(2\alpha_{\mathrm{stab}}\mu_r^{-1}\nabla_{\mathrm{sym},0}\vec{E}^*) = 2\alpha_{\mathrm{stab}}\mu_r^{-1}\vec{E}^*$).

### Resultados Principais (Malhas $15\times 15$ a $49\times 49$)

| Malha | $h_{\mathrm{nom}}$ | Fixo ($s_{\mathrm{div}}=6, c_{\mathrm{stab}}=0$) $L^2$ / Curl | Escalado ($c_{\mathrm{stab}}=0$) $L^2$ / Curl | Escalado ($c_{\mathrm{stab}}=70$, Base $\vec{J}_s$) $L^2$ / Curl | Desvio $L^2_{\mathrm{shift}}$ | $\frac{\|\mathcal{R}_h\|_{\mathcal{V}_{h,0}^*}}{\|\mathbf{e}^*\|_{\mathbf{A}}}$ | $c_{\mathrm{stab}}=70$ ($\vec{J}_s^{\mathrm{stab}}$) $L^2$ / Curl |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $15^2$ | 0.2244 | $9.62\%$ / $33.29\%$ | $9.01\%$ / $32.08\%$ | $10.00\%$ / $31.72\%$ | $8.28\%$ | $24.86\%$ | $11.12\%$ / $30.82\%$ |
| $22^2$ | 0.1496 | $12.58\%$ / $39.15\%$ | $9.15\%$ / $34.14\%$ | $8.19\%$ / $31.11\%$ | $7.71\%$ | $27.92\%$ | $9.91\%$ / $31.15\%$ |
| $29^2$ | 0.1122 | $14.10\%$ / $42.97\%$ | $8.94\%$ / $36.03\%$ | $7.41\%$ / $32.94\%$ | $7.44\%$ | $29.63\%$ | $9.18\%$ / $33.23\%$ |
| $36^2$ | 0.0898 | $11.06\%$ / $38.22\%$ | $5.19\%$ / $26.83\%$ | $4.85\%$ / $25.69\%$ | $5.30\%$ | $25.28\%$ | $5.76\%$ / $25.75\%$ |
| $43^2$ | 0.0748 | $8.62\%$ / $36.89\%$ | $3.92\%$ / $25.24\%$ | $3.76\%$ / $24.18\%$ | $3.97\%$ | $23.52\%$ | $4.42\%$ / $24.23\%$ |
| $49^2$ | 0.0654 | $10.99\%$ / $39.97\%$ | $4.45\%$ / $26.54\%$ | $4.17\%$ / $25.63\%$ | $4.52\%$ | $25.22\%$ | $4.84\%$ / $25.72\%$ |
| **Taxa $p$** | --- | **$0.04$ / $-0.09$** | **$0.73$ / $0.23$** | **$0.83$ / $0.23$** | --- | --- | **$0.80$ / $0.21$** |

Taxas adicionais em $L^2$ para os demais valores de $c_{\mathrm{stab}}$ testados em `run_conventional_volumetric_mms_study`:
- **Fonte base $\vec{J}_s$:** $p_{L^2} = 0.72$ ($c_{\mathrm{stab}}=20$), $0.75$ ($c_{\mathrm{stab}}=40$), $0.90$ ($c_{\mathrm{stab}}=100$, erro em $43^2\text{--}49^2$ de $3.83\%\text{--}4.13\%$).
- **Fonte consistente $\vec{J}_s^{\mathrm{stab}}$:** $p_{L^2} = 0.70$ ($c_{\mathrm{stab}}=20$), $0.73$ ($c_{\mathrm{stab}}=40$), $0.87$ ($c_{\mathrm{stab}}=100$, erro em $43^2\text{--}49^2$ de $4.73\%\text{--}5.06\%$).

### Interpretação Via Lema de Strang
- Com penalidade fixa ($s_{\mathrm{div}} = 6.0, c_{\mathrm{stab}} = 0$), o sistema estagna em $\sim 10\%\text{--}14\%$ ($p_{L^2} = 0.04$).
- Com a calibração escalada ($s_{\mathrm{div}} \propto h_{\mathrm{nom}}^{-1.5}$) e estabilização ortogonal, o erro em $L^2(\vec{E})$ passa a convergir com taxa empírica de primeira ordem ($p_{L^2} = 0.73\text{--}0.90$, caindo de $\approx 10\%$ para $3.76\%\text{--}4.17\%$), e o desvio de consistência em $L^2$ acompanha essa redução ($8.28\% \to 3.97\%\text{--}4.52\%$).
- Entretanto, devido ao termo $\mathcal{R}_h$ não cancelado nos arcos internos $L_{sI}$, a norma dual de energia $\|\mathcal{R}_h\|_{\mathcal{V}_{h,0}^*}/\|\mathbf{e}^*\|_{\mathbf{A}}$ permanece $\mathcal{O}(1)$ ($\approx 23.5\%\text{--}29.6\%$) e o erro rotacional $\varepsilon_{\mathbf{curl},h}$ estaciona no platô de $\approx 24\%\text{--}26\%$ ($p_{\mathbf{curl}} \approx 0.21\text{--}0.26$), contrastando diretamente com as taxas $\mathcal{O}(h_{\mathrm{nom}}^{1.93})$ em $L^2$ e $\mathcal{O}(h_{\mathrm{nom}}^{1.05})$ em rotacional observadas quando $\mathcal{R}_h \equiv 0$ sob equilíbrio local exato.

---

## 5. Arquivos Atualizados e Verificações Executadas

- **Código alterado:** [mlpg_source_problem_2d.py](file:///home/renato/Insync/rencmbr@gmail.com/Google%20Drive/doc/CNPq/Projeto2026/VNMM2D-Estudo/VNMM/2D/scripts/mlpg_source_problem_2d.py) (`_compute_subdomain_mms_loads`, `assemble_and_solve_mlpg_mms`, `run_conventional_volumetric_mms_study`, `quantify_multiplicity_rayleigh_perturbation`).
- **Manuscrito atualizado:** [main-vnmm-mlpg.tex](file:///home/renato/Insync/rencmbr@gmail.com/Google%20Drive/doc/CNPq/Projeto2026/VNMM2D-Estudo/VNMM/paper/mlpg-vnmm/main-vnmm-mlpg.tex) (Abstract, Seções 1, 2.5, 2.10.2, 3.3, 3.3.3 com Tabela 5, 3.5.7, 3.6, 4.2, 5 e Apêndices B.1, B.2 e B.4) $\rightarrow$ compilado em [main-vnmm-mlpg.pdf](file:///home/renato/Insync/rencmbr@gmail.com/Google%20Drive/doc/CNPq/Projeto2026/VNMM2D-Estudo/VNMM/paper/mlpg-vnmm/main-vnmm-mlpg.pdf).
- **Verificações:**
  - `python3 2D/scripts/mlpg_source_problem_2d.py`: executado com exit code `0`.
  - `python3 -m unittest 2D/tests/test_mlpg_source_problem_2d.py -v`: `2` testes OK (`1.324s`).
  - `tectonic --keep-logs main-vnmm-mlpg.tex`: exit code `0`, `42` páginas, `0` avisos de referências/citações indefinidas e `0` ocorrências de `??` via `pdftotext`.
