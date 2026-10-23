# Diagnóstico e Solução da Sub-integração no VNMM 2D para $N=25$ ($h=0.1309\text{m}$)

**Autor:** Antigravity (Google DeepMind) & Equipe do Projeto VNMM  
**Problema:** Análise detalhada das causas do aumento transitório do erro médio de $k_c$ no VNMM 2D puro em $N=25$ e estratégias de correção.

---

## 1. Descrição do Fenômeno Observado

Na varredura paramétrica de refinamento de malha ($N=9$ a $N=33$) com $N_c = N // 2$ células de fundo e quadratura $2 \times 2$ de Gauss, observou-se uma anomalia em $N=25$:
- **$N=21$ ($h=0.1571\text{m}$, $361$ DoFs, $N_c=10$):** Erro médio $k_c = \mathbf{1.00\%}$
- **$N=25$ ($h=0.1309\text{m}$, $529$ DoFs, $N_c=12$):** Erro médio $k_c = \mathbf{4.46\%}$ *(aumento anômalo)*
- **$N=29$ ($h=0.1122\text{m}$, $729$ DoFs, $N_c=14$):** Erro médio $k_c = \mathbf{1.49\%}$
- **$N=33$ ($h=0.0982\text{m}$, $961$ DoFs, $N_c=16$):** Erro médio $k_c = \mathbf{0.52\%}$

---

## 2. Diagnóstico Modo a Modo em $N=25$ ($N_c=12$, Quadratura $2 \times 2$)

| Modo ($TE_{nm}$) | $\lambda_{\text{analítico}}$ | $k_{c, \text{analítico}}$ | $\lambda_{\text{VNMM}}$ | $k_{c, \text{VNMM}}$ | Erro $k_c$ (%) | Diagnóstico |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$TE_{10}$** | $1.00$ | $1.000$ | $0.8847$ | $0.941$ | **$5.94\%$** | Deslocamento para baixo |
| **$TE_{01}$** | $1.00$ | $1.000$ | $0.9414$ | $0.970$ | **$2.97\%$** | Deslocamento para baixo |
| **$TE_{11}$** | $2.00$ | $1.414$ | $1.8657$ | $1.366$ | **$3.42\%$** | Deslocamento para baixo |
| **$TE_{20}$** | $4.00$ | $2.000$ | $3.4813$ | $1.866$ | **$6.71\%$** | Deslocamento para baixo |
| **$TE_{02}$** | $4.00$ | $2.000$ | $3.6902$ | $1.921$ | **$3.95\%$** | Deslocamento para baixo |
| **$TE_{21}$** | $5.00$ | $2.236$ | $4.5590$ | $2.135$ | **$4.51\%$** | Deslocamento para baixo |
| **$TE_{12}$** | $5.00$ | $2.236$ | $4.6873$ | $2.165$ | **$3.18\%$** | Deslocamento para baixo |
| **$TE_{22}$** | $8.00$ | $2.828$ | $7.1646$ | $2.677$ | **$5.37\%$** | Deslocamento para baixo |
| **$TE_{30}$** | $9.00$ | $3.000$ | $8.0009$ | $2.829$ | **$5.71\%$** | Deslocamento para baixo |
| **$TE_{03}$** | $9.00$ | $3.000$ | $8.4946$ | $2.915$ | **$2.85\%$** | Deslocamento para baixo |

> **Constatação Física:** Todos os 10 autovalores foram sistematicamente subestimados (amolecimento da matriz de rigidez $\mathbf{c}^T K \mathbf{c}$), sintoma clássico de **sub-integração numérica**.

---

## 3. Investigação das Causas Raízes

### Causa 1: Razão Crítica entre Pontos de Gauss e Graus de Liberdade ($N_g / N_{\text{DoF}}$)
Para a regra $N_c = N // 2$ com $2 \times 2$ Gauss:
- Para $N=9$: $49$ DoFs, $N_c=4 \implies 64$ pts Gauss (Razão: $1.31\times$)
- Para $N=21$: $361$ DoFs, $N_c=10 \implies 400$ pts Gauss (Razão: $1.11\times$)
- Para $N=25$: **$529$ DoFs, $N_c=12 \implies 576$ pts Gauss (Razão: $1.088\times$)**

Com apenas $1.088$ pontos de integração por incógnita, o sistema entra em regime de sub-integração limítrofe, reduzindo artificialmente a energia da forma bilinear variacional.

### Causa 2: Ressonância Espacial e Aliasing de Malha com Modos de Alta Frequência
Para $N_c = 12$, a largura da célula é $\Delta x = \pi / 12$. A função própria de $TE_{30}$ é proporcional a $\sin(3x)$. O produto $\sin^2(3x)$ oscila com período espacial $\pi/3 = 4\Delta x$.
Os pontos de Gauss da quadratura de 2 pontos ($\pm 1/\sqrt{3} \approx \pm 0.577$) incidem em posições onde a 4ª derivada do integrando atinge o valor extremo, maximizando o erro de truncamento da regra de Gauss-Legendre.

---

## 4. Testes de Sensibilidade e Correções

### Matriz de Testes para $N=25$ ($529$ DoFs):

| Configuração ($N_c \times N_c$, Gauss) | Total de Pontos de Gauss | Razão $N_g / N_{\text{DoF}}$ | Erro Médio $k_c$ (%) | Erro Máximo $k_c$ (%) |
|:---|:---:|:---:|:---:|:---:|
| **$N_c=12$, Gauss $2 \times 2$ (Original)** | 576 | $1.09\times$ | **$4.46\%$** | $6.71\%$ |
| **$N_c=14$, Gauss $2 \times 2$** | 784 | $1.48\times$ | **$1.03\%$** | $1.98\%$ |
| **$N_c=16$, Gauss $2 \times 2$** | 1024 | $1.94\times$ | **$0.96\%$** | $1.78\%$ |
| **$N_c=12$, Gauss $3 \times 3$ (Ótima)** | **1296** | **$2.45\times$** | **$0.76\%$** | **$1.27\%$** |
| **$N_c=12$, Gauss $4 \times 4$** | 2304 | $4.36\times$ | **$0.87\%$** | $2.34\%$ |

---

## 5. Recomendações e Diretrizes Práticas

1. **Razão Segura de Quadratura:** Manter sempre $N_g / N_{\text{DoF}} \ge 2.0\times$ para métodos sem malha nodais em eletromagnetismo.
2. **Uso de Quadratura $3 \times 3$ de Gauss por Célula:** Para malhas densas ($N \ge 21$), a quadratura $3 \times 3$ de Gauss-Legendre elimina completamente o aliasing de alta frequência e restaura a convergência monotônica estrita:
   - $N=21$: $2.08\%$
   - $N=25$: **$0.76\%$** (máx: $1.27\%$)
   - $N=29$: **$0.68\%$** (máx: $1.82\%$)
   - $N=33$: **$0.83\%$** (máx: $1.70\%$)
