# Taxas Assintóticas de Convergência: EFEM (Whitney) vs. VNMM L1

**Condições do Teste:**
- Mesma malha triangular conforme para ambos os métodos.
- Mesma grade fixa e invariante de amostragem interna (1600 pontos em $[0, \pi] \times [0, \pi]$).
- Interpolação pontual baseada nos graus de liberdade locais de cada triângulo (ponto médio das arestas).

## Tabela Comparativa de Erros

| Malha | $h_{\mathrm{avg}}$ [m] | Erro RMS $\vec{E}$ (VNMM $\mathcal{L}^1$) | Erro RMS $\vec{E}$ (EFEM Whitney) | Erro RMS Rot (VNMM $\mathcal{L}^1$) | Erro RMS Rot (EFEM Whitney) | $\|\vec{E}^{\mathrm{VNMM}} - \vec{E}^{\mathrm{EFEM}}\|_{\mathrm{RMS}}$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4x4 | 0.8953 | 1.6782e-01 | 1.6782e-01 | 2.5365e-01 | 2.5365e-01 | `1.49e-16` |
| 6x6 | 0.5971 | 1.1432e-01 | 1.1432e-01 | 1.6883e-01 | 1.6883e-01 | `1.50e-16` |
| 8x8 | 0.4478 | 8.6766e-02 | 8.6766e-02 | 1.2923e-01 | 1.2923e-01 | `1.57e-16` |
| 12x12 | 0.2987 | 5.7384e-02 | 5.7384e-02 | 8.9864e-02 | 8.9864e-02 | `1.53e-16` |
| 16x16 | 0.2240 | 4.3413e-02 | 4.3413e-02 | 6.6559e-02 | 6.6559e-02 | `1.60e-16` |
| 24x24 | 0.1494 | 2.9143e-02 | 2.9143e-02 | 4.6181e-02 | 4.6181e-02 | `1.58e-16` |
| 32x32 | 0.1120 | 2.0957e-02 | 2.0957e-02 | 3.3433e-02 | 3.3433e-02 | `1.58e-16` |

## Taxas Assintóticas Obtidas $\mathcal{O}(h^p)$

- **Campo Elétrico $\vec{E}$ (VNMM $\mathcal{L}^1$):** $\mathcal{O}(h^{1.00})$
- **Campo Elétrico $\vec{E}$ (EFEM Whitney):** $\mathcal{O}(h^{1.00})$
- **Rotacional $(\nabla \times \vec{E})_z$ (VNMM $\mathcal{L}^1$):** $\mathcal{O}(h^{0.96})$
- **Rotacional $(\nabla \times \vec{E})_z$ (EFEM Whitney):** $\mathcal{O}(h^{0.96})$
