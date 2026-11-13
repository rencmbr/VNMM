import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DIR_EXP = Path(r"c:\Users\DELL\Desktop\Test_github\VNMM\2D\Experimentos")

# ==============================================================================
# FIGURA 1: GUIA COM SALIÊNCIA EM 'H' (HOMOGÊNEO ANISOTRÓPICO)
# ==============================================================================
fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

x_coords = [-2.5, -1.0, -1.0,  1.0,  1.0,  2.5,  2.5,  1.0, 1.0, -1.0, -1.0, -2.5, -2.5]
y_coords = [-1.5, -1.5, -0.5, -0.5, -1.5, -1.5,  1.5,  1.5, 0.5,  0.5,  1.5,  1.5, -1.5]

ax.fill(x_coords, y_coords, color="#E3F2FD", alpha=0.85, label="Meio Homogêneo Anisotrópico")
ax.plot(x_coords, y_coords, color="#0D47A1", linewidth=2.5, label="Fronteira Condutora Perfeita (PEC)")

ax.annotate("Saliência Superior\n[-1, 1] x [0.5, 1.5]", xy=(0, 0.5), xytext=(0, 0.95),
            ha="center", fontsize=9, color="#C62828", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2))

ax.annotate("Saliência Inferior\n[-1, 1] x [-1.5, -0.5]", xy=(0, -0.5), xytext=(0, -0.95),
            ha="center", fontsize=9, color="#C62828", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2))

props_text = r"$\mu_r = 1.0$" + "\n" + r"$\varepsilon_{xx}=1, \varepsilon_{xy}=-j$" + "\n" + r"$\varepsilon_{yx}=j, \varepsilon_{yy}=4$"
ax.text(-1.75, 0.0, props_text, fontsize=9.5, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#0D47A1", alpha=0.9))

ax.set_title("Geometria do Guia de Onda com Saliência em 'H' (Teste 4.3.2)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("x (m)", fontsize=10)
ax.set_ylabel("y (m)", fontsize=10)
ax.set_xlim(-2.8, 2.8)
ax.set_ylim(-1.8, 1.8)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

plt.tight_layout()
p1 = DIR_EXP / "Geometria_Guia_H_Anisotropico.png"
plt.savefig(p1)
plt.close()
print("Salvo:", p1)

# ==============================================================================
# FIGURA 2: GUIA BIMATERIAL ANISOTRÓPICO COM CONDUTORES CIRCULARES PEC
# ==============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=300)

r1 = plt.Rectangle((-2.0, -1.0), 2.0, 2.0, facecolor="#E8F5E9", edgecolor="none", alpha=0.85, label=r"Meio 1 ($\Omega_1$)")
r2 = plt.Rectangle((0.0, -1.0), 2.0, 2.0, facecolor="#FFF3E0", edgecolor="none", alpha=0.85, label=r"Meio 2 ($\Omega_2$)")
ax.add_patch(r1)
ax.add_patch(r2)

ax.axvline(0.0, color="#2E7D32", linestyle="--", linewidth=2.2, label=r"Interface Bimaterial ($x=0$)")
ax.plot([-2, 2, 2, -2, -2], [-1, -1, 1, 1, -1], color="#1A237E", linewidth=2.2, label="Paredes Externas PEC")

r_circ = 1.0 / 3.0
c1 = plt.Circle((-1.0, 0.0), r_circ, facecolor="#CFD8DC", edgecolor="#B71C1C", linewidth=2.0, label="Cilindros PEC Internos ($r=1/3$)")
c2 = plt.Circle((1.0, 0.0), r_circ, facecolor="#CFD8DC", edgecolor="#B71C1C", linewidth=2.0)
ax.add_patch(c1)
ax.add_patch(c2)

t1 = r"$\mathbf{\Omega_1}$" + "\n" + r"$\mu_{r1} = 1.0$" + "\n" + r"$\varepsilon_{xx}=1, \varepsilon_{yy}=2$"
ax.text(-1.0, 0.55, t1, fontsize=9.0, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#2E7D32", alpha=0.9))

t2 = r"$\mathbf{\Omega_2}$" + "\n" + r"$\mu_{r2} = 3.0$" + "\n" + r"$\varepsilon_{xx}=3, \varepsilon_{xy}=-2j$" + "\n" + r"$\varepsilon_{yx}=2j, \varepsilon_{yy}=8$"
ax.text(1.0, 0.55, t2, fontsize=9.0, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#E65100", alpha=0.9))

ax.plot(-1.0, 0.0, "k+", markersize=8, markeredgewidth=1.5)
ax.plot(1.0, 0.0, "k+", markersize=8, markeredgewidth=1.5)
ax.annotate(r"$r=1/3$", xy=(-1.0 + r_circ*np.cos(np.pi/4), r_circ*np.sin(np.pi/4)),
            xytext=(-0.5, 0.25), fontsize=8.5, color="#B71C1C", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.0))

ax.set_title("Geometria do Guia Bimaterial Anisotrópico com Cilindros PEC (Teste 4.3.3)", fontsize=11, fontweight="bold", pad=12)
ax.set_xlabel("x (m)", fontsize=10)
ax.set_ylabel("y (m)", fontsize=10)
ax.set_xlim(-2.3, 2.3)
ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8, framealpha=0.9)

plt.tight_layout()
p2 = DIR_EXP / "Geometria_Guia_Bimaterial_Circ.png"
plt.savefig(p2)
plt.close()
print("Salvo:", p2)
