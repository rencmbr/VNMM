
import sys
from pathlib import Path

# Localiza a pasta 2D do projeto
DIR_2D = Path(__file__).resolve().parent.parent

# Permite importar os módulos da pasta src
sys.path.insert(0, str(DIR_2D))

# Importa a função desenvolvida no projeto VNMM
from src.gauss_quadrature_2d import get_gauss_points_weights_1d

# Calcula a quadratura de Gauss-Legendre com 3 pontos
pontos, pesos = get_gauss_points_weights_1d(3)

print("Pontos de Gauss:", pontos)
print("Pesos de Gauss:", pesos)
print("Soma dos pesos:", sum(pesos))
