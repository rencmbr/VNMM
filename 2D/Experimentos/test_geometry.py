"""
Teste 4.3.2: Guia de Onda com Saliência e Meio Homogêneo Anisotrópico
====================================================================
Arquivo: VNMM_Anisotropic_Ridge_Waveguide.py

Descrição:
----------
Seção transversal em forma de "H" (guia com saliências simétricas):
Domínio original [-2.5, 2.5] x [-1.5, 1.5] ou conforme Figura 4-15:
x in [-2.5, 2.5] ou [-2.0, 2.0], y in [-1.5, 1.5]
Vamos verificar a geometria exata na Figura 4-15 da tese.
"""

import sys
from pathlib import Path
import numpy as np
import scipy.linalg as la
from scipy.spatial import KDTree
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

DIR_EXPERIMENTOS = Path(__file__).resolve().parent
DIR_2D = DIR_EXPERIMENTOS.parent
ROOT_DIR = DIR_2D.parent

for p in [str(DIR_2D), str(ROOT_DIR), str(DIR_EXPERIMENTOS)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.support_nodes_vnmm_2d_6_p1 import support_nodes_vnmm_2d_6_p1
from src.gauss_quadrature_2d import get_gauss_points_weights_1d

print("Módulos carregados com sucesso.")
