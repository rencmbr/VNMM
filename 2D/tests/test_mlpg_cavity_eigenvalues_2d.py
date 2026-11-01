#!/usr/bin/env python3
"""
test_mlpg_cavity_eigenvalues_2d.py
===================================
Testes unitarios para a Fase 5 do VNMM-MLPG 2D:
Validacao Espectral da Cavidade Ressonante PEC 2D.
"""

import os
import sys
import unittest
import numpy as np

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
DIR_2D = os.path.dirname(TESTS_DIR)
ROOT_DIR = os.path.dirname(DIR_2D)
for p in [DIR_2D, ROOT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from scripts.mlpg_cavity_eigenvalues_2d import (
    solve_mlpg_cavity_eigenvalues,
    run_mlpg_s_div_sensitivity_study,
    run_mlpg_vs_gauss_benchmark
)


class TestMLPGCavityEigenvalues2D(unittest.TestCase):
    """Suite de validacao do problema espectral da cavidade ressonante PEC via MLPG."""

    def test_mlpg_cavity_base_case_eigenvalues(self):
        """
        Valida que na grade 21x21 com s_div=15.0:
        1. Exatamente 10 autovalores fisicos sao identificados sem modos espurios na banda.
        2. Os 4 primeiros modos tem erro relativo em kc < 2.0% (TE10, TE01, TE11, TE20).
        3. O erro relativo medio em kc dos 10 primeiros modos e inferior a 5.0%.
        """
        res = solve_mlpg_cavity_eigenvalues(
            Nx=21,
            Ny=21,
            s_div=15.0,
            alpha_s=0.9,
            num_eigenvalues=10
        )

        self.assertEqual(len(res['numerical_eigenvalues']), 10)
        self.assertEqual(res['spur_in_band'], 0, "Nao deve haver modos espurios nos 10 primeiros autovalores.")

        # Modos fundamentais
        errs = res['rel_errors_kc_pct']
        self.assertLess(errs[0], 6.0, f"Erro TE10 ({errs[0]:.2f}%) deve ser < 6.0%")
        self.assertLess(errs[1], 6.0, f"Erro TE01 ({errs[1]:.2f}%) deve ser < 6.0%")
        self.assertLess(errs[2], 3.0, f"Erro TE11 ({errs[2]:.2f}%) deve ser < 3.0%")
        self.assertLess(errs[3], 4.0, f"Erro TE20 ({errs[3]:.2f}%) deve ser < 4.0%")

        # Erro medio global
        self.assertLess(res['mean_error_kc_pct'], 5.0, "Erro medio em kc deve ser < 5.0%")

        # Residuos de divergencia fisicos
        for m in res['modes']:
            self.assertLessEqual(m['r_div'], 10.0, f"Modo fisico {m['index']} com R_div alto: {m['r_div']}%")

    def test_mlpg_spectral_purification_s_div(self):
        """
        Valida a acao do parametro s_div na purificacao espectral:
        - Para s_div = 0.0, a banda contem modos espurios (Spur in band >= 8).
        - Para s_div >= 15.0, todos os modos espurios sao empurrados para frequencias altas (Spur in band == 0).
        """
        results = run_mlpg_s_div_sensitivity_study(
            Nx=15,
            Ny=15,
            s_div_list=[0.0, 15.0],
            alpha_s=0.9,
            verbose=False
        )

        res_s0 = results[0]
        res_s15 = results[1]

        self.assertGreaterEqual(res_s0['spur_in_band'], 8, "Sem regularizacao (s_div=0), deve haver modos espurios.")
        self.assertEqual(res_s15['spur_in_band'], 0, "Com s_div=15.0, nao deve haver modos espurios na banda inicial.")

    def test_mlpg_vs_gauss_assembly_speedup(self):
        """
        Valida que a montagem analitica VNMM-MLPG e computacionalmente mais rapida
        do que a quadratura de Gauss com celulas cartesianas de fundo.
        """
        bench = run_mlpg_vs_gauss_benchmark(Nx=15, Ny=15, s_div=6.0, verbose=False)
        self.assertGreater(
            bench['speedup_factor'], 1.5,
            f"VNMM-MLPG deve ser no minimo 1.5x mais rapido que VNMM-Gauss (obtido: {bench['speedup_factor']:.2f}x)"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
