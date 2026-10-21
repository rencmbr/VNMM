import unittest
import numpy as np
from codigo.teste_patch_test_galerkin import (
    executar_patch_test_modo,
    gerar_malha_cavidade
)


class TestGalerkinPatchTest(unittest.TestCase):
    def test_galerkin_patch_test_P1_regular(self):
        """Verifica que a base completa P1 passa no Patch Test de Galerkin em malha regular."""
        coords, vectors, is_bnd = gerar_malha_cavidade(
            Nx=7, Ny=7, Lx=np.pi, Ly=np.pi, tipo_interior="alternado", jitter_frac=0.0, seed=42
        )
        
        campo_arbitrario = {
            "nome": "Campo Afim Arbitrário",
            "func": lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y]),
            "rot": 0.4 - (-0.3),
            "div": 0.8 + 1.2,
            "harmonico": False
        }
        
        res_P1 = executar_patch_test_modo(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            campo_alvo=campo_arbitrario, base="P1", s_div=6.0, Ncx=6, Ncy=6
        )
        
        self.assertLess(res_P1['erro_dof_rel'], 1e-11, f"P1 falhou no patch test regular: {res_P1['erro_dof_rel']}")
        self.assertLess(res_P1['err_L2_E'], 1e-11)
        self.assertLess(res_P1['err_L2_rot'], 1e-10)
        self.assertLess(res_P1['err_L2_div'], 1e-10)

    def test_galerkin_patch_test_P1_irregular(self):
        """Verifica que a base completa P1 passa no Patch Test de Galerkin em nuvem irregular (25% jitter)."""
        coords, vectors, is_bnd = gerar_malha_cavidade(
            Nx=7, Ny=7, Lx=np.pi, Ly=np.pi, tipo_interior="aleatorio", jitter_frac=0.25, seed=42
        )
        
        campo_arbitrario = {
            "nome": "Campo Afim Arbitrário",
            "func": lambda x, y: np.array([1.5 + 0.8 * x - 0.3 * y, -0.7 + 0.4 * x + 1.2 * y]),
            "rot": 0.4 - (-0.3),
            "div": 0.8 + 1.2,
            "harmonico": False
        }
        
        res_P1 = executar_patch_test_modo(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            campo_alvo=campo_arbitrario, base="P1", s_div=6.0, Ncx=6, Ncy=6
        )
        
        self.assertLess(res_P1['erro_dof_rel'], 1e-11, f"P1 falhou no patch test irregular: {res_P1['erro_dof_rel']}")
        self.assertLess(res_P1['err_L2_E'], 1e-11)
        self.assertLess(res_P1['err_L2_rot'], 1e-10)
        self.assertLess(res_P1['err_L2_div'], 1e-10)

    def test_galerkin_patch_test_L1_failure(self):
        """Verifica que a base incompleta L1 falha no Patch Test para deformação normal (strain longitudinal)."""
        coords, vectors, is_bnd = gerar_malha_cavidade(
            Nx=7, Ny=7, Lx=np.pi, Ly=np.pi, tipo_interior="alternado", jitter_frac=0.0, seed=42
        )
        
        campo_strain = {
            "nome": "Strain Normal",
            "func": lambda x, y: np.array([x, -y]),
            "rot": 0.0,
            "div": 0.0,
            "harmonico": True
        }
        
        res_L1 = executar_patch_test_modo(
            coords=coords, vectors=vectors, is_boundary=is_bnd,
            campo_alvo=campo_strain, base="L1", s_div=6.0, Ncx=6, Ncy=6
        )
        
        self.assertGreater(res_L1['erro_dof_rel'], 0.05, f"L1 deveria falhar no patch test para strain normal: {res_L1['erro_dof_rel']}")


if __name__ == "__main__":
    unittest.main()
