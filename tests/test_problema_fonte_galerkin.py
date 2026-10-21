import unittest
import numpy as np
from codigo.teste_problema_fonte_galerkin import (
    montar_e_resolver_mms,
    gerar_malha_cavidade
)


class TestProblemaFonteGalerkin(unittest.TestCase):
    def test_convergencia_mms_P1(self):
        """Verifica a convergência monótona e taxa assintótica O(h^2) para base P1."""
        tamanhos = [9, 13, 17, 21]
        erros_L2 = []
        erros_rot = []
        h_vals = []

        for N in tamanhos:
            coords, vectors, is_bnd = gerar_malha_cavidade(Nx=N, Ny=N, jitter_frac=0.0)
            res = montar_e_resolver_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                base="P1",
                s_div=6.0,
                kappa=1.0,
                Ncx=N - 1,
                Ncy=N - 1,
                pontos_por_dir=3
            )
            erros_L2.append(res['err_L2_E_rel'])
            erros_rot.append(res['err_L2_rot_rel'])
            h_vals.append(res['h'])

        # Verifica decaimento estrito do erro L2
        for i in range(len(erros_L2) - 1):
            self.assertLess(erros_L2[i + 1], erros_L2[i], "Erro L2 deve diminuir sob refinamento para P1")

        # Taxa assintótica de convergência em L2: deve ser >= 1.8
        taxa_L2, _ = np.polyfit(np.log(h_vals), np.log(erros_L2), 1)
        self.assertGreater(taxa_L2, 1.8, f"Taxa assintótica L2 para P1 deve ser ~ 2.0, obtido {taxa_L2:.2f}")

    def test_estagnacao_mms_L1(self):
        """Verifica que a base incompleta L1 estagna e não converge (erro > 20%)."""
        tamanhos = [9, 17]
        erros_L2 = []

        for N in tamanhos:
            coords, vectors, is_bnd = gerar_malha_cavidade(Nx=N, Ny=N, jitter_frac=0.0)
            res = montar_e_resolver_mms(
                coords=coords,
                vectors=vectors,
                is_boundary=is_bnd,
                base="L1",
                s_div=6.0,
                kappa=1.0,
                Ncx=N - 1,
                Ncy=N - 1,
                pontos_por_dir=3
            )
            erros_L2.append(res['err_L2_E_rel'])

        # L1 deve ter erro substancial em ambos os refinamentos (> 20%)
        self.assertGreater(erros_L2[0], 0.20)
        self.assertGreater(erros_L2[1], 0.20)


if __name__ == "__main__":
    unittest.main()
