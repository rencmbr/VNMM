import os
import sys
import unittest
import numpy as np

# Configuração de caminhos para os testes
DIRETORIO_TESTS = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_RAIZ = os.path.dirname(DIRETORIO_TESTS)
if DIRETORIO_RAIZ not in sys.path:
    sys.path.insert(0, DIRETORIO_RAIZ)

from src.malha_cavidade import gerar_malha_cavidade
from src.quadratura_gauss import gerar_celulas_quadratura, obter_pontos_pesos_gauss_1d
from src.montador_vnmm import montar_matrizes_vnmm_2d
from src.eigen_solver_cavity import (
    aplicar_condicao_pec, 
    resolver_problema_autovalores, 
    resolver_autovalores_cavidade,
    MODOS_ANALITICOS_REF
)


class TestMalhaCavidadePEC(unittest.TestCase):
    """Testes unitários da geração de malha e direcionalidade tangente PEC."""
    
    def test_dimensoes_e_quantidades_nos(self):
        Nx, Ny = 7, 7
        coords, vectors, is_boundary = gerar_malha_cavidade(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        self.assertEqual(coords.shape, (Nx * Ny, 2))
        self.assertEqual(vectors.shape, (Nx * Ny, 2))
        self.assertEqual(is_boundary.shape, (Nx * Ny,))
        
        # Quantidade de nós de fronteira em grade regular = 2*Nx + 2*Ny - 4
        n_borda_esperado = 2 * Nx + 2 * Ny - 4
        self.assertEqual(np.sum(is_boundary), n_borda_esperado)
        
    def test_direcionalidade_tangente_fronteiras_pec(self):
        Nx, Ny = 9, 9
        coords, vectors, is_boundary = gerar_malha_cavidade(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        tol = 1e-7
        for pt, vec, eh_borda in zip(coords, vectors, is_boundary):
            x, y = pt[0], pt[1]
            norma_vec = np.linalg.norm(vec)
            self.assertTrue(np.isclose(norma_vec, 1.0, atol=1e-12), "O vetor diretor deve ser unitário.")
            
            # Parede inferior (y=0) ou superior (y=pi)
            if np.isclose(y, 0.0, atol=tol) or np.isclose(y, np.pi, atol=tol):
                self.assertTrue(eh_borda, "Nó em y=0 ou y=pi deve ser classificado como fronteira.")
                self.assertTrue(np.isclose(abs(vec[0]), 1.0, atol=1e-10))
                self.assertTrue(np.isclose(vec[1], 0.0, atol=1e-10))
                
            # Parede lateral esquerda (x=0) ou direita (x=pi)
            elif np.isclose(x, 0.0, atol=tol) or np.isclose(x, np.pi, atol=tol):
                self.assertTrue(eh_borda, "Nó em x=0 ou x=pi deve ser classificado como fronteira.")
                self.assertTrue(np.isclose(vec[0], 0.0, atol=1e-10))
                self.assertTrue(np.isclose(abs(vec[1]), 1.0, atol=1e-10))
            else:
                self.assertFalse(eh_borda, "Nó no interior não deve ser classificado como fronteira.")


class TestQuadraturaGauss2D(unittest.TestCase):
    """Testes unitários da integração numérica em células de fundo."""
    
    def test_pesos_gauss_1d(self):
        xi, w = obter_pontos_pesos_gauss_1d(2)
        self.assertEqual(len(xi), 2)
        self.assertTrue(np.isclose(np.sum(w), 2.0))
        
    def test_integracao_area_total_dominio(self):
        Lx, Ly = np.pi, np.pi
        pontos_g, pesos_g, celulas = gerar_celulas_quadratura(Lx=Lx, Ly=Ly, Ncx=6, Ncy=6, pontos_por_dir=2)
        
        area_calculada = np.sum(pesos_g)
        area_exata = Lx * Ly
        self.assertTrue(np.isclose(area_calculada, area_exata, atol=1e-12), "A integral da função constante 1 deve ser Lx * Ly.")
        
    def test_integracao_polinomio_quadratico(self):
        # Integral de f(x, y) = x^2 + y^2 em [0, pi]^2
        # Integral exata = 2 * pi^4 / 3
        Lx, Ly = np.pi, np.pi
        pontos_g, pesos_g, _ = gerar_celulas_quadratura(Lx=Lx, Ly=Ly, Ncx=8, Ncy=8, pontos_por_dir=2)
        
        f_vals = pontos_g[:, 0]**2 + pontos_g[:, 1]**2
        integral_num = np.sum(f_vals * pesos_g)
        integral_exata = 2.0 * (np.pi**4) / 3.0
        
        self.assertTrue(np.isclose(integral_num, integral_exata, rtol=1e-10))


class TestMontadorMatrizesVNMM(unittest.TestCase):
    """Testes de propriedades das matrizes esparsas globais e reduzidas."""
    
    def test_simetria_e_positividade_matrizes_ponto_gauss(self):
        Nx, Ny = 9, 9
        coords, vectors, is_boundary = gerar_malha_cavidade(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        K, M = montar_matrizes_vnmm_2d(
            coords, vectors, base="P1", s_div=6.0, Ncx=8, Ncy=8, 
            pontos_por_dir=2, modo_suporte="ponto_gauss"
        )
        
        # 1. Simetria de K e M globais
        diff_K = K - K.T
        diff_M = M - M.T
        self.assertTrue(diff_K.nnz == 0 or np.max(np.abs(diff_K.data)) < 1e-14)
        self.assertTrue(diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-14)
        
        # 2. Redução PEC
        K_red, M_red, idx_int = aplicar_condicao_pec(K, M, is_boundary)
        self.assertEqual(K_red.shape[0], len(idx_int))
        self.assertEqual(M_red.shape[0], len(idx_int))
        
        # 3. Positividade da diagonal da matriz de massa M_red
        diag_M = M_red.diagonal()
        self.assertTrue(np.all(diag_M > 0.0), "Todos os elementos da diagonal de M_red devem ser estritamente positivos.")

    def test_simetria_matrizes_centro_celula(self):
        Nx, Ny = 9, 9
        coords, vectors, is_boundary = gerar_malha_cavidade(Nx=Nx, Ny=Ny, Lx=np.pi, Ly=np.pi)
        
        K, M = montar_matrizes_vnmm_2d(
            coords, vectors, base="P1", s_div=6.0, Ncx=8, Ncy=8, 
            pontos_por_dir=2, modo_suporte="centro_celula"
        )
        
        diff_K = K - K.T
        diff_M = M - M.T
        self.assertTrue(diff_K.nnz == 0 or np.max(np.abs(diff_K.data)) < 1e-14)
        self.assertTrue(diff_M.nnz == 0 or np.max(np.abs(diff_M.data)) < 1e-14)


class TestSolverAutovaloresCavidadePEC(unittest.TestCase):
    """Testes de validação da Tabela 4-1 da tese de Luilly Ortiz (UFMG, 2023)."""
    
    def test_autovalores_cavidade_luilly_tabela_4_1_ponto_gauss(self):
        """
        Valida os 10 primeiros autovalores numéricos do modo TEz na cavidade [0, pi]^2
        usando suporte por ponto de Gauss (estilo EFG).
        """
        resultado = resolver_autovalores_cavidade(
            Nx=21, 
            Ny=21, 
            Lx=np.pi, 
            Ly=np.pi, 
            Ncx=10,
            Ncy=10,
            base="P1", 
            tipo_interior="alternado", 
            num_autovalores=10, 
            s_div=6.0,
            pontos_por_dir=2,
            modo_suporte="ponto_gauss"
        )
        
        vals_num = resultado['autovalores_numericos']
        vals_ref = resultado['autovalores_analiticos']
        
        self.assertEqual(len(vals_num), 10, "Devem ser retornados exatamente 10 autovalores.")
        ref_esperado = np.array([1.0, 1.0, 2.0, 4.0, 4.0, 5.0, 5.0, 8.0, 9.0, 9.0])
        np.testing.assert_allclose(vals_ref, ref_esperado, atol=1e-12)
        
        # Erro médio de kc inferior a 1.5% e erro máx inferior a 2.5%
        self.assertLess(resultado['erro_medio_kc_pct'], 1.5)
        self.assertLess(resultado['erro_max_kc_pct'], 2.5)

    def test_autovalores_cavidade_luilly_tabela_4_1_centro_celula(self):
        """
        Valida os 10 primeiros autovalores numéricos do modo TEz com suporte fixado por célula.
        """
        resultado = resolver_autovalores_cavidade(
            Nx=21, 
            Ny=21, 
            Lx=np.pi, 
            Ly=np.pi, 
            Ncx=20,
            Ncy=20,
            base="P1", 
            tipo_interior="alternado", 
            num_autovalores=10, 
            s_div=6.0,
            pontos_por_dir=2,
            modo_suporte="centro_celula"
        )
        
        self.assertLess(resultado['erro_medio_kc_pct'], 2.5)
        self.assertLess(resultado['erro_max_kc_pct'], 5.0)
        
    def test_ausencia_modos_espurios(self):
        """
        Verifica se não ocorrem autovalores espúrios ou não-físicos no espectro.
        """
        resultado = resolver_autovalores_cavidade(
            Nx=17, 
            Ny=17, 
            Lx=np.pi, 
            Ly=np.pi, 
            base="P1", 
            num_autovalores=8, 
            s_div=6.0
        )
        vals_num = resultado['autovalores_numericos']
        
        # Nenhum autovalor deve ser negativo ou menor que 0.8 (o menor modo físico TE10 tem lambda = 1.0)
        self.assertTrue(np.all(vals_num >= 0.8), "Não devem existir autovalores abaixo do primeiro modo físico (lambda=1.0).")
        
        # Verifica monotonicidade estrita
        self.assertTrue(np.all(np.diff(vals_num) >= -1e-8), "O espectro de autovalores deve ser monotonicamente crescente.")


class TestFEMEdge2D(unittest.TestCase):
    """Testes unitários do solver de Elementos Finitos de Aresta Triangulares 2D (Nédélec)."""
    
    def test_solver_fem_aresta_cavidade(self):
        from src.fem_edge_2d import resolver_autovalores_fem_aresta_2d
        
        res = resolver_autovalores_fem_aresta_2d(Nex=11, Ney=11, num_autovalores=10)
        
        self.assertEqual(len(res['autovalores_numericos']), 10)
        self.assertEqual(res['N_incognitas'], 341)
        self.assertEqual(res['n_nulos_descartados'], 100)
        
        # Erro médio do número de onda kc < 1.0%
        self.assertLess(res['erro_medio_kc_pct'], 1.0)
        # Erro máximo < 2.0%
        self.assertLess(res['erro_max_kc_pct'], 2.0)

    def test_solver_fem_aresta_malha_aleatoria(self):
        from src.fem_edge_2d import resolver_autovalores_fem_aresta_2d
        
        res = resolver_autovalores_fem_aresta_2d(
            Nex=12, Ney=12, num_autovalores=10, jitter_frac=0.25, seed=42
        )
        
        self.assertEqual(len(res['autovalores_numericos']), 10)
        # Mesmo sob malha com jitter de 25%, erro médio de kc < 1.0%
        self.assertLess(res['erro_medio_kc_pct'], 1.0)
        self.assertLess(res['erro_max_kc_pct'], 2.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
