import unittest
import numpy as np
import pandas as pd
from treinamento_local.variantes_svm import criar_variantes, visoes


class TestVariantes(unittest.TestCase):
    def test_preserva_peso_e_rotulo_por_documento(self):
        df = pd.DataFrame({'entrada_a': ['título completo corpo', 'outra notícia texto'],
                           'titulo': ['título', 'outra notícia'], 'texto': ['corpo', 'texto'], 'label': [0, 1]})
        textos, y, pesos = criar_variantes(df)
        self.assertEqual(len(textos), 6)
        np.testing.assert_allclose([pesos[y == n].sum() for n in (0, 1)], [1, 1])
        self.assertEqual(textos[:3], ['título completo corpo', 'título', 'corpo'])

    def test_variantes_iguais_apos_limpeza_sao_unidas(self):
        df = pd.DataFrame({'entrada_a': ['FALSO biblioteca aberta'], 'titulo': ['Biblioteca aberta'],
                           'texto': [''], 'label': [0]})
        textos, y, pesos = criar_variantes(df)
        self.assertEqual(len(textos), 1)
        self.assertEqual(pesos.tolist(), [1.])

    def test_recorte_vem_do_corpo_e_nao_do_titulo(self):
        corpo = ' '.join('palavra' + str(n) for n in range(60))
        df = pd.DataFrame({'entrada_a': ['MANCHETE ' + corpo], 'titulo': ['MANCHETE'], 'texto': [corpo], 'label': [1]})
        self.assertEqual(len(visoes(df)['inicio_40'][0].split()), 40)
        self.assertNotIn('MANCHETE', visoes(df)['inicio_40'][0])
        textos, _, pesos = criar_variantes(df, curtos=False)
        self.assertEqual(textos, df.entrada_a.tolist())
        self.assertEqual(pesos.tolist(), [1.])


if __name__ == '__main__':
    unittest.main()
