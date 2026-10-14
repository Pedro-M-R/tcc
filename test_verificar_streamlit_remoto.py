import unittest

from verificar_streamlit_remoto import campos_formulario


def elemento(tipo, identificador, rotulo='', formulario='noticia'):
    return {'newElement': {tipo: {'id': identificador, 'label': rotulo, 'formId': formulario}}}


class TestAcaoRemota(unittest.TestCase):
    def test_distingue_consulta_e_svm_independente_da_ordem(self):
        elementos = [elemento('textInput', 'titulo'), elemento('textArea', 'texto'),
                     elemento('button', 'svm', 'Analisar notícia com SVM'),
                     elemento('button', 'consulta', 'Consultar checagens pelo texto')]
        for ordem in (elementos, list(reversed(elementos))):
            self.assertEqual(campos_formulario(ordem, 'noticia')['button'], 'svm')
            self.assertEqual(campos_formulario(ordem, 'noticia', True)['button'], 'consulta')

    def test_versao_antiga_nao_finge_ter_consulta(self):
        elementos = [elemento('textInput', 'titulo'), elemento('textArea', 'texto'),
                     elemento('button', 'svm', 'Analisar notícia')]
        self.assertEqual(campos_formulario(elementos, 'noticia')['button'], 'svm')
        with self.assertRaises(ValueError):
            campos_formulario(elementos, 'noticia', True)

    def test_link_mantem_acao_original(self):
        elementos = [elemento('textInput', 'url', formulario='link'),
                     elemento('button', 'link', 'Ler link e analisar', 'link'),
                     elemento('button', 'consulta', 'Consultar checagens pelo texto')]
        self.assertEqual(campos_formulario(elementos, 'link')['button'], 'link')


if __name__ == '__main__':
    unittest.main()
