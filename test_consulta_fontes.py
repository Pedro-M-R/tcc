"""Consulta por texto sem atribuir vereditos por semelhança de assunto."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from lxml import html

sys.path.insert(0, str(Path(__file__).resolve().parent / 'web'))
from consulta_fontes import consultar_fontes, correspondencia, consultas
from extrair_link import ErroLink

URL = 'https://www.boatos.org/brasil/biblioteca.html'
ALEGACAO = 'A prefeitura vai fechar todas as bibliotecas públicas da cidade em 2026.'
CITACAO = 'Atenção moradores, todas as bibliotecas públicas da nossa cidade vão fechar na próxima segunda-feira, segundo esta mensagem.'
BUSCA = f'<article><h2><a href="{URL}">Bibliotecas vão fechar?</a></h2></article>'.encode()


def pagina(rotulo='Fake news', alegacao=ALEGACAO):
    return (f'<html><head><meta charset="utf-8"></head><body><article><p>Boato – {alegacao}</p>'
            f'<blockquote>{CITACAO}</blockquote><p>Conclusão</p><p>{rotulo}</p>'
            '</article></body></html>').encode()


class TestConsultaFontes(unittest.TestCase):
    def consultar(self, rotulo='Fake news', texto=ALEGACAO):
        with patch('consulta_fontes.baixar_html', side_effect=[(BUSCA, 'https://www.boatos.org/?s=x'),
                                                              (pagina(rotulo), URL)]):
            return consultar_fontes('', texto)

    def test_busca_encontra_alegacao_e_conclusao_falsa(self):
        r = self.consultar()
        self.assertEqual(r['status'], 'encontrada')
        self.assertEqual(r['checagem']['rotulo'], 'fake')
        self.assertEqual(r['checagem']['url'], URL)

    def test_conclusao_verdadeira_preservada(self):
        self.assertEqual(self.consultar('Verdadeiro')['checagem']['rotulo'], 'true')

    def test_mensagem_inteira_citada(self):
        self.assertEqual(self.consultar(texto=CITACAO)['checagem']['correspondencia'], 'mensagem_citada')

    def test_markdown_e_artigo_com_alegacao_no_primeiro_paragrafo(self):
        for quebra in ('\n', '\n\n'):
            r = self.consultar(texto=f'**{ALEGACAO}**{quebra}Análise\n\nRelato sobre a mensagem.')
            self.assertEqual(r['status'], 'encontrada')

    def test_endereco_citado_na_alegacao_nao_e_descartado(self):
        alegacao = ALEGACAO + ' A inscrição está em https://example.org/inscricao.'
        checagem = {'alegacao': alegacao}
        arvore = html.fromstring(pagina(alegacao=alegacao))
        self.assertIsNone(correspondencia('', alegacao.replace('example.org', 'outro.example'), checagem, arvore))

    def test_nao_transfere_rotulo_para_negacao_data_ou_sujeito_diferente(self):
        checagem = {'alegacao': ALEGACAO}
        arvore = html.fromstring(pagina())
        alteracoes = [ALEGACAO.replace('vai fechar', 'não vai fechar'),
                      ALEGACAO.replace('2026', '2027'),
                      ALEGACAO.replace('prefeitura', 'universidade'),
                      'É falso que ' + ALEGACAO, 'Não acredite: ' + CITACAO]
        for texto in alteracoes:
            with self.subTest(texto=texto):
                self.assertIsNone(correspondencia('', texto, checagem, arvore))

    def test_sem_selo_so_sugere_leitura(self):
        def baixar(url, **kwargs):
            return (BUSCA, 'https://www.boatos.org/?s=x') if '?s=' in url else (pagina('Sem conclusão'), URL)
        with patch('consulta_fontes.baixar_html', side_effect=baixar):
            r = consultar_fontes('', ALEGACAO)
        self.assertEqual(r['status'], 'relacionadas')
        self.assertNotIn('checagem', r)

    def test_fonte_indisponivel_e_busca_vazia_nao_significam_verdadeira(self):
        with patch('consulta_fontes.baixar_html', side_effect=ErroLink('offline')):
            self.assertEqual(consultar_fontes('', ALEGACAO)['status'], 'indisponivel')
        with patch('consulta_fontes.baixar_html', return_value=(b'<html></html>', 'https://www.boatos.org/')):
            self.assertEqual(consultar_fontes('', ALEGACAO)['status'], 'nao_encontrada')

    def test_nao_segue_resultado_em_dominio_imitado(self):
        busca = BUSCA.replace(b'www.boatos.org/', b'boatos.org.example.com/')
        with patch('consulta_fontes.baixar_html', return_value=(busca, 'https://www.boatos.org/')) as baixar:
            r = consultar_fontes('', ALEGACAO)
        self.assertEqual(r['status'], 'nao_encontrada')
        self.assertTrue(all('?s=' in c.args[0] for c in baixar.call_args_list))

    def test_vereditos_conflitantes(self):
        url2 = URL.replace('biblioteca', 'outra')
        busca = BUSCA + BUSCA.replace(URL.encode(), url2.encode())
        with patch('consulta_fontes.baixar_html', side_effect=[
            (busca, 'https://www.boatos.org/'), (pagina(), URL), (pagina('Verdadeiro'), url2)]):
            self.assertEqual(consultar_fontes('', ALEGACAO)['status'], 'conflitante')

    def test_consulta_curta_nao_envia_corpo_inteiro(self):
        buscas = consultas(ALEGACAO, 'CONTEUDO NAO UTILIZADO ' * 100)
        self.assertTrue(buscas)
        self.assertTrue(all(len(b.split()) <= 6 and 'conteudo' not in b for b in buscas))

    def test_interface_consulta_sem_link_e_sem_svm(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parent / 'web/app.py')).run()
        self.assertEqual([b.label for b in app.button], ['Consultar checagens pelo texto'])
        self.assertFalse(app.selectbox)
        resultado = self.consultar()
        with patch('consulta_fontes.consultar_fontes', return_value=resultado) as buscar, \
             patch('inferencia.Classificador') as modelo:
            app.text_area[0].set_value(ALEGACAO)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.metric)
        self.assertTrue(any('alegação falsa' in m.value for m in app.markdown))
        buscar.assert_called_once_with('', ALEGACAO)
        modelo.assert_not_called()

    def test_interface_sem_correspondencia_nao_cai_em_veredito_svm(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parent / 'web/app.py')).run()
        with patch('consulta_fontes.consultar_fontes', return_value={'status': 'nao_encontrada', 'fontes': []}), \
             patch('inferencia.Classificador') as modelo:
            app.text_area[0].set_value(ALEGACAO)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.metric)
        self.assertTrue(any('inconclusiva' in w.value for w in app.warning))
        modelo.assert_not_called()


if __name__ == '__main__':
    unittest.main()
