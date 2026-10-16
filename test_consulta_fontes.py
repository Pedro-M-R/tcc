"""Consulta por texto sem atribuir vereditos por semelhança de assunto."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from lxml import html

sys.path.insert(0, str(Path(__file__).resolve().parent / 'web'))
from consulta_fontes import (consultar_boatos as consultar_fontes, consultar_g1,
                            consultar_fontes as consultar_todas, buscar_g1,
                            correspondencia, correspondencia_g1, consultas)
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
        self.assertEqual([b.label for b in app.button], ['Analisar notícia'])
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

    def test_interface_sem_correspondencia_executa_svm(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        for status in ['nao_encontrada', 'indisponivel', 'relacionadas', 'insuficiente']:
            with self.subTest(status=status):
                st.cache_resource.clear()
                app = AppTest.from_file(str(Path(__file__).resolve().parent / 'web/app.py')).run()
                modelo = Mock()
                modelo.analisar.return_value = {'rotulo': 'fake', 'margem_svm': -.7}
                with patch('consulta_fontes.consultar_fontes', return_value={'status': status, 'fontes': []}), \
                     patch('inferencia.Classificador', return_value=modelo):
                    app.text_area[0].set_value(ALEGACAO)
                    app.button[0].click().run()
                self.assertFalse(app.exception)
                modelo.analisar.assert_called_once_with('', ALEGACAO)
                self.assertTrue(any('Indícios identificados pelo modelo' in m.value for m in app.markdown))
        st.cache_resource.clear()


G1_URL = 'https://g1.globo.com/sp/noticia/2026/10/16/bibliotecas.ghtml'
G1_TITULO = 'Prefeitura anuncia novas bibliotecas públicas para os moradores da cidade'
G1_TEXTO = 'A prefeitura anunciou nesta sexta-feira a abertura de novas bibliotecas públicas para atender moradores da cidade.'
G1_HTML = (f'<html><head><meta charset="utf-8"></head><body><h1>{G1_TITULO}</h1>'
           f'<p class="content-text__container">{G1_TEXTO}</p></body></html>').encode()


class TestConsultaG1(unittest.TestCase):
    def test_noticia_por_titulo_ou_texto(self):
        for titulo, texto in [(G1_TITULO, ''), ('', G1_TEXTO), ('', G1_TITULO)]:
            with patch('consulta_fontes.buscar_g1', return_value=[{'url': G1_URL, 'titulo': G1_TITULO}]), \
                 patch('consulta_fontes.baixar_html', return_value=(G1_HTML, G1_URL)):
                r = consultar_g1(titulo, texto)
            self.assertEqual(r['status'], 'encontrada')
            self.assertEqual(r['checagem']['rotulo'], 'true')
            self.assertEqual(r['checagem']['fonte'], 'G1')

    def test_negacao_data_e_sujeito_diferentes_nao_correspondem(self):
        for texto in [G1_TEXTO.replace('anunciou', 'não anunciou'),
                      G1_TEXTO.replace('sexta-feira', 'segunda-feira'),
                      G1_TEXTO.replace('prefeitura', 'universidade')]:
            self.assertIsNone(correspondencia_g1('', texto, html.fromstring(G1_HTML), G1_URL))

    def test_fato_ou_fake_e_desmentido_nao_sao_indicios_de_verdade(self):
        self.assertIsNone(correspondencia_g1('', G1_TEXTO, html.fromstring(G1_HTML),
                                           G1_URL.replace('/sp/', '/fato-ou-fake/')))
        pagina_falsa = G1_HTML.replace(G1_TITULO.encode(), ('É falso que ' + G1_TITULO).encode())
        self.assertIsNone(correspondencia_g1('', G1_TEXTO, html.fromstring(pagina_falsa), G1_URL))

    def test_dominio_imitado_e_redirecionamento_nao_atribuem_verdade(self):
        imitado = G1_URL.replace('g1.globo.com', 'g1.globo.com.example.org')
        with patch('consulta_fontes.buscar_g1', return_value=[{'url': imitado, 'titulo': G1_TITULO}]), \
             patch('consulta_fontes.baixar_html') as baixar:
            self.assertEqual(consultar_g1('', G1_TEXTO)['status'], 'nao_encontrada')
            baixar.assert_not_called()
        with patch('consulta_fontes.buscar_g1', return_value=[{'url': G1_URL, 'titulo': G1_TITULO}]), \
             patch('consulta_fontes.baixar_html', return_value=(G1_HTML, imitado)):
            self.assertNotEqual(consultar_g1('', G1_TEXTO)['status'], 'encontrada')

    def test_api_extrai_url_original_e_filtra_outros_dominios(self):
        from urllib.parse import urlencode
        resposta = Mock()
        resposta.json.return_value = [{'result': {'hits': {'hits': [
            {'_source': {'title': G1_TITULO, 'url': 'https://measures.globo.com/v1/click?' + urlencode({'u': G1_URL})}},
            {'_source': {'title': 'Outro site', 'url': 'https://example.org/noticia/x.ghtml'}}]}}}]
        with patch('consulta_fontes.requests.post') as post:
            post.return_value.__enter__.return_value = resposta
            self.assertEqual(buscar_g1('prefeitura bibliotecas', 5), [{'titulo': G1_TITULO, 'url': G1_URL}])
            self.assertEqual(post.call_args.kwargs['json'][0]['params']['q'], 'prefeitura bibliotecas')

    def test_boatos_tem_prioridade_e_g1_e_consultado_sem_correspondencia(self):
        for status in ['encontrada', 'conflitante']:
            with patch('consulta_fontes.consultar_boatos', return_value={'status': status}) as boatos, \
                 patch('consulta_fontes.consultar_g1') as g1:
                self.assertEqual(consultar_todas('', ALEGACAO)['status'], status)
                boatos.assert_called_once()
                g1.assert_not_called()
        with patch('consulta_fontes.consultar_boatos', return_value={'status': 'nao_encontrada', 'fontes': []}), \
             patch('consulta_fontes.consultar_g1', return_value={'status': 'encontrada', 'checagem': {'fonte': 'G1'}}) as g1:
            self.assertEqual(consultar_todas('', G1_TEXTO)['checagem']['fonte'], 'G1')
            g1.assert_called_once()

    def test_interface_g1_nao_executa_modelo(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parent / 'web/app.py')).run()
        resultado = {'status': 'encontrada', 'checagem': {'fonte': 'G1', 'rotulo': 'true',
                     'alegacao': G1_TEXTO, 'url': G1_URL}}
        with patch('consulta_fontes.consultar_fontes', return_value=resultado), \
             patch('inferencia.Classificador') as modelo:
            app.text_area[0].set_value(G1_TEXTO)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('Há indícios de que seja verdadeira' in m.value for m in app.markdown))
        self.assertTrue(any('Notícia correspondente encontrada no G1' in m.value for m in app.markdown))
        modelo.assert_not_called()

    def test_interface_vazia_nao_busca_nem_executa_modelo(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parent / 'web/app.py')).run()
        with patch('consulta_fontes.consultar_fontes') as buscar, patch('inferencia.Classificador') as modelo:
            app.button[0].click().run()
        self.assertFalse(app.exception)
        buscar.assert_not_called()
        modelo.assert_not_called()


if __name__ == '__main__':
    unittest.main()
