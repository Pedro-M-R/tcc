"""Regressões para checagem de uma alegação versus previsão sobre o artigo."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))
from extrair_link import extrair_html

URL = "https://www.boatos.org/exemplo.html"
PARAGRAFO = "Este texto explica uma alegação que circulou entre moradores. A equipe consultou documentos e entrevistou responsáveis para verificar as informações, sem atribuir a conclusão a um modelo de inteligência artificial."


def pagina(selo="Fake news ❌", conclusao=True):
    return ('<html><head><meta charset="utf-8"><title>História circula na cidade?</title></head>'
            '<body><article><h1>É falsa a história que circula na cidade</h1>'
            '<p>Boato – A biblioteca da cidade será fechada definitivamente.</p>'
            f'<p>{PARAGRAFO}</p>' + ('<p>Conclusão</p>' if conclusao else '') +
            f'<p>{PARAGRAFO}</p><p>{selo}</p></article></body></html>')


class TestChecagem(unittest.TestCase):
    def test_preserva_h1_e_extrai_conclusao_explicita(self):
        r = extrair_html(pagina().encode(), URL)
        self.assertEqual(r["titulo"], "É falsa a história que circula na cidade")
        self.assertEqual(r["checagem"]["rotulo"], "fake")
        self.assertEqual(r["checagem"]["alegacao"], "A biblioteca da cidade será fechada definitivamente.")

    def test_dominio_nao_determina_rotulo(self):
        self.assertIsNone(extrair_html(pagina(selo="Verificação inconclusiva"), URL)["checagem"])
        self.assertEqual(extrair_html(pagina(selo="Verdadeiro"), URL)["checagem"]["rotulo"], "true")

    def test_sem_conclusao_ou_mencao_em_prosa_nao_gera_veredito(self):
        self.assertIsNone(extrair_html(pagina(conclusao=False), URL)["checagem"])
        self.assertIsNone(extrair_html(pagina(selo="O artigo debate o termo fake news."), URL)["checagem"])

    def test_site_diferente_nao_recebe_atribuicao_ao_boatos(self):
        for url in ("https://example.com", "https://boatos.org.example.com"):
            self.assertIsNone(extrair_html(pagina(), url)["checagem"])

    def test_vereditos_conflitantes_nao_sao_resolvidos_arbitrariamente(self):
        html = pagina().replace("</article>", "<p>Verdadeiro</p></article>")
        self.assertIsNone(extrair_html(html, URL)["checagem"])

    def test_interface_exibe_alegacao_sem_classificar_artigo_ou_fabricar_percentual(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        noticia = extrair_html(pagina(), URL)
        modelo = Mock()
        modelo.analisar.return_value = {"rotulo": "true", "truncado": True,
                                       "probabilidades": {"fake": .001, "true": .999}}
        app = AppTest.from_file(str(Path(__file__).resolve().parent / "web/app.py"), default_timeout=30).run()
        app.selectbox[0].set_value("TF-IDF + SVM").run()
        app.radio[0].set_value("Usar um link").run()
        with patch("extrair_link.extrair_noticia", return_value=noticia):
            with patch("inferencia.Classificador", return_value=modelo) as carregar:
                app.text_input[0].set_value(URL)
                app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Há indícios de que seja uma alegação falsa" in m.value and "Boatos.org" in m.value for m in app.markdown))
        self.assertFalse(app.warning)
        self.assertFalse(app.metric)
        self.assertFalse(any("Há indícios de que seja verdadeira" in m.value for m in app.markdown))
        self.assertTrue(any("não uma previsão do SVM" in e.value for e in app.caption))
        carregar.assert_not_called()
        modelo.analisar.assert_not_called()
        st.cache_resource.clear()

    def test_link_colado_no_campo_de_texto_tambem_usa_checagem(self):
        from streamlit.testing.v1 import AppTest
        noticia = extrair_html(pagina(), URL)
        app = AppTest.from_file(str(Path(__file__).resolve().parent / "web/app.py"), default_timeout=30).run()
        with patch("extrair_link.extrair_noticia", return_value=noticia) as extrair, \
             patch("inferencia.Classificador") as carregar:
            app.text_area[0].set_value(URL)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        extrair.assert_called_once_with(URL)
        carregar.assert_not_called()
        self.assertTrue(any("Há indícios de que seja uma alegação falsa" in m.value for m in app.markdown))
        self.assertFalse(app.metric)

    def test_sem_selo_nao_usa_svm_para_inventar_conclusao(self):
        from streamlit.testing.v1 import AppTest
        noticia = extrair_html(pagina(selo="Verificação inconclusiva"), URL)
        self.assertTrue(noticia["pagina_checagem"])
        app = AppTest.from_file(str(Path(__file__).resolve().parent / "web/app.py"), default_timeout=30).run()
        app.radio[0].set_value("Usar um link").run()
        with patch("extrair_link.extrair_noticia", return_value=noticia), patch("inferencia.Classificador") as carregar:
            app.text_input[0].set_value(URL)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("não identificada" in w.value for w in app.warning))
        self.assertFalse(app.metric)
        carregar.assert_not_called()

    def test_texto_colado_de_checagem_nao_autentica_conclusao(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).resolve().parent / "web/app.py"), default_timeout=30).run()
        with patch("inferencia.Classificador") as carregar:
            app.text_area[0].set_value("Boato - O parque vai fechar.\nChecagem\nRelato de investigação.\nConclusão\nFake news")
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("parece ser uma checagem" in w.value for w in app.warning))
        self.assertFalse(app.metric)
        carregar.assert_not_called()

    def test_conclusao_verdadeira_nao_vira_falsa_por_dominio(self):
        from streamlit.testing.v1 import AppTest
        noticia = extrair_html(pagina(selo="Verdadeiro"), URL)
        app = AppTest.from_file(str(Path(__file__).resolve().parent / "web/app.py"), default_timeout=30).run()
        app.radio[0].set_value("Usar um link").run()
        with patch("extrair_link.extrair_noticia", return_value=noticia):
            app.text_input[0].set_value(URL)
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Alegação verdadeira" in m.value for m in app.markdown))
        self.assertFalse(any("Há indícios de que seja uma alegação falsa" in m.value for m in app.markdown))

    @unittest.skipUnless(all((Path("resultados/revisao_casos_reais") / (nome + ".html")).exists()
                             for nome in ("eduarda", "tse")), "Exige HTML público salvo no diagnóstico local")
    def test_dois_links_reais_sem_rede_e_sem_inferir_sobre_artigo(self):
        import json
        from streamlit.testing.v1 import AppTest
        pasta = Path("resultados/revisao_casos_reais")
        for nome in ("eduarda", "tse"):
            with self.subTest(caso=nome):
                metadados = json.loads((pasta / (nome + ".json")).read_text(encoding="utf-8"))
                noticia = extrair_html((pasta / (nome + ".html")).read_bytes(), metadados["url"])
                self.assertEqual(noticia["checagem"]["rotulo"], "fake")
                app = AppTest.from_file(str(Path(__file__).resolve().parent / "web/app.py"), default_timeout=30).run()
                app.radio[0].set_value("Usar um link").run()
                with patch("extrair_link.extrair_noticia", return_value=noticia), patch("inferencia.Classificador") as carregar:
                    app.text_input[0].set_value(metadados["url"])
                    app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertFalse(app.metric)
                self.assertTrue(any("Há indícios de que seja uma alegação falsa" in m.value for m in app.markdown))
                self.assertFalse(any("Há indícios de que seja verdadeira" in m.value for m in app.markdown))
                carregar.assert_not_called()


if __name__ == "__main__":
    unittest.main()
