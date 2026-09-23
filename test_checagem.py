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

    def test_interface_separa_fonte_e_modelo_sem_fabricar_percentual(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        noticia = extrair_html(pagina(), URL)
        modelo = Mock()
        modelo.analisar.return_value = {"rotulo": "true", "truncado": True,
                                       "probabilidades": {"fake": .001, "true": .999}}
        app = AppTest.from_file(str(Path(__file__).resolve().parent / "web/app.py")).run()
        app.selectbox[0].set_value("BERTimbau Base").run()
        app.radio[0].set_value("Usar um link").run()
        with patch("extrair_link.extrair_noticia", return_value=noticia):
            with patch("inferencia.Classificador", return_value=modelo):
                app.text_input[0].set_value(URL)
                app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Alegação falsa" in m.value and "Boatos.org" in m.value for m in app.markdown))
        self.assertTrue(any("diverge" in w.value for w in app.warning))
        self.assertEqual(app.metric[0].value, "99,90%")
        self.assertTrue(any("texto da checagem" in e.label for e in app.expander))
        st.cache_resource.clear()


if __name__ == "__main__":
    unittest.main()
