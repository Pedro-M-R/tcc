"""Verifica formulário e inferência do site sem depender de SVM."""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from streamlit.testing.v1 import AppTest

WEB = Path(__file__).resolve().parent / "web"
sys.path.insert(0, str(WEB))
from inferencia import Classificador, preparar_entrada


class TestSite(unittest.TestCase):
    def test_link_extrai_e_analisa_sem_digitar_conteudo(self):
        import streamlit as st
        st.cache_resource.clear()
        modelo = Mock()
        modelo.analisar.return_value = {"rotulo": "true", "truncado": False,
                                       "probabilidades": {"true": .8, "fake": .2}}
        noticia = {"titulo": "Título extraído", "texto": "Conteúdo extraído da notícia.",
                   "url": "https://example.com/artigo", "texto_limitado": False}
        app = self.abrir()
        app.radio[0].set_value("Usar um link").run()
        with patch("extrair_link.extrair_noticia", return_value=noticia) as extrair:
            with patch("inferencia.Classificador", return_value=modelo):
                app.text_input[0].set_value(noticia["url"])
                app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        extrair.assert_called_once_with(noticia["url"])
        modelo.analisar.assert_called_once_with(noticia["titulo"], noticia["texto"])
        self.assertEqual(app.text_area[0].value, noticia["texto"])
        self.assertEqual(app.metric[0].value, "80,00%")
        from extrair_link import ErroLink
        with patch("extrair_link.extrair_noticia", side_effect=ErroLink("Página indisponível")):
            app.button[0].click().run()
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(app.warning[0].value, "Página indisponível")
        st.cache_resource.clear()

    def abrir(self, modelo=None):
        with patch("inferencia.Classificador", return_value=modelo) as carregar:
            app = AppTest.from_file(str(WEB / "app.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(carregar.call_count, 0)
        app.selectbox[0].set_value("BERTimbau Base").run()
        return app

    def test_formulario_vazio_nao_carrega_modelo(self):
        app = self.abrir()
        with patch("inferencia.Classificador") as carregar:
            app.button[0].click().run()
            self.assertEqual(len(app.warning), 1)
            carregar.assert_not_called()

    def test_somente_titulo_ou_texto_analisa_como_na_janela(self):
        import streamlit as st
        for titulo, texto in (("Título", ""), ("", "Texto da notícia")):
            st.cache_resource.clear()
            app = self.abrir()
            modelo = Mock()
            modelo.analisar.return_value = {"rotulo": "true", "truncado": False,
                                           "probabilidades": {"true": .8, "fake": .2}}
            with patch("inferencia.Classificador", return_value=modelo):
                app.text_input[0].set_value(titulo)
                app.text_area[0].set_value(texto)
                app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            modelo.analisar.assert_called_once_with(titulo, texto)
        st.cache_resource.clear()

    def test_falha_parcial_nao_e_apresentada_como_concordancia(self):
        import streamlit as st
        st.cache_resource.clear()
        app = self.abrir()
        app.selectbox[0].set_value("Comparar os dois modelos").run()
        svm = Mock()
        svm.analisar.return_value = {"rotulo": "fake", "margem_svm": -.4}
        with patch("inferencia.Classificador", side_effect=[svm, RuntimeError("modelo indisponível")]), patch("logging.Logger.exception"):
            app.text_area[0].set_value("Texto de teste")
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertTrue(any("Comparação incompleta" in x.value for x in app.warning))
        self.assertFalse(any("concordam" in x.value for x in app.info))
        st.cache_resource.clear()

    def test_comparacao_padrao_mostra_discordancia_e_margem_sem_percentual(self):
        import streamlit as st
        st.cache_resource.clear()
        inicial = AppTest.from_file(str(WEB / "app.py"), default_timeout=30).run()
        self.assertEqual(inicial.selectbox[0].value, "Comparar os dois modelos")
        app = self.abrir()
        app.selectbox[0].set_value("Comparar os dois modelos").run()
        svm, bert = Mock(), Mock()
        svm.analisar.return_value = {"rotulo": "fake", "margem_svm": -.4}
        bert.analisar.return_value = {"rotulo": "true", "truncado": False,
                                      "probabilidades": {"true": .99983, "fake": .00017}}
        with patch("inferencia.Classificador", side_effect=lambda caminho, tipo: svm if tipo == "svm" else bert):
            app.text_area[0].set_value("Mesmo texto nos dois modelos.")
            app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("discordam" in x.value for x in app.warning))
        self.assertEqual([x.value for x in app.metric], ["-0,4000", "99,98%", "0,02%"])
        for modelo in (svm, bert):
            modelo.analisar.assert_called_once_with("", "Mesmo texto nos dois modelos.")
        st.cache_resource.clear()

    def test_classes_e_texto_longo(self):
        import streamlit as st
        for rotulo, esperado, prob_true in (("true", "Verdadeira", 0.823), ("fake", "Falsa", 0.176)):
            st.cache_resource.clear()
            modelo = Mock()
            modelo.analisar.return_value = {
                "rotulo": rotulo, "truncado": True,
                "probabilidades": {"true": prob_true, "fake": 1 - prob_true},
            }
            app = self.abrir()
            with patch("inferencia.Classificador", return_value=modelo):
                app.text_input[0].set_value("Título")
                app.text_area[0].set_value("Texto da notícia.")
                app.button[0].click().run()
            self.assertEqual(len(app.exception), 0)
            modelo.analisar.assert_called_once_with("Título", "Texto da notícia.")
            self.assertTrue(any(f"<h2>{esperado}</h2>" in x.value for x in app.markdown))
            self.assertTrue(any("início do texto" in x.value for x in app.caption))
            self.assertEqual(app.metric[0].label, "Verdadeira")
            self.assertEqual(app.metric[0].value, f"{prob_true * 100:.2f}%".replace(".", ","))
            self.assertEqual(app.metric[1].label, "Falsa")
            self.assertEqual(app.metric[1].value, f"{(1 - prob_true) * 100:.2f}%".replace(".", ","))
            self.assertIn("Não considere o resultado 100% certo", app.info[0].value)
            # Nova submissão inválida não deve manter o resultado anterior.
            app.text_area[0].set_value(" ")
            app.text_input[0].set_value(" ")
            app.button[0].click().run()
            self.assertFalse(any("<h2>" in x.value for x in app.markdown))
            self.assertEqual(len(app.metric), 0)
        st.cache_resource.clear()

    def test_erro_modelo_nao_expoe_detalhes(self):
        app = self.abrir()
        with patch("inferencia.Classificador", side_effect=FileNotFoundError("arquivo interno")):
            with patch("logging.Logger.exception"):
                app.text_input[0].set_value("Título")
                app.text_area[0].set_value("Texto")
                app.button[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 1)
        self.assertNotIn("arquivo interno", app.error[0].value)

    def test_normalizacao_compativel_com_treino(self):
        from dados_modelos import juntar_texto
        titulo, texto = "  Ti\u0301tulo\n", "Texto\t com   espaços e AÇÃO."
        self.assertEqual(preparar_entrada(titulo, texto), juntar_texto(titulo, texto))

    def test_modelo_ausente(self):
        import tempfile
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaises(FileNotFoundError):
                Classificador(pasta)


if __name__ == "__main__":
    unittest.main()
