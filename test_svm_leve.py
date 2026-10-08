"""Regressões do modelo leve e do fluxo real do Streamlit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from svm_leve import ModeloSVMLeve, opiniao_pessoal
from preparacao_noticias import preparar_noticia
from web.inferencia import Classificador, modelos_do_site
from test_triagem import NOTICIA

RAIZ = Path(__file__).resolve().parent
MODELO = Path(os.environ.get("TCC_SVM_MODEL_PATH", RAIZ / "modelos/svm_leve")).resolve()


class TestSVMLeve(unittest.TestCase):
    def test_site_ignora_configuracao_antiga_bert(self):
        with patch.dict(os.environ, {"TCC_MODEL_PATH": "experimentos/bert", "BERTIMBAU_MODEL_PATH": "bert"}):
            self.assertEqual([m["tipo"] for m in modelos_do_site()], ["svm_leve"])

    def test_modelo_incompleto_nao_carrega(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaisesRegex(ValueError, "concluído"):
                ModeloSVMLeve(pasta)

    def test_texto_curto_nao_e_bloqueado_por_triagem_ou_limiar_antigo(self):
        modelo = ModeloSVMLeve.__new__(ModeloSVMLeve)
        modelo.politica = {"rotulos": ["fake", "true"], "limiares": [1.1, 1.1]}
        modelo.escores = Mock(return_value=np.array([[.49, .51]]))
        for titulo, texto in (("", "Olá"), ("Somente o título", ""), ("", NOTICIA)):
            self.assertEqual(modelo.analisar_noticia(titulo, texto)['rotulo'], "true")
        self.assertEqual(modelo.escores.call_count, 3)
        with self.assertRaisesRegex(ValueError, "Digite"):
            modelo.analisar_noticia(" ", " ")

    def test_opiniao_inteira_e_regra_explicita_sem_escore_fabricado(self):
        modelo = ModeloSVMLeve.__new__(ModeloSVMLeve)
        modelo.escores = Mock()
        for frase in ("pedro eh bonito", "pedro eh bonita", "O Pedro é bonito.", "Maria é linda", "João é muito bonito"):
            with self.subTest(frase=frase):
                r = modelo.analisar_noticia(frase, frase)
                self.assertEqual(r["rotulo"], "fake")
                self.assertEqual(r["origem"], "regra_opiniao")
                self.assertNotIn("probabilidades", r)
        modelo.escores.assert_not_called()
        for frase in ("Pedro não é bonito", "Pedro eh bonito?", "João disse Pedro é bonito",
                      "Pedro eh bonito. O prefeito anunciou uma obra."):
            self.assertFalse(opiniao_pessoal(frase, frase), frase)
        self.assertFalse(opiniao_pessoal("Pedro eh bonito", NOTICIA))
        self.assertFalse(opiniao_pessoal("Pedro eh bonito", "O hospital fechou ontem."))

    def test_carregamento_real_sem_torch(self):
        codigo = """
import sys
from web.inferencia import Classificador
m = Classificador(sys.argv[1])
assert m.tipo == 'svm_leve'
assert m.analisar('', 'Olá')['rotulo'] in ('fake', 'true')
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'prever' not in sys.modules
assert 'treinamento_local.inferencia' not in sys.modules
print('CPU OK')
"""
        r = subprocess.run([sys.executable, "-c", codigo, str(MODELO)], cwd=RAIZ, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_modelo_real_sempre_classifica_textos_preenchidos(self):
        modelo = ModeloSVMLeve(MODELO)
        for texto in (NOTICIA, "O prefeito anunciou uma obra.", "galaxias unicornios dragao nebulosa"):
            r = modelo.analisar_noticia("", texto)
            self.assertIn(r["rotulo"], ("fake", "true"))
            self.assertEqual(r["origem"], "svm")
            self.assertAlmostEqual(sum(r["probabilidades"].values()), 1.)

    def test_streamlit_caso_pedro_e_texto_curto(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        sys.path.insert(0, str(RAIZ / "web"))
        app = AppTest.from_file(str(RAIZ / "web/app.py"), default_timeout=30).run()
        app.text_input[0].set_value("pedro eh bonita")
        app.text_area[0].set_value("pedro eh bonita")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(app.warning)
        self.assertTrue(any("<h2>Falsa</h2>" in m.value for m in app.markdown))
        self.assertTrue(any("Não é uma previsão do SVM" in c.value for c in app.caption))
        self.assertFalse(app.metric)
        app.text_input[0].set_value("")
        app.text_area[0].set_value("O prefeito anunciou uma obra.")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertFalse(app.warning)
        self.assertEqual(len(app.metric), 2)

    def test_cache_recarrega_apos_atualizar_artefatos_na_mesma_pasta(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        sys.path.insert(0, str(RAIZ / "web"))
        modelo = Mock()
        modelo.analisar.return_value = {"rotulo": "true", "calibrado": True,
                                       "probabilidades": {"fake": .1, "true": .9}}
        with tempfile.TemporaryDirectory() as pasta:
            concluido = Path(pasta) / "concluido.json"
            concluido.write_text("{}", encoding="utf-8")
            with patch.dict(os.environ, {"TCC_SVM_MODEL_PATH": pasta}), \
                 patch("inferencia.Classificador", return_value=modelo) as carregar:
                app = AppTest.from_file(str(RAIZ / "web/app.py"), default_timeout=30).run()
                app.text_area[0].set_value(NOTICIA)
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(carregar.call_count, 1)
                app.button[0].click().run()
                self.assertEqual(carregar.call_count, 1)
                concluido.write_text('{"nova_versao": true}', encoding="utf-8")
                app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertEqual(carregar.call_count, 2)

    @unittest.skipUnless((RAIZ / "resultados/bases/base_atualizada_2026-10-06.csv").is_file(), "Exige base local")
    def test_mesmos_resultados_do_teste_reservado(self):
        base = pd.read_csv(RAIZ / "resultados/bases/base_atualizada_2026-10-06.csv", keep_default_na=False)
        previsoes = pd.read_csv(MODELO / "predicoes_teste.csv")
        amostra = previsoes.groupby("resultado", sort=True).head(3)
        modelo = Classificador(MODELO, "svm_leve")
        for _, linha in amostra.iterrows():
            noticia = base.iloc[int(linha.linha_csv) - 2]
            with self.subTest(linha=int(linha.linha_csv)):
                r = modelo.analisar(noticia.titulo, noticia.texto)
                self.assertEqual(r['rotulo'], linha.resultado)
                p = modelo.leve.escores([preparar_noticia(noticia.titulo, noticia.texto)])[0]
                self.assertAlmostEqual(float(p.sum()), 1.)

    @unittest.skipUnless((RAIZ / "resultados/bases/base_atualizada_2026-10-06.csv").is_file(), "Exige base local")
    def test_streamlit_real_texto_e_link(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        st.cache_resource.clear()
        sys.path.insert(0, str(RAIZ / "web"))
        base = pd.read_csv(RAIZ / "resultados/bases/base_atualizada_2026-10-06.csv", keep_default_na=False)
        previsoes = pd.read_csv(MODELO / "predicoes_teste.csv")
        linha = previsoes[previsoes.resultado != "inconclusivo"].iloc[0]
        noticia = base.iloc[int(linha.linha_csv) - 2]
        app = AppTest.from_file(str(RAIZ / 'web/app.py'), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox[0].options, ["TF-IDF + SVM"])
        app.text_input[0].set_value(noticia.titulo[:500])
        app.text_area[0].set_value(noticia.texto[:30000])
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.metric), 2)
        escores = [m.value for m in app.metric]
        app.radio[0].set_value('Usar um link').run()
        with patch('extrair_link.extrair_noticia', return_value={
            'titulo': noticia.titulo[:500], 'texto': noticia.texto[:30000],
            'url': 'https://example.org/noticia', 'texto_limitado': False}):
            app.text_input[0].set_value('https://example.org/noticia')
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual([m.value for m in app.metric], escores)
        st.cache_resource.clear()


if __name__ == '__main__':
    unittest.main()
