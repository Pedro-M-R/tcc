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

from svm_leve import ModeloSVMLeve
from preparacao_noticias import preparar_noticia
from web.inferencia import Classificador, modelos_do_site
from test_triagem import NOTICIA

RAIZ = Path(__file__).resolve().parent
MODELO = RAIZ / "modelos/svm_leve"


class TestSVMLeve(unittest.TestCase):
    def test_site_ignora_configuracao_antiga_bert(self):
        with patch.dict(os.environ, {"TCC_MODEL_PATH": "experimentos/bert", "BERTIMBAU_MODEL_PATH": "bert"}):
            self.assertEqual([m["tipo"] for m in modelos_do_site()], ["svm_leve"])

    def test_modelo_incompleto_nao_carrega(self):
        with tempfile.TemporaryDirectory() as pasta:
            with self.assertRaisesRegex(ValueError, "concluído"):
                ModeloSVMLeve(pasta)

    def test_abstencao_antes_de_classificar(self):
        modelo = ModeloSVMLeve.__new__(ModeloSVMLeve)
        modelo.dominio = Mock()
        modelo.escores = Mock()
        self.assertEqual(modelo.analisar_noticia("", "Olá")['rotulo'], "inconclusivo")
        modelo.dominio.aceitar.assert_not_called()
        modelo.dominio.aceitar.return_value = np.array([False])
        self.assertEqual(modelo.analisar_noticia("", NOTICIA)['rotulo'], "inconclusivo")
        modelo.escores.assert_not_called()

    def test_carregamento_real_sem_torch(self):
        codigo = """
import sys
from web.inferencia import Classificador
m = Classificador('modelos/svm_leve')
assert m.tipo == 'svm_leve'
assert m.analisar('', 'Olá')['rotulo'] == 'inconclusivo'
assert 'torch' not in sys.modules
assert 'transformers' not in sys.modules
assert 'prever' not in sys.modules
assert 'treinamento_local.inferencia' not in sys.modules
print('CPU OK')
"""
        r = subprocess.run([sys.executable, "-c", codigo], cwd=RAIZ, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

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
