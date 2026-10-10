"""Vazamento nos dois analisadores, persistência e isolamento das fontes."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from treinar_svm_leve import vetorizador
from treinamento_local.limpeza import limpar_texto
from treinamento_local.vazamento_svm import carregar_svm, dividir_svm, relatorio_atributos


CONFIG = {"nome": "teste", "palavras": 300, "ordem_palavras": 2,
          "caracteres": 300, "ordem_caracteres": (2, 6), "peso_caracteres": .5}


class TestLimpezaSVM(unittest.TestCase):
    def test_preserva_conteudo_e_remove_pistas_e_linhas_editoriais(self):
        texto = ("É #FALSO: montagem verificada pela Agência Lupa e Aos Fatos.\n"
                 "O governo não confirmou o golpe em 2026.\nPor: Maria Silva\n"
                 "Todos os direitos reservados.\nhttps://boatos.org/fake email@teste.com")
        limpo = limpar_texto(texto)
        self.assertIn("governo nao confirmou o golpe em 2026", limpo)
        for pista in ("falso", "montagem", "verificada", "agencia lupa", "aos fatos", "maria", "reservados", "https", "email", "#"):
            self.assertNotIn(pista, limpo)
        self.assertEqual(limpar_texto(limpo), limpo)
        self.assertEqual(limpar_texto("#FALSO #TRUE #VERIFICADO boatos org boatos.org UOL Confere escola"), "escola")
        self.assertEqual(limpar_texto("#educacao escola"), "#educacao escola")

    def test_ambos_analisadores_ignoram_pistas_antes_de_gerar_ngramas(self):
        uniao = vetorizador(CONFIG)
        for _, vetor in uniao.transformer_list:
            analisar = vetor.build_analyzer()
            self.assertEqual(analisar("falso fake boato comprovado verdadeiro checagem"), [])
            self.assertEqual(analisar("FALSO escola abriu"), analisar("escola abriu"))
        bruto = vetorizador({**CONFIG, "limpeza": False})
        self.assertIn("fake", bruto.transformer_list[0][1].build_analyzer()("fake escola"))

    def test_serializacao_em_outro_processo_e_sentido_dos_pesos(self):
        textos = [f"{'FAKE violeta lunar' if i % 2 == 0 else 'VERDADEIRO laranja solar'} noticia {i}" for i in range(20)]
        modelo = Pipeline([("tfidf", vetorizador(CONFIG)), ("svm", LinearSVC(dual="auto"))])
        modelo.fit(textos, [i % 2 for i in range(20)])
        consultas = ["falso violeta lunar", "comprovado laranja solar"]
        with tempfile.TemporaryDirectory() as tmp:
            pasta = Path(tmp)
            joblib.dump(modelo, pasta / "modelo.joblib")
            codigo = ("import json,joblib,sys; m=joblib.load(sys.argv[1]); "
                      "print(json.dumps(m.decision_function(json.loads(sys.argv[2])).tolist()))")
            resultado = subprocess.run([sys.executable, "-c", codigo, str(pasta / "modelo.joblib"), json.dumps(consultas)],
                                       capture_output=True, text=True, check=True)
            np.testing.assert_allclose(json.loads(resultado.stdout), modelo.decision_function(consultas))
            rel = relatorio_atributos(modelo, pasta / "atributos.json", 5)
            self.assertTrue(all(i["peso"] < 0 for i in rel["fake"]))
            self.assertTrue(all(i["peso"] > 0 for i in rel["true"]))

    def test_fontes_e_duplicatas_nao_atravessam_particoes(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "base.csv"
            linhas = [{"titulo": f"Relato {s} classe {r} item {i}", "texto": f"Noticia local {s} {'solar' if r == 'true' else 'lunar'} {i}",
                       "rotulo": r, "url": f"https://fonte{s}.example/noticia/{r}/{i}"}
                      for s in range(20) for r in ("fake", "true") for i in range(2)]
            # Duas fontes ligadas pela mesma matéria devem ficar juntas.
            linhas[4]["titulo"], linhas[4]["texto"] = linhas[0]["titulo"], linhas[0]["texto"]
            pd.DataFrame(linhas).to_csv(csv, index=False)
            df, _, _ = carregar_svm([csv], "fonte")
            self.assertEqual(df.iloc[0].grupo_split, df.iloc[4].grupo_split)
            partes, manifesto = dividir_svm(df, 42, "fonte")
            partes2, manifesto2 = dividir_svm(df, 42, "fonte")
            self.assertEqual(manifesto, manifesto2)
            vistos, chaves = set(), set()
            for parte in partes.values():
                self.assertFalse(vistos & set(parte.fonte_split))
                self.assertFalse(chaves & set(parte.chave))
                vistos.update(parte.fonte_split)
                chaves.update(parte.chave)
                self.assertEqual(set(parte.label), {0, 1})
            self.assertEqual(sum(map(len, partes2.values())), len(linhas))

    def test_ablation_preserva_texto_bruto_e_une_duplicatas_limpas(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "base.csv"
            linhas = [{"titulo": "Farsa escola", "texto": "montagem relato", "rotulo": "fake"},
                      {"titulo": "escola", "texto": "relato", "rotulo": "fake"},
                      {"titulo": "hospital", "texto": "atendimento publico", "rotulo": "true"}]
            pd.DataFrame(linhas).to_csv(csv, index=False)
            df, _, _ = carregar_svm([csv])
            self.assertIn("Farsa", df.iloc[0].entrada_a)
            self.assertEqual(df.iloc[0].grupo_split, df.iloc[1].grupo_split)
            with self.assertRaisesRegex(ValueError, "hostname"):
                carregar_svm([csv], "fonte")


if __name__ == "__main__":
    unittest.main()
