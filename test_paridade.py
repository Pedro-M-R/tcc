"""Confere os modelos reais da janela e do site com as mesmas entradas.

Execute: python -m unittest test_paridade -v
Exige os modelos finais e o CSV local; não treina nem baixa pesos.
"""

import json
import unittest
from pathlib import Path

import pandas as pd

from resultados_noticias import analisar_noticia, modelos_disponiveis
from web.inferencia import Classificador, modelos_do_site
from pontuacoes import formatar_probabilidade

RAIZ = Path(__file__).resolve().parent


@unittest.skipUnless((RAIZ / "fake tratada.csv").is_file(), "Exige a base local")
class TestParidade(unittest.TestCase):
    def test_janela_e_streamlit_com_modelos_reais(self):
        modelos = modelos_disponiveis()
        self.assertEqual({m["tipo"] for m in modelos}, {"svm", "bertimbau"})
        self.assertEqual(modelos_do_site(), modelos)
        classificadores = {m["tipo"]: Classificador(m["caminho"], m["tipo"]) for m in modelos}
        base = pd.read_csv(RAIZ / "fake tratada.csv", dtype=str, keep_default_na=False)
        divisao = json.loads((RAIZ / "resultados/divisao.json").read_text(encoding="utf-8"))
        # Seleção fixa e balanceada da avaliação anterior, sem escolher só acertos.
        teste = base.iloc[divisao["indices"]["teste"]]
        amostra = pd.concat([teste[teste.rotulo == r].head(12) for r in ("fake", "true")])
        casos = [(int(i), linha.titulo, linha.texto) for i, linha in amostra.iterrows()]
        casos += [("somente_texto", "", amostra.iloc[0].texto),
                  ("somente_titulo", amostra.iloc[0].titulo, ""),
                  ("acentos", "  Ti\u0301tulo\n", "Texto\t com   espaços e AÇÃO.")]
        registros = []
        for identificador, titulo, texto in casos:
            desktop = analisar_noticia(titulo, texto, modelos)["resultados"]
            for local in desktop:
                with self.subTest(caso=identificador, modelo=local["tipo"]):
                    site = classificadores[local["tipo"]].analisar(titulo, texto)
                    self.assertEqual(site["rotulo"], local["rotulo_previsto"])
                    if local["tipo"] == "svm":
                        self.assertAlmostEqual(site["margem_svm"], local["margem_svm"], places=10)
                    else:
                        for rotulo, prob in local["probabilidades_modelo"].items():
                            self.assertAlmostEqual(site["probabilidades"][rotulo], prob, places=6)
                            self.assertNotIn(formatar_probabilidade(prob), ("100%", "100,00%"))
                        self.assertLessEqual(site["tokens_lidos"], local["limite_tokens"])
                        self.assertEqual(site["truncado"], site["tokens_totais"] > local["limite_tokens"])
                    registros.append({"caso": identificador, "modelo": local["tipo"],
                                      "janela": local["rotulo_previsto"], "site": site["rotulo"],
                                      "escores": site.get("probabilidades", site.get("margem_svm"))})
        destino = RAIZ / "resultados/relatorios/paridade_streamlit.json"
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(json.dumps(registros, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
