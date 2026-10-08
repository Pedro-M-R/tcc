"""Inferência TF-IDF + SVM calibrado, sem dependência de PyTorch ou Transformers."""

import json
import re
from pathlib import Path

import joblib
import numpy as np

from preparacao_noticias import preparar_noticia, simplificar, VERSAO_PREPARACAO
from triagem_noticias import INDICIOS


def opiniao_pessoal(titulo, texto):
    """Convenção explícita do aplicativo; não é previsão nem verificação factual.

    Só aceita a frase inteira, inclusive quando repetida no título e no corpo.
    Uma notícia que apenas menciona a frase continua sendo analisada pelo SVM.
    """
    entradas = {" ".join(simplificar(s).strip(" .!").split())
                for s in (titulo, texto) if str(s).strip()}
    if len(entradas) != 1:
        return False
    frase = entradas.pop()
    padrao = re.fullmatch(
        r"(?:o |a )?(?P<pessoa>[a-z]+(?: [a-z]+){0,3}) (?:e|eh) "
        r"(?:muito )?(?:bonito|bonita|lindo|linda)", frase)
    if not padrao:
        return False
    return not set(padrao['pessoa'].split()) & {
        'nao', 'nunca', 'ninguem', 'se', 'que', 'disse', 'segundo', 'quando', 'porque', 'e', 'eh'}


class ModeloSVMLeve:
    def __init__(self, pasta):
        self.pasta = Path(pasta)
        if not (self.pasta / "concluido.json").is_file():
            raise ValueError("O modelo TF-IDF + SVM ainda não foi concluído.")
        self.politica = json.loads((self.pasta / "politica_decisao.json").read_text(encoding="utf-8"))
        if self.politica.get("preparacao") != VERSAO_PREPARACAO:
            raise ValueError("Preparação do modelo incompatível com a aplicação.")
        self.modelo = joblib.load(self.pasta / "modelo.joblib")
        self.calibrador = joblib.load(self.pasta / "calibrador.joblib")
        if list(self.modelo.classes_) != [0, 1] or list(self.calibrador.classes_) != [0, 1]:
            raise ValueError("Ordem de classes inválida para o modelo binário.")
        if self.politica["rotulos"] != ["fake", "true"]:
            raise ValueError("Rótulos incompatíveis.")

    def escores(self, entradas):
        margens = self.modelo.decision_function(entradas)
        return self.calibrador.predict_proba(np.asarray(margens).reshape(-1, 1))

    def analisar_noticia(self, titulo, texto):
        if not (str(titulo).strip() or str(texto).strip()):
            raise ValueError("Digite um título ou texto para analisar.")
        if opiniao_pessoal(titulo, texto):
            return {"rotulo": "fake", "mensagem": "Falsa — regra para opinião pessoal",
                    "origem": "regra_opiniao",
                    "motivo": "Opinião pessoal classificada como falsa por uma regra do aplicativo. "
                              "Não é uma previsão do SVM nem uma verificação objetiva."}
        entrada = preparar_noticia(titulo, texto)
        probs = self.escores([entrada])[0]
        indice = int(probs.argmax())
        rotulo = self.politica["rotulos"][indice]
        return {"rotulo": rotulo, "mensagem": INDICIOS[rotulo], "calibrado": True,
                "origem": "svm",
                "probabilidades": dict(zip(self.politica["rotulos"], map(float, probs))),
                "truncado": False,
                "motivo": "Classificação binária pelos padrões do texto; os fatos não foram verificados."}
