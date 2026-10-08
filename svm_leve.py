"""Inferência TF-IDF + SVM calibrado, sem dependência de PyTorch ou Transformers."""

import json
from pathlib import Path

import joblib
import numpy as np

from preparacao_noticias import preparar_noticia, VERSAO_PREPARACAO
from triagem_noticias import avaliar_entrada, INDICIOS


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
        self.dominio = joblib.load(self.pasta / "dominio.joblib")
        if list(self.modelo.classes_) != [0, 1] or list(self.calibrador.classes_) != [0, 1]:
            raise ValueError("Ordem de classes inválida para o modelo binário.")
        if self.politica["rotulos"] != ["fake", "true"]:
            raise ValueError("Rótulos incompatíveis.")

    def escores(self, entradas):
        margens = self.modelo.decision_function(entradas)
        return self.calibrador.predict_proba(np.asarray(margens).reshape(-1, 1))

    def analisar_noticia(self, titulo, texto):
        motivo = avaliar_entrada(titulo, texto)
        if motivo:
            return {"rotulo": "inconclusivo", "mensagem": INDICIOS["inconclusivo"], "motivo": motivo}
        entrada = preparar_noticia(titulo, texto)
        if not self.dominio.aceitar([entrada])[0]:
            return {"rotulo": "inconclusivo", "mensagem": INDICIOS["inconclusivo"],
                    "motivo": "O vocabulário do texto ou de um trecho tem pouca relação com as notícias do treinamento."}
        probs = self.escores([entrada])[0]
        indice = int(probs.argmax())
        aceito = bool(probs[indice] >= self.politica["limiares"][indice])
        rotulo = self.politica["rotulos"][indice] if aceito else "inconclusivo"
        return {"rotulo": rotulo, "mensagem": INDICIOS[rotulo], "calibrado": True,
                "probabilidades": dict(zip(self.politica["rotulos"], map(float, probs))),
                "truncado": False,
                "motivo": "Padrões do texto; os fatos não foram verificados." if aceito else
                          "O escore não atingiu o limiar definido na seleção. Confira outras fontes."}
