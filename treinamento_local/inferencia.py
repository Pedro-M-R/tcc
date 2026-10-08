"""Carrega explicitamente um experimento; não altera o modelo do site existente."""

import json
import os
from pathlib import Path

from treinamento_local.calibracao import probabilidades
from triagem_noticias import INDICIOS, avaliar_entrada
from dados_modelos import juntar_texto
from preparacao_noticias import preparar_noticia, VERSAO_PREPARACAO


class ModeloLocal:
    def __init__(self, caminho, tarefa, cpu=False):
        self.pasta = Path(caminho)
        manifesto = json.loads((self.pasta / "manifesto.json").read_text(encoding="utf-8"))
        if manifesto["tarefa"] != tarefa or not (self.pasta / "concluido.json").is_file():
            raise ValueError("Experimento não concluído ou tarefa incompatível")
        self.politica = json.loads((self.pasta / "politica_decisao.json").read_text(encoding="utf-8"))
        self.dominio = None
        if tarefa == "noticia" and self.politica.get("preparacao") == VERSAO_PREPARACAO:
            import joblib
            # Falha explícita se faltar o filtro; nunca ignora a política salva.
            self.dominio = joblib.load(self.pasta / "dominio.joblib")
        self.tarefa = tarefa
        self.max_length = manifesto["argumentos"]["max_length"]
        self._carregar(str(self.pasta), cpu, local=True)

    def _carregar(self, caminho, cpu, local):
        os.environ["USE_TF"] = "0"
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.dispositivo = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(caminho, local_files_only=local)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(caminho, local_files_only=local).to(self.dispositivo)
        self.modelo.eval()
        if [self.modelo.config.id2label[i] for i in range(self.modelo.config.num_labels)] != self.politica["rotulos"]:
            raise ValueError("Ordem dos rótulos difere da política de decisão")

    def escores(self, textos_a, textos_b=None):
        import torch
        if textos_b is not None:
            if any(len(t) > self.max_length // 2 for t in self.tokenizer(textos_b, truncation=False, verbose=False)["input_ids"]):
                raise ValueError("Alegação muito longa: divida em afirmações específicas, preservando contexto.")
        entradas = self.tokenizer(textos_a, text_pair=textos_b, truncation="only_first" if textos_b is not None else True,
                                  max_length=self.max_length, padding=True, return_tensors="pt").to(self.dispositivo)
        with torch.inference_mode():
            logits = self.modelo(**entradas).logits.float().cpu().numpy()
        return probabilidades(logits, self.politica["temperatura"])

    def analisar_noticia(self, titulo, texto):
        motivo = avaliar_entrada(titulo, texto)
        if motivo:
            return {"mensagem": INDICIOS["inconclusivo"], "motivo": motivo, "rotulo": "inconclusivo"}
        entrada = (preparar_noticia(titulo, texto) if self.politica.get("preparacao") == VERSAO_PREPARACAO
                   else juntar_texto(titulo, texto))
        if getattr(self, "dominio", None) is not None and not self.dominio.aceitar([entrada])[0]:
            return {"mensagem": INDICIOS["inconclusivo"], "rotulo": "inconclusivo",
                    "motivo": "O texto ou parte dele tem pouca relação com o material usado no treinamento. Não há suporte suficiente para classificá-lo."}
        probs = self.escores([entrada])[0]
        indice = int(probs.argmax())
        rotulo = self.politica["rotulos"][indice]
        if rotulo == "nao_verificavel" or probs[indice] < self.politica["limiares"][indice]:
            mensagem, resultado = INDICIOS["inconclusivo"], "inconclusivo"
            motivo = "O modelo reconheceu conteúdo não verificável ou o escore não atingiu o critério de aceitação."
        else:
            mensagem, resultado = INDICIOS[rotulo], rotulo
            motivo = "Indícios baseados em padrões do texto; os fatos não foram verificados."
        total = len(self.tokenizer(entrada, truncation=False, verbose=False)["input_ids"])
        return {"mensagem": mensagem, "rotulo": resultado, "motivo": motivo,
                "detalhes_tecnicos": {"escores_calibrados_na_amostra": dict(zip(self.politica["rotulos"], map(float, probs))),
                                       "tokens_totais": total, "limite_tokens": self.max_length,
                                       "truncado": total > self.max_length}}


class NLIPronto(ModeloLocal):
    """Comparador multilíngue pronto; não foi validado na base de notícias do usuário."""
    def __init__(self, cpu=False, limiar=.9):
        self.tarefa = "evidencia"
        self.max_length = 512
        self.politica = {"temperatura": 1., "limiares": [limiar]*3,
                        "rotulos": ["entailment", "neutral", "contradiction"]}
        self._carregar("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", cpu, local=False)
        self.politica["rotulos"] = ["apoia", "insuficiente", "contradiz"]
