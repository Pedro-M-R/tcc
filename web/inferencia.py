"""Inferência do site, somente com o BERTimbau final treinado."""

import os
import unicodedata
from pathlib import Path
from threading import Lock

os.environ.setdefault("USE_TF", "0")

PASTA_PROJETO = Path(__file__).resolve().parents[1]


def caminho_modelo():
    configurado = os.environ.get("BERTIMBAU_MODEL_PATH")
    caminho = Path(configurado) if configurado else Path("modelos/bertimbau_128_es")
    return caminho if caminho.is_absolute() else PASTA_PROJETO / caminho


def preparar_entrada(titulo, texto):
    # Mesma normalização e ordem de campos usadas no treinamento.
    return " ".join(unicodedata.normalize("NFC", f"{titulo}\n{texto}").split())


class Classificador:
    def __init__(self, caminho):
        caminho = Path(caminho)
        for nome in ("config.json", "tokenizer_config.json", "model.safetensors"):
            if not (caminho / nome).is_file():
                raise FileNotFoundError(f"Arquivo do modelo ausente: {nome}")
        with (caminho / "model.safetensors").open("rb") as arquivo:
            if arquivo.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError("Os pesos são um ponteiro Git LFS. Execute git lfs pull.")

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # CPU evita dependência de GPU no servidor. Um lock limita inferências simultâneas.
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(caminho, local_files_only=True)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(
            caminho, local_files_only=True, use_safetensors=True,
        ).to("cpu")
        self.modelo.eval()
        if set(self.modelo.config.id2label.values()) != {"fake", "true"}:
            raise ValueError("O modelo precisa ter as classes fake e true do treinamento.")
        self.limite = min(self.tokenizer.model_max_length, self.modelo.config.max_position_embeddings)
        self.lock = Lock()

    def analisar(self, titulo, texto):
        import torch

        entrada = preparar_entrada(titulo, texto)
        if not entrada:
            raise ValueError("Preencha a notícia antes de analisar.")
        with self.lock, torch.inference_mode():
            tokens = self.tokenizer(entrada, truncation=False, verbose=False)["input_ids"]
            entradas = self.tokenizer(
                entrada, truncation=True, max_length=self.limite, return_tensors="pt",
            )
            probs = self.modelo(**entradas).logits.softmax(dim=-1)[0].tolist()
        indice = max(range(len(probs)), key=probs.__getitem__)
        return {
            "rotulo": self.modelo.config.id2label[indice],
            "probabilidades": {
                self.modelo.config.id2label[i]: probabilidade
                for i, probabilidade in enumerate(probs)
            },
            "truncado": len(tokens) > self.limite,
        }
