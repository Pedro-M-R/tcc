"""Adaptador do site para a mesma inferência usada por testar_noticias.py."""

import os
import sys
from pathlib import Path
from threading import Lock

PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

from dados_modelos import juntar_texto as preparar_entrada
from prever import carregar_bertimbau, carregar_svm, prever_bertimbau, prever_svm, selecionar_bertimbau
from resultados_noticias import modelos_disponiveis


def caminho_modelo():
    configurado = os.environ.get("BERTIMBAU_MODEL_PATH")
    if configurado:
        caminho = Path(configurado)
        return caminho if caminho.is_absolute() else PASTA_PROJETO / caminho
    return selecionar_bertimbau(PASTA_PROJETO / "modelos") or PASTA_PROJETO / "modelos/bertimbau_128_es"


def modelos_do_site():
    modelos = modelos_disponiveis(PASTA_PROJETO / "modelos")
    # Compatibilidade com publicações antigas sem metadados de treino.
    bert = {"tipo": "bertimbau", "nome": "BERTimbau Base", "caminho": str(caminho_modelo().resolve())}
    return [m for m in modelos if m["tipo"] != "bertimbau"] + [bert]


class Classificador:
    def __init__(self, caminho, tipo="bertimbau"):
        self.caminho = str(Path(caminho).resolve())
        self.tipo = tipo
        self.lock = Lock()
        if tipo == "svm":
            carregar_svm(self.caminho)
        elif tipo == "bertimbau":
            caminho = Path(self.caminho)
            for nome in ("config.json", "tokenizer_config.json", "model.safetensors"):
                if not (caminho / nome).is_file():
                    raise FileNotFoundError(f"Arquivo do modelo ausente: {nome}")
            with (caminho / "model.safetensors").open("rb") as arquivo:
                if arquivo.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                    raise ValueError("Os pesos são um ponteiro Git LFS. Execute git lfs pull.")
            import torch

            torch.set_num_threads(2)
            _, modelo, _ = carregar_bertimbau(self.caminho, True)
            if set(modelo.config.id2label.values()) != {"fake", "true"}:
                raise ValueError("O modelo precisa ter as classes fake e true do treinamento.")
        else:
            raise ValueError(f"Tipo de modelo desconhecido: {tipo}")

    def analisar(self, titulo, texto):
        entrada = preparar_entrada(titulo, texto)
        if not entrada:
            raise ValueError("Preencha a notícia antes de analisar.")
        with self.lock:
            if self.tipo == "svm":
                resultado = prever_svm([entrada], self.caminho)[0]
            else:
                resultado = prever_bertimbau([entrada], self.caminho, batch_size=1, cpu=True, detalhes=True)[0]
        resultado["rotulo"] = resultado["rotulo_previsto"]
        if "probabilidades_modelo" in resultado:
            resultado["probabilidades"] = resultado["probabilidades_modelo"]
        return resultado
