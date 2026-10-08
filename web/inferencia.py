"""Inferência leve do site; adaptadores históricos exigem um tipo explícito."""

import os
import sys
from pathlib import Path
from threading import Lock

PASTA_PROJETO = Path(__file__).resolve().parents[1]
if str(PASTA_PROJETO) not in sys.path:
    sys.path.insert(0, str(PASTA_PROJETO))

from dados_modelos import juntar_texto as preparar_entrada
from svm_leve import ModeloSVMLeve
from preparacao_noticias import VERSAO_PREPARACAO
import json


def caminho_modelo():
    from prever import selecionar_bertimbau

    configurado = os.environ.get("BERTIMBAU_MODEL_PATH")
    if configurado:
        caminho = Path(configurado)
        return caminho if caminho.is_absolute() else PASTA_PROJETO / caminho
    return selecionar_bertimbau(PASTA_PROJETO / "modelos") or PASTA_PROJETO / "modelos/bertimbau_128_es"


def modelos_do_site():
    # BERTimbau fica fora do site mesmo se houver pesos ou uma variável antiga.
    caminho = Path(os.environ.get("TCC_SVM_MODEL_PATH", "modelos/svm_leve"))
    if not caminho.is_absolute():
        caminho = PASTA_PROJETO / caminho
    return [{"tipo": "svm_leve", "nome": "TF-IDF + SVM", "caminho": str(caminho.resolve()),
             "versao": versao_modelo(caminho)}]


def versao_modelo(caminho):
    """Invalida o cache quando pesos ou política são atualizados na mesma pasta."""
    versao = []
    for nome in ("modelo.joblib", "calibrador.joblib", "dominio.joblib",
                 "politica_decisao.json", "concluido.json"):
        try:
            estado = (Path(caminho) / nome).stat()
            versao.append((nome, estado.st_mtime_ns, estado.st_size))
        except FileNotFoundError:
            versao.append((nome, None, None))
    return tuple(versao)


class Classificador:
    def __init__(self, caminho, tipo="svm_leve"):
        self.caminho = str(Path(caminho).resolve())
        self.tipo = tipo
        self.lock = Lock()
        if tipo == "svm_leve":
            self.leve = ModeloSVMLeve(self.caminho)
        elif tipo == "local":
            from treinamento_local.inferencia import ModeloLocal

            politica = json.loads((Path(self.caminho) / "politica_decisao.json").read_text(encoding="utf-8"))
            if politica.get("preparacao") != VERSAO_PREPARACAO or not politica.get("dominio"):
                raise ValueError("Este site exige um novo experimento com palavras ignoradas e filtro de domínio.")
            self.local = ModeloLocal(self.caminho, "noticia", cpu=True)
        elif tipo == "svm":
            from prever import carregar_svm

            carregar_svm(self.caminho)
        elif tipo == "bertimbau":
            from prever import carregar_bertimbau

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
            if self.tipo == "svm_leve":
                return self.leve.analisar_noticia(titulo, texto)
            if self.tipo == "local":
                resultado = self.local.analisar_noticia(titulo, texto)
                detalhes = resultado.get("detalhes_tecnicos", {})
                if detalhes:
                    resultado["probabilidades"] = detalhes["escores_calibrados_na_amostra"]
                    resultado["truncado"] = detalhes["truncado"]
                    resultado["calibrado"] = True
                return resultado
            from prever import prever_noticias

            resultado = prever_noticias([titulo], [texto], self.caminho, self.tipo,
                                       batch_size=1, cpu=True, detalhes=True)[0]
        resultado["rotulo"] = resultado["rotulo_previsto"]
        if "probabilidades_modelo" in resultado:
            resultado["probabilidades"] = resultado["probabilidades_modelo"]
        return resultado
