"""Preparação compartilhada: os dois modelos usam exatamente as mesmas notícias."""

import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


ROTULOS = {0: "fake", 1: "true"}


def normalizar_texto(valor):
    return " ".join(unicodedata.normalize("NFC", str(valor)).split())


def juntar_texto(titulo="", texto=""):
    # Preserva acentos, pontuação e maiúsculas para o BERTimbau cased.
    return normalizar_texto(f"{titulo}\n{texto}")


def salvar_json(caminho, conteudo):
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(conteudo, ensure_ascii=False, indent=2), encoding="utf-8")


def carregar_base(caminho):
    df = pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    faltando = {"titulo", "texto", "rotulo"} - set(df.columns)
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {sorted(faltando)}")
    df["id_linha"] = np.arange(len(df))  # Índice do registro original, começando em zero.
    df["entrada"] = [juntar_texto(t, x) for t, x in zip(df.titulo, df.texto)]
    df["rotulo"] = df.rotulo.str.strip().str.lower()
    desconhecidos = set(df.rotulo) - set(ROTULOS.values())
    if desconhecidos:
        raise ValueError(f"Rótulos desconhecidos: {sorted(desconhecidos)}. Esperados: fake e true.")
    df = df.loc[df.entrada.ne("")].copy()
    df["chave"] = df.entrada.str.casefold()
    if (df.groupby("chave").rotulo.nunique() > 1).any():
        raise ValueError("Há notícias idênticas com rótulos diferentes. Corrija a base antes do treino.")
    df = df.drop_duplicates("chave").reset_index(drop=True)
    df["label"] = df.rotulo.map({v: k for k, v in ROTULOS.items()})
    if len(df.label.unique()) != 2 or df.label.value_counts().min() < 10:
        raise ValueError("A base precisa de pelo menos 10 notícias de cada classe para a divisão.")
    return df


def carregar_divisao(csv, arquivo="resultados/divisao.json", seed=42):
    csv = Path(csv)
    df = carregar_base(csv)
    assinatura = hashlib.sha256(csv.read_bytes()).hexdigest()
    arquivo = Path(arquivo)
    if arquivo.exists():
        manifesto = json.loads(arquivo.read_text(encoding="utf-8"))
        if manifesto["sha256_csv"] != assinatura or manifesto["seed"] != seed:
            raise ValueError("CSV ou seed diferente da divisão salva. Use outro caminho em --divisao.")
    else:
        # Quantidades inteiras mais próximas de 70/15/15; treino recebe o restante.
        n_teste = round(len(df) * 0.15)
        n_validacao = round(len(df) * 0.15)
        treino, restante = train_test_split(
            df.index.to_numpy(), test_size=n_teste + n_validacao,
            stratify=df.label, random_state=seed,
        )
        validacao, teste = train_test_split(
            restante, test_size=n_teste, stratify=df.loc[restante, "label"], random_state=seed,
        )
        manifesto = {
            "sha256_csv": assinatura, "seed": seed,
            "entrada": "titulo + texto; normalização NFC e espaços; duplicatas exatas removidas",
            "rotulos": {str(k): v for k, v in ROTULOS.items()},
            "indices": {k: df.loc[v, "id_linha"].tolist() for k, v in
                        (("treino", treino), ("validacao", validacao), ("teste", teste))},
        }
        salvar_json(arquivo, manifesto)
    indices = manifesto["indices"]
    todos = [i for parte in indices.values() for i in parte]
    if set(indices) != {"treino", "validacao", "teste"} or len(todos) != len(set(todos)) or set(todos) != set(df.id_linha):
        raise ValueError("Divisão inválida: registros ausentes, repetidos ou sobrepostos.")
    por_id = df.set_index("id_linha", drop=False)
    partes = {k: por_id.loc[v].reset_index(drop=True) for k, v in indices.items()}
    for nome, parte in partes.items():
        print(f"{nome}: {len(parte)} ({len(parte)/len(df):.2%}), classes: {parte.rotulo.value_counts().to_dict()}")
    return partes, manifesto


def metricas(y, previsto):
    return {
        "acuracia": float(accuracy_score(y, previsto)),
        "f1_macro": float(f1_score(y, previsto, average="macro", zero_division=0)),
        "ordem_classes": list(ROTULOS.values()),
        "matriz_confusao": confusion_matrix(y, previsto, labels=[0, 1]).tolist(),
        "por_classe": classification_report(y, previsto, labels=[0, 1],
                                            target_names=list(ROTULOS.values()), output_dict=True, zero_division=0),
    }


def salvar_avaliacao(pasta, nome, parte, previsto):
    resultado = metricas(parte.label, previsto)
    salvar_json(Path(pasta) / f"metricas_{nome}.json", resultado)
    saida = parte[["id_linha", "rotulo"]].copy()
    saida["previsto"] = [ROTULOS[int(p)] for p in previsto]
    saida.to_csv(Path(pasta) / f"predicoes_{nome}.csv", index=False, encoding="utf-8-sig")
    print(f"{nome}: acurácia={resultado['acuracia']:.4f}; F1 macro={resultado['f1_macro']:.4f}")
    return resultado
