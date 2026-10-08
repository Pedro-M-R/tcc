"""Dados revisados e cinco partições disjuntas por grupo, sem executar treino."""

import hashlib
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dados_modelos import juntar_texto, normalizar_texto
from preparacao_noticias import pistas_rotulo, preparar_noticia, VERSAO_PREPARACAO

ROTULOS_NOTICIA = ["fake", "true", "nao_verificavel"]
# Mantém a ordem semântica do mDeBERTa NLI original.
ROTULOS_EVIDENCIA = ["apoia", "insuficiente", "contradiz"]
PARTICOES = ("treino", "validacao", "calibracao", "selecao", "teste")


def chave(texto):
    return re.sub(r"\W+", " ", normalizar_texto(texto).casefold()).strip()


def url_canonica(valor):
    p = urlsplit(str(valor).strip())
    if not p.hostname:
        return ""
    # Query é preservada: em alguns portais, ela identifica a matéria.
    return urlunsplit(("https", p.netloc.lower(), p.path.rstrip("/"), p.query, ""))


def agrupar(df, tarefa):
    """Componentes conexos: mesmo grupo, URL, corpo, título ou afirmação não vazios."""
    pais = list(range(len(df)))

    def raiz(i):
        while pais[i] != i:
            pais[i] = pais[pais[i]]
            i = pais[i]
        return i

    vistos = {}
    for i, row in df.iterrows():
        chaves = [("manual", row.get("grupo", ""))]
        if tarefa == "noticia":
            chaves += [("url", url_canonica(row.get("url", ""))),
                       ("titulo", chave(row.titulo)), ("corpo", chave(row.texto))]
        else:
            chaves += [("alegacao", chave(row.alegacao)),
                       ("evidencia", chave(row.evidencia)),
                       ("url_evidencia", url_canonica(row.url_evidencia))]
        for tipo, valor in chaves:
            if str(valor).strip():
                k = (tipo, str(valor).strip())
                if k in vistos:
                    pais[raiz(i)] = raiz(vistos[k])
                else:
                    vistos[k] = i
    return [raiz(i) for i in range(len(df))]


def carregar(caminhos, tarefa):
    arquivos, partes = [], []
    for caminho in map(Path, caminhos):
        d = pd.read_csv(caminho, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        d["origem_csv"] = caminho.name
        d["linha_csv"] = np.arange(len(d)) + 2
        partes.append(d)
        arquivos.append({"arquivo": str(caminho.resolve()), "sha256": hashlib.sha256(caminho.read_bytes()).hexdigest()})
    df = pd.concat(partes, ignore_index=True).fillna("")
    obrigatorias = {"titulo", "texto", "rotulo"} if tarefa == "noticia" else {
        "alegacao", "evidencia", "rotulo", "url_evidencia", "revisado"}
    if obrigatorias - set(df):
        raise ValueError(f"Colunas ausentes: {sorted(obrigatorias - set(df))}")
    df.rotulo = df.rotulo.str.strip().str.lower()
    rotulos = (ROTULOS_NOTICIA if "nao_verificavel" in set(df.rotulo) else ROTULOS_NOTICIA[:2]) if tarefa == "noticia" else ROTULOS_EVIDENCIA
    if set(df.rotulo) != set(rotulos):
        raise ValueError(f"Exigidas as classes {rotulos}; encontrados {sorted(set(df.rotulo))}. Não rotule coleta automaticamente.")
    if "uso" in df and df.uso.str.lower().eq("exemplo").any():
        raise ValueError("Exemplos didáticos não são base de treino. Substitua por dados reais revisados.")
    if tarefa == "evidencia" and not df.revisado.str.lower().isin(["sim", "true", "1"]).all():
        raise ValueError("Todos os pares de evidência precisam de revisão humana (revisado=sim).")
    quantidade_original = len(df)
    alterados = []
    if tarefa == "noticia":
        pistas = [pistas_rotulo(t, x) for t, x in zip(df.titulo, df.texto)]
        alterados = [{"origem_csv": r.origem_csv, "linha_csv": int(r.linha_csv), "palavras_ignoradas": p}
                     for (_, r), p in zip(df.iterrows(), pistas) if p]
        df["entrada_a"] = [preparar_noticia(t, x) for t, x in zip(df.titulo, df.texto)]
        df["entrada_b"] = ""
    else:
        # Premissa primeiro, hipótese depois. Inverter muda a tarefa NLI.
        df["entrada_a"] = df.evidencia.map(normalizar_texto)
        df["entrada_b"] = df.alegacao.map(normalizar_texto)
        if df.entrada_b.eq("").any() or not df.url_evidencia.map(url_canonica).ne("").all():
            raise ValueError("Cada par exige alegação e URL de evidência.")
    if df.entrada_a.eq("").any():
        raise ValueError("Há entradas sem texto.")
    df["chave"] = [hashlib.sha256((chave(a) + "\n" + chave(b)).encode()).hexdigest()
                   for a, b in zip(df.entrada_a, df.entrada_b)]
    if (df.groupby("chave").rotulo.nunique() > 1).any():
        raise ValueError("Entradas idênticas com rótulos conflitantes; revise a base.")
    # Preserva todas as notícias. Entradas iguais após ignorar palavras ficam no
    # mesmo grupo, sem atravessar treino/validação/teste.
    df["grupo_split"] = agrupar(df, tarefa)
    if tarefa == "noticia":
        # Liga também componentes que só ficaram iguais após a preparação.
        pais = {g: g for g in df.grupo_split}
        def raiz(g):
            while pais[g] != g:
                pais[g] = pais[pais[g]]
                g = pais[g]
            return g
        vistos = {}
        for chave_entrada, grupo in zip(df.chave, df.grupo_split):
            if chave_entrada in vistos:
                pais[raiz(grupo)] = raiz(vistos[chave_entrada])
            else:
                vistos[chave_entrada] = grupo
        df["grupo_split"] = df.grupo_split.map(raiz)
    else:
        df = df.drop_duplicates("chave").reset_index(drop=True)
    df["label"] = df.rotulo.map({r: i for i, r in enumerate(rotulos)})
    return df, rotulos, {"arquivos": arquivos, "registros_originais": quantidade_original,
                        "preparacao": VERSAO_PREPARACAO if tarefa == "noticia" else "pares_revisados",
                        "registros_excluidos_por_pistas": 0,
                        "registros_com_palavras_ignoradas": len(alterados), "preparacoes": alterados,
                        "duplicatas_removidas": quantidade_original - len(df),
                        "registros_utilizados": len(df), "classes_utilizadas": df.rotulo.value_counts().to_dict()}


def dividir(df, seed=42):
    por_classe = df.groupby("rotulo").grupo_split.nunique()
    if (por_classe < 10).any():
        raise ValueError(f"São necessários pelo menos 10 grupos por classe para separar os cinco usos: {por_classe.to_dict()}")
    folds = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    indices = list(folds.split(df, df.label, groups=df.grupo_split))
    partes = {"treino": np.concatenate([indices[i][1] for i in range(6)]),
              **{nome: indices[i][1] for i, nome in enumerate(PARTICOES[1:], 6)}}
    conjuntos = {nome: df.iloc[idx].copy() for nome, idx in partes.items()}
    todas_classes = set(df.label)
    for nome, parte in conjuntos.items():
        if set(parte.label) != todas_classes:
            raise ValueError(f"A partição {nome} ficou sem alguma classe. Amplie os grupos da base.")
    manifesto = {"metodo": "StratifiedGroupKFold/10: 6 treino + 1 validação + 1 calibração + 1 seleção + 1 teste",
                 "seed": seed, "particoes": {nome: parte.chave.tolist() for nome, parte in conjuntos.items()},
                 "contagens": {nome: {"total": len(p), "classes": p.rotulo.value_counts().to_dict(),
                                       "grupos": int(p.grupo_split.nunique())} for nome, p in conjuntos.items()}}
    return conjuntos, manifesto
