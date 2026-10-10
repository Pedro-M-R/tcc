"""Dados brutos, grupos conservadores e diagnósticos de atalhos do SVM."""

import hashlib
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from dados_modelos import salvar_json
from treinamento_local.dados import carregar, dividir, PARTICOES
from treinamento_local.limpeza import limpar_texto, VERSAO_LIMPEZA


def fonte_url(url):
    host = (urlsplit(str(url)).hostname or "").lower()
    return host.removeprefix("www.")


def carregar_svm(caminhos, separacao="grupo"):
    # Mantém validações, rótulos e vínculos de duplicatas do carregador existente.
    df, rotulos, auditoria = carregar(caminhos, "noticia")
    df["entrada_a"] = df.titulo + "\n" + df.texto
    df["titulo_modelo"] = df.titulo
    limpos = df.entrada_a.map(limpar_texto)
    if limpos.eq("").any():
        raise ValueError("Há notícias vazias após a limpeza; revise as linhas "
                         + str(df.loc[limpos.eq(""), ["origem_csv", "linha_csv"]].to_dict("records")[:10]))
    df["chave"] = limpos.map(lambda t: hashlib.sha256(t.encode()).hexdigest())
    df["fonte_split"] = df.get("url", pd.Series("", index=df.index)).map(fonte_url)
    if separacao == "fonte" and df.fonte_split.eq("").any():
        raise ValueError("Separação por fonte exige URL com hostname em todas as notícias.")
    pais = {g: g for g in df.grupo_split}

    def raiz(g):
        while pais[g] != g:
            pais[g] = pais[pais[g]]
            g = pais[g]
        return g

    # Igualdade após limpeza também une grupos na ablação SEM limpeza.
    # Ao separar por fonte, duplicatas entre fontes unem essas fontes inteiras.
    colunas = ["chave"] + (["fonte_split"] if separacao == "fonte" else [])
    for coluna in colunas:
        vistos = {}
        for valor, grupo in zip(df[coluna], df.grupo_split):
            if valor in vistos:
                pais[raiz(grupo)] = raiz(vistos[valor])
            else:
                vistos[valor] = grupo
    df["grupo_split"] = df.grupo_split.map(raiz)
    tabela = pd.crosstab(df.fonte_split.replace("", "sem_url"), df.rotulo)
    auditoria.update({
        "entrada_svm": "texto original; preparação dentro do Pipeline",
        "limpeza_para_agrupar": VERSAO_LIMPEZA,
        "separacao": separacao,
        "por_fonte": tabela.to_dict(orient="index"),
        "fontes_com_uma_classe": tabela.index[(tabela > 0).sum(axis=1) == 1].tolist(),
        "grupos_por_classe": df.groupby("rotulo").grupo_split.nunique().to_dict(),
        "entradas_alteradas_pela_limpeza": int((df.entrada_a != limpos).sum()),
        "chaves_limpas_com_rotulos_conflitantes": int((df.groupby("chave").rotulo.nunique() > 1).sum()),
    })
    return df, rotulos, auditoria


def dividir_svm(df, seed=42, separacao="grupo"):
    if separacao == "grupo":
        partes, manifesto = dividir(df, seed)
    else:
        contagens = df.groupby("rotulo").grupo_split.nunique()
        if contagens.min() < 5:
            raise ValueError("Separação por fonte inviável: são necessários pelo menos cinco componentes "
                             "independentes de fontes/duplicatas por classe. Encontrados: " + str(contagens.to_dict()))
        n = min(10, int(df.grupo_split.nunique()))
        partes = None
        # Só usa presença de classes para viabilidade; não consulta escores de modelos.
        for tentativa in range(50):
            cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed + tentativa)
            indices = [iv for _, iv in cv.split(df, df.label, df.grupo_split)]
            candidatos = {"treino": df.iloc[np.concatenate(indices[:n-4])].copy(),
                          **{nome: df.iloc[iv].copy() for nome, iv in zip(PARTICOES[1:], indices[n-4:])}}
            if all(set(p.label) == set(df.label) for p in candidatos.values()):
                partes = candidatos
                break
        if partes is None:
            raise ValueError("Não foi possível manter ambas as classes nas cinco partições por fonte. Amplie a base.")
        manifesto = {"metodo": f"StratifiedGroupKFold/{n} por componentes de fontes e duplicatas",
                     "seed": seed, "seed_divisao": seed + tentativa,
                     "particoes": {nome: p.chave.tolist() for nome, p in partes.items()},
                     "contagens": {nome: {"total": len(p), "classes": p.rotulo.value_counts().to_dict(),
                                           "grupos": int(p.grupo_split.nunique())} for nome, p in partes.items()}}
    manifesto["separacao"] = separacao
    manifesto["fontes"] = {nome: sorted(set(p.fonte_split)) for nome, p in partes.items()}
    return partes, manifesto


def relatorio_atributos(modelo, caminho, quantidade=40):
    tfidf, svm = modelo.named_steps["tfidf"], modelo.named_steps["svm"]
    if list(svm.classes_) != [0, 1]:
        raise ValueError("Relatório exige ordem 0=fake, 1=true.")
    nomes, pesos = np.asarray(tfidf.get_feature_names_out()), svm.coef_[0]
    ordem = np.argsort(pesos)
    def itens(indices):
        return [{"atributo": str(nomes[i]), "peso": float(pesos[i])} for i in indices]
    rel = {"fake": itens([i for i in ordem if pesos[i] < 0][:quantidade]),
           "true": itens([i for i in ordem[::-1] if pesos[i] > 0][:quantidade]),
           "nota": "Associações lineares, não prova de vazamento ou causalidade. Revise usando treino/validação; "
                   "não ajuste a limpeza pelos resultados do teste."}
    salvar_json(caminho, rel)
    for classe in ("fake", "true"):
        print(f"Atributos mais associados a {classe}:", [i["atributo"] for i in rel[classe]], flush=True)
    return rel
