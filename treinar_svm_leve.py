"""Treina e avalia um SVM leve em CPU, com calibração e abstenção independentes."""

import argparse
import gc
import platform
import time
import warnings
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from dados_modelos import salvar_json
from preparacao_noticias import VERSAO_PREPARACAO
from treinamento_local.dados import carregar, dividir
from treinamento_local.dominio import FiltroDominio
from treinamento_local.calibracao import selecionar_limiares, avaliar, aceitos
from triagem_noticias import avaliar_entrada


def vetorizador(caracteres=False):
    palavras = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
                              max_features=60000, sublinear_tf=True,
                              strip_accents="unicode", dtype=np.float32)
    if not caracteres:
        return palavras
    letras = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                            max_features=40000, sublinear_tf=True,
                            strip_accents="unicode", dtype=np.float32)
    return FeatureUnion([("palavras", palavras), ("caracteres", letras)],
                        transformer_weights={"palavras": 1.0, "caracteres": .5}, n_jobs=1)


def elegiveis(parte):
    return np.array([avaliar_entrada(t, x) is None for t, x in zip(parte.titulo, parte.texto)])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", nargs="+", default=["resultados/bases/base_atualizada_2026-10-06.csv"])
    p.add_argument("--saida", type=Path, default=Path("modelos/svm_leve"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alvo-precisao", type=float, default=.9)
    p.add_argument("--minimo-selecao", type=int, default=30)
    a = p.parse_args(argv)
    if not .5 <= a.alvo_precisao < 1 or a.minimo_selecao < 1:
        p.error("Alvo deve estar entre 0.5 e 1 (exclusive); mínimo deve ser positivo.")
    if a.saida.exists() and any(a.saida.iterdir()):
        p.error("Saída ocupada; escolha uma nova pasta para preservar o modelo anterior.")
    inicio = time.perf_counter()
    print("Preparando base e separação por grupos…", flush=True)
    df, rotulos, auditoria = carregar(a.csv, "noticia")
    if rotulos != ["fake", "true"]:
        p.error("Este SVM exige classes fake e true; a abstenção é uma política separada.")
    partes, divisao = dividir(df, a.seed)
    a.saida.mkdir(parents=True, exist_ok=True)
    salvar_json(a.saida / "manifesto.json", {"modelo": "TF-IDF + LinearSVC calibrado",
                "dados": auditoria, "divisao": divisao,
                "argumentos": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}})
    print({k: len(v) for k, v in partes.items()}, flush=True)
    treino, validacao = partes["treino"], partes["validacao"]
    melhor_f1, melhor, escolha, candidatos = -1.0, None, None, []
    for caracteres in (False, True):
        nome = "palavras+caracteres" if caracteres else "palavras"
        print(f"TF-IDF: {nome}", flush=True)
        vetor = vetorizador(caracteres)
        x = vetor.fit_transform(treino.entrada_a)
        xv = vetor.transform(validacao.entrada_a)
        for c in (.25, 1., 4.):
            svm = LinearSVC(C=c, class_weight="balanced", dual="auto", max_iter=20000, random_state=a.seed)
            with warnings.catch_warnings(record=True) as avisos:
                warnings.simplefilter("always", ConvergenceWarning)
                svm.fit(x, treino.label)
            convergiu = not any(issubclass(w.category, ConvergenceWarning) for w in avisos)
            f1 = float(f1_score(validacao.label, svm.predict(xv), average="macro"))
            candidato = {"atributos": nome, "C": c, "f1_macro_validacao": f1,
                         "convergiu": convergiu, "dimensoes": x.shape[1]}
            candidatos.append(candidato)
            print(candidato, flush=True)
            if convergiu and f1 > melhor_f1:
                melhor_f1, escolha = f1, candidato
                melhor = Pipeline([("tfidf", vetor), ("svm", svm)])
        del x, xv
        gc.collect()
    if melhor is None:
        raise RuntimeError("Nenhum candidato convergiu; ajuste o treino antes de publicar.")
    salvar_json(a.saida / "selecao_modelo.json", {"melhor": escolha, "candidatos": candidatos})
    joblib.dump(melhor, a.saida / "modelo.joblib", compress=3)

    print("Calibrando os escores em amostra separada…", flush=True)
    cal = partes["calibracao"]
    calibrador = LogisticRegression(C=1.0, max_iter=2000, random_state=a.seed)
    calibrador.fit(melhor.decision_function(cal.entrada_a).reshape(-1, 1), cal.label)
    joblib.dump(calibrador, a.saida / "calibrador.joblib", compress=3)

    print("Ajustando filtro de domínio com treino e validação…", flush=True)
    # O piso histórico de 0.15 rejeitava notícias que a própria validação
    # considerava familiares. Para o SVM, aprenda o limite na validação.
    entradas_validacao = validacao.loc[elegiveis(validacao), "entrada_a"]
    if entradas_validacao.empty:
        raise ValueError("Sem notícias elegíveis na validação para ajustar o filtro de domínio.")
    dominio = FiltroDominio(piso_similaridade=0).ajustar(
        treino.loc[elegiveis(treino), "entrada_a"], entradas_validacao)
    joblib.dump(dominio, a.saida / "dominio.joblib", compress=3)
    salvar_json(a.saida / "dominio.json", dominio.descrever())

    def prever(parte):
        margens = melhor.decision_function(parte.entrada_a)
        return calibrador.predict_proba(margens.reshape(-1, 1))

    def mascara(parte):
        return elegiveis(parte) & dominio.aceitar(parte.entrada_a)

    print("Selecionando limiares em partição própria…", flush=True)
    selecao = partes["selecao"]
    limiares, detalhes = selecionar_limiares(prever(selecao), selecao.label, mascara(selecao),
                                            a.alvo_precisao, a.minimo_selecao)
    salvar_json(a.saida / "politica_decisao.json", {
        "preparacao": VERSAO_PREPARACAO, "rotulos": rotulos, "limiares": limiares,
        "calibracao": "Regressão logística sobre a margem do SVM; somente partição de calibração",
        "dominio": dominio.descrever(), "alvo_selecao": a.alvo_precisao,
        "minimo_selecao": a.minimo_selecao, "detalhes_selecao": detalhes,
        "nota": "Escores calibrados na amostra; não são probabilidades de verdade factual."})
    # O teste só é avaliado com a configuração e a política congeladas.
    print("Avaliando o teste reservado…", flush=True)
    teste = partes["teste"]
    probs, elegivel = prever(teste), mascara(teste)
    metricas = avaliar(probs, teste.label, limiares, elegivel, rotulos)
    metricas["por_classe_sem_abstencao"] = classification_report(teste.label, probs.argmax(axis=1),
        labels=[0, 1], target_names=rotulos, output_dict=True, zero_division=0)
    salvar_json(a.saida / "metricas_teste.json", metricas)
    hosts = teste.url.map(lambda u: urlsplit(u).hostname or "sem_url").to_numpy()
    fontes = {}
    for host in sorted(set(hosts)):
        pos = hosts == host
        fontes[host] = avaliar(probs[pos], teste.label.to_numpy()[pos], limiares, elegivel[pos], rotulos)
    salvar_json(a.saida / "metricas_teste_por_fonte.json", fontes)
    relatorio = teste[["origem_csv", "linha_csv", "rotulo", "grupo_split"]].copy()
    previstos = [rotulos[i] for i in probs.argmax(axis=1)]
    relatorio["previsto_bruto"] = previstos
    relatorio["resultado"] = [r if ok else "inconclusivo" for r, ok in zip(previstos, aceitos(probs, limiares, elegivel))]
    relatorio.to_csv(a.saida / "predicoes_teste.csv", index=False, encoding="utf-8-sig")
    salvar_json(a.saida / "ambiente.json", {"python": platform.python_version(),
        "bibliotecas": {n: version(n) for n in ("scikit-learn", "numpy", "scipy", "pandas", "joblib")},
        "duracao_segundos": time.perf_counter() - inicio})
    salvar_json(a.saida / "concluido.json", {"tarefa": "noticia", "tipo": "svm_leve", "pesos_e_politica_prontos": True})
    print("Concluído:", a.saida, flush=True)
    print(metricas, flush=True)


if __name__ == "__main__":
    main()
