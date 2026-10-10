"""Treina TF-IDF + SVM binário em CPU, com seleção e calibração separadas."""

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
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from dados_modelos import salvar_json
from preparacao_noticias import VERSAO_PREPARACAO
from treinamento_local.vazamento_svm import carregar_svm, dividir_svm, relatorio_atributos
from treinamento_local.limpeza import limpar_texto, VERSAO_LIMPEZA
from treinamento_local.calibracao import avaliar


CONFIGURACOES = [
    {"nome": "referencia", "palavras": 60000, "ordem_palavras": 2,
     "caracteres": 40000, "ordem_caracteres": (3, 5), "peso_caracteres": .5},
    {"nome": "amplo", "palavras": 150000, "ordem_palavras": 2,
     "caracteres": 150000, "ordem_caracteres": (3, 5), "peso_caracteres": 1.0},
    {"nome": "contexto", "palavras": 200000, "ordem_palavras": 3,
     "caracteres": 200000, "ordem_caracteres": (3, 6), "peso_caracteres": .5},
    {"nome": "caracteres", "palavras": 100000, "ordem_palavras": 2,
     "caracteres": 250000, "ordem_caracteres": (2, 5), "peso_caracteres": .75},
]


def vetorizador(config):
    preprocessor = limpar_texto if config.get("limpeza", True) else None
    palavras = TfidfVectorizer(ngram_range=(1, config["ordem_palavras"]), min_df=2, max_df=.98,
                              max_features=config["palavras"], sublinear_tf=True,
                              preprocessor=preprocessor, strip_accents="unicode", dtype=np.float32)
    letras = TfidfVectorizer(analyzer="char_wb", ngram_range=config["ordem_caracteres"], min_df=3,
                            max_features=config["caracteres"], sublinear_tf=True,
                            preprocessor=preprocessor, strip_accents="unicode", dtype=np.float32)
    return FeatureUnion([("palavras", palavras), ("caracteres", letras)],
                        transformer_weights={"palavras": 1.0, "caracteres": config["peso_caracteres"]}, n_jobs=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", nargs="+", default=["resultados/bases/base_atualizada_2026-10-06.csv"])
    p.add_argument("--saida", type=Path, default=Path("modelos/svm_leve"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--perfil", choices=("leve", "completo"), default="completo")
    p.add_argument("--sem-limpeza", action="store_true", help="Controle de ablação com texto original")
    p.add_argument("--separacao", choices=("grupo", "fonte"), default="grupo")
    p.add_argument("--validar-dados", action="store_true", help="Audita fontes e partições sem treinar")
    a = p.parse_args(argv)
    if a.saida.exists() and any(a.saida.iterdir()):
        p.error("Saída ocupada; escolha uma nova pasta para preservar o modelo anterior.")
    inicio = time.perf_counter()
    print("Preparando base e separação por grupos…", flush=True)
    df, rotulos, auditoria = carregar_svm(a.csv, a.separacao)
    if rotulos != ["fake", "true"]:
        p.error("Este SVM exige classes fake e true.")
    a.saida.mkdir(parents=True, exist_ok=True)
    salvar_json(a.saida / "auditoria_vazamento.json", auditoria)
    partes, divisao = dividir_svm(df, a.seed, a.separacao)
    salvar_json(a.saida / "manifesto.json", {"modelo": "TF-IDF + LinearSVC binário calibrado",
                "dados": auditoria, "divisao": divisao,
                "usos": {"busca": ["treino", "selecao"], "escolha": ["validacao"],
                         "ajuste_final": ["treino", "selecao", "validacao"],
                         "calibracao": ["calibracao"], "avaliacao_final": ["teste"]},
                "argumentos": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}})
    print({k: len(v) for k, v in partes.items()}, flush=True)
    if a.validar_dados:
        print("Dados validados; nenhum modelo treinado.", flush=True)
        return
    # A antiga partição de seleção de abstenção agora faz parte do treino.
    # Calibração e teste continuam isolados, com os mesmos grupos de antes.
    treino = pd.concat([partes["treino"], partes["selecao"]])
    validacao = partes["validacao"]
    melhor_f1, escolha, candidatos = -1.0, None, []
    configs = CONFIGURACOES if a.perfil == "completo" else CONFIGURACOES[:1]
    configs = [{**c, "limpeza": not a.sem_limpeza} for c in configs]
    valores_c = (.25, 1., 4., 10.) if a.perfil == "completo" else (.25, 1., 4.)
    pesos = ("balanced", None) if a.perfil == "completo" else ("balanced",)
    for config in configs:
        print(f"TF-IDF: {config['nome']}", flush=True)
        vetor = vetorizador(config)
        x = vetor.fit_transform(treino.entrada_a)
        xv = vetor.transform(validacao.entrada_a)
        for peso in pesos:
            for c in valores_c:
                svm = LinearSVC(C=c, class_weight=peso, dual="auto", max_iter=20000, random_state=a.seed)
                with warnings.catch_warnings(record=True) as avisos:
                    warnings.simplefilter("always", ConvergenceWarning)
                    svm.fit(x, treino.label)
                convergiu = not any(issubclass(w.category, ConvergenceWarning) for w in avisos)
                f1 = float(f1_score(validacao.label, svm.predict(xv), average="macro"))
                candidato = {"atributos": config["nome"], "config": config, "C": c,
                             "class_weight": peso, "f1_macro_validacao": f1,
                             "convergiu": convergiu, "dimensoes": x.shape[1]}
                candidatos.append(candidato)
                print(candidato, flush=True)
                if convergiu and f1 > melhor_f1:
                    melhor_f1, escolha = f1, candidato
        del x, xv, vetor, svm
        gc.collect()
    if escolha is None:
        raise RuntimeError("Nenhum candidato convergiu; ajuste o treino antes de publicar.")
    salvar_json(a.saida / "selecao_modelo.json", {"melhor": escolha, "candidatos": candidatos})
    print("Reajustando a configuração escolhida com treino e validação…", flush=True)
    final = pd.concat([treino, validacao])
    melhor = Pipeline([("tfidf", vetorizador(escolha["config"])),
                       ("svm", LinearSVC(C=escolha["C"], class_weight=escolha["class_weight"],
                                         dual="auto", max_iter=20000, random_state=a.seed))])
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always", ConvergenceWarning)
        melhor.fit(final.entrada_a, final.label)
    if any(issubclass(w.category, ConvergenceWarning) for w in avisos):
        raise RuntimeError("O ajuste final não convergiu; o modelo não será marcado como concluído.")
    joblib.dump(melhor, a.saida / "modelo.joblib", compress=3)
    relatorio_atributos(melhor, a.saida / "atributos_mais_influentes.json")

    print("Calibrando os escores em amostra separada…", flush=True)
    cal = partes["calibracao"]
    calibrador = LogisticRegression(C=1.0, max_iter=2000, random_state=a.seed)
    calibrador.fit(melhor.decision_function(cal.entrada_a).reshape(-1, 1), cal.label)
    joblib.dump(calibrador, a.saida / "calibrador.joblib", compress=3)

    def prever(parte):
        margens = melhor.decision_function(parte.entrada_a)
        return calibrador.predict_proba(margens.reshape(-1, 1))

    limiares = [.5, .5]
    salvar_json(a.saida / "politica_decisao.json", {
        "preparacao": VERSAO_PREPARACAO, "rotulos": rotulos, "limiares": limiares,
        "entrada_pipeline": "bruta", "limpeza": None if a.sem_limpeza else VERSAO_LIMPEZA,
        "calibracao": "Regressão logística sobre a margem do SVM; somente partição de calibração",
        "decisao": "binaria_sem_abstencao", "filtro_dominio": False,
        "regra_opiniao": "Frases pessoais de beleza inteiras: fake por convenção explícita, sem escore.",
        "nota": "Escores calibrados na amostra; não são probabilidades de verdade factual."})
    # O teste só é avaliado com a configuração e a política congeladas.
    print("Avaliando o teste reservado…", flush=True)
    teste = partes["teste"]
    probs, elegivel = prever(teste), np.ones(len(teste), dtype=bool)
    metricas = avaliar(probs, teste.label, limiares, elegivel, rotulos)
    metricas["escopo"] = "Classificador estatístico nas notícias; regra de opinião avaliada em testes separados."
    metricas["por_classe_sem_abstencao"] = classification_report(teste.label, probs.argmax(axis=1),
        labels=[0, 1], target_names=rotulos, output_dict=True, zero_division=0)
    salvar_json(a.saida / "metricas_teste.json", metricas)
    hosts = teste.get("url", pd.Series("", index=teste.index)).map(lambda u: urlsplit(u).hostname or "sem_url").to_numpy()
    fontes = {}
    for host in sorted(set(hosts)):
        pos = hosts == host
        fontes[host] = avaliar(probs[pos], teste.label.to_numpy()[pos], limiares, elegivel[pos], rotulos)
    salvar_json(a.saida / "metricas_teste_por_fonte.json", fontes)
    relatorio = teste[["origem_csv", "linha_csv", "rotulo", "grupo_split"]].copy()
    previstos = [rotulos[i] for i in probs.argmax(axis=1)]
    relatorio["previsto_bruto"] = previstos
    relatorio["resultado"] = previstos
    relatorio.to_csv(a.saida / "predicoes_teste.csv", index=False, encoding="utf-8-sig")
    salvar_json(a.saida / "ambiente.json", {"python": platform.python_version(),
        "bibliotecas": {n: version(n) for n in ("scikit-learn", "numpy", "scipy", "pandas", "joblib")},
        "duracao_segundos": time.perf_counter() - inicio})
    salvar_json(a.saida / "concluido.json", {"tarefa": "noticia", "tipo": "svm_leve", "pesos_e_politica_prontos": True})
    print("Concluído:", a.saida, flush=True)
    print(metricas, flush=True)


if __name__ == "__main__":
    main()
