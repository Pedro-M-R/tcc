"""Previsão e relatórios legíveis usando somente artefatos já treinados."""

import json
from datetime import datetime
from pathlib import Path

from dados_modelos import juntar_texto
from prever import prever_bertimbau, prever_svm, selecionar_bertimbau
from pontuacoes import formatar_probabilidade


RAIZ = Path(__file__).resolve().parent
NOMES = {"fake": "FALSA", "true": "VERDADEIRA"}


def ler_json(caminho):
    return json.loads(Path(caminho).read_text(encoding="utf-8"))


def modelos_disponiveis(pasta=None):
    pasta = Path(pasta or RAIZ / "modelos")
    modelos = []
    if (pasta / "svm" / "modelo.joblib").is_file():
        modelos.append({"tipo": "svm", "nome": "TF-IDF + SVM linear", "caminho": str((pasta / "svm").resolve())})
    escolhido = selecionar_bertimbau(pasta)
    if escolhido:
        modelos.append({"tipo": "bertimbau", "nome": "BERTimbau Base", "caminho": str(escolhido.resolve())})
    return modelos


def analisar_noticia(titulo, texto, modelos):
    entrada = juntar_texto(titulo, texto)
    if not entrada:
        raise ValueError("Cole o texto da notícia ou informe um título antes de analisar.")
    if not modelos:
        raise ValueError("Nenhum modelo treinado foi encontrado na pasta modelos.")
    resultado = {"data": datetime.now().astimezone().isoformat(timespec="seconds"),
                 "titulo": titulo.strip(), "texto": texto.strip(), "resultados": []}
    for modelo in modelos:
        if modelo["tipo"] == "svm":
            predicao = prever_svm([entrada], modelo["caminho"])[0]
        else:
            predicao = prever_bertimbau([entrada], modelo["caminho"], batch_size=1, cpu=False)[0]
        item = {**modelo, **predicao}
        if modelo["tipo"] == "bertimbau":
            item["limite_tokens"] = ler_json(Path(modelo["caminho"]) / "configuracao_treino.json")["max_length"]
        resultado["resultados"].append(item)
    return resultado


def porcentagem(valor):
    return f"{valor * 100:.2f}%".replace(".", ",")


def formatar_analise(resultado, incluir_noticia=True):
    linhas = ["ANÁLISE DE NOTÍCIA", "=" * 64, f"Data: {resultado['data']}",
              f"Título: {resultado['titulo'] or '(não informado)'}", ""]
    for item in resultado["resultados"]:
        linhas.extend([item["nome"], "-" * 64,
                       f"Classificação prevista: {NOMES[item['rotulo_previsto']]}"])
        if "probabilidades_modelo" in item:
            probs = item["probabilidades_modelo"]
            linhas.extend([f"Escore para falsa:      {formatar_probabilidade(probs['fake'])}",
                           f"Escore para verdadeira: {formatar_probabilidade(probs['true'])}",
                           "Escores softmax não calibrados; não medem a certeza sobre os fatos.",
                           f"Leitura: até {item['limite_tokens']} tokens do início de título + texto."])
        else:
            linhas.extend([f"Margem de decisão: {item['margem_svm']:+.4f}",
                           "Margem positiva favorece verdadeira; negativa favorece falsa.",
                           "A margem não é uma probabilidade ou porcentagem."])
        linhas.extend([f"Modelo utilizado: {Path(item['caminho']).name}", ""])
    previsoes = {item["rotulo_previsto"] for item in resultado["resultados"]}
    if len(resultado["resultados"]) > 1:
        linhas.extend(["Comparação: " + ("os modelos concordam." if len(previsoes) == 1 else "os modelos discordam."), ""])
    linhas.extend(["COMO INTERPRETAR", "A classificação representa padrões aprendidos na base de notícias.",
                   "Sem um rótulo verdadeiro conhecido, não é possível afirmar se esta previsão acertou.",
                   "O resultado não substitui a checagem dos fatos."])
    if incluir_noticia:
        linhas.extend(["", "TEXTO ANALISADO", "-" * 64, resultado["texto"] or "(somente título)"])
    return "\n".join(linhas)


def relatorio_desempenho(modelos):
    linhas = ["AVALIAÇÃO NO CONJUNTO DE TESTE", "=" * 72,
              "Resultados já salvos ao fim dos treinamentos.",
              "Esta tela não treina os modelos nem repete a avaliação.", ""]
    registros = []
    for modelo in modelos:
        pasta = Path(modelo["caminho"])
        if not (pasta / "metricas_teste.json").is_file():
            linhas.extend([f"{modelo['nome']}: métricas de teste não encontradas.", ""])
            continue
        metrica = ler_json(pasta / "metricas_teste.json")
        config_arquivo = pasta / ("configuracao_treino.json" if modelo["tipo"] == "bertimbau" else "configuracao.json")
        config = ler_json(config_arquivo) if config_arquivo.is_file() else {}
        registros.append((modelo, metrica, config))
    linhas.append(f"{'Modelo':<25} {'Acurácia':>10} {'F1 macro':>10} {'Acertos':>10} {'Erros':>8}")
    for modelo, m, _ in registros:
        cm = m["matriz_confusao"]
        total = sum(sum(linha) for linha in cm)
        acertos = sum(cm[i][i] for i in range(len(cm)))
        linhas.append(f"{modelo['nome']:<25} {porcentagem(m['acuracia']):>10} {m['f1_macro']:>10.4f} {acertos:>10} {total-acertos:>8}")
    if len(registros) > 1:
        divisoes = [c.get("divisao") for _, _, c in registros]
        mesma = all(d and d.get("sha256_csv") == divisoes[0].get("sha256_csv")
                    and d.get("indices") == divisoes[0].get("indices") for d in divisoes) if divisoes[0] else False
        linhas.extend(["", "Comparação: " + ("os modelos usaram a mesma base e os mesmos conjuntos."
                                           if mesma else "não foi possível confirmar divisões idênticas; compare com cautela.")])
    for modelo, m, config in registros:
        ordem = m["ordem_classes"]
        cm = m["matriz_confusao"]
        total = sum(sum(linha) for linha in cm)
        linhas.extend(["", modelo["nome"].upper(), "-" * 72, f"Notícias no teste: {total}",
                       "Matriz de confusão: linhas = rótulo real; colunas = previsão.",
                       f"{'':<22}" + "".join(f"{NOMES[r]:>16}" for r in ordem)])
        for rotulo, linha in zip(ordem, cm):
            linhas.append(f"{'Real: ' + NOMES[rotulo]:<22}" + "".join(f"{valor:>16}" for valor in linha))
        linhas.extend(["", f"{'Classe':<15} {'Precisão':>12} {'Recall':>12} {'F1':>12} {'Notícias':>10}"])
        for rotulo in ordem:
            c = m["por_classe"][rotulo]
            linhas.append(f"{NOMES[rotulo]:<15} {porcentagem(c['precision']):>12} {porcentagem(c['recall']):>12} {c['f1-score']:>12.4f} {int(c['support']):>10}")
        divisao = config.get("divisao", {}).get("indices", {})
        if divisao:
            linhas.append("Divisão: " + "; ".join(f"{nome} = {len(divisao[nome])}" for nome in ("treino", "validacao", "teste")))
        if modelo["tipo"] == "bertimbau":
            linhas.extend([f"Épocas executadas: {config.get('epocas_executadas', 'não registrado')}; limite: {config.get('epocas', 'não registrado')}",
                           f"Limite de tokens: {config.get('max_length', 'não registrado')}; dropout do classificador: {config.get('classifier_dropout', 'não registrado')}"])
            historico = Path(modelo["caminho"]) / "historico_treino.json"
            if historico.is_file() and config.get("melhor_checkpoint"):
                passo = int(Path(config["melhor_checkpoint"].replace("\\", "/")).name.split("-")[-1])
                melhor = next((r for r in ler_json(historico) if r.get("step") == passo and "eval_f1_macro" in r), None)
                if melhor:
                    linhas.append(f"Checkpoint escolhido pela validação: época {melhor['epoch']:g} (F1 macro = {melhor['eval_f1_macro']:.4f}).")
        linhas.append(f"Fonte: {Path(modelo['caminho']) / 'metricas_teste.json'}")
    linhas.extend(["", "GUIA PARA EXPLICAR", "-" * 72,
                   "Acurácia: proporção de notícias classificadas corretamente no teste.",
                   "Precisão de uma classe: entre as previsões dessa classe, quantas acertaram.",
                   "Recall de uma classe: entre as notícias reais dessa classe, quantas foram identificadas.",
                   "F1: combinação de precisão e recall; F1 macro dá o mesmo peso às duas classes.",
                   "Na matriz, a diagonal contém os acertos; fora da diagonal estão os erros.",
                   "As métricas descrevem este conjunto de teste e não garantem acertos em notícias novas."])
    return "\n".join(linhas)
