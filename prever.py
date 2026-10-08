"""Classifica novas notícias com os modelos treinados localmente."""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from dados_modelos import ROTULOS, juntar_texto
from triagem_noticias import avaliar_entrada, apresentar_predicao, resultado_inconclusivo


def prever_noticias(titulos, corpos, caminho, tipo, batch_size=8, cpu=True, detalhes=False):
    """Entrada pública com abstenção. Funções prever_svm/bertimbau são escores brutos."""
    titulos, corpos = list(titulos), list(corpos)
    if len(titulos) != len(corpos):
        raise ValueError("Cada texto precisa corresponder a um título.")
    resultados, indices, entradas = [], [], []
    for titulo, corpo in zip(titulos, corpos):
        motivo = avaliar_entrada(titulo, corpo)
        resultados.append(resultado_inconclusivo(motivo) if motivo else None)
        if not motivo:
            indices.append(len(resultados) - 1)
            entradas.append(juntar_texto(titulo, corpo))
    if entradas:
        if tipo == "svm":
            predicoes = prever_svm(entradas, caminho)
        elif tipo == "bertimbau":
            predicoes = prever_bertimbau(entradas, caminho, batch_size, cpu, detalhes)
        else:
            raise ValueError(f"Modelo desconhecido: {tipo}")
        for indice, predicao in zip(indices, predicoes):
            resultados[indice] = apresentar_predicao(predicao)
    return resultados


def selecionar_bertimbau(pasta):
    candidatos = [p for p in Path(pasta).glob("*") if p.is_dir() and (p / "config.json").is_file()
                  and (p / "configuracao_treino.json").is_file()
                  and ((p / "model.safetensors").is_file() or (p / "pytorch_model.bin").is_file())]
    return max(candidatos, key=lambda p: (p / "configuracao_treino.json").stat().st_mtime_ns) if candidatos else None


@lru_cache(maxsize=4)
def carregar_svm(caminho):
    import joblib

    return joblib.load(Path(caminho) / "modelo.joblib")


def prever_svm(textos, caminho):
    modelo = carregar_svm(str(Path(caminho).resolve()))
    previsto = modelo.predict(textos)
    margens = modelo.decision_function(textos)
    return [{"rotulo_previsto": ROTULOS[int(p)], "margem_svm": float(m)} for p, m in zip(previsto, margens)]


@lru_cache(maxsize=2)
def carregar_bertimbau(caminho, cpu):
    import os

    os.environ["USE_TF"] = "0"
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dispositivo = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(caminho, local_files_only=True)
    modelo = AutoModelForSequenceClassification.from_pretrained(caminho, local_files_only=True).to(dispositivo)
    modelo.eval()
    return tokenizer, modelo, dispositivo


def prever_bertimbau(textos, caminho, batch_size, cpu, detalhes=False):
    import torch

    tokenizer, modelo, dispositivo = carregar_bertimbau(str(Path(caminho).resolve()), cpu)
    resultados = []
    limite = min(tokenizer.model_max_length, modelo.config.max_position_embeddings, 512)
    with torch.inference_mode():
        for inicio in range(0, len(textos), batch_size):
            entradas = tokenizer(textos[inicio:inicio + batch_size], padding=True, truncation=True,
                                 max_length=limite, return_tensors="pt").to(dispositivo)
            probabilidades = modelo(**entradas).logits.softmax(dim=-1).cpu().tolist()
            for posicao, probs in enumerate(probabilidades):
                indice = max(range(len(probs)), key=probs.__getitem__)
                resultado = {"rotulo_previsto": modelo.config.id2label[indice],
                             "probabilidades_modelo": {modelo.config.id2label[i]: p for i, p in enumerate(probs)}}
                if detalhes:
                    total = len(tokenizer(textos[inicio + posicao], truncation=False, verbose=False)["input_ids"])
                    lidos = int(entradas["attention_mask"][posicao].sum().item())
                    resultado.update(truncado=total > limite, tokens_lidos=lidos, tokens_totais=total,
                                     trecho_lido=tokenizer.decode(entradas["input_ids"][posicao], skip_special_tokens=True))
                resultados.append(resultado)
    return resultados


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", choices=["svm", "bertimbau"], required=True)
    parser.add_argument("--caminho-modelo", help="Pasta do modelo; por padrão, SVM salvo ou BERTimbau final mais recente.")
    entrada = parser.add_mutually_exclusive_group(required=True)
    entrada.add_argument("--texto", help="Texto de uma notícia.")
    entrada.add_argument("--arquivo", help="Arquivo TXT em UTF-8.")
    entrada.add_argument("--csv", help="CSV em UTF-8 com coluna texto e, opcionalmente, titulo.")
    parser.add_argument("--titulo", default="", help="Título para --texto ou --arquivo.")
    parser.add_argument("--saida", help="Grava CSV quando a entrada é CSV; caso contrário grava JSON.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size deve ser positivo.")
    if args.csv and args.titulo:
        parser.error("Use a coluna titulo do CSV em vez de --titulo.")
    pasta = Path(__file__).resolve().parent / "modelos"
    padrao = (selecionar_bertimbau(pasta) or pasta / "bertimbau") if args.modelo == "bertimbau" else pasta / "svm"
    caminho = Path(args.caminho_modelo or padrao)
    esperado = "modelo.joblib" if args.modelo == "svm" else "config.json"
    if not (caminho / esperado).is_file():
        parser.error(f"Modelo treinado não encontrado em {caminho}. Execute treinar_{args.modelo}.py primeiro.")
    if args.csv:
        if args.saida and Path(args.saida).resolve() == Path(args.csv).resolve():
            parser.error("A saída deve ter nome diferente do CSV de entrada.")
        df = pd.read_csv(args.csv, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        if "texto" not in df:
            parser.error("O CSV precisa da coluna texto.")
        titulos = df["titulo"] if "titulo" in df else [""] * len(df)
        corpos = df.texto.tolist()
        textos = [juntar_texto(t, x) for t, x in zip(titulos, corpos)]
    else:
        texto = Path(args.arquivo).read_text(encoding="utf-8-sig") if args.arquivo else args.texto
        titulos, corpos = [args.titulo], [texto]
        textos = [juntar_texto(args.titulo, texto)]
    if not textos or any(not t for t in textos):
        parser.error("Há notícias vazias. Informe um título ou texto não vazio para cada registro.")
    resultados = prever_noticias(titulos, corpos, caminho, args.modelo, args.batch_size, args.cpu)
    if args.saida:
        destino = Path(args.saida)
        destino.parent.mkdir(parents=True, exist_ok=True)
        if args.csv:
            df["rotulo_previsto"] = [r["rotulo_previsto"] for r in resultados]
            df["analise"] = [r["mensagem"] for r in resultados]
            df["motivo"] = [r.get("motivo", "") for r in resultados]
            if args.modelo == "svm":
                df["margem_svm"] = [r.get("margem_svm") for r in resultados]
            else:
                for rotulo in ROTULOS.values():
                    df[f"probabilidade_{rotulo}"] = [r.get("probabilidades_modelo", {}).get(rotulo) for r in resultados]
            df.to_csv(destino, index=False, encoding="utf-8-sig")
        else:
            destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Resultado salvo em: {destino}")
    else:
        print(json.dumps(resultados, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
