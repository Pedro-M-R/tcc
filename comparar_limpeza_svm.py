"""Executa ablação pareada em pastas novas, sem promover modelos para o site."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from dados_modelos import salvar_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", nargs="+", required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--treinador", type=Path, default=Path(__file__).with_name("treinar_svm_leve.py"))
    p.add_argument("--perfil", choices=("leve", "completo"), default="leve")
    p.add_argument("--separacao", choices=("grupo", "fonte"), default="grupo")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args(argv)
    if a.saida.exists() and any(a.saida.iterdir()):
        p.error("Saída ocupada; escolha outra pasta para preservar a comparação anterior.")
    a.saida.mkdir(parents=True, exist_ok=True)
    resultados, divisoes = {}, []
    for nome in ("com_limpeza", "sem_limpeza"):
        destino = a.saida / nome
        comando = [sys.executable, str(a.treinador.resolve()), "--csv", *[str(Path(c).resolve()) for c in a.csv],
                   "--saida", str(destino.resolve()), "--perfil", a.perfil, "--separacao", a.separacao,
                   "--seed", str(a.seed)]
        if nome == "sem_limpeza":
            comando.append("--sem-limpeza")
        print(f"Treinando {nome}; progresso em {a.saida / (nome + '.log')}", flush=True)
        with (a.saida / (nome + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(comando, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        def ler(arquivo):
            return json.loads((destino / arquivo).read_text(encoding="utf-8"))
        metricas = ler("metricas_teste.json")
        resultados[nome] = {"modelo": str(destino.resolve()),
                            "f1_macro": metricas["f1_macro_sem_abstencao"],
                            "acuracia": metricas["acuracia_sem_abstencao"]}
        divisoes.append(ler("manifesto.json")["divisao"])
    if divisoes[0] != divisoes[1]:
        raise RuntimeError("Ablação inválida: as partições diferem entre as execuções.")
    resumo = {"particoes_identicas": True, "separacao": a.separacao, "perfil": a.perfil,
              "resultados": resultados,
              "delta_f1_com_menos_sem": resultados["com_limpeza"]["f1_macro"] - resultados["sem_limpeza"]["f1_macro"],
              "nota": "Comparação do pacote de limpeza (termos, URLs e linhas editoriais) com texto bruto. "
                      "Mesmo protocolo de busca, calibração e teste; parâmetros podem diferir. "
                      "Queda de F1 não prova vazamento, nem estabilidade prova compreensão do conteúdo. "
                      "Teste interno reutilizado: resultados exploratórios; não ajuste regras olhando este teste."}
    salvar_json(a.saida / "comparacao.json", resumo)
    print(json.dumps(resumo, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
