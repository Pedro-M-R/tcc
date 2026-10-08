"""Formatação compartilhada dos escores, sem arredondar para certeza absoluta."""

import math


def formatar_probabilidade(probabilidade):
    if not math.isfinite(probabilidade) or not 0 <= probabilidade <= 1:
        raise ValueError("Probabilidade inválida retornada pelo modelo.")
    percentual = probabilidade * 100
    for casas in (2, 3, 4, 6):
        if 0 < round(percentual, casas) < 100:
            return f"{percentual:.{casas}f}%".replace(".", ",")
    return ">99,999999%" if probabilidade > .5 else "<0,000001%"
