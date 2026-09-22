"""Formatação das pontuações sem arredondá-las para certeza absoluta."""

import math


def formatar_probabilidade(probabilidade):
    if not math.isfinite(probabilidade) or not 0 <= probabilidade <= 1:
        raise ValueError("Probabilidade inválida retornada pelo modelo.")
    if probabilidade >= 0.9995:
        return ">99,9%"
    if probabilidade <= 0.0005:
        return "<0,1%"
    return f"{probabilidade * 100:.1f}%".replace(".", ",")
