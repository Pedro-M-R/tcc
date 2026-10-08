"""Calibração e abstenção. Escores não constituem probabilidade factual."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def probabilidades(logits, temperatura=1.0):
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or not np.isfinite(logits).all() or not np.isfinite(temperatura) or temperatura <= 0:
        raise ValueError("Logits ou temperatura inválidos")
    return softmax(logits / temperatura, axis=1)


def ajustar_temperatura(logits, y):
    logits, y = np.asarray(logits, float), np.asarray(y, int)
    probabilidades(logits)  # Validação antes de otimizar.

    def perda(log_t):
        z = logits / np.exp(log_t)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))

    resultado = minimize_scalar(perda, bounds=(-3, 4), method="bounded")
    if not resultado.success or perda(resultado.x) > perda(0):
        return 1.0
    return float(np.exp(resultado.x))


def limite_wilson(acertos, n, z=1.96):
    if not n:
        return 0.0
    p = acertos / n
    return float((p + z*z/(2*n) - z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


def selecionar_limiares(probs, y, elegivel, alvo=.9, minimo=30):
    """Escolhe no conjunto de seleção; teste nunca participa.

    Wilson por classe é critério de seleção, não garantia fora da amostra.
    1.1 bloqueia uma classe quando não há amostra/evidência suficiente.
    """
    probs, y, elegivel = np.asarray(probs), np.asarray(y), np.asarray(elegivel, bool)
    previstos, escores = probs.argmax(axis=1), probs.max(axis=1)
    limiares, detalhes = [], []
    for classe in range(probs.shape[1]):
        escolhido, detalhe = 1.1, {"aceitos": 0, "motivo": "Critério não atingido; abstenção nesta classe"}
        for limite in np.unique(np.r_[.5, escores[(previstos == classe) & elegivel & (escores >= .5)]]):
            mascara = (previstos == classe) & (escores >= limite) & elegivel
            n = int(mascara.sum())
            if n < minimo:
                continue
            acertos = int((y[mascara] == classe).sum())
            inferior = limite_wilson(acertos, n)
            if inferior >= alvo:
                escolhido = float(limite)
                detalhe = {"aceitos": n, "acertos": acertos, "limite_inferior_wilson": inferior}
                break
        limiares.append(escolhido)
        detalhes.append(detalhe)
    return limiares, detalhes


def aceitos(probs, limiares, elegivel):
    previstos = np.argmax(probs, axis=1)
    return (np.max(probs, axis=1) >= np.asarray(limiares)[previstos]) & np.asarray(elegivel, bool)


def avaliar(probs, y, limiares, elegivel, rotulos):
    y = np.asarray(y)
    previstos = probs.argmax(axis=1)
    mascara = aceitos(probs, limiares, elegivel)
    n = int(mascara.sum())
    return {"total": len(y), "acuracia_sem_abstencao": float(accuracy_score(y, previstos)),
            "f1_macro_sem_abstencao": float(f1_score(y, previstos, average="macro", zero_division=0)),
            "brier_multiclasse": float(np.mean(np.sum((probs - np.eye(len(rotulos))[y])**2, axis=1))),
            "aceitos": n, "abstencoes": len(y)-n, "cobertura": n/len(y),
            "acuracia_aceitos": float(accuracy_score(y[mascara], previstos[mascara])) if n else None,
            "ordem_classes": rotulos, "matriz_confusao_sem_abstencao": confusion_matrix(y, previstos, labels=range(len(rotulos))).tolist(),
            "nota": "Desempenho desta amostra; não garante factualidade, domínio novo ou desempenho futuro."}
