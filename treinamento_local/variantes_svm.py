"""Entradas curtas geradas somente dentro de cada partição já separada."""
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from treinamento_local.limpeza import limpar_texto


def visoes(parte):
    return {
        "completo": parte.entrada_a.tolist(),
        "titulo": parte.titulo.tolist(),
        "inicio_40": [" ".join(str(t).split()[:40]) for t in parte.texto],
    }


def criar_variantes(parte, curtos=True, limpeza=True):
    """Peso total unitário por notícia, inclusive quando variantes coincidem.

    O rótulo é herdado do documento e pode perder contexto no recorte.
    Não use exemplos de teste para ajustar esses pesos ou escolher configurações.
    """
    textos, rotulos, pesos = [], [], []
    entradas = visoes(parte)
    for i, label in enumerate(parte.label):
        opcoes = [(entradas['completo'][i], .6 if curtos else 1.)]
        if curtos:
            opcoes += [(entradas['titulo'][i], .2), (entradas['inicio_40'][i], .2)]
        unicas = {}
        for texto, peso in opcoes:
            chave = limpar_texto(texto) if limpeza else " ".join(str(texto).casefold().split())
            if chave:
                if chave not in unicas:
                    unicas[chave] = [texto, 0.]
                unicas[chave][1] += peso
        if not unicas:
            raise ValueError('Notícia sem texto utilizável para gerar variantes.')
        total = sum(peso for _, peso in unicas.values())
        for texto, peso in unicas.values():
            textos.append(texto)
            rotulos.append(int(label))
            pesos.append(peso / total)
    return textos, np.asarray(rotulos), np.asarray(pesos)


def avaliar_entradas(modelo, calibrador, parte):
    relatorios = {}
    for nome, entradas in visoes(parte).items():
        previstos = calibrador.predict(modelo.decision_function(entradas).reshape(-1, 1))
        relatorios[nome] = {
            'total': len(parte),
            'f1_macro': float(f1_score(parte.label, previstos, average='macro', zero_division=0)),
            'matriz_confusao': confusion_matrix(parte.label, previstos, labels=[0, 1]).tolist(),
            'por_classe': classification_report(parte.label, previstos, labels=[0, 1],
                target_names=['fake', 'true'], output_dict=True, zero_division=0),
        }
    return relatorios
