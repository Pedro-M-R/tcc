"""Abstenção lexical: referência aprendida somente no treino, sem rótulos.

Similaridade mede familiaridade do vocabulário, nunca apoio factual. Não há
promessa de detectar toda entrada fora do domínio. Salvo junto de cada modelo.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

STOPWORDS = "a ao aos aquela aquele aqui as ate com como da das de dela dele do dos e ela ele em entre era essa esse esta este eu foi ha isso ja la mais mas me mesmo meu minha muito na nas nao no nos nossa nosso num numa o os ou para pela pelo por qual quando que quem se sem seu sua tambem tem um uma voce".split()


class FiltroDominio:
    def __init__(self, piso_similaridade=.15):
        if not 0 <= piso_similaridade <= 1:
            raise ValueError("Piso de similaridade deve estar entre zero e um.")
        self.piso_similaridade = float(piso_similaridade)
        self.vetor = TfidfVectorizer(strip_accents="unicode", lowercase=True, stop_words=STOPWORDS,
                                     token_pattern=r"(?u)\b[^\W\d_]{3,}\b", sublinear_tf=True,
                                     max_features=50000, ngram_range=(1, 1))
        self.limiar_similaridade = self.piso_similaridade
        self.limiar_cobertura = .35

    def ajustar(self, textos_treino, textos_validacao):
        textos_treino = list(textos_treino)
        if not textos_treino:
            raise ValueError("Sem notícias elegíveis para construir o filtro de domínio.")
        # Transposta em CSR uma única vez; evita converter toda a referência
        # a cada notícia na seleção, no teste e no site.
        self.referencia = self.vetor.fit_transform(textos_treino).T.tocsr()
        self.quantidade_treino = len(textos_treino)
        scores = self.medidas(list(textos_validacao))
        if len(scores):
            self.limiar_similaridade = max(self.piso_similaridade, float(np.quantile(scores[:, 0], .02)))
            self.limiar_cobertura = max(.35, float(np.quantile(scores[:, 1], .02)))
        return self

    def medidas(self, textos):
        analisar = self.vetor.build_analyzer()
        resultados = []
        for texto in textos:
            palavras = texto.split()
            # Verifica também o final para não aceitar um parágrafo noticioso
            # seguido de uma grande quantidade de texto alheio ao domínio.
            blocos = [" ".join(palavras[i:i+120]) for i in range(0, len(palavras), 120)] or [""]
            if len(blocos) > 1 and len(blocos[-1].split()) < 30:
                blocos[-2:] = [blocos[-2] + " " + blocos[-1]]
            matriz = self.vetor.transform(blocos)
            similaridades = (matriz @ self.referencia).max(axis=1).toarray().ravel()
            coberturas = []
            for bloco in blocos:
                termos = set(analisar(bloco))
                coberturas.append(sum(t in self.vetor.vocabulary_ for t in termos) / max(1, len(termos)))
            resultados.append([float(min(similaridades)), min(coberturas)])
        return np.asarray(resultados, dtype=float).reshape(-1, 2)

    def aceitar(self, textos):
        scores = self.medidas(textos)
        return (scores[:, 0] >= self.limiar_similaridade) & (scores[:, 1] >= self.limiar_cobertura)

    def descrever(self):
        return {"metodo": "TF-IDF por trechos; referência só do treino; percentil 2 da validação",
                "piso_similaridade": getattr(self, "piso_similaridade", .15),
                "piso_cobertura": .35,
                "referencias_treino": self.quantidade_treino,
                "limiar_similaridade": self.limiar_similaridade, "limiar_cobertura": self.limiar_cobertura,
                "nota": "Filtro de familiaridade lexical; não comprova fatos nem identifica todo texto inadequado."}
