"""Triagem conservadora de entrada; regras explícitas, não checagem de fatos."""

import re
import unicodedata

from preparacao_noticias import preparar_noticia


INDICIOS = {
    "true": "Há indícios de que seja verdadeira",
    "fake": "Há indícios de que não seja verdadeira",
    "inconclusivo": "Análise inconclusiva",
}
MIN_PALAVRAS = 40
MIN_DISTINTAS = 20


def simplificar(texto):
    return "".join(c for c in unicodedata.normalize("NFKD", texto.casefold())
                   if not unicodedata.combining(c))


def avaliar_entrada(titulo, texto):
    """Devolve motivo para abstenção, ou None; título repetido não aumenta contexto.

    Os limites são heurísticos e não foram calibrados como detector de domínio.
    Uma entrada aceita ainda pode ser opinião ou estar fora da base de treino.
    """
    corpo = preparar_noticia("", texto)
    titulo = preparar_noticia("", titulo)
    if simplificar(corpo).startswith(simplificar(titulo)) and titulo:
        corpo = corpo[len(titulo):].lstrip(" .:;-–—")
    palavras = re.findall(r"[^\W\d_]+", simplificar(corpo), re.UNICODE)
    entrada = simplificar(f"{titulo} {corpo}")
    opiniao = re.search(
        r"\b(?:e|eh|sao|parece|(?:acho|considero)(?:\s+\w+){0,3})\s+(?:(?:muito|mais|tao|super|o|a)\s+)*"
        r"(?:bonit[oa]s?|fei[oa]s?|lind[oa]s?|maravilhos[oa]s?|chato|chata)\b", entrada)
    if opiniao and len(palavras) < MIN_PALAVRAS:
        return "O conteúdo parece uma opinião ou avaliação subjetiva, como beleza pessoal. Envie uma notícia com fatos e contexto verificáveis."
    if len(palavras) < MIN_PALAVRAS:
        return "O texto tem pouco contexto para esta análise. Envie o corpo da notícia com pelo menos 40 palavras, incluindo o acontecimento, os envolvidos e a fonte. Um título isolado não basta."
    distintas = len(set(palavras))
    if distintas < MIN_DISTINTAS or (len(palavras) >= 80 and distintas / len(palavras) < .15):
        return "O conteúdo tem pouca diversidade de palavras ou muita repetição. Envie o texto completo de uma notícia."
    # Ser longo não transforma conversa, receita ou opinião em relato noticioso.
    # Heurística conservadora; não é um detector semântico nem prova factual.
    atribuicao = re.search(r"\b(?:segundo|de acordo com|afirmou|declarou|informou|disse|relatou)\b", entrada)
    pessoal = re.search(r"\b(?:eu acho|na minha opiniao|para mim|eu acredito|meu dia|minha rotina)\b", entrada)
    instrucao = re.search(r"\b(?:modo de preparo|misture os|preaqueca|pre aque[cç]a|adicione os ingredientes|"
                          r"ignore (?:as|todas)|responda apenas|classifique (?:como|este)|voce e um assistente)\b", entrada)
    if instrucao or (pessoal and not atribuicao) or (opiniao and not atribuicao):
        return "O conteúdo parece uma opinião, conversa ou instrução. Envie um relato de acontecimento com contexto verificável."
    acontecimento = re.search(r"\b(?:anunci\w*|aprov\w*|inaugur\w*|public\w*|registr\w*|decidi\w*|"
                              r"investig\w*|prend\w*|pres[oa]s?|morr\w*|morte\w*|ferid\w*|"
                              r"ocorre\w*|acontece\w*|realiz\w*|assin\w*|vot\w*|elei\w*|"
                              r"aument\w*|reduz\w*|caiu|subiu|divulg\w*|confirm\w*|"
                              r"receb\w*|entreg\w*|cancel\w*|suspend\w*|abr\w*|fech\w*|"
                              r"venceu|perdeu|empat\w*|conquist\w*|lanc\w*|detect\w*)\b", entrada)
    contexto = re.search(r"\b(?:segundo|de acordo com|prefeitura|governo|ministerio|policia|"
                         r"tribunal|secretaria|universidade|hospital|empresa|pesquisa|"
                         r"municipio|cidade|bairro|estado|congresso|camara|senado|"
                         r"feira|ontem|hoje|amanha|ano|mes|semana)\b|\b\d{4}\b", entrada)
    if not atribuicao and not (acontecimento and contexto):
        return "Não foi identificado contexto suficiente de um acontecimento noticioso. A análise ficará inconclusiva."
    return None


def resultado_inconclusivo(motivo):
    return {"rotulo_previsto": "inconclusivo", "status": "inconclusivo",
            "mensagem": INDICIOS["inconclusivo"], "motivo": motivo}


def apresentar_predicao(resultado):
    return {**resultado, "status": "analisado", "mensagem": INDICIOS[resultado["rotulo_previsto"]]}
