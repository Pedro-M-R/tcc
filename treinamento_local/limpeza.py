"""Preparação serializável do SVM; regras fixas, nunca aprendidas no teste."""

import re
import unicodedata

from preparacao_noticias import preparar_noticia, PISTAS

VERSAO_LIMPEZA = "svm_limpeza_v3_sem_secao_checagem"
TERMOS_ROTULO = (
    r"fals\w*", r"fake\w*", r"farsa\w*", r"boato\w*", r"enganos\w*",
    r"desinforma\w*", r"hoax\w*", r"mentira\w*", r"montage(?:m|ns)",
    r"verdadeir\w*", r"verdade", r"comprovad\w*", r"checage(?:m|ns)",
    r"verificad\w*",
)
_TERMOS = re.compile(r"(?<!\w)#?(?:" + "|".join(TERMOS_ROTULO) + r")\b")
_HASHTAGS_ROTULO = re.compile(r"#(?:" + PISTAS.pattern + r")")
_FONTES = re.compile(r"\b(?:agencia\s+lupa|aos\s+fatos|boatos(?:\s*\.\s*|\s+)org|uol\s+confere)\b")
# Somente linhas explicitamente editoriais; não corta todo o restante da notícia.
_EDITORIAL = re.compile(
    r"^\s*(?:(?:por|autor(?:a)?|reportagem|edicao|revisao)\s*:\s*.+|"
    r"por\s+[a-z]+(?:\s+[a-z]+){1,4}\s*|"
    r"(?:todos os direitos reservados|assine nossa newsletter|"
    r"siga(?:-nos)? nas redes sociais|compartilhe (?:esta noticia|este conteudo)).*)$",
    re.MULTILINE,
)


def remover_secao_checagem(texto):
    """Corta a seção editorial e tudo que a segue, preservando o texto anterior.

    Reconhece cabeçalhos e o formato achatado Boato / Análise / Checagem
    da base tratada. Uma menção comum a checagem não é um cabeçalho.
    A regra independe do rótulo e também é aplicada na inferência.
    """
    texto = str(texto)
    flags = re.IGNORECASE | re.MULTILINE
    cortes = []
    cabecalho = re.search(
        r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*)?checagem(?:\*\*)?"
        r"[ \t]*(?::|[–—-]|\r?$)", texto, flags)
    if cabecalho:
        cortes.append(cabecalho.start())
    # No CSV original, pontuação e quebras de linha já foram removidas.
    estrutura = re.search(r"^[ \t]*boatos?\b.*?\ban[aá]lise\b", texto, flags | re.DOTALL)
    if estrutura:
        marcador = re.search(r"\bchecagem\b", texto[estrutura.end():], re.IGNORECASE)
        if marcador:
            cortes.append(estrutura.end() + marcador.start())
    # Formato anterior do mesmo corpus: "checagem parte checagem ...".
    parte = re.search(
        r"\b(?:checagem\s+)?parte\s+(?:de\s+)?checagem\b", texto, re.IGNORECASE)
    if parte:
        cortes.append(parte.start())
    return texto[:min(cortes)].rstrip() if cortes else texto


def limpar_texto(texto):
    """Normaliza e remove pistas ANTES dos analisadores de palavras e caracteres.

    Preserva datas, negações e nomes de veículos em geral. Termos como 'golpe'
    podem ser o próprio acontecimento; ampliar regras exige revisão do treino.
    A remoção não garante ausência de viés ou compreensão factual.
    """
    texto = remover_secao_checagem(texto)
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    # Antes de remover nomes de fontes: não pode sobrar um fragmento 'https://'.
    texto = re.sub(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", texto)
    texto = _EDITORIAL.sub(" ", texto)
    texto = _FONTES.sub(" ", texto)
    texto = _HASHTAGS_ROTULO.sub(" ", texto)
    texto = _TERMOS.sub(" ", texto)
    # Inclui URLs, e-mails e as pistas já usadas na versão anterior.
    texto = preparar_noticia("", texto)
    return " ".join(texto.split())
