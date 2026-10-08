"""Política compartilhada do novo treino e da análise, independente do rótulo."""

import re
import unicodedata
from functools import lru_cache

VERSAO_PREPARACAO = "ignorar_palavras_v1"
PISTAS = re.compile(
    r"\b(?:fakes?|fals[oa]s?|falks[oa]s?|boatos?|mentiras?|enganos[oa]s?|"
    r"desinformacao|verdade|verdadeir[oa]s?|veridic[oa]s?|desmentid[oa]s?|"
    r"desmentir|desmente|desmentiu|checagem|hoax|debunk(?:ed|ing)?|"
    r"fact[ -]?check(?:ing)?|true|false)\b"
)


def simplificar(texto):
    return "".join(c for c in unicodedata.normalize("NFKD", str(texto).casefold())
                   if not unicodedata.combining(c))


def pistas_rotulo(titulo, texto):
    return sorted(set(PISTAS.findall(simplificar(f"{titulo} {texto}"))))


@lru_cache(maxsize=32768)
def _ignorar_token(token):
    return bool(PISTAS.fullmatch(simplificar(token)))


def preparar_noticia(titulo, texto):
    # URLs e e-mails não são atributos do classificador. Não insere marcadores
    # de substituição que possam revelar qual classe sofreu mais remoções.
    entrada = unicodedata.normalize("NFC", f"{titulo}\n{texto}")
    entrada = re.sub(r"(?:https?://|www\.)\S+|\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", entrada)
    # Substitui nos tokens originais: preserva acentos, negações e palavras
    # vizinhas, e não introduz um marcador que denuncie onde havia uma pista.
    entrada = re.sub(r"\b\w+(?:[ -]+check(?:ing)?)?\b",
                     lambda m: " " if _ignorar_token(m.group()) else m.group(),
                     entrada, flags=re.IGNORECASE)
    return " ".join(entrada.split())
