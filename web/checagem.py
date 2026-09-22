"""Reconhece conclusões explícitas de checagens, sem alterar a previsão da IA."""

import re
import unicodedata
from urllib.parse import urlsplit


def extrair_checagem(arvore, url):
    # O domínio sozinho nunca define um veredito. É necessário encontrar a
    # alegação e um selo explícito depois da seção Conclusão no mesmo artigo.
    if urlsplit(url).hostname not in {"boatos.org", "www.boatos.org"}:
        return None
    for artigo in arvore.xpath("//article"):
        alegacao = None
        conclusao = False
        vereditos = set()
        for elemento in artigo.xpath(".//p|.//h2|.//h3"):
            texto = " ".join(elemento.text_content().split())
            marcador = re.fullmatch(r"Boato\s*[–—-]\s*(.+)", texto, re.IGNORECASE)
            if marcador and alegacao is None:
                alegacao = marcador.group(1)
            simples = unicodedata.normalize("NFKD", texto.casefold())
            simples = "".join(c for c in simples if c.isalpha() or c.isspace())
            simples = " ".join(simples.split())
            if simples == "conclusao":
                conclusao = True
            elif conclusao:
                if simples in {"fake news", "falso", "falsa"}:
                    vereditos.add("fake")
                elif simples in {"verdadeiro", "verdadeira", "e verdade"}:
                    vereditos.add("true")
        if alegacao and len(vereditos) == 1:
            return {"fonte": "Boatos.org", "url": url, "alegacao": alegacao,
                    "rotulo": vereditos.pop()}
    return None
