"""Busca pública por texto; atribui veredito apenas a uma alegação reconhecida.

O SVM não participa da busca. Correspondências aproximadas são sugestões de
leitura, nunca prova de que as alegações são iguais. Não exige chave de API.
"""
import html
import re
import time
import unicodedata
from urllib.parse import urlencode, urljoin, urlsplit

from lxml import html as parser_html

from checagem import extrair_checagem
from extrair_link import baixar_html, ErroLink

FONTE = 'https://www.boatos.org/'
HOSTS = {'boatos.org', 'www.boatos.org'}
IGNORAR = set('a o as os um uma de da do das dos em no na nos nas e que se para por com '
              'ao aos como mais seu sua seus suas estaria seria vai vao esta estao foi sao '
              'noticia analise boato boatos versao mensagens circulam redes sociais afirmam '
              'dentro campanha tentar alerta brasil'.split())


def normalizar(texto):
    texto = html.unescape(str(texto))
    texto = unicodedata.normalize('NFKD', texto.casefold())
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    return ' '.join(re.findall(r'\w+', texto))


def consultas(titulo, texto):
    # Busca somente termos do título/início; nunca envia o corpo inteiro.
    inicio = (titulo.strip() or texto.strip())[:700]
    inicio = re.sub(r'https?://\S+', ' ', inicio)
    termos = list(dict.fromkeys(t for t in normalizar(inicio).split()
                               if t not in IGNORAR and len(t) > 2))
    if len(termos) < 3:
        return []
    return list(dict.fromkeys(' '.join(termos[:n]) for n in (6, 3)))


def url_da_fonte(url):
    try:
        p = urlsplit(url)
        return (p.scheme == 'https' and p.hostname in HOSTS and not p.username
                and not p.password and p.port in (None, 443))
    except ValueError:
        return False


def retirar_marcador(texto):
    texto = texto.strip().strip('*_> \t')
    return re.sub(r'^(?:Boato\s*[–—:-]|Vers[aã]o\s+\d+\s*:)\s*', '', texto,
                  flags=re.IGNORECASE).strip()


def correspondencia(titulo, texto, checagem, arvore):
    """Igualdade lexical completa, preservando negações, números e nomes.

    Aceita uma alegação inteira no título/primeiro parágrafo ou a mensagem
    inteira citada na fonte. Não aceita conter a alegação dentro de uma frase
    que a nega, nem apenas grande sobreposição de palavras.
    """
    candidatos = [titulo, texto]
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    if linhas:
        candidatos.append(linhas[0])
    paragrafos = [p.strip() for p in re.split(r'\n\s*\n', texto) if p.strip()]
    if paragrafos:
        candidatos.append(paragrafos[0])
    alegacao = normalizar(checagem['alegacao'])
    alvos = [(alegacao, 'alegacao')]
    for bloco in arvore.xpath('//article//blockquote'):
        citado = bloco.text_content().strip()
        versoes = re.split(r'\bVers[aã]o\s+\d+\s*:', citado, flags=re.IGNORECASE)
        for versao in versoes:
            alvo = normalizar(retirar_marcador(versao))
            if len(alvo.split()) >= 12:
                alvos.append((alvo, 'mensagem_citada'))
    for candidato in candidatos:
        entrada = normalizar(retirar_marcador(candidato))
        if len(entrada.split()) < 8:
            continue
        for alvo, tipo in alvos:
            if entrada == alvo:
                return tipo
    return None


def consultar_fontes(titulo, texto, tempo_limite=20):
    limite = time.monotonic() + tempo_limite
    relacionados, correspondencias, vistos = [], [], set()
    falhas = 0
    buscas = consultas(titulo, texto)
    if not buscas:
        return {'status': 'insuficiente', 'fontes': []}
    for consulta in buscas:
        if time.monotonic() >= limite or len(vistos) >= 5:
            break
        try:
            pagina, destino = baixar_html(FONTE + '?' + urlencode({'s': consulta}),
                                          tempo_limite=limite - time.monotonic())
            if not url_da_fonte(destino):
                raise ErroLink('A busca saiu da fonte configurada.')
            arvore = parser_html.fromstring(pagina)
            links = arvore.xpath('//article//h2//a[@href]')
            for link in links:
                url = urljoin(destino, link.get('href'))
                if not url_da_fonte(url) or url in vistos:
                    continue
                if len(vistos) >= 5 or time.monotonic() >= limite:
                    break
                vistos.add(url)
                relacionados.append({'titulo': ' '.join(link.text_content().split()), 'url': url})
                try:
                    conteudo, final = baixar_html(url, tempo_limite=limite - time.monotonic())
                    if not url_da_fonte(final):
                        continue
                    artigo = parser_html.fromstring(conteudo)
                    checagem = extrair_checagem(artigo, final)
                    if checagem:
                        tipo = correspondencia(titulo, texto, checagem, artigo)
                        if tipo:
                            correspondencias.append({**checagem, 'correspondencia': tipo})
                except (ErroLink, ValueError, parser_html.etree.ParserError):
                    falhas += 1
            # Todos os candidatos desta busca foram examinados; não escolhe
            # arbitrariamente entre conclusões conflitantes encontradas.
            if correspondencias:
                break
        except (ErroLink, ValueError, parser_html.etree.ParserError):
            falhas += 1
    if correspondencias:
        if len({c['rotulo'] for c in correspondencias}) > 1:
            return {'status': 'conflitante', 'fontes': relacionados}
        return {'status': 'encontrada', 'checagem': correspondencias[0], 'fontes': relacionados}
    status = 'relacionadas' if relacionados else ('indisponivel' if falhas or time.monotonic() >= limite else 'nao_encontrada')
    return {'status': status, 'fontes': relacionados, 'falhas': falhas}
