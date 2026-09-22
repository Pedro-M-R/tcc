"""Lê artigos públicos, com limites de rede e bloqueio de endereços internos."""

import http.client
import ipaddress
import socket
import ssl
import time
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

MAX_BYTES = 2_000_000
MAX_TEXTO = 30_000


class ErroLink(ValueError):
    """Mensagem de leitura que pode ser apresentada ao visitante."""


def validar_destino(url):
    url = url.strip()
    if not url or len(url) > 2048 or any(c.isspace() for c in url) or "\\" in url:
        raise ErroLink("Cole um link completo e válido, começando com https://.")
    try:
        partes = urlsplit(url)
        if partes.scheme not in {"http", "https"} or not partes.hostname or partes.username or partes.password:
            raise ValueError()
        porta = partes.port or (443 if partes.scheme == "https" else 80)
        if porta != (443 if partes.scheme == "https" else 80):
            raise ValueError()
        host = partes.hostname.encode("idna").decode("ascii")
        registros = socket.getaddrinfo(host, porta, type=socket.SOCK_STREAM)
        enderecos = {ipaddress.ip_address(r[4][0]) for r in registros}
    except (ValueError, UnicodeError, OSError):
        raise ErroLink("Não foi possível localizar esse link. Confira o endereço e tente novamente.") from None
    if not enderecos or any(not ip.is_global for ip in enderecos):
        raise ErroLink("Use um link público de uma notícia.")
    # Conecta diretamente ao IP validado, impedindo uma segunda resolução de DNS.
    ip = str(sorted(enderecos, key=lambda item: (item.version, int(item)))[0])
    host_header = f"[{host}]" if ":" in host else host
    caminho = quote(partes.path or "/", safe="/%:@!$&'()*+,;=-._~")
    consulta = quote(partes.query, safe="%=&?/:@!$'()*+,;~-._")
    url_limpa = urlunsplit((partes.scheme, host_header, caminho, consulta, ""))
    destino = caminho + ("?" + consulta if consulta else "")
    return url_limpa, partes.scheme, host, host_header, porta, ip, destino


class ConexaoSegura(http.client.HTTPSConnection):
    def __init__(self, host, porta, ip, timeout):
        super().__init__(host, porta, timeout=timeout)
        self.ip_validado = ip

    def connect(self):
        sock = socket.create_connection((self.ip_validado, self.port), timeout=self.timeout)
        try:
            self.sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        except Exception:
            sock.close()
            raise


def baixar_html(url):
    limite_tempo = time.monotonic() + 25
    for _ in range(5):
        url, esquema, host, host_header, porta, ip, destino = validar_destino(url)
        restante = limite_tempo - time.monotonic()
        if restante <= 0:
            raise ErroLink("A página demorou para responder. Tente novamente ou cole o texto.")
        timeout = min(8, restante)
        conexao = (ConexaoSegura(host, porta, ip, timeout) if esquema == "https" else
                   http.client.HTTPConnection(ip, porta, timeout=timeout))
        try:
            conexao.request("GET", destino, headers={
                "Host": host_header, "User-Agent": "NoticiaEmAnalise/1.0 (leitor de artigos publicos)",
                "Accept": "text/html,application/xhtml+xml", "Accept-Encoding": "identity",
            })
            resposta = conexao.getresponse()
            if resposta.status in {301, 302, 303, 307, 308}:
                local = resposta.getheader("Location")
                if not local:
                    raise ErroLink("Esse link não levou a uma notícia. Cole o link direto da publicação.")
                url = urljoin(url, local)
                continue
            if resposta.status != 200:
                raise ErroLink("A página não permitiu a leitura. Ela pode exigir login, assinatura ou estar indisponível. "
                               "Use a opção de colar o título e o texto.")
            tipo = resposta.getheader("Content-Type", "").split(";")[0].strip().lower()
            if tipo not in {"text/html", "application/xhtml+xml"}:
                raise ErroLink("Use o link de uma página de notícia. PDFs, imagens e vídeos não são lidos aqui.")
            if resposta.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ErroLink("Não foi possível ler o formato dessa página. Cole o título e o texto.")
            partes = []
            tamanho = 0
            while True:
                restante = limite_tempo - time.monotonic()
                if restante <= 0:
                    raise ErroLink("A página demorou para responder. Tente novamente ou cole o texto.")
                if conexao.sock is not None:
                    conexao.sock.settimeout(min(8, restante))
                bloco = resposta.read1(64 * 1024)
                if not bloco:
                    return b"".join(partes), url
                tamanho += len(bloco)
                if tamanho > MAX_BYTES:
                    raise ErroLink("Essa página é muito grande para leitura automática. Cole o título e o texto.")
                partes.append(bloco)
        except (OSError, http.client.HTTPException):
            raise ErroLink("Não foi possível acessar a página agora. Tente novamente ou cole o título e o texto.") from None
        finally:
            conexao.close()
    raise ErroLink("Esse link redireciona muitas vezes. Cole o endereço direto da notícia.")


def extrair_html(html, url):
    from lxml import html as parser_html
    from readability import Document

    documento = Document(html)
    titulo = " ".join(documento.short_title().split())
    corpo = parser_html.fromstring(documento.summary(html_partial=True))
    # Mantém separação entre parágrafos, sem enviar HTML à interface ou ao modelo.
    for elemento in corpo.xpath(".//script|.//style|.//nav|.//footer|.//form"):
        elemento.drop_tree()
    paragrafos = corpo.xpath(".//p|.//h2|.//h3|.//li|.//blockquote[not(.//p)]")
    texto = "\n\n".join(" ".join(p.text_content().split()) for p in paragrafos)
    if not texto:
        texto = " ".join(corpo.text_content().split())
    if not titulo or len(texto) < 120 or len(texto.split()) < 20:
        raise ErroLink("Não encontrei título e texto suficientes para analisar. "
                       "A página pode exigir login ou carregar o conteúdo dinamicamente. Cole o conteúdo na outra opção.")
    return {"titulo": titulo[:500], "texto": texto[:MAX_TEXTO], "url": url,
            "texto_limitado": len(texto) > MAX_TEXTO}


def extrair_noticia(url):
    html, url_final = baixar_html(url)
    return extrair_html(html, url_final)
