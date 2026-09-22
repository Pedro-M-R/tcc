"""Testes de extração e limites de acesso a links públicos."""

import socket
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))
from extrair_link import ErroLink, baixar_html, extrair_html, validar_destino


def dns(ip):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


class TestLinks(unittest.TestCase):
    def test_nao_aceita_esquemas_e_credenciais(self):
        for url in ("", "file:///etc/passwd", "ftp://example.com", "https://u:p@example.com", "https://example.com:8080"):
            with self.subTest(url=url), self.assertRaises(ErroLink):
                validar_destino(url)

    def test_bloqueia_redes_privadas_e_metadados(self):
        for ip in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "192.168.1.1", "::1", "fe80::1", "::ffff:127.0.0.1"):
            with self.subTest(ip=ip), patch("extrair_link.socket.getaddrinfo", return_value=dns(ip)):
                with self.assertRaises(ErroLink):
                    validar_destino("https://example.com/artigo")

    def test_endereco_publico_e_preservacao_de_consulta(self):
        with patch("extrair_link.socket.getaddrinfo", return_value=dns("93.184.215.14")):
            resultado = validar_destino("https://example.com/notícia?q=ação#inicio")
        self.assertEqual(resultado[5], "93.184.215.14")
        self.assertEqual(resultado[-1], "/not%C3%ADcia?q=a%C3%A7%C3%A3o")

    def test_redirect_para_endereco_interno_e_bloqueado(self):
        resposta = Mock(status=302)
        resposta.getheader.return_value = "http://127.0.0.1/segredo"
        conexao = Mock()
        conexao.getresponse.return_value = resposta
        with patch("extrair_link.socket.getaddrinfo", side_effect=[dns("93.184.215.14"), dns("127.0.0.1")]):
            with patch("extrair_link.ConexaoSegura", return_value=conexao):
                with patch("extrair_link.http.client.HTTPConnection") as http:
                    with self.assertRaises(ErroLink):
                        baixar_html("https://example.com")
                    http.assert_not_called()
        conexao.close.assert_called_once()

    def test_tamanho_e_tipo_limitados(self):
        for tipo, blocos in (("image/png", []), ("text/html", [b"x" * 101])):
            resposta = Mock(status=200)
            resposta.getheader.side_effect = lambda k, default=None: {"Content-Type": tipo}.get(k, default)
            resposta.read1.side_effect = blocos
            conexao = Mock()
            conexao.getresponse.return_value = resposta
            with patch("extrair_link.socket.getaddrinfo", return_value=dns("93.184.215.14")):
                with patch("extrair_link.ConexaoSegura", return_value=conexao), patch("extrair_link.MAX_BYTES", 100):
                    with self.assertRaises(ErroLink):
                        baixar_html("https://example.com")

    def test_extrai_artigo_sem_menu_ou_script(self):
        paragrafo = "Pesquisadores analisaram os resultados do estudo durante dois anos. Os dados foram publicados com os métodos utilizados, permitindo que outras equipes confiram os resultados apresentados."
        html = f'<html><head><meta charset="utf-8"><title>Pesquisa sobre educação</title></head><body><nav>MENU DO SITE</nav><article><h1>Pesquisa sobre educação</h1><p>{paragrafo}</p><p>{paragrafo}</p></article><script>CODIGO_INTERNO</script></body></html>'
        resultado = extrair_html(html.encode(), "https://example.com/artigo")
        self.assertEqual(resultado["titulo"], "Pesquisa sobre educação")
        self.assertIn(paragrafo, resultado["texto"])
        self.assertNotIn("MENU DO SITE", resultado["texto"])
        self.assertNotIn("CODIGO_INTERNO", resultado["texto"])

    def test_pagina_sem_texto_nao_vira_noticia(self):
        with self.assertRaises(ErroLink):
            extrair_html(b"<html><title>Login</title><body>Entre para continuar</body></html>", "https://example.com")


if __name__ == "__main__":
    unittest.main()
