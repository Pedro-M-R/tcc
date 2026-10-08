"""Regressões de abstenção: opinião, título, repetição e notícia contextualizada."""

import unittest
from unittest.mock import patch

from prever import prever_noticias
from resultados_noticias import analisar_noticia, formatar_analise
from triagem_noticias import avaliar_entrada, INDICIOS

NOTICIA = ("A prefeitura anunciou nesta terça-feira a abertura de uma biblioteca pública no bairro central. "
           "Segundo o comunicado divulgado pela secretaria de cultura, o espaço receberá estudantes e moradores "
           "a partir de segunda-feira, das oito às dezessete horas. A equipe informou que o acervo inicial "
           "terá cinco mil livros e acesso gratuito mediante cadastro presencial.")


class TestTriagem(unittest.TestCase):
    def test_opinioes_variantes_sem_carregar_modelos(self):
        for frase in ("o Pedro é bonito", "O PEDRO EH BONITO", "Maria é linda", "Acho João muito bonito"):
            with self.subTest(frase=frase), patch("prever.carregar_svm") as svm, patch("prever.carregar_bertimbau") as bert:
                for tipo in ("svm", "bertimbau"):
                    r = prever_noticias([frase], [frase], "inexistente", tipo)[0]
                    self.assertEqual(r["rotulo_previsto"], "inconclusivo")
                    self.assertIn("subjetiva", r["motivo"])
                    self.assertNotIn("probabilidades_modelo", r)
                svm.assert_not_called()
                bert.assert_not_called()

    def test_titulo_texto_curto_e_repeticao(self):
        for titulo, texto in ((NOTICIA, ""), ("", "bom dia"), ("", "Pedro é bonito. " * 50),
                              (NOTICIA, NOTICIA), ("", "1234 !!! ???")):
            self.assertIsNotNone(avaliar_entrada(titulo, texto))

    def test_noticia_e_citacao_com_contexto_passam(self):
        self.assertIsNone(avaliar_entrada("Biblioteca abre na segunda", NOTICIA))
        self.assertIsNone(avaliar_entrada("Morador considera o prédio bonito", NOTICIA + ' Um morador disse: "é bonito".'))

    def test_lote_preserva_ordem_e_so_avalia_entradas_adequadas(self):
        with patch("prever.prever_svm", return_value=[{"rotulo_previsto": "fake", "margem_svm": -.2}]) as bruto:
            rs = prever_noticias(["", "", ""], ["Pedro eh bonito", NOTICIA, "teste"], "modelo", "svm")
        self.assertEqual([r["rotulo_previsto"] for r in rs], ["inconclusivo", "fake", "inconclusivo"])
        self.assertEqual(rs[1]["mensagem"], INDICIOS["fake"])
        self.assertEqual(len(bruto.call_args.args[0]), 1)

    def test_relatorio_opiniao_sem_pontuacao(self):
        r = analisar_noticia("Pedro é bonito", "Pedro é bonito", [
            {"tipo": "svm", "nome": "SVM", "caminho": "inexistente"},
            {"tipo": "bertimbau", "nome": "BERTimbau", "caminho": "inexistente"}])
        texto = formatar_analise(r)
        self.assertIn("Análise inconclusiva", texto)
        self.assertNotIn("%", texto)
        self.assertNotIn("Classificação prevista:", texto)


if __name__ == "__main__":
    unittest.main()
