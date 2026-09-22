import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "web"))
from apresentacao import formatar_probabilidade


class TestApresentacao(unittest.TestCase):
    def test_saturacao_nao_e_exibida_como_certeza(self):
        self.assertEqual(formatar_probabilidade(0.9998304843902588), ">99,9%")
        self.assertEqual(formatar_probabilidade(0.0001694978418527171), "<0,1%")
        self.assertEqual(formatar_probabilidade(1.0), ">99,9%")
        self.assertEqual(formatar_probabilidade(0.0), "<0,1%")

    def test_valores_intermediarios_preservados(self):
        self.assertEqual(formatar_probabilidade(.823), "82,3%")
        self.assertEqual(formatar_probabilidade(.999), "99,9%")

    def test_valores_invalidos_sao_rejeitados(self):
        for p in (float('nan'), float('inf'), -1, 2):
            with self.assertRaises(ValueError):
                formatar_probabilidade(p)


if __name__ == '__main__':
    unittest.main()
