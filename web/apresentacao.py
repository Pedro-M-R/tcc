"""Mesma formatação de escores usada no relatório da janela local."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from pontuacoes import formatar_probabilidade
