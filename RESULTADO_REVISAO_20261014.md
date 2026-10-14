# Revisão de 14/10/2026

A consulta de checagens pelo texto está integrada ao formulário, sem exigir URL.
Use **Colar título e texto → Consultar checagens pelo texto**. O resultado mostra
a alegação identificada, a conclusão publicada e o link da fonte. O botão de
SVM continua disponível como análise estatística separada.

O treinamento com variantes terminou e foi comparado com o modelo ativo.
Melhorou o F1 macro de títulos e recortes, mas reduziu o F1 de notícias completas
de 94,01% para 92,20% e continuou errando nas duas mensagens fornecidas.
O modelo ativo foi preservado. Veja [a comparação e as limitações](CONSULTA_POR_TEXTO.md).

Validação concluída:

- Suíte da pasta de publicação: 70 testes, sem falhas, 3 dispensados pelas
  condições locais; inclui interface, rede simulada, links e variantes.
- Consulta real: cinco formatos do exemplo reconhecidos pela fonte como `fake`.
- Interfaces local e de publicação: conclusão da fonte exibida sem escore do SVM.
- Verificador remoto: seleção explícita entre consulta, SVM e leitura de link,
  evitando que o segundo botão substitua silenciosamente o primeiro; os três
  testes adicionais passaram.

As alterações estão nos arquivos locais. Esta revisão não foi enviada ao GitHub
nem implantada no Streamlit público.
