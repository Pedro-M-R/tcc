# Consulta por texto: triagem, fontes e SVM

Revisão de 16/10/2026 — `2026.10.16-triagem-restaurada`.

No formulário **Colar título e texto**, cole a alegação ou mensagem recebida
e clique em **Analisar notícia**. Não é necessário informar URL.
Primeiro, a triagem exige corpo com pelo menos 40 palavras, 20 palavras
distintas e contexto noticioso. Frases como “elefante rosa no Piauí”, títulos
isolados e repetições retornam **Análise inconclusiva**, sem consulta externa
nem escore. Essa proteção também é aplicada na inferência pública do SVM.
A regra preexistente para opiniões pessoais permanece separada.
A mensagem de andamento é **Analisando a notícia…**; os detalhes das fontes
ficam em **Como a análise funciona**.

Após a triagem, o sistema consulta o **Boatos.org**. Se reconhecer a alegação e uma
conclusão explícita, exibe o resultado da fonte. Uma conclusão falsa mostra
“Há indícios de que seja uma alegação falsa”. Uma checagem verdadeira mantém
seu resultado; o domínio sozinho não define falsidade.

Sem correspondência no Boatos.org, consulta o **G1**. Uma correspondência com
a manchete, primeiro parágrafo ou corpo da notícia mostra “Há indícios de que
seja verdadeira”, acompanhada do link. Páginas de Fato ou Fake e manchetes
com marcadores explícitos de desmentido não recebem esse resultado.

Sem correspondência confirmada, executa o **SVM** no texto original somente
se a entrada passar pela triagem. Falhas de rede também permitem essa análise. A interface
informa que esse resultado vem dos padrões do modelo, que pode errar.
Os pesos e o treinamento não foram alterados. O fluxo de links mantém a
triagem de contexto existente. O filtro usa regras de conteúdo e tamanho;
não detecta toda afirmação inventada nem comprova a veracidade de textos aceitos.

A consulta utiliza as buscas públicas do **Boatos.org** e do **G1**, sem chave de API. Envia
até seis termos extraídos do título ou do início do texto. Não envia o corpo
inteiro. Examina até cinco páginas por fonte, com orçamento de aproximadamente 24 segundos
(a resolução DNS depende do sistema operacional).

O sistema só apresenta uma conclusão automaticamente se encontrar:

- Uma alegação completa correspondente no título ou primeiro parágrafo/linha;
  ou uma mensagem completa citada em um bloco de citação da fonte.
- A alegação e um selo de conclusão explícito, sem conflito, no artigo original.

A comparação ignora caixa, acentos e pontuação. Mantém palavras, negações,
números e endereços citados. Não interpreta paráfrases. Similaridade de assunto
ou os mesmos nomes nunca bastam para atribuir verdadeiro/falso.

O resultado identifica a alegação examinada e oferece o link da checagem. Não
classifica toda a reportagem como falsa por ela descrever um boato. Matérias
relacionadas sem correspondência são sugestões de leitura. Busca vazia ou falha
de rede deixam a consulta inconclusiva e acionam o SVM. Conclusões conflitantes
do Boatos.org permanecem inconclusivas. Entradas vazias pedem preenchimento.
Artigos de checagem colados sem correspondência pedem o link original para
evitar classificar a explicação como se fosse a alegação.

## Relação com o treinamento

Este recurso consulta uma fonte externa; ele não é uma melhora de acurácia do
SVM. Os resultados da consulta não devem ser contados como acertos do modelo
no TCC. O SVM continua podendo errar em notícias novas.

O treinador `treinar_svm_leve.py` desta pasta passou a gerar variantes após a
separação dos grupos: texto completo (60%), título (20%) e primeiras 40 palavras
do corpo (20%). Variantes iguais após a limpeza são unidas e cada notícia mantém
peso total 1. A validação e a calibração também incluem as variantes. A decisão
continua em 0,5; não foi ajustada para acertar uma pessoa ou um exemplo.

`--sem-variantes` controla somente a ampliação do treino; validação e calibração
continuam incluindo entradas curtas. `metricas_teste_por_entrada.json` separa os
resultados por formato. Os recortes herdam o rótulo documental e precisam de
revisão de contexto. O teste interno já foi reutilizado; as métricas são
exploratórias e não estimam, sozinhas, desempenho em notícias novas.

O caso informado pelo usuário fica fora do treino, em um diagnóstico local.
Novos experimentos são gravados em outra pasta; executar o treinador não
substitui automaticamente o modelo ativo nem publica o site.

## Resultado do experimento concluído

O treinamento `experimentos/svm_variantes_20261014` terminou em 528 segundos.
A comparação nos mesmos 1.042 documentos de teste ficou assim:

| Entrada | F1 macro do modelo ativo | F1 macro com variantes |
|---|---:|---:|
| Notícia completa | 94,01% | 92,20% |
| Somente título | 45,97% | 74,95% |
| Primeiras 40 palavras | 83,40% | 89,70% |

Em notícias completas, ambos reconheceram 449 das 488 falsas. Os erros que
classificam verdadeiras como falsas subiram de 23 para 42, entre 554 verdadeiras.
O modelo com variantes também continuou classificando a alegação curta e as
duas mensagens do exemplo como `true`. Por isso, **o modelo ativo foi mantido**.
O experimento continua disponível para estudo; não é a solução desse caso.

A consulta real por texto reconheceu cinco entradas: alegação curta, texto
longo, título com corpo separado e as duas mensagens. Todas retornaram `fake`
como conclusão publicada pelo Boatos.org, com o link da checagem. As interfaces
da raiz e da pasta de publicação também passaram na verificação ao vivo.

Os registros locais ficam em `resultados/revisao_textos_20261014/comparacao.json`
e `consulta_ao_vivo.json`. Esses resultados são diagnósticos, não uma garantia
para paráfrases, outras alegações ou indisponibilidade da fonte.

## Validação

Na pasta `.publicacao-svm`, com o ambiente Python do projeto:

```powershell
..\.venv-svm\Scripts\python.exe -m unittest test_consulta_fontes test_variantes_svm test_checagem test_links -v
```

Os testes isolam a rede e cobrem conclusões falsas/verdadeiras, mensagens
citadas, mudanças de negação/data/sujeito, ausência de selo, fonte indisponível,
domínio imitado e conflitos. Testes de interface verificam que uma
correspondência não chama o SVM, que o G1 mostra o indício favorável e que
ausência de correspondência ou indisponibilidade executam o modelo apenas
com contexto suficiente. `test_svm_leve` cobre a abstenção no formulário,
nos links e na inferência direta, inclusive após um resultado favorável anterior.

A consulta depende da disponibilidade e do HTML da fonte. Notícias para as
quais não exista checagem nesse site e textos reformulados podem não ser
reconhecidos. Não existe promessa de classificar toda notícia falsa.
