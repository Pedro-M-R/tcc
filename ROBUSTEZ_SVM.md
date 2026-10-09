# Revisão do SVM — 9 de outubro de 2026

O SVM usado no site é treinado por `treinar_svm_leve.py`. `treinar_local.py`
treina os experimentos Transformer e não atualiza esse SVM.

## Diagnóstico reproduzido

O modelo anterior (`modelos/svm_leve`, antes desta revisão) não previa tudo como
verdadeiro nas notícias completas: acertava 457 de 488 falsas no teste interno.
Ao receber apenas os títulos dessas mesmas notícias, acertava somente 53 de 488
falsas e chamava as outras 435 de verdadeiras. Com as primeiras 40 palavras,
acertava 287 de 488 falsas. A avaliação anterior ocultava essa diferença porque
considerava apenas notícias completas.

A ordem das classes está correta: `0 = fake`, `1 = true`. O arquivo joblib
contém o Pipeline com o TF-IDF ajustado e o SVM. A inferência não ajusta novamente
o vocabulário. A base tem 5.542 verdadeiras e 4.891 falsas (53,1% e 46,9%);
o pequeno desequilíbrio não explica sozinho a falha nas entradas curtas.

## O que mudou

- Validação cruzada estratificada por grupos, com três folds por padrão. O TF-IDF
  é ajustado dentro do treino de cada fold; duplicatas, títulos, corpos e URLs
  iguais continuam agrupados pelo carregador compartilhado.
- Busca de duas representações de palavras/caracteres, três valores de C,
  três opções de pesos de classe e uso opcional de títulos/recortes no treino:
  36 candidatos, cada um avaliado em três folds.
- As variantes são geradas depois da divisão. Cada notícia mantém peso total
  unitário, repartido entre texto completo (60%), título (20%) e primeiras
  40 palavras (20%). Variantes iguais são unidas. Duplicatas preparadas recebem
  peso inverso à multiplicidade; os CSVs originais permanecem preservados.
- Escolha por F1 macro: 50% do peso para notícias completas, 25% para títulos
  e 25% para recortes. São rejeitados candidatos sem convergência ou que
  prevejam uma única classe em algum cenário/fold.
- Calibração em grupos separados, incluindo as variantes de entrada.
- Ajuste do limiar para favorecer a recuperação de falsas, usando F2 da classe
  `fake`. A seleção exige precisão de falsas de pelo menos 90% nos textos
  completos e limita a perda de F1 macro a um ponto percentual em relação a
  0,5, tanto em textos completos quanto no resultado ponderado. Se nenhum limiar
  atende, mantém 0,5 e registra `criterio_atingido=false`. Esse piso é um critério
  na amostra de seleção, não uma promessa sobre novas notícias.
- O site usa o mesmo limiar salvo que a avaliação. Modelos antigos continuam
  usando 0,5. Os escores permanecem os do calibrador, sem alterações artificiais.
- Relatórios por classe, entrada e fonte, distribuição de previsões, comparação
  com a classe majoritária e identificação explícita de colapso.

Os grupos de `treino` + `validacao` (70%) são usados na busca e no ajuste final;
`calibracao` (10%) ajusta os escores; `selecao` (10%) escolhe o limiar; `teste`
(10%) mede o resultado após congelar as decisões. O manifesto registra os usos.
Ajustar limiar em dados diferentes dos usados para ajustar o classificador evita
essa forma de sobreajuste, conforme a
[documentação do scikit-learn](https://scikit-learn.org/1.5/modules/classification_threshold.html).

## Reproduzir

Na raiz do projeto, use o ambiente do site (scikit-learn 1.8.0):

```powershell
.\.venv-svm\Scripts\python.exe -m unittest test_svm_robusto -v
$saida = "modelos/svm_robusto_" + (Get-Date -Format "yyyyMMdd_HHmmss")
.\.venv-svm\Scripts\python.exe treinar_svm_leve.py --saida $saida --perfil completo --comparar-com modelos/svm_leve
if ($LASTEXITCODE -ne 0) { throw "Treinamento falhou; verifique os relatórios." }
$env:TCC_SVM_MODEL_PATH = $saida
.\.venv-svm\Scripts\python.exe -m streamlit run web/app.py
```

`--perfil leve` reduz a busca para dois candidatos, mantendo a separação por
grupos. `--csv` aceita uma ou mais bases com `titulo,texto,rotulo`; `url` é
opcional. Uma pasta de saída ocupada nunca é sobrescrita.

Relatórios principais: `selecao_modelo.json`, `selecao_limiar.json`,
`metricas_teste_por_entrada.json`, `metricas_teste_por_fonte.json`,
`comparacao_anterior.json`, `predicoes_teste.csv` e `auditoria_base.json`.

Para usar o mesmo modelo e limiar do site pelo terminal:

```powershell
.\.venv-svm\Scripts\python.exe prever.py --modelo svm_leve --texto "Cole aqui a notícia."
```

O modo antigo `--modelo svm` e a janela histórica `testar_noticias.py` continuam
usando os modelos históricos; não representam o novo treino do site.

## Limitações da base

As fontes são correlacionadas aos rótulos. Por exemplo, a base contém 1.359
linhas de `noticias.uol.com.br` como falsas e 16 como verdadeiras. Isso exige
revisar se os textos contêm a alegação falsa original, uma checagem sobre ela
ou uma notícia legítima. A URL sozinha não permite corrigir o rótulo. Nenhum
rótulo foi trocado automaticamente.

Todos os 4.891 corpos falsos estão sem pontuação; entre os verdadeiros, 86,95%
estão nessa condição. Muitos corpos já chegaram sem palavras funcionais, enquanto os
títulos e entradas do aplicativo têm linguagem natural. O treino com variantes
reduz essa diferença, mas não recupera o texto original. Títulos e recortes
herdam o rótulo da notícia e podem perder contexto; suas métricas são testes
de robustez, não uma validação independente da veracidade de cada frase.

O teste interno foi reutilizado por versões anteriores. Os resultados não são
uma estimativa independente de desempenho em notícias novas. Uma próxima
avaliação externa deve usar alegações revisadas, fontes e períodos novos, sem
misturar esses exemplos no ajuste. O SVM identifica padrões de texto e não
consulta evidências factuais. A regra preexistente de opinião pessoal permanece
separada e não participa das métricas do SVM.

## Resultado executado nesta revisão

Experimento: `modelos/svm_robusto_20261009`. Configuração escolhida pela CV:
palavras de 1–2 unidades e caracteres de 2–5, C=1, sem peso adicional de classe,
com títulos e recortes no treino. F1 macro ponderado na CV: 0,86169, desvio
entre folds 0,00353. O melhor candidato sem treino com recortes obteve 0,76210.
O limiar selecionado para `fake` foi 0,52; os critérios de seleção foram atendidos.

Comparação no mesmo teste interno (1.042 notícias, sendo 488 falsas):

| Entrada / medida | Anterior | Novo |
|---|---:|---:|
| Completa — acurácia | 95,11% | 92,42% |
| Completa — recall de falsas | 93,65% (457/488) | 91,80% (448/488) |
| Completa — precisão de falsas | 95,81% | 91,99% |
| Título — recall de falsas | 10,86% (53/488) | 65,37% (319/488) |
| Título — precisão de falsas | 88,33% | 81,79% |
| Primeiras 40 palavras — recall de falsas | 58,81% (287/488) | 85,45% (417/488) |
| Primeiras 40 palavras — precisão de falsas | 79,28% | 85,63% |

O ganho é de robustez nas entradas curtas, com regressão nos textos completos.
Não houve ganho uniforme de precisão. Recall significa a proporção de falsas
identificadas; precisão significa a proporção correta entre as previsões de falsa.
O modelo novo ainda deixou passar 169 dos 488 títulos falsos do teste.

Os arquivos em `comparacao_anterior.json` registram as métricas completas antes
e depois. `erros_para_revisao.csv` reúne os 79 erros em notícias completas,
incluindo texto, URL e rótulos da base, para revisão humana. Esses casos não
foram adicionados ao treinamento nem usados para escolher o limiar.

O modelo anterior fica preservado em `modelos/svm_leve_antes_robustez_20261009`.
O novo experimento permanece em sua própria pasta e sua cópia ativa fica em
`modelos/svm_leve`, usada por `abrir_site.cmd` e pelo modo CLI `svm_leve`.
Reinicie o aplicativo local após atualizar o código. Nenhum site remoto foi
publicado nesta revisão; uma publicação precisa levar também `svm_leve.py`
atualizado para respeitar o limiar salvo.

## Correção do fluxo de checagens — 9 de outubro de 2026

Os dois links enviados (Eduarda Campopiano e atraso na apuração/TSE) foram
preservados em `resultados/revisao_casos_reais`, fora do treinamento.
`diagnostico_local.json` registra que o SVM retorna `true` para ambos os artigos
que desmentem os boatos. Ao receber apenas a alegação, retorna `fake` no caso
Eduarda e `true` no caso TSE. O erro do TSE permanece no classificador estatístico.

A revisão `2026.10.09-checagem1` apresenta a conclusão explícita do Boatos.org
sobre a alegação e não executa o SVM sobre o texto que a desmente. Não basta
pertencer ao domínio: o leitor exige alegação e selo único após a conclusão.
Sem esse selo, uma checagem reconhecida não recebe veredito do modelo.
URLs isoladas no formulário de texto também usam o leitor de páginas.

`test_checagem.py` cobre as duas páginas salvas, conclusões falsas e verdadeiras,
selos ausentes/conflitantes e texto colado sem origem autenticada.
`validar_publicacao_svm.py` verifica os dois links ao vivo nos dois formulários,
além das entradas comuns. Esses testes validam o fluxo da aplicação; não
constituem evidência de melhora na generalização do SVM. Pesos, limiar e métricas
de treinamento permanecem os mesmos nesta correção.

A revisão foi publicada em `tccpedro.streamlit.app` e validada ao vivo em
09/10/2026: sete cenários passaram. Os dois links mostram “Alegação falsa”,
atribuída ao Boatos.org, tanto em “Usar um link” quanto quando a URL é colada
no campo de texto. Nenhum escore do SVM é exibido nesses quatro cenários.
O registro local está em `resultados/revisao_casos_reais/publicacao_checagem1.json`.

## Triagem de contexto no site — revisão `2026.10.09-triagem1`

A triagem existente em `triagem_noticias.py` estava sendo ignorada pelo SVM na
interface. Agora ela também é aplicada antes de carregar esse modelo. O exemplo
“elefante rosa no ceara”, inclusive no título ou repetido muitas vezes, retorna
“Análise inconclusiva”, pede contexto e não recebe escore de veracidade.

O filtro exige corpo com 40 palavras, ao menos 20 distintas e sinais de contexto
noticioso; também trata repetição e alguns padrões de opinião/instrução. A regra
preexistente de opinião pessoal do aplicativo permanece separada. Checagens com
conclusão explícita continuam seguindo o fluxo atribuído à fonte.

Esses limites são heurísticos e podem rejeitar notícias legítimas curtas ou
aceitar textos inventados com aparência de notícia. Não é um detector semântico
de absurdos. Pesos, limiar, classificador bruto e métricas de treinamento não
mudaram; as métricas antigas não avaliam a cobertura deste filtro da interface.
`test_svm_leve.py` verifica as entradas bloqueadas sem carregar o SVM e permite
um relato contextualizado. A validação remota é salva em
`resultados/revisao_casos_reais/publicacao_triagem1.json`.
