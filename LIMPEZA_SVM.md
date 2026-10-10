# Limpeza e diagnóstico de atalhos do SVM

O treinamento principal (`treinar_svm_leve.py` e `treinamento_local/treino_svm.py`)
e o script da pasta `.publicacao-svm` usam `treinamento_local/limpeza.py` como
`preprocessor` dos TF-IDFs de palavras **e** caracteres. O Pipeline salvo aplica
a limpeza ao texto bruto, inclusive ao carregar o joblib em outro processo.
Os modelos antigos continuam usando a preparação anterior.

A limpeza normaliza caixa e acentos, remove os termos de rótulo configurados,
URLs, e-mails, referências explícitas a Agência Lupa/Aos Fatos/Boatos.org/UOL Confere e
linhas editoriais reconhecidas (assinaturas e rodapés). Os padrões de linhas
dependem de quebras de linha preservadas no texto original. Não se removem
indiscriminadamente datas, veículos ou palavras como “golpe”: podem ser fatos
relevantes. Mesmo termos removidos, como “montagem”, podem ter uso legítimo.

A versão `svm_limpeza_v2` remove também o marcador de hashtags de rótulo
(por exemplo, `#FALSO`) e reconhece `boatos org` sem pontuação. Hashtags de
conteúdo, como `#educacao`, são preservadas. Modelos com outra versão de limpeza
são recusados pelo adaptador, para evitar usar pesos com uma preparação diferente.

O carregador preserva o texto bruto para o classificador. Duplicatas originais
e entradas iguais após a limpeza ficam no mesmo grupo **nas duas condições**
da ablação. Rótulos e CSVs não são alterados.

## Treinar e comparar

Na raiz do projeto, com uma pasta de saída nova:

```powershell
.\.venv-svm\Scripts\python.exe comparar_limpeza_svm.py --csv resultados/bases/base_atualizada_2026-10-06.csv --saida experimentos/minha_ablacao --perfil leve
```

Para usar especificamente o script aberto na pasta de publicação, acrescente
`--treinador .publicacao-svm/treinar_svm_leve.py`. Os protocolos dos dois scripts
continuam distintos: o principal usa CV e seleção de limiar; a publicação usa
a validação fixa e decisão a 0,5. Compare condições usando o mesmo script.

O comando executa treino, calibração e teste completos duas vezes, gera logs
separados e `comparacao.json`, e verifica que as partições são idênticas.
`--perfil leve` limita a busca; `--perfil completo` executa a busca mais ampla.
Também é possível treinar individualmente com `--sem-limpeza` para o controle.
O controle mantém termos, URLs e linhas editoriais, com a normalização padrão
do TF-IDF. Portanto, a ablação mede o pacote inteiro de limpeza.

O modelo novo não substitui automaticamente `modelos/svm_leve` nem é publicado.
Os novos artefatos requerem também o módulo `treinamento_local/limpeza.py` e
`preparacao_noticias.py` no ambiente em que forem carregados.

## Fontes e relatórios

Use `--separacao fonte` para reservar hostnames inteiros, normalizando o prefixo
`www.`. Subdomínios diferentes continuam fontes diferentes; isso não garante
separação por grupo empresarial. Duplicatas que ligam duas fontes unem essas
fontes no mesmo componente. URLs ausentes e falta de componentes suficientes
geram erro explícito, sem voltar silenciosamente à divisão por notícia.

`--validar-dados` no script de treino salva auditoria e manifesto sem treinar.
Na separação por fonte, tentativas determinísticas de divisão procuram somente
presença das duas classes nas cinco partições, nunca a melhor métrica.
Fontes grandes podem produzir partições muito desiguais; confira o manifesto.

Cada modelo gera:

- `atributos_mais_influentes.json`: até 40 pesos negativos (fake) e positivos
  (true), com identificação do analisador de palavras/caracteres.
- `auditoria_vazamento.json`: distribuição de classes por fonte, fontes de
  classe única, grupos e conflitos de rótulo após a limpeza.
- `manifesto.json`: notícias e fontes em cada partição.
- `metricas_teste.json` e `metricas_teste_por_fonte.json`: avaliação reservada.

Revise atributos com treino/validação antes de congelar as regras. Alterar
regras olhando o teste contamina a avaliação. Queda de F1 não prova vazamento;
estabilidade de F1 não prova compreensão do conteúdo. O teste histórico desta
base já foi reutilizado: a comparação é exploratória e requer confirmação
externa. As probabilidades calibradas não medem verdade factual.

## Prever diretamente

```python
import joblib

pasta = "experimentos/minha_ablacao/com_limpeza"
modelo = joblib.load(f"{pasta}/modelo.joblib")
calibrador = joblib.load(f"{pasta}/calibrador.joblib")
margem = modelo.decision_function(["Texto original da notícia aqui"]).reshape(-1, 1)
print(dict(zip(["fake", "true"], calibrador.predict_proba(margem)[0])))
```

Para reproduzir também o limiar e as regras do aplicativo, use
`svm_leve.ModeloSVMLeve(pasta).analisar_noticia(titulo, texto)`.

## Resultados concluídos em 10/10/2026

Execuções com `svm_limpeza_v2`, perfil `leve`, seed 42 e o treinador
`.publicacao-svm/treinar_svm_leve.py` (validação fixa, decisão a 0,5).
Base: `resultados/bases/base_atualizada_2026-10-06.csv`, 10.433 notícias;
SHA-256: `6a73076684f18643eaa64b84f190703b19f946cd087d9305970788243272fbac`.
Estes números não são resultados do protocolo de CV do treinador da raiz.

| Experimento | Notícias no teste | F1 macro | Acurácia |
| --- | ---: | ---: | ---: |
| Grupos de notícias, com limpeza | 1.042 | 94,50% | 94,53% |
| Grupos de notícias, sem limpeza | 1.042 | 95,75% | 95,78% |
| Fontes separadas, com limpeza | 53 | 32,05% | 47,17% |

A ablação está em `experimentos/ablacao_limpeza_v2_2026-10-10/comparacao.json`.
Os manifestos confirmam os mesmos arquivos e partições nas duas condições;
nenhuma chave de texto limpo aparece em partições diferentes. A diferença
de F1 (com menos sem limpeza) foi **−1,26 ponto percentual**. A seleção pela
validação escolheu C=1 com limpeza e C=4 sem limpeza; ambos convergiram.
Portanto, a comparação mede o procedimento completo com o mesmo espaço de
busca, permitindo que cada condição escolha seu parâmetro.

O diagnóstico por fonte está em `experimentos/svm_limpo_fontes_v2_2026-10-10`.
A base contém 27 hostnames normalizados, dos quais 24 têm uma só classe.
Duplicatas uniram fontes em 26 componentes independentes. Nenhum hostname
atravessa as partições desse experimento, mas a distribuição ficou desigual:

| Partição por fonte | Total | Fake | True |
| --- | ---: | ---: | ---: |
| Treino | 10.188 | 4.783 | 5.405 |
| Validação | 78 | 14 | 64 |
| Calibração | 54 | 53 | 1 |
| Seleção (incorporada ao treino neste protocolo) | 60 | 16 | 44 |
| Teste | 53 | 25 | 28 |

O classificador calibrado previu **fake para todas as 53 notícias do teste**:
acertou 25 falsas e errou as 28 verdadeiras. A calibração com apenas uma
notícia verdadeira e o teste pequeno limitam esse resultado. Além disso,
o teste por fonte contém notícias diferentes do teste por grupos: a diferença
entre os dois cenários não isola causalmente o efeito da fonte. Este ensaio
não sustenta uma estimativa geral de desempenho em fontes novas.

Para reproduzir o ensaio por fonte em outra pasta, execute na raiz:

```powershell
.\.venv-svm\Scripts\python.exe .publicacao-svm/treinar_svm_leve.py --csv resultados/bases/base_atualizada_2026-10-06.csv --saida experimentos/minhas_fontes --perfil leve --separacao fonte --seed 42
```

Cada pasta concluída contém os pesos, calibrador, política de decisão,
manifesto, métricas, predições e 40 atributos por classe. Os experimentos
anteriores sem `v2` no nome usam revisões anteriores e não devem ser
misturados com esta comparação. Nenhum desses modelos foi promovido para o site.

Na retomada, passaram 43 testes na raiz (`test_limpeza_svm`, `test_svm_leve`,
`test_svm_robusto`, `test_treinamento_local`, `test_config_treino_local`) e
5 testes de `test_limpeza_svm` na pasta de publicação. O modelo real com
limpeza foi recarregado em outro processo: os dois analisadores preservam
o preprocessador e as probabilidades do adaptador coincidem com as do pipeline.

Os resultados são exploratórios porque o teste interno já foi reutilizado.
As regras permanecem congeladas nesta versão; a confirmação exige uma base
externa independente, com rótulos revisados e melhor cobertura de classes
por fonte. A queda de F1 observada não comprova vazamento.
