# Notícia em análise — TF-IDF + SVM

O site usa apenas o modelo leve em `modelos/svm_leve`, executado em CPU.
BERTimbau não é carregado nem oferecido na interface. O código histórico permanece
no projeto local para consulta; nenhum treinamento BERT é iniciado pelo site.

## Testar neste computador

Abra `abrir_site.cmd` e acesse http://localhost:8501.
Escolha **Colar título e texto** ou **Usar um link**. Informe o texto completo
da notícia ou um link público e clique em analisar.

Em uma instalação nova, use Python 3.13 para reproduzir o ambiente validado:

```powershell
python -m venv .venv-svm
.\.venv-svm\Scripts\python.exe -m pip install -r web/requirements.txt
.\.venv-svm\Scripts\python.exe -m streamlit run web/app.py
```

Não precisa de GPU, banco de dados, secrets, PyTorch ou Transformers para analisar.
Os artefatos são incluídos no repositório; não há download de pesos na inicialização.

## Modelo e avaliação

Treino com TF-IDF de palavras e caracteres e LinearSVC com classes balanceadas.
A configuração é escolhida pelo F1 macro de validação. Escores são calibrados com
regressão logística sobre a margem em outra partição. Um filtro de familiaridade
lexical e limiares selecionados separadamente permitem responder **inconclusivo**.

São cinco partições por grupos: treino, validação, calibração, seleção e teste.
Notícias com título, corpo, URL ou entrada preparada idênticos ficam no mesmo grupo.
As 10.433 linhas originais são preservadas; palavras que entregam o rótulo e URLs
são ignoradas na entrada do modelo. A mesma preparação é usada pelo site.

Os resultados finais ficam em `modelos/svm_leve/metricas_teste.json`, incluindo
acurácia, F1 macro, cobertura, abstenções e matriz de confusão. A configuração e
versões usadas ficam em `selecao_modelo.json` e `ambiente.json`.

Modelo treinado em 08/10/2026 (scikit-learn 1.8.0):

| Medida | Resultado |
|---|---:|
| Notícias no teste reservado | 1.042 |
| Acurácia sem abstenção | 94,43% |
| F1 macro sem abstenção | 94,40% |
| Notícias com resposta aceita | 550 (52,78%) |
| Notícias inconclusivas | 492 (47,22%) |
| Acurácia nas respostas aceitas | 96,00% |
| Modelo, calibrador e filtro em disco | 10,2 MB |

Os 36 testes de regressão passaram, incluindo 6 testes do novo modelo,
inferência real no Streamlit, interface, leitura de links e triagem.

Os escores medem padrões aprendidos, não a probabilidade de um fato ser verdadeiro.
O teste é uma amostra da base, não uma avaliação prospectiva nem uma validação com
fontes inteiramente novas. Os resultados podem mudar com outras notícias e fontes.
Uma resposta inconclusiva não significa notícia falsa. Confira sempre as fontes.

## Treinar novamente

Use uma nova pasta para preservar o modelo existente:

```powershell
.\.venv-svm\Scripts\python.exe treinar_svm_leve.py --csv resultados/bases/base_atualizada_2026-10-06.csv --saida modelos/svm_novo
```

Para testar esse novo artefato sem substituir o padrão:

```powershell
$env:TCC_SVM_MODEL_PATH = "modelos/svm_novo"
.\.venv-svm\Scripts\python.exe -m streamlit run web/app.py
```

## Streamlit Community Cloud

App: [tccpedro.streamlit.app](https://tccpedro.streamlit.app/).
Repositório `Pedro-M-R/tcc`, branch `main`, arquivo `web/app.py`.
Dependências em `web/requirements.txt`; ambiente local validado com Python 3.13.2.
Veja [PUBLICAR_STREAMLIT.md](PUBLICAR_STREAMLIT.md) para os arquivos e a publicação.
A atualização do repositório é usada pela implantação existente. O rodapé da
versão leve identifica **2026.10.08-svm**. Não configure `TCC_MODEL_PATH` para
reativar BERT: o site usa exclusivamente `TCC_SVM_MODEL_PATH` ou o SVM padrão.

Links privados, com login ou protegidos contra leitura automática podem falhar.
Nesses casos, cole o texto. A conclusão explícita encontrada em uma checagem do
Boatos.org aparece separada da previsão estatística para o texto da página.
