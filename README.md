# Notícia em análise — TF-IDF + SVM binário

Limpeza no Pipeline, ablação e avaliação por fontes: [guia de limpeza do SVM](LIMPEZA_SVM.md).

A revisão **2026.10.08-svm3** classifica entradas preenchidas em `fake` ou
`true`, sem abstenção por tamanho, domínio ou escore. Não carrega BERTimbau.
O modelo padrão continua em `modelos/svm_leve`, executado em CPU.

Frases inteiras como **“pedro eh bonito”** e **“pedro eh bonita”**, inclusive
repetidas no título e no corpo, retornam **Falsa** por uma regra explícita de
opinião pessoal. A interface identifica a regra e não fabrica porcentagem nem
atribui essa resposta ao SVM. Isso é uma convenção solicitada para o aplicativo,
não uma verificação objetiva de beleza. Notícias que apenas mencionem a frase
continuam sendo analisadas pelo modelo.

## Testar neste computador

Abra `abrir_site.cmd` e acesse http://localhost:8501.
Escolha **Colar título e texto** ou **Usar um link**. Textos curtos e títulos
isolados também recebem classificação. Uma entrada vazia pede preenchimento.

Em outra instalação, use Python 3.13 para reproduzir o ambiente validado:

```powershell
python -m venv .venv-svm
.\.venv-svm\Scripts\python.exe -m pip install -r web/requirements.txt
.\.venv-svm\Scripts\python.exe -m streamlit run web/app.py
```

## Treinamento e avaliação

O perfil completo compara **32 configurações** de TF-IDF + LinearSVC:
quatro combinações de palavras e caracteres, quatro valores de C e duas
opções de balanceamento de classes. A escolha usa F1 macro na validação.

A configuração escolhida combina palavras de uma a duas unidades com sequências
de dois a cinco caracteres, peso dos caracteres 0,75, C=10 e classes balanceadas.
O ajuste final tem **180.810 características**. Os dois artefatos joblib ocupam
aproximadamente **3,1 MB**; não existe filtro de domínio para carregar.

As 10.433 linhas são separadas por grupos, mantendo notícias com título, corpo,
URL ou entrada preparada idênticos no mesmo grupo. A antiga partição de seleção
de abstenção agora participa do treino. A busca usa 7.304 registros e a validação
1.043. Depois de escolher a configuração, o ajuste final usa esses 8.347 registros.
A calibração usa outros 1.044 registros; o teste mantém os mesmos 1.042 registros
reservados das versões anteriores. O manifesto registra esses usos.

| Medida no teste | Resultado |
|---|---:|
| Notícias avaliadas | 1.042 |
| Notícias classificadas | 1.042 (100%) |
| Análises inconclusivas | 0 |
| Acurácia | 95,11% |
| F1 macro | 95,08% |
| Acurácia do SVM anterior sem abstenção | 94,43% |

As métricas acima medem o classificador estatístico nas notícias, não a regra
de opinião pessoal. Essa regra tem testes separados. As previsões representam
padrões aprendidos, não comprovação de fatos; textos curtos ou assuntos novos
podem receber uma resposta errada. Estes resultados são de uma amostra interna
reutilizada para comparação, não de uma avaliação externa nova.

Os relatórios ficam em `modelos/svm_leve/metricas_teste.json` e
`metricas_teste_por_fonte.json`. A seleção fica em `selecao_modelo.json`.
O último treino completo levou cerca de 5 minutos e 46 segundos nesta máquina.

## Treinar novamente

Execute na raiz do projeto. Cada treino cria uma pasta nova:

```powershell
$saida = "modelos/svm_" + (Get-Date -Format "yyyyMMdd_HHmmss")
.\.venv-svm\Scripts\python.exe treinar_svm_leve.py --csv resultados/bases/base_atualizada_2026-10-06.csv --saida $saida --perfil completo
if ($LASTEXITCODE -ne 0) { throw "O treinamento falhou; confira a mensagem acima." }
$env:TCC_SVM_MODEL_PATH = $saida
.\.venv-svm\Scripts\python.exe -m streamlit run web/app.py
```

O perfil `leve` compara apenas três configurações; `completo` é o padrão.
Não use `treinar_local.py`, que pertence aos experimentos BERT.
Repetir exatamente a mesma base e configuração não garante ganho de desempenho.

## Publicação

App: [tccpedro.streamlit.app](https://tccpedro.streamlit.app/).
Repositório `Pedro-M-R/tcc`, branch `main`, arquivo `web/app.py`.
Use `web/requirements.txt`, com scikit-learn 1.8.0.
Veja [PUBLICAR_STREAMLIT.md](PUBLICAR_STREAMLIT.md) para atualização dos arquivos.

O Streamlit descarta o cache quando pesos ou política mudam na mesma pasta.
Não configure `TCC_MODEL_PATH` para reativar BERT: o site usa exclusivamente
`TCC_SVM_MODEL_PATH` ou o SVM padrão. A base de treino e o manifesto local não
são publicados.

Links que bloqueiem extração podem exigir colar o texto. A conclusão explícita
de uma checagem do Boatos.org permanece separada da classificação estatística.
