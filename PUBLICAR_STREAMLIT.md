# Publicar o Streamlit com TF-IDF + SVM

O site atual usa somente o modelo treinado em `modelos/svm_leve`.
Não precisa do BERTimbau, de GPU, de SageMaker nem de um novo treinamento.
App: [tccpedro.streamlit.app](https://tccpedro.streamlit.app/).
Repositório: `Pedro-M-R/tcc`, branch `main`, arquivo principal `web/app.py`.

## Testar localmente

Abra `abrir_site.cmd` na raiz do projeto e acesse http://localhost:8501.
Use **Colar título e texto** ou **Usar um link** e clique em analisar.

O ambiente `.venv-svm` já existe neste computador. Para outra instalação:

```powershell
python -m venv .venv-svm
.\.venv-svm\Scripts\python.exe -m pip install -r web/requirements.txt
.\.venv-svm\Scripts\python.exe -m streamlit run web/app.py
```

O modelo foi salvo com Python 3.13.2 e scikit-learn 1.8.0.
Use Python 3.13 para reproduzir o ambiente validado e mantenha a versão de
scikit-learn de `web/requirements.txt`.

## Atualizar a publicação existente

Neste computador, a cópia Git de publicação está em `.publicacao-svm`,
dentro do projeto original. `site_svm_leve.zip` contém os arquivos de
publicação, incluindo o modelo, sem base de treinamento ou checkpoints.

1. Atualize os arquivos na cópia Git e confira `git diff`.
2. Envie o commit para `Pedro-M-R/tcc`, branch `main`.
3. No Streamlit, confira o arquivo principal `web/app.py` e a instalação de
   `web/requirements.txt`. Esse arquivo fica ao lado do app, conforme a
   [documentação de dependências](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).
4. Abra o app e confirme a única opção **TF-IDF + SVM** e o rodapé
   **2026.10.08-svm**. Faça uma análise; abrir a página inicial não basta para
   comprovar que o modelo carregou.

Não use `requirements.txt` da raiz do projeto original para instalar o site:
ele pertence aos experimentos históricos e inclui PyTorch e Transformers.
O pacote leve usa `web/requirements.txt` e `requirements-svm.txt`.

## Modelo e configuração

O carregamento padrão usa `modelos/svm_leve`. A pasta precisa conter
`modelo.joblib`, `calibrador.joblib`, `dominio.joblib`, `politica_decisao.json`
e `concluido.json`. O pacote também inclui os relatórios e as versões de treino.
Preserve os módulos Python e a estrutura de pastas do pacote.

Não há download de pesos ao abrir o site. `TCC_MODEL_PATH` e
`BERTIMBAU_MODEL_PATH` não selecionam modelos no app atual.
Para testar outro SVM treinado por `treinar_svm_leve.py`, configure
`TCC_SVM_MODEL_PATH` com o caminho da pasta desse modelo.

O resultado pode ser inconclusivo por falta de contexto, vocabulário fora do
domínio ou escore abaixo do limiar. Isso não significa notícia falsa.
Se um link bloquear a extração, cole o corpo da notícia.

Consulte [STREAMLIT_SVM.md](STREAMLIT_SVM.md) para métricas, limitações e retreino.