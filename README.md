# TCC — Classificação de notícias com BERTimbau

Site em Streamlit que recebe o título e o texto de uma notícia e mostra a classificação **Verdadeira** ou **Falsa**, com os percentuais atribuídos pelo BERTimbau a cada classe.

Escolha **Usar um link**, cole o endereço público e clique em **Ler link e analisar**. O site extrai o título e o texto, mostra o conteúdo encontrado e executa a classificação automaticamente. A opção **Colar título e texto** continua disponível.

Links de mensagens privadas, páginas com login, paywall ou conteúdo carregado por JavaScript podem não ser lidos. A extração é automática e pode incluir trechos incorretos; confira o conteúdo mostrado. PDFs, imagens e vídeos não são aceitos. A análise continua limitada ao início do texto, conforme o modelo treinado.

**É apenas um modelo de inteligência artificial e pode errar.** Os percentuais são escores do modelo, não uma garantia de veracidade. Confira a notícia em fontes confiáveis antes de acreditar ou compartilhar.

## Publicar no Streamlit

1. Entre em [Streamlit Community Cloud](https://share.streamlit.io/) e clique em **Create app**.
2. Escolha **Yup, I have an app** e preencha:
   - **Repository:** `Pedro-M-R/tcc`
   - **Branch:** `main`
   - **Main file path:** `web/app.py`
3. Em **Advanced settings**, selecione **Python 3.14**. As dependências também são compatíveis com Python 3.11 a 3.13.
4. Clique em **Deploy** e aguarde a instalação das dependências e o carregamento do modelo.

Também é possível usar **Paste GitHub URL** com `https://github.com/Pedro-M-R/tcc/blob/main/web/app.py`.

As dependências estão em `web/requirements.txt`. Não são necessários secrets ou acesso a banco de dados. O modelo usa CPU e é carregado na primeira análise. A disponibilidade de memória e o funcionamento na nuvem devem ser verificados nos logs após a publicação.

## Executar localmente

Instale Python 3.11 a 3.14, Git e Git LFS. No terminal:

```bash
git clone https://github.com/Pedro-M-R/tcc.git
cd tcc
git lfs install
git lfs pull
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py
```

Abra `http://localhost:8501`. No Windows, após instalar as dependências, também pode usar `abrir_site.cmd`.

## Modelo

Usa exclusivamente o BERTimbau ajustado, salvo em `modelos/bertimbau_128_es`. Os pesos de aproximadamente 436 MB estão versionados com **Git LFS**, compatível com o Streamlit Community Cloud. A base de treinamento, o SVM e os checkpoints intermediários não são necessários para executar este site.

O modelo lê até 128 tokens, conforme o treinamento; o site avisa quando considera apenas o título e o início do texto. A aplicação não grava as notícias em arquivos ou banco de dados.

Referências: [publicação no Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [organização dos arquivos e suporte a Git LFS](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).
