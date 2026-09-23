# Site de notícias — comparação SVM + BERTimbau

O site usa a mesma normalização, seleção de modelos finais e funções de previsão de `testar_noticias.py`. A opção inicial é **Comparar os dois modelos**; também é possível escolher SVM ou BERTimbau separadamente. O título é opcional. A interface mostra os resultados de cada modelo e informa se concordam ou discordam. O CSV de treino, o banco e os checkpoints não são necessários. A previsão é uma classificação estatística, não uma checagem factual.

O BERTimbau mostra os escores para **Verdadeira** e **Falsa**, preservando casas decimais suficientes para não arredondar valores altos para 100% (por exemplo, 99,98%). São escores softmax não calibrados. O SVM mostra somente sua margem de decisão, como na janela local; essa margem não é uma porcentagem. A formatação não altera as previsões nem os pesos.

A opção **Usar um link** lê uma página pública e analisa o título e o texto automaticamente. O conteúdo extraído aparece na tela para conferência. Mensagens privadas, páginas com login ou assinatura e sites que dependem de JavaScript podem exigir que o usuário cole o conteúdo manualmente. A leitura aceita páginas HTML de até 2 MB, com limite de 30 mil caracteres de texto extraído.

Em checagens do Boatos.org com alegação e conclusão explícitas, a interface separa a **conclusão da fonte** da **previsão do BERTimbau sobre o artigo**. O resultado da fonte tem atribuição e link, sem porcentagem criada. A previsão original do modelo pode ser consultada em uma seção própria; eventuais divergências são sinalizadas. Isso não altera os pesos nem melhora, por si só, a capacidade do modelo de verificar notícias novas.

## Abrir no computador

Na pasta do projeto, com **Python 3.13** (também compatível com 3.11 e 3.12):

```powershell
python -m venv .venv-site
.\.venv-site\Scripts\python.exe -m pip install -r web/requirements.txt
.\.venv-site\Scripts\python.exe -m streamlit run web/app.py
```

Abra `http://localhost:8501`. Se as dependências já estiverem no Python padrão, basta dar dois cliques em `abrir_site.cmd` ou executar `python -m streamlit run web/app.py`.

## Publicar no Streamlit Community Cloud

A versão anterior do site foi publicada em [Pedro-M-R/tcc](https://github.com/Pedro-M-R/tcc), apenas com BERTimbau. Esta revisão precisa incluir também o SVM e os módulos compartilhados listados abaixo. Use **Main file path: `web/app.py`** e **Python 3.13**.

**Se o aplicativo atual usa Python 3.14, prepare outra publicação com Python 3.13 antes de substituir a versão online.** O scikit-learn 1.5.2 é a versão usada para salvar o SVM e não fornece wheel para Python 3.14. Não atualize essa biblioteca arbitrariamente para carregar o pickle antigo. O Streamlit não permite trocar o Python de uma publicação existente; veja a [orientação oficial sobre a versão do Python](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/upgrade-python).

O arquivo de pesos tem aproximadamente **436 MB**. Use **Git LFS** para enviá-lo ao GitHub; arrastar esse arquivo pela página de upload não é o fluxo adequado. Instale Git e Git LFS antes dos comandos abaixo. O arquivo `.gitattributes` já configura os pesos para LFS e `.gitignore` inclui somente os artefatos finais necessários.

1. Crie um repositório vazio no GitHub e copie a URL dele.
2. Na raiz deste projeto, execute os comandos abaixo. Substitua `SEU_USUARIO/SEU_REPOSITORIO` pela URL real. Os comandos `git add` são intencionalmente explícitos para publicar somente o site e seu modelo.

```powershell
git init
git lfs install
git add .gitignore .gitattributes .streamlit/config.toml web/app.py web/estilo.css web/inferencia.py web/extrair_link.py web/checagem.py web/apresentacao.py web/requirements.txt PUBLICAR_STREAMLIT.md
git add dados_modelos.py prever.py resultados_noticias.py pontuacoes.py modelos/svm/modelo.joblib modelos/bertimbau_128_es/configuracao_treino.json
git add modelos/bertimbau_128_es/config.json modelos/bertimbau_128_es/model.safetensors modelos/bertimbau_128_es/tokenizer.json modelos/bertimbau_128_es/tokenizer_config.json modelos/bertimbau_128_es/special_tokens_map.json modelos/bertimbau_128_es/vocab.txt
git lfs ls-files
git commit -m "Adiciona site de classificação com BERTimbau"
git branch -M main
git remote add origin https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git
git push -u origin main
```

Antes do commit, `git lfs ls-files` deve listar `model.safetensors`. O armazenamento e os downloads seguem as cotas de Git LFS da conta.

3. Entre em [Streamlit Community Cloud](https://share.streamlit.io/) com acesso ao repositório e escolha **Create app**.
4. Selecione o repositório, branch `main` e **Main file path: `web/app.py`**.
5. Nas configurações avançadas, escolha **Python 3.13** e publique com **Deploy**.
6. Aguarde a instalação e teste uma notícia no endereço `https://…streamlit.app` gerado pela plataforma.

O Streamlit usa `web/requirements.txt`, que fica ao lado do arquivo principal, e obtém os arquivos armazenados com Git LFS. O modelo roda em CPU e é carregado somente na primeira análise, ficando compartilhado em memória para os próximos acessos. Não há treinamento durante a publicação. As dependências do site fixam PyTorch para CPU; o projeto original mantém suas dependências de treinamento.

O endereço local não é um link público. Confirme na publicação a opção **Comparar os dois modelos** e o rodapé **Revisão 2026.09.23-1**. O funcionamento na nuvem deve ser confirmado após o deploy, inclusive a memória disponível para os modelos e o volume de acessos.

## Configuração e diagnóstico

- Por padrão, usa o SVM em `modelos/svm` e o BERTimbau final mais recente, como a janela local. Nesta entrega, o BERTimbau está em `modelos/bertimbau_128_es`. Para outro BERTimbau final compatível, defina `BERTIMBAU_MODEL_PATH`.
- Se o SVM estiver ausente, o site mostra um aviso e oferece somente BERTimbau. Isso não equivale à comparação dos dois modelos.
- Publique os módulos compartilhados `dados_modelos.py`, `prever.py`, `resultados_noticias.py` e `pontuacoes.py` na raiz junto da pasta `web`.
- Se aparecer erro de arquivo ausente, confira os seis arquivos do modelo adicionados acima. Não utilize apenas o BERTimbau base sem o ajuste deste projeto.
- Se o peso estiver como ponteiro de texto, execute `git lfs pull` e confira o upload LFS do repositório.
- Consulte os logs do Streamlit para detalhes de falhas. O site não grava as notícias em arquivos nem em banco de dados.
- O modelo atual lê até 128 tokens, como no treinamento. Notícias maiores são truncadas; o site informa quando somente o início foi considerado.
- Para verificar a equivalência local com os modelos reais: `python -m unittest test_paridade -v`. A base local é usada apenas nessa verificação, não na publicação.

Referências oficiais: [organização de arquivos e Git LFS](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization), [dependências](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies), [publicação](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
