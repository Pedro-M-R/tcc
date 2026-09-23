# TCC — Comparação SVM + BERTimbau

O site usa as mesmas funções de previsão e a mesma normalização de `testar_noticias.py`. A opção inicial é **Comparar os dois modelos**. Cada resultado identifica o modelo e a interface informa quando eles discordam. Também é possível escolher SVM ou BERTimbau separadamente.

O título é opcional. O SVM mostra sua margem de decisão, sem convertê-la em porcentagem. O BERTimbau mostra os escores com casas decimais suficientes para evitar arredondamento para 100%. A classificação é uma previsão estatística e pode errar; os escores não comprovam os fatos.

## Publicação

Use **Python 3.13**, repositório `Pedro-M-R/tcc` e arquivo principal `web/app.py`. Esta atualização está na branch `corrigir-comparacao-streamlit`. O SVM exige scikit-learn 1.5.2, a versão usada para salvar o modelo. Antes de substituir uma instalação em Python 3.14, consulte [PUBLICAR_STREAMLIT.md](PUBLICAR_STREAMLIT.md).

O rodapé desta versão mostra **Revisão 2026.09.23-1**. Os modelos finais ficam em `modelos/svm` e `modelos/bertimbau_128_es`; os pesos do BERTimbau continuam no Git LFS. Não publique CSV, banco de dados ou checkpoints.

## Executar localmente

```powershell
python -m pip install -r web/requirements.txt
python -m streamlit run web/app.py
```

Abra `http://localhost:8501`. Os modelos são carregados na primeira análise e reutilizados em memória. O BERTimbau lê os primeiros 128 tokens, como no treinamento; o site mostra quando houve truncamento e permite conferir o trecho analisado.

A opção **Usar um link** extrai título e texto de uma página pública. Quando há uma conclusão explícita em uma checagem do Boatos.org, ela aparece com atribuição à fonte, separada das previsões dos modelos sobre o texto do artigo.

## Validação

```powershell
python -m unittest test_apresentacao test_site test_checagem test_links -v
```

Os 25 testes do site passaram com Streamlit 1.64.0. A validação local com modelos reais comparou 27 entradas nos dois modelos e encontrou as mesmas classes e escores da janela. A base usada nessa verificação não é publicada. O teste adicional `test_paridade.py` só roda quando a base e a divisão local estão presentes.
