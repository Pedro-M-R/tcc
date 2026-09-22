"""Execute a partir da raiz: python -m streamlit run web/app.py."""

import logging

import streamlit as st

from inferencia import Classificador, caminho_modelo
from extrair_link import ErroLink, extrair_noticia


st.set_page_config(page_title="Notícia em análise", page_icon="📰", layout="centered")


@st.cache_resource(show_spinner=False)
def carregar_modelo(caminho):
    return Classificador(caminho)


st.markdown("""
<style>
    .block-container {max-width: 820px; padding-top: 3rem; padding-bottom: 2rem;}
    h1 {letter-spacing: -0.045em; font-weight: 750 !important;}
    [data-testid="stForm"] {border-radius: 18px; padding: 1.5rem;}
    .eyebrow {color: #167568; font-size: .78rem; font-weight: 700;
              letter-spacing: .14em; margin-bottom: .6rem;}
    .intro {color: #546173; font-size: 1.08rem; margin-bottom: 1.8rem;}
    .resultado {border-radius: 16px; padding: 1.5rem; margin-top: 1rem;}
    .resultado p {margin: 0 0 .35rem; font-size: .9rem;}
    .resultado h2 {padding: 0; margin: 0; font-size: 2rem;}
    .verdadeira {background: #e7f4ed; border: 1px solid #acd5ba; color: #165c34;}
    .falsa {background: #fff0ee; border: 1px solid #efbcb6; color: #9b3027;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">ANÁLISE DE NOTÍCIAS · BERTIMBAU</div>', unsafe_allow_html=True)
st.title("O que a notícia diz?")
st.markdown('<p class="intro">Envie um link ou cole o conteúdo para analisar a notícia.</p>',
            unsafe_allow_html=True)

modo = st.radio("Como você quer analisar?", ["Colar título e texto", "Usar um link"], horizontal=True)
titulo, texto = "", ""
noticia_extraida = None
if modo == "Colar título e texto":
    with st.form("noticia"):
        titulo = st.text_input("Título da notícia", placeholder="Digite ou cole o título", max_chars=500)
        texto = st.text_area("Texto da notícia", placeholder="Cole o conteúdo da notícia aqui…",
                             height=240, max_chars=30000)
        enviar = st.form_submit_button("Analisar notícia", type="primary", use_container_width=True)
else:
    st.caption("Use o link público da notícia. Mensagens privadas, páginas com login e alguns sites "
               "podem não permitir a leitura automática.")
    with st.form("link"):
        url = st.text_input("Link da notícia ou publicação", placeholder="https://site.com/noticia", max_chars=2048)
        enviar = st.form_submit_button("Ler link e analisar", type="primary", use_container_width=True)
    if enviar:
        try:
            with st.spinner("Lendo o título e o texto da página…"):
                noticia_extraida = extrair_noticia(url)
                titulo, texto = noticia_extraida["titulo"], noticia_extraida["texto"]
        except ErroLink as erro:
            st.warning(str(erro))
            enviar = False
        except Exception:
            st.error("Não foi possível ler o conteúdo desse link. Use a opção de colar o título e o texto.")
            enviar = False

if enviar:
    if not titulo.strip() or not texto.strip():
        st.warning("Preencha o título e o texto da notícia para continuar.")
    else:
        if noticia_extraida:
            with st.expander("Título e texto encontrados", expanded=True):
                st.text(titulo)
                st.text_area("Texto extraído da página", value=texto, height=180, disabled=True)
                st.caption("Confira se este é o conteúdo que você queria analisar. Se a extração estiver incorreta, "
                           "use a opção de colar o título e o texto.")
                if noticia_extraida["texto_limitado"]:
                    st.caption("A leitura foi limitada aos primeiros 30 mil caracteres da página.")
        try:
            with st.spinner("Analisando a notícia… O primeiro acesso pode levar alguns instantes."):
                classificador = carregar_modelo(str(caminho_modelo()))
                resultado = classificador.analisar(titulo, texto)
        except Exception:
            logging.getLogger(__name__).exception("Não foi possível executar o BERTimbau.")
            st.error("Não foi possível analisar agora. Tente novamente em instantes. "
                     "Se o problema continuar, avise o responsável pelo site.")
        else:
            verdadeira = resultado["rotulo"] == "true"
            classe = "verdadeira" if verdadeira else "falsa"
            rotulo = "Verdadeira" if verdadeira else "Falsa"
            st.markdown(
                f'<section class="resultado {classe}" role="status">'
                f'<p>Classificação do modelo</p><h2>{rotulo}</h2></section>',
                unsafe_allow_html=True,
            )
            st.markdown("#### Estimativa do modelo")
            coluna_verdadeira, coluna_falsa = st.columns(2)
            porcentagem_verdadeira = resultado["probabilidades"]["true"] * 100
            porcentagem_falsa = resultado["probabilidades"]["fake"] * 100
            coluna_verdadeira.metric("Verdadeira", f"{porcentagem_verdadeira:.1f}%".replace(".", ","))
            coluna_falsa.metric("Falsa", f"{porcentagem_falsa:.1f}%".replace(".", ","))
            st.caption("Os percentuais representam a pontuação atribuída pelo modelo a cada classe. "
                       "Eles não medem quanto da notícia foi comprovado nem garantem a chance real de ela ser verdadeira.")
            if resultado["truncado"]:
                st.caption("Esta notícia ultrapassa o tamanho de leitura do modelo. "
                           "O resultado considera o título e o início do texto.")

st.info("Este é apenas um modelo de inteligência artificial e pode errar. "
        "Não considere o resultado 100% certo, mesmo quando o percentual for alto. "
        "Confira a notícia em fontes confiáveis antes de acreditar ou compartilhar.")
