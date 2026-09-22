"""Execute a partir da raiz: python -m streamlit run web/app.py."""

import logging
from html import escape
from pathlib import Path

import streamlit as st

from inferencia import Classificador, caminho_modelo
from extrair_link import ErroLink, extrair_noticia
from apresentacao import formatar_probabilidade


st.set_page_config(page_title="Notícia em análise", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner=False)
def carregar_modelo(caminho):
    return Classificador(caminho)


def mostrar_checagem(checagem):
    rotulo = "Alegação falsa" if checagem["rotulo"] == "fake" else "Alegação verdadeira"
    classe = "falsa" if checagem["rotulo"] == "fake" else "verdadeira"
    st.markdown(
        f'<section class="resultado {classe}" role="status"><div>'
        f'<p>Conclusão publicada pelo Boatos.org</p><h2>{rotulo}</h2>'
        '</div></section>', unsafe_allow_html=True,
    )
    st.caption("Alegação examinada pela fonte:")
    st.text(checagem["alegacao"])
    st.link_button("Ler a checagem no Boatos.org", checagem["url"])
    st.caption("O resultado acima foi extraído do selo de conclusão da página. "
               "A pontuação do BERTimbau, quando disponível abaixo, se refere ao texto da checagem.")


def mostrar_resultado(resultado):
    verdadeira = resultado["rotulo"] == "true"
    classe = "verdadeira" if verdadeira else "falsa"
    rotulo = "Verdadeira" if verdadeira else "Falsa"
    prob_true = resultado["probabilidades"]["true"] * 100
    prob_fake = resultado["probabilidades"]["fake"] * 100
    destaque = prob_true if verdadeira else prob_fake
    percentual = escape(formatar_probabilidade(resultado["probabilidades"][resultado["rotulo"]]))
    st.markdown(
        f'<section class="resultado {classe}" role="status">'
        f'<div><p>Classificação do modelo</p><h2>{rotulo}</h2>'
        '<p class="nota">Resultado baseado nos padrões do texto.</p></div>'
        f'<div class="anel" style="--angulo:{destaque * 3.6:.3f}deg" '
        f'role="img" aria-label="Pontuação para {rotulo}: {percentual}">'
        f'<div class="anel-centro"><strong>{percentual}</strong><small>PARA {rotulo.upper()}</small></div>'
        '</div></section>', unsafe_allow_html=True,
    )
    st.caption("Distribuição da pontuação entre as duas classes")
    st.markdown(
        '<div class="distribuicao" aria-hidden="true">'
        f'<span class="barra-true" style="width:{prob_true:.4f}%"></span>'
        f'<span class="barra-fake" style="width:{prob_fake:.4f}%"></span></div>',
        unsafe_allow_html=True,
    )
    coluna_true, coluna_fake = st.columns(2)
    coluna_true.metric("Verdadeira", formatar_probabilidade(resultado["probabilidades"]["true"]))
    coluna_fake.metric("Falsa", formatar_probabilidade(resultado["probabilidades"]["fake"]))
    st.caption("Os percentuais representam a pontuação atribuída pelo modelo a cada classe. "
               "Eles não medem quanto da notícia foi comprovado nem garantem a chance real de ela ser verdadeira.")
    if resultado["truncado"]:
        st.caption("Esta notícia ultrapassa o tamanho de leitura do modelo. "
                   "O resultado considera o título e o início do texto.")
    if resultado.get("trecho_lido"):
        with st.expander("Conferir o que o modelo analisou"):
            st.caption(f"Foram lidos {resultado['tokens_lidos']} de {resultado['tokens_totais']} tokens "
                       "(partes de palavras, incluindo marcadores do modelo).")
            st.text(resultado["trecho_lido"])
            st.caption("Trecho reconstruído a partir da entrada do modelo; os espaços podem diferir do original.")
            st.caption("Pontuações com mais casas decimais, sem calibração de confiança:")
            st.json(resultado["probabilidades"])


st.markdown(Path(__file__).with_name("estilo.css").read_text(encoding="utf-8-sig"), unsafe_allow_html=True)
st.markdown('''
<div class="marca"><span class="marca-simbolo" aria-hidden="true">n.</span>
<span>notícia em análise.</span><small>Um olhar a mais sobre a informação</small></div>
<section class="hero">
  <div><p class="sobretitulo">LEITURA CRÍTICA · INTELIGÊNCIA ARTIFICIAL</p>
  <h1>Antes de compartilhar,<br><span>olhe mais de perto.</span></h1>
  <p class="descricao">Um link ou um texto. Uma nova perspectiva sobre a notícia.
  Explore a classificação da inteligência artificial e tire suas próprias conclusões.</p></div>
  <div class="ilustracao" aria-hidden="true"><div class="orbita"></div>
    <div class="folha"><div class="folha-topo"></div><div class="folha-linha titulo"></div>
    <div class="folha-linha titulo curta"></div><div class="folha-linha"></div>
    <div class="folha-linha"></div><div class="folha-linha curta"></div></div>
    <div class="lupa"><span>?</span></div>
  </div>
</section>
<div class="fluxo"><span><b>01</b> Envie a notícia</span><span><b>02</b> Explore o resultado</span>
<span><b>03</b> Confira as fontes</span></div>
''', unsafe_allow_html=True)

coluna_entrada, coluna_resultado = st.columns([1.15, 1], gap="large")
titulo, texto = "", ""
noticia_extraida = None
erro_link = None

with coluna_entrada, st.container(border=True, key="painel_entrada"):
    st.markdown('<div class="painel-cabecalho"><span class="numero">01</span><h3>Qual é a notícia?</h3></div>'
                '<p class="painel-subtitulo">Escolha como enviar o conteúdo para análise.</p>', unsafe_allow_html=True)
    modo = st.radio("Como você quer analisar?", ["Colar título e texto", "Usar um link"],
                    horizontal=True, label_visibility="collapsed")
    if modo == "Colar título e texto":
        with st.form("noticia"):
            titulo = st.text_input("Título da notícia", placeholder="Qual é a manchete?", max_chars=500)
            texto = st.text_area("Texto da notícia", placeholder="Cole o conteúdo que você quer analisar…",
                                 height=220, max_chars=30000)
            enviar = st.form_submit_button("Analisar notícia", type="primary", use_container_width=True)
    else:
        with st.form("link"):
            url = st.text_input("Link da notícia ou publicação", placeholder="https://site.com/noticia", max_chars=2048)
            st.caption("Vamos buscar o título e o texto da página para você.")
            enviar = st.form_submit_button("Ler link e analisar", type="primary", use_container_width=True)
        st.caption("Use um link público. Mensagens privadas, páginas com login e alguns sites "
                   "podem não permitir a leitura automática.")
        if enviar:
            try:
                with st.spinner("Lendo o título e o texto da página…"):
                    noticia_extraida = extrair_noticia(url)
                    titulo, texto = noticia_extraida["titulo"], noticia_extraida["texto"]
            except ErroLink as erro:
                erro_link = str(erro)
            except Exception:
                erro_link = "Não foi possível ler o conteúdo desse link. Use a opção de colar o título e o texto."
    if noticia_extraida:
        with st.expander("Título e texto encontrados", expanded=False):
            st.text(titulo)
            st.text_area("Texto extraído da página", value=texto, height=180, disabled=True)
            st.caption("Confira se este é o conteúdo que você queria analisar. Se a extração estiver incorreta, "
                       "use a opção de colar o título e o texto.")
            if noticia_extraida["texto_limitado"]:
                st.caption("A leitura foi limitada aos primeiros 30 mil caracteres da página.")

with coluna_resultado, st.container(border=True, key="painel_resultado"):
    st.markdown('<div class="painel-cabecalho"><span class="numero">02</span><h3>Sua análise</h3></div>'
                '<p class="painel-subtitulo">Uma estimativa para ajudar na sua leitura crítica.</p>', unsafe_allow_html=True)
    if erro_link:
        st.warning(erro_link)
    elif enviar:
        if not titulo.strip() or not texto.strip():
            st.warning("Preencha o título e o texto da notícia para continuar.")
        else:
            checagem = noticia_extraida.get("checagem") if noticia_extraida else None
            if checagem:
                mostrar_checagem(checagem)
            try:
                with st.spinner("Analisando a notícia… O primeiro acesso pode levar alguns instantes."):
                    classificador = carregar_modelo(str(caminho_modelo()))
                    resultado = classificador.analisar(titulo, texto)
            except Exception:
                logging.getLogger(__name__).exception("Não foi possível executar o BERTimbau.")
                st.error("Não foi possível analisar agora. Tente novamente em instantes. "
                         "Se o problema continuar, avise o responsável pelo site.")
            else:
                if checagem:
                    if resultado["rotulo"] != checagem["rotulo"]:
                        st.warning("A previsão do BERTimbau diverge da conclusão da fonte. "
                                   "O modelo avaliou o texto da checagem e não verificou os fatos da alegação.")
                    with st.expander("Ver previsão do BERTimbau para o texto da checagem"):
                        mostrar_resultado(resultado)
                else:
                    mostrar_resultado(resultado)
    else:
        st.markdown('''<div class="vazio"><div class="vazio-icone" aria-hidden="true"><span>◎</span></div>
        <h3>Todo resultado começa<br>com uma boa leitura.</h3>
        <p>Envie uma notícia para análise. A classificação e as pontuações do modelo vão aparecer aqui.</p></div>
        <div class="vazio-legenda"><span><i class="ponto"></i>Verdadeira</span>
        <span><i class="ponto coral"></i>Falsa</span></div>''', unsafe_allow_html=True)

with st.container(key="aviso_modelo"):
    st.info("Este é apenas um modelo de inteligência artificial e pode errar. "
            "Não considere o resultado 100% certo, mesmo quando o percentual for alto. "
            "Confira a notícia em fontes confiáveis antes de acreditar ou compartilhar.")
st.markdown('<div class="rodape"><span>Notícia em análise · Projeto acadêmico</span>'
            '<span>BERTimbau · Revisão 2026.09.22-3</span></div>', unsafe_allow_html=True)
