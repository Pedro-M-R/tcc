"""Execute a partir da raiz: python -m streamlit run web/app.py."""

import logging
from html import escape
from pathlib import Path

import streamlit as st

from inferencia import Classificador, modelos_do_site
from extrair_link import ErroLink, extrair_noticia
from apresentacao import formatar_probabilidade
from triagem_noticias import INDICIOS, avaliar_entrada


st.set_page_config(page_title="Notícia em análise", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner=False)
def carregar_modelo(caminho, tipo):
    return Classificador(caminho, tipo)


def mostrar_checagem(checagem):
    rotulo = "A fonte contesta a alegação" if checagem["rotulo"] == "fake" else "A fonte sustenta a alegação"
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
               "As previsões dos modelos abaixo se referem ao texto da checagem.")


def mostrar_resultado(resultado, nome="TF-IDF + SVM"):
    st.subheader(nome)
    rotulo = resultado["rotulo"]
    if rotulo == "inconclusivo":
        st.warning("Análise inconclusiva")
        st.write(resultado["motivo"])
        return
    classe = "verdadeira" if rotulo == "true" else "falsa"
    st.markdown(
        f'<section class="resultado {classe}" role="status">'
        f'<div><p>Indícios identificados pelo modelo</p><h2>{INDICIOS[rotulo]}</h2>'
        '<p class="nota">Padrões do texto; os fatos não foram verificados.</p></div></section>',
        unsafe_allow_html=True,
    )
    st.caption("Esses indícios são padrões aprendidos na base de treinamento. "
               "Confira a fonte, a data e outras coberturas antes de concluir.")
    with st.expander("Detalhes técnicos do modelo"):
        if "margem_svm" in resultado:
            st.metric("Margem de decisão do SVM", f"{resultado['margem_svm']:+.4f}".replace(".", ","))
            st.caption("Margem positiva favorece indícios favoráveis; negativa, indícios contrários. "
                       "Não é uma probabilidade de veracidade.")
        else:
            st.caption("Escores calibrados na amostra de avaliação; não comprovam os fatos." if resultado.get("calibrado")
                       else "Escores softmax não calibrados; não representam a chance de a notícia ser verdadeira.")
            a, b = st.columns(2)
            a.metric("Escore técnico · indícios favoráveis", formatar_probabilidade(resultado["probabilidades"]["true"]))
            b.metric("Escore técnico · indícios contrários", formatar_probabilidade(resultado["probabilidades"]["fake"]))
            if resultado.get("truncado"):
                st.caption("Esta notícia ultrapassa o tamanho de leitura do modelo. "
                           "O resultado considera o título e o início do texto.")
            if resultado.get("trecho_lido"):
                st.caption(f"Foram lidos {resultado['tokens_lidos']} de {resultado['tokens_totais']} tokens.")
                st.text(resultado["trecho_lido"])


def mostrar_comparacao(resultados):
    if len(resultados) > 1:
        if len({r["rotulo"] for _, r in resultados}) == 1:
            st.info("Os modelos apontam os mesmos indícios; isso não comprova os fatos.")
        else:
            st.warning("Os modelos discordam. Confira os resultados de cada um e verifique as fontes.")
    for modelo, resultado in resultados:
        mostrar_resultado(resultado, modelo["nome"])
        st.caption(f"Modelo utilizado: {Path(modelo['caminho']).name}")


st.markdown(Path(__file__).with_name("estilo.css").read_text(encoding="utf-8-sig"), unsafe_allow_html=True)
st.markdown('''
<div class="marca"><span class="marca-simbolo" aria-hidden="true">n.</span>
<span>notícia em análise.</span><small>Um olhar a mais sobre a informação</small></div>
<section class="hero">
  <div><p class="sobretitulo">LEITURA CRÍTICA · INTELIGÊNCIA ARTIFICIAL</p>
  <h1>Antes de compartilhar,<br><span>olhe mais de perto.</span></h1>
  <p class="descricao">Um link ou um texto. Uma nova perspectiva sobre a notícia.
  Explore os indícios identificados pela inteligência artificial e confira as fontes.</p></div>
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
    modelos = modelos_do_site()
    opcoes = {"Comparar os dois modelos": modelos} if len(modelos) > 1 else {}
    opcoes.update({m["nome"]: [m] for m in modelos})
    selecao = st.selectbox("Modelos para análise", list(opcoes))
    if len(modelos) == 1 and modelos[0]["tipo"] == "svm_leve":
        st.caption("Análise com TF-IDF + SVM, executada em CPU.")
    elif len(modelos) < 2 and modelos[0]["tipo"] not in {"local", "svm", "svm_leve"}:
        st.warning("O SVM não está disponível nesta instalação. Inclua modelos/svm/modelo.joblib "
                   "para comparar os dois modelos, como no testar_noticias.")
    if modo == "Colar título e texto":
        with st.form("noticia"):
            titulo = st.text_input("Título da notícia (opcional)", placeholder="Qual é a manchete?", max_chars=500)
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
        if not (titulo.strip() or texto.strip()):
            st.warning("Cole o texto da notícia ou informe um título para continuar.")
        elif motivo := avaliar_entrada(titulo, texto):
            st.warning("Análise inconclusiva")
            st.write(motivo)
            st.caption("Nenhum escore de veracidade foi atribuído. A triagem usa regras de tamanho e conteúdo e também pode falhar.")
        else:
            checagem = noticia_extraida.get("checagem") if noticia_extraida else None
            if checagem:
                mostrar_checagem(checagem)
            resultados = []
            with st.spinner("Analisando a notícia… O primeiro acesso pode levar alguns instantes."):
                for modelo in opcoes[selecao]:
                    try:
                        classificador = carregar_modelo(modelo["caminho"], modelo["tipo"])
                        resultados.append((modelo, classificador.analisar(titulo, texto)))
                    except Exception:
                        logging.getLogger(__name__).exception("Não foi possível executar %s.", modelo["nome"])
                        st.error(f"Não foi possível analisar com {modelo['nome']}. "
                                 "Tente novamente em instantes ou avise o responsável pelo site.")
            if len(resultados) != len(opcoes[selecao]) and resultados:
                st.warning("Comparação incompleta: um dos modelos não pôde ser executado.")
            if resultados:
                if checagem:
                    if any(r["rotulo"] != checagem["rotulo"] for _, r in resultados):
                        st.warning("Uma previsão dos modelos diverge da conclusão da fonte. "
                                   "Os modelos avaliaram o texto da checagem e não verificaram os fatos da alegação.")
                    with st.container(border=True):
                        st.caption("Indícios dos modelos para o texto da checagem")
                        mostrar_comparacao(resultados)
                else:
                    mostrar_comparacao(resultados)
    else:
        st.markdown('''<div class="vazio"><div class="vazio-icone" aria-hidden="true"><span>◎</span></div>
        <h3>Todo resultado começa<br>com uma boa leitura.</h3>
        <p>Envie uma notícia para análise. Os indícios encontrados ou uma orientação para completar o texto vão aparecer aqui.</p></div>
        <div class="vazio-legenda"><span><i class="ponto"></i>Indícios favoráveis</span>
        <span><i class="ponto coral"></i>Indícios contrários</span></div>''', unsafe_allow_html=True)

with st.container(key="aviso_modelo"):
    st.info("Este é apenas um modelo de inteligência artificial e pode errar. "
            "Não considere o resultado 100% certo. A análise pode ser inconclusiva quando falta contexto ou suporte do modelo. "
            "Confira a notícia em fontes confiáveis antes de acreditar ou compartilhar.")
st.markdown('<div class="rodape"><span>Notícia em análise · Projeto acadêmico</span>'
            '<span>TF-IDF + SVM · Revisão 2026.10.08-svm</span></div>', unsafe_allow_html=True)
