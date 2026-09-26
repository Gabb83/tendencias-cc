import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer


DATA_PATH = Path(__file__).parent / "data" / "manifestacoes.json"
MODELOS = [
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet-base-v2",
    "distiluse-base-multilingual-cased-v2",
]
CORES_CATEGORIA = {
    "infraestrutura": "#187f78",
    "saúde": "#d26a50",
    "segurança": "#405a7a",
    "educação": "#bd8a26",
    "meio ambiente": "#708d48",
}


@st.cache_data
def carregar_manifestacoes() -> pd.DataFrame:
    with DATA_PATH.open("r", encoding="utf-8") as arquivo:
        registros = json.load(arquivo)
    dados = pd.DataFrame(registros)
    colunas_obrigatorias = {"id", "data", "categoria_oficial", "texto"}
    if not colunas_obrigatorias.issubset(dados.columns):
        raise ValueError("A base precisa conter id, data, categoria_oficial e texto.")
    return dados


@st.cache_resource
def carregar_modelo(nome: str) -> SentenceTransformer:
    return SentenceTransformer(nome)


@st.cache_data(show_spinner="Calculando embeddings das manifestações...")
def gerar_embeddings(nome_modelo: str, textos: tuple[str, ...]) -> np.ndarray:
    modelo = carregar_modelo(nome_modelo)
    return modelo.encode(
        list(textos),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def dividir_texto(texto: str, estrategia: str, tamanho: int, sobreposicao: int) -> list[str]:
    texto = texto.strip()
    if not texto:
        return []

    if estrategia == "Recursivo (LangChain)":
        divisor = RecursiveCharacterTextSplitter(
            chunk_size=tamanho,
            chunk_overlap=sobreposicao,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return divisor.split_text(texto)

    if estrategia == "Janela fixa":
        passo = tamanho - sobreposicao
        return [texto[inicio : inicio + tamanho] for inicio in range(0, len(texto), passo)]

    frases = re.split(r"(?<=[.!?])\s+", texto)
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=tamanho,
        chunk_overlap=sobreposicao,
        separators=[" ", ""],
    )
    unidades = [
        parte
        for frase in frases
        for parte in (divisor.split_text(frase) if len(frase) > tamanho else [frase])
    ]
    chunks: list[str] = []
    atual = ""
    for unidade in unidades:
        candidato = f"{atual} {unidade}".strip()
        if atual and len(candidato) > tamanho:
            chunks.append(atual)
            sobra = atual[-sobreposicao:] if sobreposicao else ""
            limite_sobra = max(0, tamanho - len(unidade) - 1)
            atual = f"{sobra[-limite_sobra:]} {unidade}".strip() if limite_sobra else unidade
        else:
            atual = candidato
    if atual:
        chunks.append(atual)
    return chunks


st.set_page_config(
    page_title="Ouvidoria Inteligente",
    page_icon="📬",
    layout="wide",
)
st.markdown(
    """
    <style>
    .block-container {max-width: 1440px; padding-top: 2rem;}
    [data-testid="stMetric"] {background: #f3f6f2; padding: 14px 18px; border-radius: 6px;}
    .score {font-size: 1.05rem; font-weight: 700;}
    .score-high {color: #187f58;}
    .score-medium {color: #a66a00;}
    .score-low {color: #b2443b;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Ouvidoria Inteligente")
st.caption("Busca semântica, análise exploratória e chunking de manifestações cidadãs.")

with st.sidebar:
    st.header("Configuração")
    modelo_selecionado = st.selectbox("Modelo de embedding", MODELOS, index=0)

try:
    df = carregar_manifestacoes()
except (OSError, ValueError, json.JSONDecodeError) as erro:
    st.error(f"Não foi possível carregar a base de manifestações: {erro}")
    st.stop()

with st.sidebar:
    top_k = st.slider("Manifestações no resultado (top-k)", 1, len(df), min(5, len(df)))
    st.caption(f"Base carregada: {len(df)} manifestações")

try:
    with st.spinner(f"Preparando o modelo {modelo_selecionado}..."):
        modelo = carregar_modelo(modelo_selecionado)
        textos_base = tuple(df["texto"].astype(str))
        embeddings = gerar_embeddings(modelo_selecionado, textos_base)
except Exception as erro:
    st.error(
        "Não foi possível carregar o modelo ou gerar os embeddings. "
        "Verifique a conexão com a internet, as dependências e o espaço em disco."
    )
    st.exception(erro)
    st.stop()

aba_busca, aba_base, aba_vetores, aba_chunking = st.tabs(
    ["Busca Semântica", "Base Completa", "Espaço Vetorial", "Chunking"]
)

with aba_busca:
    st.subheader("Encontre manifestações relacionadas")
    with st.form("form_busca"):
        consulta = st.text_area(
            "Descreva o problema",
            placeholder="Ex.: buracos e falta de iluminação na rua do meu bairro",
            height=110,
        )
        buscar = st.form_submit_button("Buscar manifestações", type="primary")

    if buscar:
        if not consulta.strip():
            st.warning("Digite uma descrição para iniciar a busca.")
        else:
            vetor_consulta = modelo.encode(
                [consulta.strip()], convert_to_numpy=True, normalize_embeddings=True
            )[0]
            scores = embeddings @ vetor_consulta
            indices = np.argsort(scores)[::-1][:top_k]
            st.caption("Similaridade de cosseno. Verde: > 0,7; âmbar: > 0,5; vermelho: demais.")
            for posicao, indice in enumerate(indices, start=1):
                registro = df.iloc[indice]
                score = float(scores[indice])
                classe = "score-high" if score > 0.7 else "score-medium" if score > 0.5 else "score-low"
                with st.container(border=True):
                    esquerda, direita = st.columns([5, 1])
                    with esquerda:
                        st.markdown(
                            f"**{posicao}. {registro['id']}** · {registro['categoria_oficial']} · {registro['data']}"
                        )
                        st.write(registro["texto"])
                    with direita:
                        st.markdown(
                            f'<p class="score {classe}">{score:.3f}</p>',
                            unsafe_allow_html=True,
                        )

with aba_base:
    st.subheader("Manifestações registradas")
    metricas = st.columns(3)
    metricas[0].metric("Registros", len(df))
    metricas[1].metric("Categorias", df["categoria_oficial"].nunique())
    metricas[2].metric("Modelo", modelo_selecionado.split("-")[0])
    st.dataframe(df, use_container_width=True, hide_index=True)

    if st.button("Gerar matriz de similaridade", key="gerar_matriz"):
        matriz = embeddings @ embeddings.T
        rotulos = df["id"].tolist()
        figura = px.imshow(
            matriz,
            x=rotulos,
            y=rotulos,
            color_continuous_scale="YlGnBu",
            zmin=0,
            zmax=1,
            aspect="auto",
            labels={"x": "Manifestação", "y": "Manifestação", "color": "Similaridade"},
        )
        figura.update_layout(height=760, margin=dict(l=10, r=10, t=25, b=10))
        st.plotly_chart(figura, use_container_width=True)

with aba_vetores:
    st.subheader("Distribuição semântica das manifestações")
    metodo = st.radio("Redução de dimensionalidade", ["PCA", "t-SNE"], horizontal=True)
    if metodo == "PCA":
        coordenadas = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    else:
        perplexidade = min(30, len(df) - 1)
        coordenadas = TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexidade,
            init="pca",
        ).fit_transform(embeddings)

    dados_plot = df.copy()
    dados_plot["Dimensão 1"] = coordenadas[:, 0]
    dados_plot["Dimensão 2"] = coordenadas[:, 1]
    figura = px.scatter(
        dados_plot,
        x="Dimensão 1",
        y="Dimensão 2",
        color="categoria_oficial",
        color_discrete_map=CORES_CATEGORIA,
        hover_name="id",
        hover_data={"texto": True, "categoria_oficial": True, "Dimensão 1": False, "Dimensão 2": False},
        labels={"categoria_oficial": "Categoria oficial"},
    )
    figura.update_traces(marker=dict(size=11, opacity=0.82, line=dict(width=0.7, color="white")))
    figura.update_layout(height=560, legend_title_text="Categoria oficial")
    st.plotly_chart(figura, use_container_width=True)
    st.markdown("**Reflexão do aluno**")
    st.text_area(
        "Os clusters semânticos coincidem com as categorias oficiais? Justifique com exemplos.",
        key="reflexao_clusters",
        placeholder="Compare a proximidade dos pontos com as cores das categorias e registre casos de sobreposição.",
        height=110,
    )

with aba_chunking:
    st.subheader("Divida uma manifestação longa")
    texto_longo = st.text_area(
        "Texto da manifestação",
        height=180,
        placeholder="Cole aqui uma manifestação longa para explorar diferentes estratégias de chunking.",
        key="texto_chunking",
    )
    controles = st.columns([2, 1, 1])
    estrategia = controles[0].selectbox(
        "Estratégia",
        ["Recursivo (LangChain)", "Janela fixa", "Por frases"],
    )
    tamanho = controles[1].number_input("Tamanho do chunk (caracteres)", 50, 2000, 300, step=50)
    sobreposicao = controles[2].number_input(
        "Sobreposição (caracteres)", 0, int(tamanho) - 1, min(40, int(tamanho) - 1), step=10
    )

    if st.button("Gerar chunks e embeddings", type="primary", key="gerar_chunks"):
        if not texto_longo.strip():
            st.warning("Cole uma manifestação antes de gerar os chunks.")
        else:
            chunks = dividir_texto(texto_longo, estrategia, int(tamanho), int(sobreposicao))
            vetores_chunks = modelo.encode(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            st.success(f"{len(chunks)} chunks gerados com {estrategia.lower()}.")
            if len(chunks) > 1:
                coords_chunks = PCA(n_components=2, random_state=42).fit_transform(vetores_chunks)
                grafico_chunks = pd.DataFrame(
                    {
                        "Componente 1": coords_chunks[:, 0],
                        "Componente 2": coords_chunks[:, 1],
                        "Chunk": [f"Chunk {indice + 1}" for indice in range(len(chunks))],
                        "Texto": chunks,
                    }
                )
                st.plotly_chart(
                    px.scatter(
                        grafico_chunks,
                        x="Componente 1",
                        y="Componente 2",
                        text="Chunk",
                        hover_data={"Texto": True, "Componente 1": False, "Componente 2": False},
                        title="Projeção PCA dos embeddings dos chunks",
                    ).update_traces(marker=dict(size=13, color="#187f78")),
                    use_container_width=True,
                )

            registros_chunks = []
            for indice, (chunk, vetor) in enumerate(zip(chunks, vetores_chunks)):
                previa = ", ".join(f"{valor:.3f}" for valor in vetor[:8])
                registros_chunks.append(
                    {
                        "Chunk": indice + 1,
                        "Caracteres": len(chunk),
                        "Texto": chunk,
                        "Embedding (8 primeiras dimensões)": f"[{previa}, ...]",
                        "Norma": f"{np.linalg.norm(vetor):.3f}",
                    }
                )
            st.dataframe(pd.DataFrame(registros_chunks), use_container_width=True, hide_index=True)