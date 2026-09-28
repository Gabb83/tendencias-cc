import re
import pandas as pd
import nltk
import streamlit as st
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from corpus import CORPUS_MEDICO

st.set_page_config(page_title="HealthSearch", page_icon="🏥", layout="wide")

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOP_WORDS_PT = set(stopwords.words('portuguese'))


# fase 1: ingestao e pre-processamento
def preprocessar_texto(texto: str) -> list:
    """
    Realiza a limpeza e tokenização do texto:
    - Converte para minúsculas
    - Remove caracteres especiais (mantendo hífens e acentos)
    - Elimina stopwords em português
    """
    texto_lc = texto.lower()
    texto_limpo = re.sub(r'[^a-zA-Z0-9\sçáàâãéèêíóòôõúü-]', '', texto_lc)
    tokens = texto_limpo.split()
    return [token for token in tokens if token not in STOP_WORDS_PT]

df_corpus = pd.DataFrame(CORPUS_MEDICO)
df_corpus['tokens'] = df_corpus['conteudo'].apply(preprocessar_texto)


# fase 3: carregamento dos embeddings
@st.cache_resource
def carregar_modelo_semantico():
    return SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

modelo_semantico = carregar_modelo_semantico()
doc_embeddings = modelo_semantico.encode(df_corpus['conteudo'].tolist())
st.title("🏥 HealthSearch — Motor de Busca Híbrido")


# fase 2: parametros BM25
st.sidebar.header("⚙️ Parâmetros BM25 (Fase 2)")
k1 = st.sidebar.slider("Saturação de Frequência (k1)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
b = st.sidebar.slider("Normalização pelo Comprimento (b)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

corpus_tokens = df_corpus['tokens'].tolist()
bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
query = st.text_input("🔍 Digite a sua consulta clínica/médica:", value="CÓD-ECG-12D infarto")

tab1, tab2, tab3 = st.tabs([
    "📄 Fase 1: Corpus e Tokens", 
    "📊 Fase 2: Ranking BM25", 
    "🧠 Fase 3: Busca Semântica"
])

# fase 1
with tab1:
    st.header("Fase 1: Ingestão e Pré-processamento do Corpus Médico")
    st.dataframe(df_corpus[['id', 'titulo', 'conteudo', 'tokens']], use_container_width=True)

# fase 2
with tab2:
    st.header("Fase 2: Resultados do Motor Léxico (BM25)")
    
    if query.strip():
        query_tokens = preprocessar_texto(query)
        st.write(f"**Tokens da consulta:** `{query_tokens}`")
        
        doc_scores = bm25.get_scores(query_tokens)
        
        df_bm25 = df_corpus.copy()
        df_bm25['score_bm25'] = doc_scores
        df_bm25 = df_bm25.sort_values(by='score_bm25', ascending=False)
        df_bm25['rank_bm25'] = range(1, len(df_bm25) + 1)
        
        st.dataframe(
            df_bm25[['rank_bm25', 'id', 'titulo', 'score_bm25', 'conteudo']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Insira um termo na caixa de pesquisa para calcular o BM25.")

# fase 3
with tab3:
    st.header("Fase 3: Resultados Semânticos (Embeddings + Similaridade de Cosseno)")
    
    if query.strip():
        query_embedding = modelo_semantico.encode([query])
        similaridades = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        df_semantico = df_corpus.copy()
        df_semantico['score_semantico'] = similaridades
        df_semantico = df_semantico.sort_values(by='score_semantico', ascending=False)
        df_semantico['rank_semantico'] = range(1, len(df_semantico) + 1)
        
        st.dataframe(
            df_semantico[['rank_semantico', 'id', 'titulo', 'score_semantico', 'conteudo']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Insira um termo na caixa de pesquisa para calcular a similaridade semântica.")