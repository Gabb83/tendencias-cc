import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from corpus import CORPUS_MEDICO

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

STOP_WORDS_PT = set(stopwords.words('portuguese'))


# fase 1: ingestão e pre-processamento
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

st.title("HealthSearch - Motor de Busca Híbrido")
st.header("Fase 1: Ingestão e Pré-processamento do Corpus Médico")

st.dataframe(df_corpus[['id', 'titulo', 'conteudo', 'tokens']], use_container_width=True)


# fase 2: motor lexico com parametros interativos
st.title("🏥 HealthSearch — Motor de Busca Híbrido")
st.sidebar.header("⚙️ Parâmetros BM25 (Fase 2)")
k1 = st.sidebar.slider("Saturação de Frequência (k1)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
b = st.sidebar.slider("Normalização pelo Comprimento (b)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)


corpus_tokens = df_corpus['tokens'].tolist()
bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
query = st.text_input("🔍 Digite a sua consulta clínica/médica:", value="CÓD-ECG-12D infarto")
tab1, tab2 = st.tabs(["📄 Fase 1: Corpus e Tokens", "📊 Fase 2: Ranking BM25"])

with tab1:
    st.header("Fase 1: Ingestão e Pré-processamento")
    st.dataframe(df_corpus[['id', 'titulo', 'conteudo', 'tokens']], use_container_width=True)

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
        st.info("Insira um termo na caixa de pesquisa para calcular as pontuações BM25.")