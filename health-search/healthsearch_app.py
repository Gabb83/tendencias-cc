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