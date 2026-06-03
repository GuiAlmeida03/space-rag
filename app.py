import os
import streamlit as st
from dotenv import load_dotenv
from src.retriever import retrieve
from src.chain import answer
from src.ingest import build_index
from src.config import VECTOR_STORE_DIR

load_dotenv()

st.set_page_config(
    page_title="Space RAG",
    page_icon="🛰️",
    layout="centered",
)

st.title("🛰️ Assistente da Nova Economia Espacial")
st.caption("Respostas baseadas em documentos reais da NASA, ESA, INPE e mais.")


def _ensure_index():
    index_file = os.path.join(VECTOR_STORE_DIR, "index.faiss")
    if not os.path.exists(index_file):
        with st.spinner("Construindo índice de conhecimento (primeira execução)..."):
            build_index()


_ensure_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Faça uma pergunta sobre espaço, satélites, clima..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando documentos relevantes..."):
            chunks = retrieve(prompt)
            result = answer(prompt, chunks)

        st.markdown(result["response"])

        if result["chunks"]:
            st.success(f"🟢 {len(result['chunks'])} chunk(s) relevante(s) recuperado(s)")
            with st.expander(f"📄 Ver chunks recuperados ({len(result['chunks'])})"):
                for i, chunk in enumerate(result["chunks"], 1):
                    source = os.path.basename(chunk["source"])
                    score = chunk["score"]
                    st.markdown(f"**[{i}] `{source}` — score: `{score}`**")
                    st.text(chunk["content"])
                    st.divider()
        else:
            st.error("🔴 Nenhum documento relevante encontrado (abaixo do threshold)")

    st.session_state.messages.append({"role": "assistant", "content": result["response"]})
