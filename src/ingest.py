import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL,
    DOCS_DIR, VECTOR_STORE_DIR,
)


def _load_documents(docs_dir: str):
    docs = []
    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        try:
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filename.endswith((".txt", ".md")):
                loader = TextLoader(filepath, encoding="utf-8")
            else:
                continue
            docs.extend(loader.load())
        except Exception as e:
            print(f"[ingest] Ignorando {filename}: {e}")
    return docs


def build_index():
    print("[ingest] Carregando documentos...")
    docs = _load_documents(DOCS_DIR)
    if not docs:
        raise ValueError(f"Nenhum documento encontrado em {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] {len(docs)} documentos -> {len(chunks)} chunks")

    print("[ingest] Gerando embeddings (primeira execução pode demorar)...")
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    db = FAISS.from_documents(chunks, embedder)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    db.save_local(VECTOR_STORE_DIR)
    print(f"[ingest] Índice salvo em {VECTOR_STORE_DIR}")


if __name__ == "__main__":
    build_index()
