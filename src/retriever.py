from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import (
    EMBEDDING_MODEL, VECTOR_STORE_DIR,
    TOP_K, SIMILARITY_THRESHOLD,
)


def _load_db():
    embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTOR_STORE_DIR,
        embedder,
        allow_dangerous_deserialization=True,
    )


def _filter_by_threshold(docs_and_scores: list) -> list[dict]:
    return [
        {
            "content": doc.page_content,
            "source": doc.metadata.get("source", "desconhecido"),
            "score": round(score, 4),
        }
        for doc, score in docs_and_scores
        if score >= SIMILARITY_THRESHOLD
    ]


def retrieve(query: str) -> list[dict]:
    db = _load_db()
    docs_and_scores = db.similarity_search_with_relevance_scores(query, k=TOP_K)
    return _filter_by_threshold(docs_and_scores)
