from src.config import (
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K,
    SIMILARITY_THRESHOLD, EMBEDDING_MODEL,
    GROQ_MODEL, DOCS_DIR, VECTOR_STORE_DIR
)

def test_chunk_size_larger_than_overlap():
    assert CHUNK_SIZE > CHUNK_OVERLAP

def test_top_k_positive():
    assert TOP_K > 0

def test_threshold_between_zero_and_one():
    assert 0.0 < SIMILARITY_THRESHOLD <= 1.0

def test_dirs_are_strings():
    assert isinstance(DOCS_DIR, str)
    assert isinstance(VECTOR_STORE_DIR, str)
