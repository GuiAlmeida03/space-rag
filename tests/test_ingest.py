from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def test_chunks_respect_size_and_overlap():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    long_text = "Palavra " * 300  # ~600 tokens
    chunks = splitter.split_text(long_text)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk) <= CHUNK_SIZE * 6  # margem de caracteres por token


def test_chunks_have_overlap():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    text = "palavra " * 200
    chunks = splitter.split_text(text)
    if len(chunks) >= 2:
        # Último token do chunk 0 deve aparecer no início do chunk 1
        end_of_first = chunks[0][-50:]
        assert any(word in chunks[1] for word in end_of_first.split()[-3:])
