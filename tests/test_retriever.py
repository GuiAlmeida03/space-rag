from unittest.mock import MagicMock, patch
from src.retriever import retrieve, _filter_by_threshold
from src.config import SIMILARITY_THRESHOLD


def _make_doc(content: str, source: str):
    doc = MagicMock()
    doc.page_content = content
    doc.metadata = {"source": source}
    return doc


def test_filter_keeps_above_threshold():
    doc = _make_doc("conteúdo relevante", "nasa.txt")
    docs_and_scores = [(doc, 0.85), (_make_doc("outro", "x.txt"), 0.3)]
    result = _filter_by_threshold(docs_and_scores)
    assert len(result) == 1
    assert result[0]["score"] == 0.85


def test_filter_returns_empty_below_threshold():
    doc = _make_doc("pouco relevante", "esa.txt")
    docs_and_scores = [(doc, 0.3)]
    result = _filter_by_threshold(docs_and_scores)
    assert result == []


def test_filter_preserves_metadata():
    doc = _make_doc("starlink satélites", "starlink.txt")
    result = _filter_by_threshold([(doc, 0.9)])
    assert result[0]["source"] == "starlink.txt"
    assert result[0]["content"] == "starlink satélites"


def test_retrieve_returns_empty_for_irrelevant_query(tmp_path):
    mock_db = MagicMock()
    mock_db.similarity_search_with_relevance_scores.return_value = [
        (_make_doc("texto qualquer", "doc.txt"), 0.2)
    ]
    with patch("src.retriever._load_db", return_value=mock_db):
        result = retrieve("qual a receita do bolo de cenoura")
    assert result == []
