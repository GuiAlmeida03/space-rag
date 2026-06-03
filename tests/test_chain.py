from unittest.mock import MagicMock, patch
from src.chain import answer, _build_context, OUT_OF_SCOPE_MSG


def _make_chunk(content: str, source: str, score: float) -> dict:
    return {"content": content, "source": source, "score": score}


def test_out_of_scope_when_no_chunks():
    result = answer("pergunta fora do escopo", chunks=[])
    assert result["response"] == OUT_OF_SCOPE_MSG
    assert result["chunks"] == []


def test_out_of_scope_does_not_call_llm():
    with patch("src.chain.ChatGroq") as mock_llm:
        answer("pergunta irrelevante", chunks=[])
    mock_llm.assert_not_called()


def test_build_context_formats_correctly():
    chunks = [
        _make_chunk("Artemis vai à Lua", "nasa.txt", 0.92),
        _make_chunk("ESA monitora queimadas", "esa.txt", 0.81),
    ]
    ctx = _build_context(chunks)
    assert "nasa.txt" in ctx
    assert "Artemis vai à Lua" in ctx
    assert "0.92" in ctx


def test_answer_returns_response_and_chunks():
    chunks = [_make_chunk("Starlink tem 6000 satélites", "starlink.txt", 0.88)]
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "O Starlink possui mais de 6000 satélites."
    with patch("src.chain._build_chain", return_value=mock_chain):
        result = answer("quantos satélites tem o Starlink", chunks=chunks)
    assert "Starlink" in result["response"]
    assert result["chunks"] == chunks
