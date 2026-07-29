"""Unit tests for rag.retriever.Retriever."""

from rag.retriever import Retriever


def test_retrieve_returns_hits(mock_embedder, mock_indexer):
    """Retriever should encode the question and search the index."""
    retriever = Retriever(mock_embedder, mock_indexer)
    hits = retriever.retrieve("science fiction books", top_k=1)

    assert len(hits) == 1
    assert hits[0][0]["title"] == "Dune"
    assert hits[0][1] == 0.12
    mock_embedder.encode.assert_called_once_with(["science fiction books"])
    mock_indexer.search.assert_called_once()


def test_retrieve_passes_top_k(mock_embedder, mock_indexer):
    """The top_k parameter should be forwarded to the indexer."""
    retriever = Retriever(mock_embedder, mock_indexer)
    retriever.retrieve("any question", top_k=10)

    call_args = mock_indexer.search.call_args
    assert call_args[0][1] == 10  # second positional arg is top_k
