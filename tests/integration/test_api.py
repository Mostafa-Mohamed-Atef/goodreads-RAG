"""Integration tests for the FastAPI application."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(mock_embedder, mock_indexer, mock_generator):
    """Create a TestClient with all ML components mocked out."""
    # Patch at the import locations used by api.main
    with (
        patch("api.main.Embedder", return_value=mock_embedder),
        patch("api.main.FAISSIndexer") as MockIndexer,
        patch("api.main.Generator", return_value=mock_generator),
    ):
        mock_indexer_instance = MockIndexer.return_value
        mock_indexer_instance.load = MagicMock()
        mock_indexer_instance.is_built.return_value = True
        mock_indexer_instance.search = mock_indexer.search

        # Need to import fresh to pick up the patches
        from api.main import create_app

        app = create_app()
        # Manually set state that lifespan would set
        from rag.retriever import Retriever

        app.state.retriever = Retriever(mock_embedder, mock_indexer)
        app.state.generator = mock_generator

        yield TestClient(app, raise_server_exceptions=True)


def test_query_endpoint(client):
    """POST /api/v1/query should return answer, sources, and latency."""
    resp = client.post("/api/v1/query", json={"question": "Best sci-fi books?"})
    assert resp.status_code == 200

    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert "latency_ms" in body
    assert body["latency_ms"] >= 0
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["title"] == "Dune"


def test_query_too_short(client):
    """A question shorter than 3 characters should return 422."""
    resp = client.post("/api/v1/query", json={"question": "Hi"})
    assert resp.status_code == 422


def test_query_with_custom_top_k(client):
    """top_k should be forwarded correctly."""
    resp = client.post("/api/v1/query", json={"question": "Fantasy books?", "top_k": 3})
    assert resp.status_code == 200


def test_health_endpoint(client):
    """GET /health should always return 200."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_endpoint(client):
    """GET /ready should return 200 when the retriever is loaded."""
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"
