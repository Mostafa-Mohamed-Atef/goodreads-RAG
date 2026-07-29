"""Shared test fixtures — mock ML components for fast, deterministic tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


@pytest.fixture
def mock_embedder():
    """Mock Embedder that returns a fixed random vector."""
    emb = MagicMock()
    emb.encode.return_value = np.random.rand(1, 384).astype("float32")
    return emb


@pytest.fixture
def mock_indexer():
    """Mock FAISSIndexer that returns a fixed search result."""
    idx = MagicMock()
    idx.search.return_value = [
        (
            {
                "title": "Dune",
                "author": "Frank Herbert",
                "text": "A sci-fi epic about politics and desert planets.",
                "rating": "4.5",
            },
            0.12,
        ),
    ]
    idx.is_built.return_value = True
    return idx


@pytest.fixture
def mock_generator():
    """Mock Generator that returns a canned answer."""
    gen = MagicMock()
    gen.answer.return_value = "Dune is a great sci-fi novel by Frank Herbert."
    return gen
