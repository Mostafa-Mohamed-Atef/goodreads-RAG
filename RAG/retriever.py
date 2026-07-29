"""Semantic retrieval — combines Embedder + FAISSIndexer."""

from __future__ import annotations

import logging

from rag.embedder import Embedder
from rag.indexer import FAISSIndexer

log = logging.getLogger(__name__)


class Retriever:
    """Encode a user question and retrieve the closest book entries."""

    def __init__(self, embedder: Embedder, indexer: FAISSIndexer) -> None:
        self.embedder = embedder
        self.indexer = indexer

    def retrieve(self, question: str, top_k: int) -> list[tuple[dict, float]]:
        """Return a ranked list of (metadata_dict, distance) for *question*."""
        log.debug("Retrieving: question=%r top_k=%d", question[:80], top_k)
        vec = self.embedder.encode([question])
        results = self.indexer.search(vec, top_k)
        log.debug("Retrieved %d hit(s)", len(results))
        return results
