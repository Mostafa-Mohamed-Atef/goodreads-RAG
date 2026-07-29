"""SentenceTransformer wrapper — single responsibility: text → vectors."""

from __future__ import annotations

import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import settings

log = logging.getLogger(__name__)


class Embedder:
    """Thin wrapper around SentenceTransformer for encoding text."""

    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or settings.embedding_model
        log.info("Loading SentenceTransformer model: %s", name)
        t0 = time.perf_counter()
        self.model = SentenceTransformer(name)
        elapsed = time.perf_counter() - t0
        log.info("Model loaded in %.2fs", elapsed)

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of strings into a numpy array of embeddings."""
        log.debug("Encoding %d text(s)…", len(texts))
        t0 = time.perf_counter()
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        elapsed = time.perf_counter() - t0
        log.debug("Encoded %d text(s) in %.2fs → shape %s", len(texts), elapsed, vectors.shape)
        return vectors
