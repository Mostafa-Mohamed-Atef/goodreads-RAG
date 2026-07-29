"""FAISS index management — build, save, load, search."""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import faiss
import numpy as np

from rag.config import settings

log = logging.getLogger(__name__)


class FAISSIndexer:
    """Build, persist, load, and search a FAISS flat-L2 index."""

    INDEX_FILE = "faiss.index"
    META_FILE = "metadata.pkl"

    def __init__(self, index_dir: Path | None = None) -> None:
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: list[dict] = []
        log.debug("FAISSIndexer initialised — index_dir=%s", self.index_dir)

    # ── Path helpers ──────────────────────────────────────────

    @property
    def _index_path(self) -> Path:
        return self.index_dir / self.INDEX_FILE

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / self.META_FILE

    # ── Public API ────────────────────────────────────────────

    def is_built(self) -> bool:
        """Check whether a persisted index exists on disk."""
        return self._index_path.exists() and self._meta_path.exists()

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Build a new FAISS index from embeddings and persist to disk."""
        dim = embeddings.shape[1]
        log.info("Building FAISS index: %d vectors, dim=%d", len(metadata), dim)
        t0 = time.perf_counter()

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata = metadata

        # Persist
        faiss.write_index(self.index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(metadata, f)

        elapsed = time.perf_counter() - t0
        log.info(
            "Index built and saved in %.2fs — %d vectors → %s",
            elapsed,
            self.index.ntotal,
            self._index_path,
        )

    def load(self) -> None:
        """Load a previously persisted index from disk."""
        if not self.is_built():
            raise FileNotFoundError(
                f"Index not found at {self.index_dir}. "
                "Run `python -m rag.build_index` first."
            )

        log.info("Loading FAISS index from %s…", self.index_dir)
        t0 = time.perf_counter()

        self.index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        elapsed = time.perf_counter() - t0
        log.info(
            "Index loaded in %.2fs — %d vectors, %d metadata records",
            elapsed,
            self.index.ntotal,
            len(self.metadata),
        )

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        """Search the index and return (metadata_dict, distance) pairs."""
        if self.index is None:
            raise RuntimeError("Index not loaded — call .load() or .build() first.")

        log.debug("Searching index: top_k=%d", top_k)
        t0 = time.perf_counter()
        distances, indices = self.index.search(query_vec, top_k)
        elapsed = time.perf_counter() - t0

        results = [
            (self.metadata[i], float(d))
            for i, d in zip(indices[0], distances[0])
        ]
        log.debug("Search completed in %.4fs — %d results", elapsed, len(results))
        return results
