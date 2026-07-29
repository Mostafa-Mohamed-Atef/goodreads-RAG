"""RAG pipeline orchestrator — wires Embedder, FAISSIndexer, Retriever, Generator."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from rag.config import settings
from rag.embedder import Embedder
from rag.generator import Generator
from rag.indexer import FAISSIndexer
from rag.retriever import Retriever

log = logging.getLogger(__name__)


class RAGPipeline:
    """High-level orchestrator for the full RAG flow."""

    def __init__(
        self,
        *,
        index_dir: Path | None = None,
        auto_load: bool = True,
    ) -> None:
        log.info("Initialising RAG pipeline…")
        t0 = time.perf_counter()

        self.embedder = Embedder()
        self.indexer = FAISSIndexer(index_dir=index_dir)

        if auto_load and self.indexer.is_built():
            self.indexer.load()
        elif auto_load:
            log.warning(
                "No persisted index found at %s — run `python -m rag.build_index` first.",
                self.indexer.index_dir,
            )

        self.retriever = Retriever(self.embedder, self.indexer)
        self.generator = Generator()

        elapsed = time.perf_counter() - t0
        log.info("Pipeline ready in %.2fs", elapsed)

    def query(self, question: str, top_k: int | None = None) -> dict:
        """Run the full RAG pipeline: retrieve → generate → return structured result."""
        k = top_k or settings.default_top_k

        log.info("Pipeline query: %r (top_k=%d)", question[:80], k)
        t0 = time.perf_counter()

        hits = self.retriever.retrieve(question, k)
        chunks = [h[0].get("text", "") for h in hits]
        answer = self.generator.answer(question, chunks)

        elapsed = time.perf_counter() - t0
        log.info("Pipeline query completed in %.2fs", elapsed)

        return {
            "answer": answer,
            "hits": hits,
            "latency_ms": round(elapsed * 1000, 2),
        }
