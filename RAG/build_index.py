"""
One-time script: embed all CSV data and persist a FAISS index.

Usage:
    python -m rag.build_index
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from rag.config import settings
from rag.embedder import Embedder
from rag.indexer import FAISSIndexer

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

COLS_TO_EMBED = [
    "title",
    "author",
    "rating",
    "ratings_count",
    "reviews_count",
    "description",
    "format",
    "language",
    "published",
]


def build() -> None:
    """Load CSVs, embed, build FAISS index, and persist to disk."""
    t_total = time.perf_counter()

    # ── Load CSVs ─────────────────────────────────────────────
    csv_files = list(settings.data_dir.glob("goodreads_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {settings.data_dir}")

    log.info("Found %d CSV file(s): %s", len(csv_files), [f.name for f in csv_files])
    data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    log.info("Total rows loaded: %d", len(data))

    # ── Deduplicate ───────────────────────────────────────────
    if "url" in data.columns:
        before = len(data)
        data = data.drop_duplicates(subset=["url"])
        log.info("Deduplicated: %d → %d rows", before, len(data))

    # ── Build text column ─────────────────────────────────────
    available = [c for c in COLS_TO_EMBED if c in data.columns]
    log.info("Embedding columns: %s", available)

    data["text"] = data.apply(
        lambda row: "\n".join(
            f"{c.capitalize()}: {row[c]}" for c in available if pd.notna(row[c])
        ),
        axis=1,
    )

    texts = data["text"].tolist()
    metadata = data.to_dict(orient="records")

    # ── Encode ────────────────────────────────────────────────
    log.info("Encoding %d documents…", len(texts))
    embedder = Embedder()
    embeddings = embedder.encode(texts)
    log.info("Embeddings shape: %s", embeddings.shape)

    # ── Build & persist index ─────────────────────────────────
    indexer = FAISSIndexer()
    indexer.build(embeddings, metadata)

    total = time.perf_counter() - t_total
    log.info("✓ Build complete in %.2fs — index saved to %s", total, settings.index_dir)


if __name__ == "__main__":
    build()
