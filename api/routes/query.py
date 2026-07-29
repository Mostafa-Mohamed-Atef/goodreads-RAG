"""POST /api/v1/query — the main RAG endpoint."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Request

from api.schemas import BookHit, QueryRequest, QueryResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_books(req: Request, body: QueryRequest) -> QueryResponse:
    """Retrieve relevant books and generate an LLM answer."""
    log.info("Query received: %r (top_k=%d)", body.question[:80], body.top_k)
    t0 = time.perf_counter()

    # ── Retrieve ──────────────────────────────────────────────
    hits = req.app.state.retriever.retrieve(body.question, body.top_k)
    chunks = [h[0].get("text", "") for h in hits]
    log.debug("Retrieved %d chunks, total context length: %d chars", len(chunks), sum(len(c) for c in chunks))

    # ── Generate ──────────────────────────────────────────────
    answer = req.app.state.generator.answer(body.question, chunks)

    # ── Build response ────────────────────────────────────────
    sources = [
        BookHit(
            title=h[0].get("title"),
            author=h[0].get("author"),
            rating=str(h[0].get("rating")) if h[0].get("rating") is not None else None,
            distance=h[1],
            snippet=(h[0].get("text") or "")[:400],
        )
        for h in hits
    ]

    latency = round((time.perf_counter() - t0) * 1000, 2)
    log.info("Query completed in %.2fms — answer length: %d chars", latency, len(answer))

    return QueryResponse(answer=answer, sources=sources, latency_ms=latency)
