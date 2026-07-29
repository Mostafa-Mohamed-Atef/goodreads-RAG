"""Health and readiness endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


@router.get("/ready", response_model=None)
async def ready(req: Request):
    """Readiness probe — returns 503 if the FAISS index has not loaded yet."""
    if not hasattr(req.app.state, "retriever"):
        log.warning("Readiness check failed — retriever not loaded")
        return JSONResponse({"status": "not ready"}, status_code=503)
    return {"status": "ready"}
