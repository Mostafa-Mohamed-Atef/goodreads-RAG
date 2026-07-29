"""Request/response logging middleware for FastAPI."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

log = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log every request and response with method, path, status, and timing."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # ── Request info ──────────────────────────────────────
        method = request.method
        path = request.url.path
        query = str(request.url.query) if request.url.query else ""
        client = request.client.host if request.client else "unknown"

        body_size = request.headers.get("content-length", "0")
        log.info("→ %s %s%s [client=%s, body=%sB]", method, path, f"?{query}" if query else "", client, body_size)

        # ── Process request ───────────────────────────────────
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = time.perf_counter() - t0
            log.exception("✗ %s %s — unhandled exception after %.0fms", method, path, elapsed * 1000)
            raise

        elapsed = time.perf_counter() - t0

        # ── Response info ─────────────────────────────────────
        log.info(
            "← %s %s — %d [%.0fms]",
            method,
            path,
            response.status_code,
            elapsed * 1000,
        )

        return response
