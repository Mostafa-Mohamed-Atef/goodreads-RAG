"""FastAPI application factory — lifespan, middleware, routers, static files."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.middleware import LoggingMiddleware
from api.routes import health, query
from rag.config import settings
from rag.embedder import Embedder
from rag.generator import Generator
from rag.indexer import FAISSIndexer
from rag.retriever import Retriever

# ── Logging configuration ────────────────────────────────────
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Frontend path ─────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the FAISS index and initialise ML components on startup."""
    log.info("=== Application starting ===")

    log.info("Loading Embedder…")
    embedder = Embedder()

    log.info("Loading FAISS index…")
    indexer = FAISSIndexer()
    indexer.load()

    app.state.retriever = Retriever(embedder, indexer)
    app.state.generator = Generator()

    log.info("=== Application ready — all components loaded ===")
    yield
    log.info("=== Application shutting down ===")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Goodreads RAG API",
        version="1.0.0",
        description="Retrieval-Augmented Generation over scraped Goodreads book corpus",
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routers ───────────────────────────────────────────────
    app.include_router(query.router, prefix="/api/v1", tags=["Query"])
    app.include_router(health.router, tags=["Health"])

    # ── Serve frontend static files ───────────────────────────
    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
        log.info("Frontend mounted from %s", FRONTEND_DIR)
    else:
        log.warning("Frontend directory not found at %s — API-only mode", FRONTEND_DIR)

    return app


app = create_app()
