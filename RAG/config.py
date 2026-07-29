"""Centralised application settings loaded from environment / .env file."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    """All configuration lives here — no hardcoded constants elsewhere."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Groq LLM ──────────────────────────────────────────────
    groq_api_key: str
    gen_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7

    # ── Embedding ─────────────────────────────────────────────
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    default_top_k: int = 10

    # ── Paths ─────────────────────────────────────────────────
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    index_dir: Path = PROJECT_ROOT / "data" / "index"

    # ── Server ────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"


settings = Settings()

log.info(
    "Settings loaded — model=%s, embedding=%s, data=%s, index=%s",
    settings.gen_model,
    settings.embedding_model,
    settings.data_dir,
    settings.index_dir,
)
