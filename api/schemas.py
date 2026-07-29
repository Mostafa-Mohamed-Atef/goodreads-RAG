"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming query from the user."""

    question: str = Field(..., min_length=3, max_length=1000, description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of book results to retrieve")


class BookHit(BaseModel):
    """A single retrieved book entry."""

    title: str | None = None
    author: str | None = None
    rating: str | None = None
    distance: float
    snippet: str


class QueryResponse(BaseModel):
    """Full response returned by POST /api/v1/query."""

    answer: str
    sources: list[BookHit]
    latency_ms: float
