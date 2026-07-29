# 🚀 Goodreads RAG — Production-Level Implementation Guide

> **Goal:** Transform this prototype into a CV-worthy, production-grade ML system.  
> **Stack:** Python · FAISS · SentenceTransformers · Groq (LLaMA 3) · FastAPI · Docker · GitHub Actions · Streamlit

---

## Table of Contents

1. [Current State Assessment](#1-current-state-assessment)
2. [Target Architecture](#2-target-architecture)
3. [Phase 1 — Code Quality & Project Structure](#phase-1--code-quality--project-structure)
4. [Phase 2 — Persistent FAISS Index & Caching](#phase-2--persistent-faiss-index--caching)
5. [Phase 3 — Production API with FastAPI](#phase-3--production-api-with-fastapi)
6. [Phase 4 — Testing Strategy](#phase-4--testing-strategy)
7. [Phase 5 — CI/CD with GitHub Actions](#phase-5--cicd-with-github-actions)
8. [Phase 6 — Docker & Containerization](#phase-6--docker--containerization)
9. [Phase 7 — Cloud Deployment](#phase-7--cloud-deployment)
10. [Phase 8 — RAG Evaluation & Metrics](#phase-8--rag-evaluation--metrics)
11. [Phase 9 — Observability & Logging](#phase-9--observability--logging)
12. [Phase 10 — README & CV Presentation](#phase-10--readme--cv-presentation)
13. [Priority Order for CV Impact](#priority-order-for-cv-impact)

---

## 1. Current State Assessment

| Area | Current State | Gap |
|---|---|---|
| **Architecture** | Single `pipeline.py` monolith | No separation of concerns |
| **Indexing** | Re-embeds all data on every startup (~30–60s cold start) | No persistence |
| **API** | Streamlit only (no REST API) | No programmatic access |
| **Testing** | Zero tests | No confidence in correctness |
| **Config** | Hardcoded constants in source | Not env-driven |
| **Errors** | Bare `except Exception` blocks | Silent failures in production |
| **CI/CD** | None | Manual deploy only |
| **Docker** | None | Cannot run anywhere reliably |
| **Evaluation** | None | Cannot measure RAG quality |
| **Logging** | `print()` statements | Not queryable or persistent |

---

## 2. Target Architecture

```
goodreads-rag/
├── api/                        # FastAPI REST layer
│   ├── __init__.py
│   ├── main.py                 # App factory, lifespan, CORS
│   ├── routes/
│   │   ├── query.py            # POST /query
│   │   └── health.py           # GET /health, GET /ready
│   └── schemas.py              # Pydantic request/response models
├── rag/                        # Core ML logic (renamed from RAG/)
│   ├── __init__.py
│   ├── config.py               # Settings via pydantic-settings
│   ├── embedder.py             # SentenceTransformer wrapper
│   ├── indexer.py              # FAISS build + persist + load
│   ├── retriever.py            # Semantic search
│   ├── generator.py            # Groq LLM calls
│   ├── pipeline.py             # Orchestrates the above
│   └── build_index.py          # One-time index builder script
├── data/
│   ├── scripts/                # Scrapers (existing)
│   ├── raw/                    # Raw CSV/JSON (gitignored)
│   └── index/                  # Persisted FAISS index + metadata (gitignored)
├── tests/
│   ├── unit/
│   │   ├── test_embedder.py
│   │   ├── test_retriever.py
│   │   └── test_generator.py
│   ├── integration/
│   │   └── test_pipeline.py
│   └── conftest.py
├── notebooks/                  # EDA, evaluation experiments
│   └── rag_evaluation.ipynb
├── streamlit_app.py            # UI (keep, thin wrapper)
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml              # Replaces requirements.txt
├── .env.example
└── README.md
```

---

## Phase 1 — Code Quality & Project Structure

### 1.1 Switch to `pyproject.toml`

Replace `requirements.txt` with a proper `pyproject.toml`:

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "goodreads-rag"
version = "1.0.0"
description = "Production RAG system over scraped Goodreads corpus"
requires-python = ">=3.10"

dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "faiss-cpu>=1.8",
    "sentence-transformers>=3.0",
    "groq>=0.9",
    "pandas>=2.2",
    "python-dotenv>=1.0",
    "streamlit>=1.36",
    "requests>=2.32",
    "beautifulsoup4>=4.12",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.2",
    "pytest-asyncio>=0.23",
    "pytest-cov>=5.0",
    "httpx>=0.27",
    "ruff>=0.4",
    "mypy>=1.10",
    "pre-commit>=3.7",
    "ragas>=0.1",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=rag --cov-report=term-missing"
```

### 1.2 Centralised Settings with `pydantic-settings`

Create `rag/config.py`:

```python
# rag/config.py
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Groq
    groq_api_key: str
    gen_model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.7

    # Embedding
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    default_top_k: int = 10

    # Paths
    data_dir: Path = PROJECT_ROOT / "data" / "raw"
    index_dir: Path = PROJECT_ROOT / "data" / "index"

    # Server
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

settings = Settings()
```

### 1.3 Split the Monolith into Modules

**`rag/embedder.py`** — wraps SentenceTransformer:

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from rag.config import settings

class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(settings.embedding_model)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
```

**`rag/indexer.py`** — build, save, load FAISS:

```python
import faiss, pickle
import numpy as np
from pathlib import Path
from rag.config import settings

class FAISSIndexer:
    INDEX_FILE = "faiss.index"
    META_FILE  = "metadata.pkl"

    def __init__(self, index_dir: Path | None = None):
        self.index_dir = index_dir or settings.index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.IndexFlatL2 | None = None
        self.metadata: list[dict] = []

    @property
    def _index_path(self) -> Path:
        return self.index_dir / self.INDEX_FILE

    @property
    def _meta_path(self) -> Path:
        return self.index_dir / self.META_FILE

    def is_built(self) -> bool:
        return self._index_path.exists() and self._meta_path.exists()

    def build(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.metadata = metadata
        faiss.write_index(self.index, str(self._index_path))
        with open(self._meta_path, "wb") as f:
            pickle.dump(metadata, f)

    def load(self) -> None:
        if not self.is_built():
            raise FileNotFoundError("Index not built. Run `python -m rag.build_index` first.")
        self.index = faiss.read_index(str(self._index_path))
        with open(self._meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
        assert self.index is not None, "Index not loaded."
        distances, indices = self.index.search(query_vec, top_k)
        return [(self.metadata[i], float(d)) for i, d in zip(indices[0], distances[0])]
```

**`rag/retriever.py`**:

```python
from rag.embedder import Embedder
from rag.indexer import FAISSIndexer

class Retriever:
    def __init__(self, embedder: Embedder, indexer: FAISSIndexer):
        self.embedder = embedder
        self.indexer  = indexer

    def retrieve(self, question: str, top_k: int) -> list[tuple[dict, float]]:
        vec = self.embedder.encode([question])
        return self.indexer.search(vec, top_k)
```

**`rag/generator.py`**:

```python
from groq import Groq
from rag.config import settings

SYSTEM_PROMPT = "You are a helpful assistant specialising in book recommendations."

class Generator:
    def __init__(self):
        self.client = Groq(api_key=settings.groq_api_key)

    def answer(self, question: str, chunks: list[str]) -> str:
        context  = "\n\n".join(chunks)
        user_msg = (
            f"Use ONLY the context below to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}"
        )
        resp = self.client.chat.completions.create(
            model=settings.gen_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=settings.temperature,
        )
        return resp.choices[0].message.content.strip()
```

### 1.4 Add Pre-commit Hooks

```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.8
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
```

```bash
pre-commit install
```

---

## Phase 2 — Persistent FAISS Index & Caching

The biggest production win: **build the index once, load it on every subsequent startup**.

### 2.1 Build Script

Create `rag/build_index.py`:

```python
"""
One-time script: embed all CSV data and persist FAISS index.
Run: python -m rag.build_index
"""
import logging
import pandas as pd
from rag.config import settings
from rag.embedder import Embedder
from rag.indexer import FAISSIndexer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

COLS_TO_EMBED = [
    "title", "author", "rating", "ratings_count",
    "reviews_count", "description", "format", "language", "published",
]

def build() -> None:
    csv_files = list(settings.data_dir.glob("goodreads_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {settings.data_dir}")

    log.info("Loading %d CSV file(s)...", len(csv_files))
    data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if "url" in data.columns:
        before = len(data)
        data = data.drop_duplicates(subset=["url"])
        log.info("Deduplicated: %d -> %d rows", before, len(data))

    available = [c for c in COLS_TO_EMBED if c in data.columns]
    data["text"] = data.apply(
        lambda row: "\n".join(
            f"{c.capitalize()}: {row[c]}" for c in available if pd.notna(row[c])
        ),
        axis=1,
    )

    texts    = data["text"].tolist()
    metadata = data.to_dict(orient="records")

    log.info("Encoding %d documents...", len(texts))
    embedder   = Embedder()
    embeddings = embedder.encode(texts)

    log.info("Building and persisting FAISS index...")
    indexer = FAISSIndexer()
    indexer.build(embeddings, metadata)
    log.info("Done. Index saved to %s", settings.index_dir)

if __name__ == "__main__":
    build()
```

After running this **once**, cold-start time drops from **~60 seconds to ~1 second**.

```bash
python -m rag.build_index
```

---

## Phase 3 — Production API with FastAPI

### 3.1 Pydantic Schemas

```python
# api/schemas.py
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int    = Field(default=5, ge=1, le=20)

class BookHit(BaseModel):
    title:    str | None
    author:   str | None
    rating:   str | None
    distance: float
    snippet:  str

class QueryResponse(BaseModel):
    answer:     str
    sources:    list[BookHit]
    latency_ms: float
```

### 3.2 FastAPI App

```python
# api/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.embedder import Embedder
from rag.indexer import FAISSIndexer
from rag.retriever import Retriever
from rag.generator import Generator
from api.routes import query, health

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading FAISS index...")
    embedder = Embedder()
    indexer  = FAISSIndexer()
    indexer.load()
    app.state.retriever = Retriever(embedder, indexer)
    app.state.generator = Generator()
    log.info("Ready.")
    yield

app = FastAPI(title="Goodreads RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router,  prefix="/api/v1")
app.include_router(health.router)
```

```python
# api/routes/query.py
import time
from fastapi import APIRouter, Request
from api.schemas import QueryRequest, QueryResponse, BookHit

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query_books(req: Request, body: QueryRequest):
    t0 = time.perf_counter()

    hits   = req.app.state.retriever.retrieve(body.question, body.top_k)
    chunks = [h[0]["text"] for h in hits]
    answer = req.app.state.generator.answer(body.question, chunks)

    sources = [
        BookHit(
            title    = h[0].get("title"),
            author   = h[0].get("author"),
            rating   = h[0].get("rating"),
            distance = h[1],
            snippet  = (h[0].get("text") or "")[:400],
        )
        for h in hits
    ]

    return QueryResponse(
        answer     = answer,
        sources    = sources,
        latency_ms = round((time.perf_counter() - t0) * 1000, 2),
    )
```

```python
# api/routes/health.py
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/ready")
async def ready(req: Request):
    """Returns 503 if the index has not been loaded yet."""
    if not hasattr(req.app.state, "retriever"):
        return JSONResponse({"status": "not ready"}, status_code=503)
    return {"status": "ready"}
```

**Run the API:**

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Interactive Swagger UI:** `http://localhost:8000/docs`

---

## Phase 4 — Testing Strategy

### 4.1 `tests/conftest.py`

```python
import pytest
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def mock_embedder():
    emb = MagicMock()
    emb.encode.return_value = np.random.rand(1, 384).astype("float32")
    return emb

@pytest.fixture
def mock_indexer():
    idx = MagicMock()
    idx.search.return_value = [
        ({"title": "Dune", "author": "Herbert", "text": "A sci-fi epic.", "rating": "4.5"}, 0.12),
    ]
    return idx

@pytest.fixture
def mock_generator():
    gen = MagicMock()
    gen.answer.return_value = "Dune is a great sci-fi novel."
    return gen
```

### 4.2 Unit Tests

```python
# tests/unit/test_retriever.py
from rag.retriever import Retriever

def test_retrieve_returns_hits(mock_embedder, mock_indexer):
    retriever = Retriever(mock_embedder, mock_indexer)
    hits = retriever.retrieve("science fiction books", top_k=1)
    assert len(hits) == 1
    assert hits[0][0]["title"] == "Dune"
    mock_embedder.encode.assert_called_once_with(["science fiction books"])
```

```python
# tests/unit/test_generator.py
from unittest.mock import patch, MagicMock
from rag.generator import Generator

def test_answer_calls_groq():
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "  Dune is a classic.  "

    with patch("rag.generator.Groq") as MockGroq:
        MockGroq.return_value.chat.completions.create.return_value = mock_resp
        gen    = Generator()
        result = gen.answer("What is Dune?", ["Dune is a sci-fi novel by Frank Herbert."])

    assert result == "Dune is a classic."
```

### 4.3 Integration Tests

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

@pytest.fixture
def client(mock_embedder, mock_indexer, mock_generator):
    with patch("api.main.Embedder", return_value=mock_embedder), \
         patch("api.main.FAISSIndexer", return_value=mock_indexer), \
         patch("api.main.Generator", return_value=mock_generator):
        from api.main import app
        yield TestClient(app)

def test_query_endpoint(client):
    resp = client.post("/api/v1/query", json={"question": "Best sci-fi books?"})
    assert resp.status_code == 200
    body = resp.json()
    assert "answer" in body
    assert "sources" in body
    assert body["latency_ms"] >= 0

def test_query_too_short(client):
    resp = client.post("/api/v1/query", json={"question": "Hi"})
    assert resp.status_code == 422   # Pydantic validation error

def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
```

**Run all tests with coverage:**

```bash
pytest --cov=rag --cov-report=html
```

Target: **>= 80% coverage**.

---

## Phase 5 — CI/CD with GitHub Actions

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Lint with Ruff
        run: ruff check . && ruff format --check .

      - name: Type check with mypy
        run: mypy rag/ api/

      - name: Run tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: pytest --cov=rag --cov-fail-under=80

      - name: Upload coverage report
        uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

Add a **coverage badge** to your README:

```
[![CI](https://github.com/YOUR/goodreads-rag/actions/workflows/ci.yml/badge.svg)](...)
[![Coverage](https://codecov.io/gh/YOUR/goodreads-rag/branch/main/graph/badge.svg)](...)
```

---

## Phase 6 — Docker & Containerization

### 6.1 Dockerfile (Multi-stage)

```dockerfile
# ---- builder stage ----
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir build && python -m build --wheel -o dist .

# ---- runtime stage ----
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY --from=builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

COPY rag/ ./rag/
COPY api/ ./api/
COPY data/index/ ./data/index/

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 6.2 `docker-compose.yml`

```yaml
version: "3.9"

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
    volumes:
      - ./data/index:/app/data/index

  streamlit:
    build: .
    command: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
    ports:
      - "8501:8501"
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
    depends_on:
      - api
```

**Build and run:**

```bash
docker compose up --build
```

### 6.3 `.env.example`

```bash
GROQ_API_KEY=your_groq_api_key_here
GEN_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

---

## Phase 7 — Cloud Deployment

### Option A — Streamlit Community Cloud *(Free, easiest)*

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `GROQ_API_KEY` in secrets
4. Deploy → get a public live demo URL

> **Important:** Pre-build your FAISS index and commit `data/index/` to the repo (remove it from `.gitignore`), or configure a GitHub Action to build and cache it automatically.

### Option B — Render.com *(Free tier, Docker support)*

Create `render.yaml`:

```yaml
services:
  - type: web
    name: goodreads-rag-api
    runtime: docker
    envVars:
      - key: GROQ_API_KEY
        sync: false
    healthCheckPath: /health
```

Connect GitHub and it will auto-deploy on every push.

### Option C — Hugging Face Spaces *(Free, ML-community audience)*

1. Create a new Space with the **Streamlit** SDK
2. Connect your GitHub repository
3. Add `GROQ_API_KEY` as a Space secret
4. Live demo URL: `https://huggingface.co/spaces/YOUR_USERNAME/goodreads-rag`

**Best for ML roles:** Recruiters for ML/AI positions actively browse HF Spaces.

---

## Phase 8 — RAG Evaluation & Metrics

This is what separates a serious ML project from a hobby project.

### 8.1 Install RAGAS

```bash
pip install ragas datasets
```

### 8.2 Evaluation Script

Create `scripts/evaluate_rag.py`:

```python
"""
Evaluate RAG pipeline using RAGAS metrics.
Metrics: faithfulness, answer_relevancy, context_recall, context_precision
"""
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from rag.pipeline import RAGPipeline

# Ground-truth test set — manually curate at least 15-20 examples
TEST_SET = [
    {
        "question": "What is the highest-rated book in the corpus?",
        "ground_truth": "The book with the highest rating is ...",
    },
    {
        "question": "Which Arabic books are in the dataset?",
        "ground_truth": "The Arabic books include ...",
    },
    # ... add more examples
]

def run_evaluation():
    pipeline = RAGPipeline()
    records  = []
    for item in TEST_SET:
        hits   = pipeline.retrieve_with_meta(item["question"], top_k=5)
        chunks = [h[0]["text"] for h in hits]
        answer = pipeline.generate_answer(item["question"], chunks)
        records.append({
            "question":     item["question"],
            "answer":       answer,
            "contexts":     chunks,
            "ground_truth": item["ground_truth"],
        })

    dataset = Dataset.from_list(records)
    result  = evaluate(dataset, metrics=[
        faithfulness, answer_relevancy, context_precision, context_recall
    ])
    print(result)
    result.to_pandas().to_csv("evaluation_results.csv", index=False)

if __name__ == "__main__":
    run_evaluation()
```

### 8.3 Report Results in Your README

```markdown
## Evaluation Results (RAGAS)

| Metric | Score |
|---|---|
| Faithfulness | 0.87 |
| Answer Relevancy | 0.83 |
| Context Precision | 0.79 |
| Context Recall | 0.81 |

*Evaluated on 20 manually curated questions.*
```

> **This is the single biggest signal to a hiring manager** that you think like an ML engineer, not just a developer.

---

## Phase 9 — Observability & Logging

### 9.1 Structured Logging

Replace all `print()` calls with Python's `logging` module:

```python
import logging
log = logging.getLogger(__name__)

log.info("Encoding %d documents", len(texts))
log.warning("Index not found, rebuilding...")
log.error("Groq API call failed: %s", exc)
```

Configure once in `api/main.py`:

```python
import logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
```

### 9.2 Request Latency Tracking

Already included in `QueryResponse.latency_ms`. Log it too:

```python
log.info("query=%r top_k=%d latency_ms=%.1f", body.question[:50], body.top_k, latency_ms)
```

### 9.3 Optional — Prometheus Metrics *(Advanced)*

```bash
pip install prometheus-fastapi-instrumentator
```

```python
# api/main.py
from prometheus_fastapi_instrumentator import Instrumentator
Instrumentator().instrument(app).expose(app)
```

Adds a `/metrics` endpoint compatible with Grafana dashboards.

---

## Phase 10 — README & CV Presentation

### 10.1 README Structure

```markdown
# Goodreads RAG — Semantic Book Q&A System

[![CI](https://github.com/YOUR/goodreads-rag/actions/workflows/ci.yml/badge.svg)](...)
[![Coverage](https://codecov.io/gh/YOUR/goodreads-rag/...)](...)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-FF4B4B)](YOUR_DEMO_URL)

> A production-grade Retrieval-Augmented Generation (RAG) system that answers
> natural-language questions about 2,000+ books scraped from Goodreads.
> Powered by FAISS vector search, multilingual sentence embeddings, and LLaMA 3.3-70B via Groq.

## Architecture
[diagram or screenshot]

## Evaluation (RAGAS)
[table of metrics]

## Quick Start
[clear setup instructions]

## Tech Stack
[table: Component | Technology | Why]

## Project Structure
[tree]
```

### 10.2 Architecture Diagram

```
User Query
    |
    v
+-------------------+     +---------------------------+
|   Streamlit UI    |---->|    FastAPI REST API        |
|   (port 8501)     |     |    (port 8000)             |
+-------------------+     +-------------+-------------+
                                        |
                         +--------------v--------------+
                         |        RAG Pipeline          |
                         |                              |
                         |  1. Embedder (MiniLM)        |
                         |  2. FAISS Retriever           |
                         |  3. Generator (LLaMA 3 Groq) |
                         +------------------------------+
```

### 10.3 CV Bullet Points

Copy these into your CV (fill in your actual metric values after running RAGAS):

```
Goodreads RAG — Semantic Book Q&A System                     [GitHub] [Live Demo]

* Built a production-grade RAG pipeline over 2,000+ scraped Goodreads records,
  combining FAISS vector search with LLaMA 3.3-70B (Groq API) for semantic Q&A.

* Engineered a multilingual embedding layer (paraphrase-multilingual-MiniLM-L12-v2)
  supporting Arabic and English queries with <200ms retrieval latency.

* Evaluated system quality using RAGAS metrics (faithfulness 0.87, answer relevancy 0.83)
  on a manually curated 20-question test set.

* Shipped a REST API (FastAPI + Pydantic v2) alongside a Streamlit demo, containerised
  with Docker, and deployed with CI/CD via GitHub Actions (80%+ test coverage).

* Designed persistent FAISS index storage that reduced cold-start time from ~60s to ~1s.
```

---

## Priority Order for CV Impact

Do these in order — each one adds measurable signal to a recruiter:

| Priority | Task | Est. Time | CV Signal |
|---|---|---|---|
| **1** | Persistent FAISS index (Phase 2) | 1 hr | High — systems thinking |
| **2** | Structured project layout (Phase 1.1-1.2) | 2 hrs | High — engineering maturity |
| **3** | Live demo deployment (Phase 7) | 30 min | High — shipping ability |
| **4** | FastAPI REST layer (Phase 3) | 3 hrs | High — API design skills |
| **5** | Tests + CI/CD (Phases 4 & 5) | 3 hrs | High — professional habits |
| **6** | RAGAS evaluation (Phase 8) | 2 hrs | Very High — ML rigor |
| **7** | Docker (Phase 6) | 1 hr | Medium — DevOps awareness |
| **8** | Polished README with badges | 1 hr | Medium — first impression |
| **9** | Prometheus metrics (Phase 9.3) | 1 hr | Low-Med — ops awareness |

---

## Quick-Start Checklist

```bash
# 1. Refactor project structure
mkdir -p api/routes rag tests/unit tests/integration data/raw data/index

# 2. Move existing CSV data
move Data\*.csv data\raw\

# 3. Install dev dependencies
pip install -e ".[dev]"

# 4. Set up pre-commit
pre-commit install

# 5. Build the FAISS index once
python -m rag.build_index

# 6. Run the API
uvicorn api.main:app --reload

# 7. Run tests
pytest

# 8. Build Docker image
docker compose up --build

# 9. Deploy (Streamlit Cloud / HF Spaces / Render)
# See Phase 7 above
```

---

*Last updated: July 2026*
