# Walkthrough — Phases 1–4 Complete

## What Changed

### Project Structure (before → after)

```diff
 goodreads-rag/
-├── RAG/
-│   ├── __init__.py
-│   └── pipeline.py          ← monolith (166 lines)
-├── Data/
-│   ├── *.csv, *.json
-│   └── scripts/
-├── streamlit_app.py
-└── requirements.txt
+├── rag/                     ← modular package (8 files)
+│   ├── config.py            — pydantic-settings
+│   ├── embedder.py          — SentenceTransformer wrapper
+│   ├── indexer.py           — FAISS build/save/load/search
+│   ├── retriever.py         — semantic search
+│   ├── generator.py         — Groq LLM calls
+│   ├── pipeline.py          — orchestrator
+│   └── build_index.py       — one-time index builder
+├── api/                     ← FastAPI REST layer
+│   ├── main.py              — app factory + logging config
+│   ├── middleware.py         — request/response logging
+│   ├── schemas.py           — Pydantic models
+│   └── routes/
+│       ├── query.py         — POST /api/v1/query
+│       └── health.py        — GET /health, /ready
+├── frontend/                ← HTML/CSS/JS (dark theme, glassmorphism)
+│   ├── index.html
+│   ├── css/style.css
+│   └── js/app.js
+├── data/
+│   ├── raw/                 — CSVs + JSONs (moved)
+│   ├── index/               — persisted FAISS (gitignored)
+│   └── scripts/             — scrapers (moved)
+├── tests/                   ← unit + integration tests
+│   ├── conftest.py
+│   ├── unit/
+│   └── integration/
+├── pyproject.toml           ← replaces requirements.txt
+├── .env.example
+├── .pre-commit-config.yaml
+└── .gitignore               ← updated
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `pydantic-settings` for config | Type-safe, validates on startup, loads from `.env` automatically |
| Per-module `logging.getLogger(__name__)` | Granular log filtering — set `LOG_LEVEL=DEBUG` to see everything |
| Request/response logging middleware | Every HTTP call logged with method, path, status, timing |
| FastAPI serves frontend as static files | Single server at port 8000 — no CORS issues, no separate process |
| FAISS index persistence | Build once with `build_index.py`, load in ~1s on every restart |
| Mock-based tests | All ML components mocked — tests run in <1s without GPU/API keys |

### Logging Output Example (DEBUG level)

```
2026-07-29 20:05:12 | INFO     | api.middleware | → POST /api/v1/query [client=127.0.0.1, body=47B]
2026-07-29 20:05:12 | DEBUG    | rag.retriever  | Retrieving: question='best sci-fi books' top_k=5
2026-07-29 20:05:12 | DEBUG    | rag.embedder   | Encoding 1 text(s)…
2026-07-29 20:05:12 | DEBUG    | rag.embedder   | Encoded 1 text(s) in 0.01s → shape (1, 384)
2026-07-29 20:05:12 | DEBUG    | rag.indexer    | Searching index: top_k=5
2026-07-29 20:05:12 | DEBUG    | rag.indexer    | Search completed in 0.0002s — 5 results
2026-07-29 20:05:12 | DEBUG    | rag.generator  | Generating answer — model=llama-3.3-70b-versatile, prompt_len=2341 chars
2026-07-29 20:05:13 | DEBUG    | rag.generator  | Answer generated in 1.12s — 287 chars
2026-07-29 20:05:13 | INFO     | api.middleware | ← POST /api/v1/query — 200 [1247ms]
```

---

## Setup Instructions

Run these commands from the project root:

```bash
# 1. Install all dependencies (including dev tools)
pip install -e ".[dev]"

# 2. Build the FAISS index (one-time, ~30-60s)
python -m rag.build_index

# 3. Start the API + frontend
uvicorn api.main:app --reload

# 4. Open in browser
# → http://localhost:8000       (frontend)
# → http://localhost:8000/docs  (Swagger API docs)

# 5. Run tests
pytest

# 6. (Optional) Clean up old directories
Remove-Item -Recurse -Force RAG_old, Data_old
Remove-Item streamlit_app.py
```

> [!TIP]
> Set `LOG_LEVEL=DEBUG` in your `.env` file to see full logging output during development.
