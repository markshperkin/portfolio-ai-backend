# portfolio-ai-backend

Python FastAPI backend for Mark's GPT — a RAG-backed AI assistant that answers questions about Mark Shperkin's work, skills, and experience. Handles embedding, vector retrieval, LLM streaming, rate limiting, and abuse detection.

## What It Does

Receives a chat message, retrieves the most relevant chunks from a Chroma vector store, injects them into a system prompt, and streams a grounded response from Anthropic Claude Haiku over Server-Sent Events (SSE). Refuses to answer if no relevant context is found.

## Architecture

```
POST /api/chat
    ↓
Slash command? → short-circuit response
    ↓
Abuse classifier (regex + SQLite log)
    ↓
Rate limiter (per-IP, in-process)
    ↓
Embed query → Voyage AI voyage-3-large (1024-dim)
    ↓
ChromaDB HNSW cosine search → top-5 chunks
    ↓
Threshold gate: score < 0.35 → refuse, ≥ 0.35 → proceed
    ↓
Build system prompt (persona + guardrails + retrieved context)
    ↓
Stream Claude Haiku → SSE delta/citation/done events
```

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI 0.111+, Uvicorn 0.29+ |
| LLM | Anthropic Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) |
| Embeddings | Voyage AI `voyage-3-large` (1024 dims) |
| Vector DB | ChromaDB (HNSW index, cosine similarity, file-based) |
| Async DB | aiosqlite (abuse log) |
| Validation | Pydantic v2 |
| Language | Python 3.11 |
| Deployment | Docker, Caddy reverse proxy, Hostinger VPS |

## Project Structure

```
app/
├── api/          # Routes: /api/chat, /api/health, /api/resume.pdf
├── rag/          # embedding.py, retrieval.py, chunking.py, store.py
├── llm/          # Anthropic streaming client
├── security/     # Rate limiter, abuse classifier, IP hashing (HMAC-SHA256)
├── commands/     # Slash command handler (whoami, /help, sudo hire-mark, cat resume.pdf)
├── prompt/       # System prompt builder + contact constants
├── reindex/      # Corpus loader + indexing CLI
└── models.py     # SSE event schemas (Pydantic)
infra/
├── docker-compose.yml   # Full stack: Caddy, frontend, backend (test + prod)
└── Caddyfile            # Reverse proxy routing + Let's Encrypt
static/
└── resume.pdf
data/
├── chroma_db/    # Persistent vector index
└── abuse_log.db  # SQLite jailbreak log
```

## SSE Event Protocol

| Event | Payload | Meaning |
|---|---|---|
| `retrieval_step` | `{step: "retrieving"\|"searching"\|"synthesizing"}` | Pipeline status |
| `delta` | `{text: string}` | Streamed token |
| `citation` | `{sources: [{title}]}` | Source documents used |
| `done` | `{}` | Stream complete |
| `error` | `{code, message}` | Error condition |
| `action` | `{action_type: "download"\|"open", url}` | Trigger client action |

## Reindexing the Knowledge Base

Run after updating the knowledge corpus:

```bash
python -m app.reindex --corpus /path/to/portfolio-ai-knowledge
```

Chunks each doc, embeds via Voyage AI (batch size 8, 21s delay between batches for free-tier rate limits), and atomically replaces the Chroma collection.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes | Anthropic API key |
| `VOYAGE_API_KEY` | Yes | Voyage AI API key |
| `IP_HASH_SALT` | Yes | HMAC-SHA256 salt for IP anonymisation |
| `CORPUS_PATH` | Yes | Path to knowledge corpus (for reindex CLI) |
| `CHROMA_PATH` | No | Defaults to `data/chroma_db` |

## Running Locally

```bash
pip install .
uvicorn app.main:app --reload --port 8000
```

## Deployment

Push to `test` or `prod` branch → GitHub Actions builds Docker image → pushes to GHCR (`ghcr.io/markshperkin/portfolio-ai-backend:<branch>`) → SSH-deploys to VPS via `docker compose up -d`.
