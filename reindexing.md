# Reindexing Runbook

## Prerequisites

- Python 3.11+ (global install, no venv)
- Packages installed: run `pip install -e .` from `portfolio-ai-backend/` if not already
- `.env` in `portfolio-ai-backend/` must contain:

```
VOYAGE_API_KEY=<your key>
CORPUS_PATH=C:/Users/marks/Documents/portfolio/portfolio-ai-knowledge/clean_data
```

`CHROMA_PATH` is optional — defaults to `data/chroma_db`.

---

## Command

```bash
cd portfolio-ai-backend
python -m app.reindex
```

Override the corpus path for a one-off run:

```bash
python -m app.reindex --corpus /path/to/other/corpus
```

---

## What it does

1. Walks `CORPUS_PATH` recursively for `*.md` files
   - Skips: `README.md`, `INDEX.md`, `.github/`, `scripts/`
2. Infers metadata from directory structure (no frontmatter required):
   - `projects/` → category `project`
   - `research/` → category `paper`
   - Tags derived from subdirectory name
3. Chunks each doc (~400 tokens, 70-token overlap)
4. Embeds via Voyage `voyage-3-large` (1024 dims)
5. **Deletes** the existing Chroma collection and recreates it (fully idempotent)
6. Writes index to `data/chroma_db/`

---

## Rate limit

Voyage free tier: **3 RPM, 10K TPM**. The script uses batch size 8 with a 21s delay between batches.

- 183 chunks → 23 batches → ~8 minutes total
- If the account is upgraded: increase `BATCH_SIZE` and reduce/remove `BATCH_DELAY_SECONDS` in `app/reindex/__main__.py`

---

## Expected output

```
Loading corpus from ...clean_data…
  73 document(s) found
  183 chunk(s) to embed
  Embedding batch 1/23…
  ...
  Embedding batch 23/23…
Writing to Chroma…
Done. 183 chunk(s) indexed.
```

---

## Verify the index

```bash
python -c "
import chromadb
from app.rag.store import COLLECTION_NAME
c = chromadb.PersistentClient(path='data/chroma_db')
col = c.get_collection(COLLECTION_NAME)
print('Chunks indexed:', col.count())
"
```

Should print `183` (updates after corpus changes).

---

## Adding new content

Drop new `.md` files into `portfolio-ai-knowledge/clean_data/projects/` or `.../research/` and re-run the command. No frontmatter needed — metadata is inferred from the directory structure.
