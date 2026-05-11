"""Reindex CLI: python -m app.reindex --corpus /path/to/knowledge

Walks the knowledge corpus, chunks each doc, embeds via Voyage, and
replaces the Chroma collection. Idempotent: same corpus → same chunk count.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from app.rag.chunking import chunk_text
from app.rag.embedding import embed_batch
from app.rag.store import COLLECTION_NAME
from app.reindex.loader import load_corpus

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

BATCH_SIZE = 8
# Voyage free tier: 3 RPM. Sleep between batches to avoid rate-limit errors.
BATCH_DELAY_SECONDS = 21


def parse_args() -> argparse.Namespace:
    default_corpus = os.environ.get("CORPUS_PATH")
    p = argparse.ArgumentParser(description="Reindex knowledge corpus into Chroma")
    p.add_argument(
        "--corpus",
        type=Path,
        default=Path(default_corpus) if default_corpus else None,
        required=not default_corpus,
        help="Path to clean_data corpus root (overrides CORPUS_PATH env var)",
    )
    p.add_argument("--chunk-tokens", type=int, default=400)
    p.add_argument("--overlap-tokens", type=int, default=70)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"ERROR: corpus path {corpus} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    chroma_path = os.environ.get("CHROMA_PATH", "data/chroma_db")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    print(f"Loading corpus from {corpus}…")
    docs = load_corpus(corpus)
    print(f"  {len(docs)} document(s) found")

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for doc in docs:
        chunks = chunk_text(doc.body, args.chunk_tokens, args.overlap_tokens)
        for i, chunk in enumerate(chunks):
            ids.append(f"{doc.path}::{i}")
            documents.append(chunk)
            metadatas.append(
                {
                    "path": doc.path,
                    "title": doc.title,
                    "category": doc.category,
                    "tags": ",".join(doc.tags),  # Chroma metadata values must be scalar
                    "last_updated": doc.last_updated,
                    "weight": doc.weight,
                    "chunk_index": i,
                }
            )

    print(f"  {len(ids)} chunk(s) to embed")

    vectors: list[list[float]] = []
    total_batches = (len(ids) - 1) // BATCH_SIZE + 1 if ids else 0
    for batch_num, batch_start in enumerate(range(0, len(ids), BATCH_SIZE)):
        if batch_num > 0:
            time.sleep(BATCH_DELAY_SECONDS)
        batch = documents[batch_start : batch_start + BATCH_SIZE]
        print(f"  Embedding batch {batch_num + 1}/{total_batches}…")
        vectors.extend(embed_batch(batch))

    print("Writing to Chroma…")
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)  # type: ignore[arg-type]

    print(f"Done. {len(ids)} chunk(s) indexed.")


if __name__ == "__main__":
    main()
