"""Reindex CLI: python -m app.reindex --corpus /knowledge

Walks the knowledge corpus, chunks each doc, embeds via Voyage, and
truncate-rewrites the chunks table. Idempotent: same corpus → same chunk count.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

import asyncpg

from app.rag.chunking import chunk_text
from app.rag.embedding import embed_batch
from app.reindex.loader import load_corpus

BATCH_SIZE = 32  # Voyage API batch size


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reindex knowledge corpus into pgvector")
    p.add_argument("--corpus", required=True, type=Path, help="Path to knowledge repo root")
    p.add_argument("--chunk-tokens", type=int, default=800)
    p.add_argument("--overlap-tokens", type=int, default=100)
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    corpus = args.corpus.resolve()
    if not corpus.is_dir():
        print(f"ERROR: corpus path {corpus} is not a directory", file=sys.stderr)
        raise SystemExit(1)

    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL environment variable not set")

    print(f"Loading corpus from {corpus}…")
    docs = load_corpus(corpus)
    print(f"  {len(docs)} document(s) found")

    # Build (source_path, chunk_index, content, metadata) rows
    rows: list[tuple[str, int, str, dict, str]] = []  # (src, idx, content, meta, chunk_text)
    for doc in docs:
        chunks = chunk_text(doc.body, args.chunk_tokens, args.overlap_tokens)
        for i, chunk in enumerate(chunks):
            rows.append((doc.path, i, chunk, doc.metadata))

    print(f"  {len(rows)} chunk(s) to embed")

    # Embed in batches
    texts = [r[2] for r in rows]
    vectors: list[list[float]] = []
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        print(f"  Embedding batch {batch_start // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1}…")
        vectors.extend(embed_batch(batch))

    print("Writing to database…")
    conn = await asyncpg.connect(dsn)
    try:
        async with conn.transaction():
            await conn.execute("TRUNCATE TABLE chunks")
            for (src, idx, content, meta), vector in zip(rows, vectors):
                vec_str = "[" + ",".join(str(v) for v in vector) + "]"
                await conn.execute(
                    """
                    INSERT INTO chunks (id, source_path, chunk_index, content, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6::vector)
                    """,
                    uuid.uuid4(),
                    src,
                    idx,
                    content,
                    json.dumps(meta),
                    vec_str,
                )
    finally:
        await conn.close()

    print(f"Done. {len(rows)} chunk(s) indexed.")


if __name__ == "__main__":
    asyncio.run(main())
