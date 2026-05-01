"""pgvector cosine-similarity retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass

from app.db import get_pool
from app.rag.embedding import embed_query


@dataclass
class ChunkResult:
    source_path: str
    title: str
    content: str
    score: float


async def retrieve(query: str, top_k: int = 5) -> list[ChunkResult]:
    """Embed query and retrieve top_k chunks ordered by cosine similarity."""
    vector = embed_query(query)
    vec_str = "[" + ",".join(str(v) for v in vector) + "]"

    pool = get_pool()
    rows = await pool.fetch(
        """
        SELECT source_path, content, metadata,
               1 - (embedding <=> $1::vector) AS score
        FROM chunks
        ORDER BY embedding <=> $1::vector
        LIMIT $2
        """,
        vec_str,
        top_k,
    )

    return [
        ChunkResult(
            source_path=r["source_path"],
            title=json.loads(r["metadata"]).get("title", r["source_path"]),
            content=r["content"],
            score=float(r["score"]),
        )
        for r in rows
    ]
