"""Chroma cosine-similarity retrieval."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.rag.embedding import embed_query
from app.rag.store import get_collection


@dataclass
class ChunkResult:
    source_path: str
    title: str
    content: str
    score: float


async def retrieve(query: str, top_k: int = 5) -> list[ChunkResult]:
    """Embed query and retrieve top_k chunks ordered by cosine similarity."""
    vector = embed_query(query)
    return await asyncio.to_thread(_query_chroma, vector, top_k)


def _query_chroma(vector: list[float], top_k: int) -> list[ChunkResult]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    res = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[ChunkResult] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        score = 1.0 - dist  # cosine space: distance = 1 - similarity
        chunks.append(ChunkResult(
            source_path=meta.get("path", ""),
            title=meta.get("title", meta.get("path", "")),
            content=doc,
            score=score,
        ))
    return chunks
