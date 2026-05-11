"""Chunk a document body into overlapping word-token windows."""

from __future__ import annotations


def chunk_text(text: str, chunk_tokens: int = 800, overlap_tokens: int = 100) -> list[str]:
    """Split text into chunks of ~chunk_tokens words with overlap_tokens overlap.

    Uses whitespace-split word count as a proxy for tokens (accurate to ±20%
    for typical English prose with Claude/Voyage tokenizers).
    """
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap_tokens  # slide back by overlap

    return chunks
