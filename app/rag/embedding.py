"""Embed text chunks using Voyage AI voyage-3-large (1024 dims)."""

from __future__ import annotations

import os

import voyageai

MODEL = "voyage-3-large"
DIMS = 1024
_client: voyageai.Client | None = None


def get_client() -> voyageai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY environment variable not set")
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Returns list of 1024-dim float vectors."""
    if not texts:
        return []
    client = get_client()
    result = client.embed(texts, model=MODEL, input_type="document")
    return result.embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string with query input_type."""
    client = get_client()
    result = client.embed([text], model=MODEL, input_type="query")
    return result.embeddings[0]
