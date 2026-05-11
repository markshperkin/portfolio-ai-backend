"""Embed text chunks using Voyage AI voyage-3-large (1024 dims, TASK-24-BE error handling)."""

from __future__ import annotations

import logging
import os

import voyageai

MODEL = "voyage-3-large"
DIMS = 1024
log = logging.getLogger(__name__)

_client: voyageai.Client | None = None


class EmbeddingError(Exception):
    """Raised when the embedding call fails."""


def get_client() -> voyageai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY environment variable not set")
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. Raises EmbeddingError on failure."""
    if not texts:
        return []
    try:
        client = get_client()
        result = client.embed(texts, model=MODEL, input_type="document")
        return result.embeddings  # type: ignore[return-value]
    except Exception as e:
        log.error("Voyage embed_batch error: %s", e)
        raise EmbeddingError("batch") from e


def embed_query(text: str) -> list[float]:
    """Embed a single query string. Raises EmbeddingError on failure."""
    try:
        client = get_client()
        result = client.embed([text], model=MODEL, input_type="query")
        return result.embeddings[0]  # type: ignore[return-value]
    except Exception as e:
        log.error("Voyage embed_query error: %s", e)
        raise EmbeddingError("query") from e
