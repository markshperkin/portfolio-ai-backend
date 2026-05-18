"""Chroma persistent client singleton."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb

COLLECTION_NAME = "chunks"
_client: chromadb.ClientAPI | None = None


def get_collection() -> chromadb.Collection:
    global _client
    if _client is None:
        path = os.environ.get("CHROMA_PATH", "data/chroma_db")
        Path(path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=path)
    return _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
