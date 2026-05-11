"""Chroma persistent client singleton."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb

COLLECTION_NAME = "chunks"
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        path = os.environ.get("CHROMA_PATH", "data/chroma_db")
        Path(path).mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=path)
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection
