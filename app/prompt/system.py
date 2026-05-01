"""Build the system prompt injected with retrieved context."""

from __future__ import annotations

from app.rag.retrieval import ChunkResult

PERSONA = """\
You are Mark's GPT — an AI assistant that answers questions about Mark Shperkin based \
on provided context. Mark is a software engineer specialising in applied AI.

Answer based on the context below. Be specific and grounded — do not invent details \
not present in the context. If the context does not contain enough information to answer, \
say so clearly and suggest the visitor contact Mark directly.

When the context spans multiple sources, synthesise across them rather than answering \
from a single chunk."""


def build_system_prompt(chunks: list[ChunkResult]) -> str:
    if not chunks:
        return PERSONA + "\n\nNo context retrieved. Tell the visitor you don't have enough information on that topic."

    blocks = []
    seen: set[str] = set()
    for chunk in chunks:
        header = f"[Source: {chunk.title}]"
        if chunk.source_path not in seen:
            seen.add(chunk.source_path)
        blocks.append(f"{header}\n{chunk.content}")

    context = "\n\n---\n\n".join(blocks)
    return f"{PERSONA}\n\nContext:\n\n{context}"
