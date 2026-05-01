"""Anthropic Haiku streaming client."""

from __future__ import annotations

import os
from typing import AsyncGenerator

import anthropic

MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.AsyncAnthropic(api_key=api_key)
    return _client


async def stream_completion(
    messages: list[dict],
    system: str,
) -> AsyncGenerator[str, None]:
    """Stream text tokens from Anthropic. Yields raw text chunks."""
    client = get_client()
    async with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            yield text
