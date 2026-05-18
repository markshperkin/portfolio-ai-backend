"""Anthropic Haiku streaming client with timeout and error mapping (TASK-24-BE)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator

import anthropic

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
MAX_TOKENS = 1024
STREAM_TIMEOUT = 30  # seconds before giving up on a stalled stream

log = logging.getLogger(__name__)

_client: anthropic.AsyncAnthropic | None = None


def get_client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.AsyncAnthropic(api_key=api_key)
    return _client


class LLMError(Exception):
    """Raised when the LLM call fails in a way chat.py should handle."""


async def stream_completion(
    model: str,
    messages: list[dict[str, Any]],
    system: str,
) -> AsyncGenerator[str, None]:
    """Stream text tokens from Anthropic. Raises LLMError on failure."""
    client = get_client()
    try:
        async with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,  # type: ignore[arg-type]
        ) as stream:
            async for text in stream.text_stream:
                yield text
    except anthropic.APIConnectionError as e:
        log.error("Anthropic connection error: %s", e)
        raise LLMError("connection") from e
    except anthropic.RateLimitError as e:
        log.error("Anthropic rate limit: %s", e)
        raise LLMError("rate_limit") from e
    except anthropic.APIStatusError as e:
        log.error("Anthropic API error %s: %s", e.status_code, e.message)
        raise LLMError("api_error") from e
    except asyncio.TimeoutError:
        log.error("Anthropic stream timeout after %ss", STREAM_TIMEOUT)
        raise LLMError("timeout")
