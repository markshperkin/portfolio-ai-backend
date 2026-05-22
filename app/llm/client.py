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
MAX_TOKENS_JDFIT = 4096
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


async def call_tool(
    model: str,
    messages: list[dict[str, Any]],
    system: str,
    tool: dict[str, Any],
    tool_name: str,
) -> dict[str, Any]:
    """Non-streaming Anthropic tool-use call. Returns the validated tool input dict.

    Forces the model to call tool_name. Raises LLMError on failure.
    """
    client = get_client()
    try:
        response = await client.messages.create(  # type: ignore[call-overload]
            model=model,
            max_tokens=MAX_TOKENS_JDFIT,
            system=system,
            messages=messages,  # type: ignore[arg-type]
            tools=[tool],  # type: ignore[arg-type]
            tool_choice={"type": "tool", "name": tool_name},
        )
        if response.stop_reason == "max_tokens":
            log.error("call_tool max_tokens: model=%s tool=%s", model, tool_name)
            raise LLMError("max_tokens")
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input  # type: ignore[return-value]
        raise LLMError("tool_not_called")
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
        log.error("call_tool timeout for model=%s tool=%s", model, tool_name)
        raise LLMError("timeout")
