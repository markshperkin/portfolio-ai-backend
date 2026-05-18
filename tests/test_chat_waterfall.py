"""Tests for LLM waterfall fallback: haiku → sonnet → both down."""

import json

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, patch

from app.api.chat import router
from app.llm.client import HAIKU, LLMError
from app.rag.retrieval import ChunkResult

_app = FastAPI()
_app.include_router(router)

_CHUNKS = [ChunkResult(source_path="a.md", title="A", content="content", score=0.9)]

_REQUEST = {"messages": [{"role": "user", "content": "tell me about mark"}]}


def _parse_sse(content: bytes) -> list[dict]:
    events = []
    for block in content.split(b"\n\n"):
        if not block.strip():
            continue
        event_type = None
        data = None
        for line in block.strip().split(b"\n"):
            if line.startswith(b"event: "):
                event_type = line[7:].decode()
            elif line.startswith(b"data: "):
                data = json.loads(line[6:])
        if event_type and data is not None:
            events.append({"type": event_type, **data})
    return events


def _base_patches():
    return [
        patch("app.api.chat.check_and_log_abuse", new=AsyncMock(return_value=(False, ""))),
        patch("app.api.chat.check_rate_limit", new=AsyncMock(return_value=(True, ""))),
        patch("app.api.chat.retrieve", new=AsyncMock(return_value=_CHUNKS)),
        patch("app.api.chat.build_system_prompt", return_value="system"),
    ]


async def _post(patches: list, gen_fn):
    with patch("app.api.chat.stream_completion", side_effect=gen_fn):
        for p in patches:
            p.start()
        try:
            async with AsyncClient(transport=ASGITransport(app=_app), base_url="http://test") as ac:
                r = await ac.post("/api/chat", json=_REQUEST)
            return _parse_sse(r.content)
        finally:
            for p in patches:
                p.stop()


@pytest.mark.asyncio
async def test_haiku_success():
    async def gen(model_id, messages, system):
        yield "hello"
        yield " mark"

    events = await _post(_base_patches(), gen)

    model_events = [e for e in events if e["type"] == "model"]
    delta_events = [e for e in events if e["type"] == "delta"]

    assert len(model_events) == 1
    assert model_events[0]["model"] == "haiku"
    assert "".join(e["text"] for e in delta_events) == "hello mark"


@pytest.mark.asyncio
async def test_haiku_fails_sonnet_succeeds():
    async def gen(model_id, messages, system):
        if model_id == HAIKU:
            raise LLMError("haiku down")
            yield  # make it a generator
        yield "sonnet response"

    events = await _post(_base_patches(), gen)

    model_events = [e for e in events if e["type"] == "model"]
    delta_events = [e for e in events if e["type"] == "delta"]

    assert len(model_events) == 1
    assert model_events[0]["model"] == "sonnet"
    assert delta_events[0]["text"] == "sonnet response"


@pytest.mark.asyncio
async def test_both_models_fail():
    async def gen(model_id, messages, system):
        raise LLMError("all down")
        yield

    events = await _post(_base_patches(), gen)

    delta_events = [e for e in events if e["type"] == "delta"]
    done_events = [e for e in events if e["type"] == "done"]

    assert any("unavailable" in e["text"].lower() for e in delta_events)
    assert len(done_events) == 1


@pytest.mark.asyncio
async def test_model_event_emitted_once_on_haiku():
    call_count = 0

    async def gen(model_id, messages, system):
        nonlocal call_count
        call_count += 1
        for word in ["one", "two", "three"]:
            yield word

    events = await _post(_base_patches(), gen)

    assert call_count == 1
    assert sum(1 for e in events if e["type"] == "model") == 1
