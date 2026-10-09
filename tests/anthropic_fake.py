"""A fake Anthropic API behind the real SDK: requests go through `anthropic.AsyncAnthropic`
and a mocked HTTP transport, so tests see the real wire bodies and real response parsing."""

from __future__ import annotations

import json
from typing import Callable

import anthropic
import httpx

from app.llm import client as llm
from app.llm.client import HAIKU

Responder = Callable[[dict], httpx.Response]


class FakeApi:
    def __init__(self, respond: Responder):
        self.respond = respond
        self.bodies: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        return self.respond(body)

    def models(self) -> list[str]:
        return [b["model"] for b in self.bodies]


def install(monkeypatch, respond: Responder) -> FakeApi:
    fake = FakeApi(respond)
    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)),
    )
    monkeypatch.setattr(llm, "_client", client)
    return fake


def sse_response(
    blocks: list[dict],
    stop_reason: str = "end_turn",
    model: str = HAIKU,
    stop_details: dict | None = None,
) -> httpx.Response:
    """A streamed message. Blocks: {"type": "thinking"} or {"type": "text", "pieces": [...]}."""
    events: list[dict] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 120, "output_tokens": 1},
            },
        }
    ]
    for i, block in enumerate(blocks):
        if block["type"] == "thinking":
            events.append(
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                }
            )
            events.append(
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                }
            )
        else:
            events.append(
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                }
            )
            for piece in block["pieces"]:
                events.append(
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": piece},
                    }
                )
        events.append({"type": "content_block_stop", "index": i})
    delta: dict = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    events.append({"type": "message_delta", "delta": delta, "usage": {"output_tokens": 300}})
    events.append({"type": "message_stop"})
    payload = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=payload.encode()
    )


def message_response(
    content: list[dict],
    stop_reason: str,
    model: str = HAIKU,
    stop_details: dict | None = None,
) -> httpx.Response:
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 400, "output_tokens": 900},
    }
    if stop_details is not None:
        message["stop_details"] = stop_details
    return httpx.Response(200, json=message)


def error_response(status: int) -> httpx.Response:
    return httpx.Response(status, json={"type": "error", "error": {"type": "api_error"}})


def thinking() -> dict:
    return {"type": "thinking", "thinking": "", "signature": "sig"}


def tool_use(name: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": "toolu_1", "name": name, "input": tool_input}


def refusal_details(category: str = "cyber") -> dict:
    return {"type": "refusal", "category": category, "explanation": "declined"}
