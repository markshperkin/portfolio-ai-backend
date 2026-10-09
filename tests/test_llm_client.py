"""LLM client against the real Anthropic SDK over a mocked HTTP transport (ADR 009).

The SDK builds the request bodies and parses the responses, so these tests check
what actually goes over the wire for both models.
"""

import json

import anthropic
import httpx
import pytest

from app.jdfit.prompts import EXTRACT_SYSTEM, EXTRACT_TOOL, SYNTHESIZE_SYSTEM, SYNTHESIZE_TOOL
from app.llm import client as llm
from app.llm.client import HAIKU, SONNET, LLMError, call_tool, stream_completion
from app.rag.query_planner import _SYSTEM as PLANNER_SYSTEM
from app.rag.query_planner import _TOOL as PLANNER_TOOL

_SAMPLING_AND_THINKING = ("temperature", "top_p", "top_k", "thinking")

_CONVERSATIONS = {
    "single_question": [{"role": "user", "content": "What did Mark build at RGIS?"}],
    "follow_up": [
        {"role": "user", "content": "What projects has Mark worked on?"},
        {"role": "assistant", "content": "He built Mark's GPT and an agentic investment firm."},
        {"role": "user", "content": "Tell me more about the investment one."},
    ],
    "after_lost_connection": [
        {"role": "assistant", "content": "Hey — ask me anything about Mark."},
        {"role": "user", "content": "what's his stack?"},
        {"role": "assistant", "content": "[Connection lost]"},
        {"role": "user", "content": "what's his stack?"},
    ],
}

_TOOL_STEPS = {
    "planner": (PLANNER_SYSTEM, PLANNER_TOOL),
    "jdfit_extract": (EXTRACT_SYSTEM, EXTRACT_TOOL),
    "jdfit_report": (SYNTHESIZE_SYSTEM, SYNTHESIZE_TOOL),
}


class _FakeApi:
    def __init__(self, respond):
        self.respond = respond
        self.bodies: list[dict] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        return self.respond(body)


@pytest.fixture
def api(monkeypatch):
    def install(respond) -> _FakeApi:
        fake = _FakeApi(respond)
        client = anthropic.AsyncAnthropic(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake.handle)),
        )
        monkeypatch.setattr(llm, "_client", client)
        return fake

    return install


def _sse_response(blocks: list[dict], stop_reason: str = "end_turn", model: str = HAIKU):
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
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 300},
        }
    )
    events.append({"type": "message_stop"})
    payload = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content=payload.encode()
    )


def _message_response(content: list[dict], stop_reason: str, model: str = HAIKU):
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 400, "output_tokens": 900},
        },
    )


def _thinking() -> dict:
    return {"type": "thinking", "thinking": "", "signature": "sig"}


def _tool_use(name: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": "toolu_1", "name": name, "input": tool_input}


def _assert_shared_shape(body: dict, model: str, system: str, messages: list[dict]) -> None:
    assert body["model"] == model
    assert body["max_tokens"] == 16000
    assert body["output_config"] == {"effort": "high"}
    assert body["system"] == system
    assert body["messages"] == messages
    assert body["messages"][-1]["role"] == "user"
    for key in _SAMPLING_AND_THINKING:
        assert key not in body


async def _drain(model: str, messages: list[dict], system: str) -> str:
    return "".join([t async for t in stream_completion(model, messages, system)])


# --- chat answer stream ---


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [HAIKU, SONNET])
@pytest.mark.parametrize("conversation", _CONVERSATIONS.values(), ids=_CONVERSATIONS.keys())
async def test_stream_request_shape(api, model, conversation):
    fake = api(lambda body: _sse_response([{"type": "text", "pieces": ["ok"]}]))

    await _drain(model, conversation, "persona + context")

    assert len(fake.bodies) == 1
    body = fake.bodies[0]
    _assert_shared_shape(body, model, "persona + context", conversation)
    assert body["stream"] is True
    assert "tools" not in body and "tool_choice" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", _CONVERSATIONS.values(), ids=_CONVERSATIONS.keys())
async def test_stream_identical_for_both_models(api, conversation):
    fake = api(lambda body: _sse_response([{"type": "text", "pieces": ["ok"]}]))

    await _drain(HAIKU, conversation, "system")
    await _drain(SONNET, conversation, "system")

    haiku_body, sonnet_body = fake.bodies
    assert haiku_body.pop("model") == HAIKU
    assert sonnet_body.pop("model") == SONNET
    assert haiku_body == sonnet_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocks, expected",
    [
        (
            [{"type": "thinking"}, {"type": "text", "pieces": ["Mark built ", "a RAG bot."]}],
            "Mark built a RAG bot.",
        ),
        ([{"type": "text", "pieces": ["His email is on the ", "contact card."]}], None),
        (
            [
                {"type": "thinking"},
                {"type": "text", "pieces": ["He led ", "the RGIS ", "lead-finder ", "project."]},
            ],
            "He led the RGIS lead-finder project.",
        ),
    ],
    ids=["thinking_then_text", "no_thinking", "thinking_then_long_text"],
)
async def test_stream_yields_only_answer_text(api, blocks, expected):
    api(lambda body: _sse_response(blocks))
    text_blocks = [b for b in blocks if b["type"] == "text"]
    expected = expected or "".join(p for b in text_blocks for p in b["pieces"])

    out = await _drain(HAIKU, _CONVERSATIONS["single_question"], "system")

    assert out == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, code",
    [(429, "rate_limit"), (500, "api_error"), (529, "api_error"), (400, "api_error")],
)
async def test_stream_maps_api_errors(api, status, code):
    api(lambda body: httpx.Response(status, json={"type": "error", "error": {"type": "x"}}))

    with pytest.raises(LLMError, match=code):
        await _drain(HAIKU, _CONVERSATIONS["single_question"], "system")


# --- tool steps (planner, /jdfit extract, /jdfit report) ---


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [HAIKU, SONNET])
@pytest.mark.parametrize("step", _TOOL_STEPS.keys())
async def test_call_tool_request_shape(api, model, step):
    system, tool = _TOOL_STEPS[step]
    messages = [{"role": "user", "content": "Senior Python engineer, FastAPI, RAG, 3+ years"}]
    fake = api(lambda body: _message_response([_tool_use(tool["name"], {})], "tool_use"))

    await call_tool(model, messages, system, tool, tool["name"])

    body = fake.bodies[0]
    _assert_shared_shape(body, model, system, messages)
    assert body["tool_choice"] == {"type": "auto"}
    assert body["tools"] == [tool]
    assert "stream" not in body or body["stream"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("step", _TOOL_STEPS.keys())
async def test_call_tool_identical_for_both_models(api, step):
    system, tool = _TOOL_STEPS[step]
    messages = [{"role": "user", "content": "how do I reach Mark?"}]
    fake = api(lambda body: _message_response([_tool_use(tool["name"], {})], "tool_use"))

    await call_tool(HAIKU, messages, system, tool, tool["name"])
    await call_tool(SONNET, messages, system, tool, tool["name"])

    haiku_body, sonnet_body = fake.bodies
    assert haiku_body.pop("model") == HAIKU
    assert sonnet_body.pop("model") == SONNET
    assert haiku_body == sonnet_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    [
        [_tool_use("plan_queries", {"is_abusive": False, "queries": ["RGIS lead finder"]})],
        [
            _thinking(),
            _tool_use("plan_queries", {"is_abusive": False, "queries": ["RGIS lead finder"]}),
        ],
        [
            _thinking(),
            {"type": "text", "text": "Planning the search."},
            _tool_use("plan_queries", {"is_abusive": False, "queries": ["RGIS lead finder"]}),
        ],
    ],
    ids=["tool_only", "thinking_then_tool", "thinking_text_then_tool"],
)
async def test_call_tool_returns_tool_input(api, content):
    api(lambda body: _message_response(content, "tool_use"))

    result = await call_tool(
        HAIKU,
        [{"role": "user", "content": "what did he do at RGIS?"}],
        PLANNER_SYSTEM,
        PLANNER_TOOL,
        "plan_queries",
    )

    assert result == {"is_abusive": False, "queries": ["RGIS lead finder"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content, stop_reason, code",
    [
        ([{"type": "text", "text": "Queries: RGIS, leads"}], "end_turn", "tool_not_called"),
        ([_thinking(), {"type": "text", "text": "Search RGIS."}], "end_turn", "tool_not_called"),
        ([_thinking()], "max_tokens", "max_tokens"),
    ],
    ids=["text_reply", "thinking_then_text_reply", "cut_off"],
)
async def test_call_tool_no_result_raises_llm_error(api, content, stop_reason, code):
    api(lambda body: _message_response(content, stop_reason))

    with pytest.raises(LLMError, match=code):
        await call_tool(
            HAIKU,
            [{"role": "user", "content": "what did he do at RGIS?"}],
            PLANNER_SYSTEM,
            PLANNER_TOOL,
            "plan_queries",
        )
