"""LLM client against the real Anthropic SDK over a mocked HTTP transport (ADR 009).

The SDK builds the request bodies and parses the responses, so these tests check
what actually goes over the wire for both models.
"""

import httpx
import pytest

from app.jdfit.prompts import EXTRACT_SYSTEM, EXTRACT_TOOL, SYNTHESIZE_SYSTEM, SYNTHESIZE_TOOL
from app.llm.client import HAIKU, SONNET, LLMError, call_tool, stream_completion
from app.rag.query_planner import _SYSTEM as PLANNER_SYSTEM
from app.rag.query_planner import _TOOL as PLANNER_TOOL
from tests import anthropic_fake as fake
from tests.anthropic_fake import message_response, sse_response, thinking, tool_use

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


@pytest.fixture
def api(monkeypatch):
    return lambda respond: fake.install(monkeypatch, respond)


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
    fake = api(lambda body: sse_response([{"type": "text", "pieces": ["ok"]}]))

    await _drain(model, conversation, "persona + context")

    assert len(fake.bodies) == 1
    body = fake.bodies[0]
    _assert_shared_shape(body, model, "persona + context", conversation)
    assert body["stream"] is True
    assert "tools" not in body and "tool_choice" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize("conversation", _CONVERSATIONS.values(), ids=_CONVERSATIONS.keys())
async def test_stream_identical_for_both_models(api, conversation):
    fake = api(lambda body: sse_response([{"type": "text", "pieces": ["ok"]}]))

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
    api(lambda body: sse_response(blocks))
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
    fake = api(lambda body: message_response([tool_use(tool["name"], {})], "tool_use"))

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
    fake = api(lambda body: message_response([tool_use(tool["name"], {})], "tool_use"))

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
        [tool_use("plan_queries", {"is_abusive": False, "queries": ["RGIS lead finder"]})],
        [
            thinking(),
            tool_use("plan_queries", {"is_abusive": False, "queries": ["RGIS lead finder"]}),
        ],
        [
            thinking(),
            {"type": "text", "text": "Planning the search."},
            tool_use("plan_queries", {"is_abusive": False, "queries": ["RGIS lead finder"]}),
        ],
    ],
    ids=["tool_only", "thinking_then_tool", "thinking_text_then_tool"],
)
async def test_call_tool_returns_tool_input(api, content):
    api(lambda body: message_response(content, "tool_use"))

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
        ([thinking(), {"type": "text", "text": "Search RGIS."}], "end_turn", "tool_not_called"),
        ([thinking()], "max_tokens", "max_tokens"),
    ],
    ids=["text_reply", "thinking_then_text_reply", "cut_off"],
)
async def test_call_tool_no_result_raises_llm_error(api, content, stop_reason, code):
    api(lambda body: message_response(content, stop_reason))

    with pytest.raises(LLMError, match=code):
        await call_tool(
            HAIKU,
            [{"role": "user", "content": "what did he do at RGIS?"}],
            PLANNER_SYSTEM,
            PLANNER_TOOL,
            "plan_queries",
        )
