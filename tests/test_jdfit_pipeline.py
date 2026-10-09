"""/jdfit on Haiku 5.5 → Sonnet 5.5 with tool choice `auto` (ADR 009)."""

from unittest.mock import AsyncMock, patch

import pytest

from app.jdfit.pipeline import run_jdfit
from app.llm.client import HAIKU, SONNET
from app.rag.retrieval import ChunkResult
from tests import anthropic_fake as fake
from tests.anthropic_fake import error_response, message_response, thinking, tool_use
from tests.sse_events import of_type, parse_sse, text_of

_JD = """Senior Backend Engineer — AI Platform
Requirements:
- 3+ years of Python in production
- FastAPI or similar async web framework
- Experience building RAG / LLM applications
- Docker and Linux server administration
Nice to have:
- AWS
Soft skills:
- Explaining technical trade-offs to non-engineers
"""

_EXTRACTED = {
    "requirements": [
        {"spec": "Python", "category": "must_have"},
        {"spec": "FastAPI", "category": "must_have"},
        {"spec": "RAG / LLM applications", "category": "must_have"},
        {"spec": "Docker + Linux ops", "category": "must_have"},
        {"spec": "AWS", "category": "nice_to_have"},
        {"spec": "Explaining technical trade-offs", "category": "soft"},
    ]
}

_EVIDENCE = [
    [ChunkResult("projects/marks_gpt/overview.md", "Mark's GPT", "FastAPI backend", 0.81)],
    [
        ChunkResult("projects/marks_gpt/architecture.md", "Mark's GPT architecture", "SSE", 0.77),
        ChunkResult("projects/marks_gpt/overview.md", "Mark's GPT", "FastAPI backend", 0.6),
    ],
    [ChunkResult("projects/marks_gpt/rag_pipeline.md", "RAG pipeline", "Voyage + Chroma", 0.88)],
    [ChunkResult("experience/rgis/architecture.md", "RGIS architecture", "Docker on VPS", 0.52)],
    [ChunkResult("experience/rgis/results.md", "RGIS results", "unrelated", 0.21)],
    [],
]


def _report(gaps: list[str]) -> dict:
    return {
        "requirements": [
            {
                "spec": "Python",
                "category": "must_have",
                "score": 9,
                "reasoning": "Built Mark's GPT.",
            },
            {"spec": "FastAPI", "category": "must_have", "score": 8, "reasoning": "Backend in it."},
            {
                "spec": "RAG / LLM applications",
                "category": "must_have",
                "score": 9,
                "reasoning": "Shipped a RAG pipeline.",
            },
            {
                "spec": "Docker + Linux ops",
                "category": "must_have",
                "score": 7,
                "reasoning": "Runs containers on a VPS.",
            },
            {"spec": "AWS", "category": "nice_to_have", "score": 2, "reasoning": "No evidence."},
            {
                "spec": "Explaining technical trade-offs",
                "category": "soft",
                "score": 3,
                "reasoning": "Only indirect evidence.",
            },
        ],
        "overall_score": 8,
        "strengths": ["Production RAG", "Python + FastAPI"],
        "gaps": gaps,
        "summary": "Strong fit on the core stack. Cloud depth is unproven.",
    }


def _responder(report: dict, haiku_failure_at: str | None = None):
    def respond(body):
        tool = body["tools"][0]["name"]
        if body["model"] == HAIKU and haiku_failure_at == tool:
            return message_response(
                [thinking(), {"type": "text", "text": "Here is my analysis..."}], "end_turn"
            )
        result = _EXTRACTED if tool == "extract_requirements" else report
        return message_response([thinking(), tool_use(tool, result)], "tool_use", body["model"])

    return respond


async def _run(jd: str = _JD) -> list[dict]:
    with patch("app.jdfit.pipeline.retrieve_many", new=AsyncMock(return_value=_EVIDENCE)):
        return parse_sse("".join([chunk async for chunk in run_jdfit(jd)]))


@pytest.mark.asyncio
@pytest.mark.parametrize("gaps", [["No AWS evidence"], []], ids=["with_gaps", "no_gaps"])
async def test_report_layout_unchanged(monkeypatch, gaps):
    api = fake.install(monkeypatch, _responder(_report(gaps)))

    events = await _run()

    markdown = text_of(events)
    assert markdown.startswith("## Job Fit Report")
    sections = ["### Must-have requirements", "### Nice-to-have", "### Soft skills"]
    assert [markdown.index(s) for s in sections] == sorted(markdown.index(s) for s in sections)
    assert "**Python (9/10)**\nBuilt Mark's GPT." in markdown
    assert "**AWS (2/10)**\nNo evidence." in markdown
    assert "**Overall Fit: 8/10**" in markdown
    assert "**Strengths**\n- Production RAG\n- Python + FastAPI" in markdown
    assert ("**Gaps**\n- No AWS evidence" in markdown) == bool(gaps)
    assert markdown.endswith("Strong fit on the core stack. Cloud depth is unproven.")
    assert of_type(events, "model") == [{"type": "model", "model": "haiku"}]
    titles = [s["title"] for s in of_type(events, "citation")[0]["sources"]]
    assert titles == ["Mark's GPT", "Mark's GPT architecture", "RAG pipeline", "RGIS architecture"]
    assert of_type(events, "done") == [{"type": "done"}]
    assert api.models() == [HAIKU, HAIKU]


@pytest.mark.asyncio
async def test_steps_send_jd_and_evidence(monkeypatch):
    api = fake.install(monkeypatch, _responder(_report([])))

    await _run()

    extract_body, report_body = api.bodies
    assert extract_body["messages"] == [{"role": "user", "content": f"<JD>\n{_JD}\n</JD>"}]
    report_input = report_body["messages"][0]["content"]
    assert "Requirement: Python (category: must_have)" in report_input
    assert "Evidence 1 [RAG pipeline]: Voyage + Chroma" in report_input
    assert "Requirement: AWS (category: nice_to_have)\n  Evidence: none found" in report_input


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_step, expected_models, label",
    [
        ("extract_requirements", [HAIKU, SONNET, HAIKU], "haiku"),
        ("submit_jdfit_report", [HAIKU, HAIKU, SONNET], "sonnet"),
    ],
)
async def test_text_reply_falls_back_to_sonnet(monkeypatch, failing_step, expected_models, label):
    api = fake.install(monkeypatch, _responder(_report([]), haiku_failure_at=failing_step))

    events = await _run()

    assert api.models() == expected_models
    assert text_of(events).startswith("## Job Fit Report")
    assert of_type(events, "model") == [{"type": "model", "model": label}]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 529])
async def test_haiku_overloaded_falls_back_to_sonnet(monkeypatch, status):
    def respond(body):
        if body["model"] == HAIKU:
            return error_response(status)
        return _responder(_report([]))(body)

    api = fake.install(monkeypatch, respond)

    events = await _run()

    assert api.models() == [HAIKU, SONNET, HAIKU, SONNET]
    assert text_of(events).startswith("## Job Fit Report")


@pytest.mark.asyncio
async def test_both_models_reply_in_text_gives_friendly_error(monkeypatch):
    fake.install(
        monkeypatch,
        lambda body: message_response([{"type": "text", "text": "Looks like a fit!"}], "end_turn"),
    )

    events = await _run()

    assert text_of(events) == "Couldn't reach the AI right now — try again in a moment."
    assert of_type(events, "done") == [{"type": "done"}]
