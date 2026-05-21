"""Agentic /jdfit pipeline: extract → retrieve → synthesize → render markdown."""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, Literal

from app.jdfit.prompts import (
    EXTRACT_SYSTEM,
    EXTRACT_TOOL,
    SYNTHESIZE_SYSTEM,
    SYNTHESIZE_TOOL,
)
from app.jdfit.schemas import ExtractionResult, JdfitReport, ScoredRequirement
from app.llm.client import HAIKU, SONNET, LLMError, call_tool
from app.models import (
    CitationEvent,
    CitationSource,
    DeltaEvent,
    DoneEvent,
    ModelEvent,
    RetrievalStepEvent,
    sse_format,
)
from app.rag.retrieval import ChunkResult, retrieve

log = logging.getLogger(__name__)

_T_WEAK = 0.35
_NO_REQS_MSG = (
    "Couldn't find any job requirements in that text. Paste a real job description and try again."
)
_FALLBACK_MSG = "Something went wrong analysing the job description. Try again in a moment."

_CATEGORY_LABELS = {
    "must_have": "Must-have requirements",
    "nice_to_have": "Nice-to-have",
    "soft": "Soft skills",
}
_CATEGORY_ORDER = ["must_have", "nice_to_have", "soft"]


_FALLBACK_MODELS: list[tuple[str, Literal["haiku", "sonnet"]]] = [
    (HAIKU, "haiku"),
    (SONNET, "sonnet"),
]


async def _call_with_fallback(
    messages: list[dict],
    system: str,
    tool: dict,
    tool_name: str,
) -> tuple[dict, Literal["haiku", "sonnet"]]:
    """Try Haiku, fall back to Sonnet. Returns (input_dict, model_name)."""
    for model_id, model_name in _FALLBACK_MODELS:
        try:
            result = await call_tool(model_id, messages, system, tool, tool_name)
            return result, model_name
        except LLMError:
            continue
    raise LLMError("both_models_failed")


def _render_markdown(report: JdfitReport) -> str:
    lines = ["## Job Fit Report", ""]
    by_category: dict[str, list[ScoredRequirement]] = {k: [] for k in _CATEGORY_ORDER}
    for req in report.requirements:
        by_category.get(req.category, by_category["soft"]).append(req)

    first_section = True
    for cat_key in _CATEGORY_ORDER:
        reqs = by_category[cat_key]
        if not reqs:
            continue
        if not first_section:
            lines.append("")
        lines.append(f"### {_CATEGORY_LABELS[cat_key]}")
        for req in reqs:
            lines.append(f"**{req.spec} ({req.score}/10)**")
            lines.append(req.reasoning)
            lines.append("")
        first_section = False

    lines.append("---")
    lines.append("**Summary**")
    lines.append(report.summary)
    return "\n".join(lines)


async def run_jdfit(jd: str) -> AsyncGenerator[str, None]:
    """Full /jdfit pipeline. Yields SSE-formatted event strings."""

    # Step 1 — extract requirements
    yield sse_format(RetrievalStepEvent(step="extracting", detail="reading job description"))

    extract_messages = [{"role": "user", "content": f"<JD>\n{jd}\n</JD>"}]
    try:
        raw, _ = await _call_with_fallback(
            extract_messages, EXTRACT_SYSTEM, EXTRACT_TOOL, "extract_requirements"
        )
        extraction = ExtractionResult.model_validate(raw)
    except (LLMError, Exception):
        log.exception("jdfit extract step failed")
        yield sse_format(DeltaEvent(text=_FALLBACK_MSG))
        yield sse_format(DoneEvent())
        return

    if not extraction.requirements:
        yield sse_format(DeltaEvent(text=_NO_REQS_MSG))
        yield sse_format(DoneEvent())
        return

    # Step 2 — retrieve evidence per requirement in parallel
    n = len(extraction.requirements)
    yield sse_format(
        RetrievalStepEvent(step="retrieving", detail=f"searching evidence for {n} requirements")
    )

    specs = [r.spec for r in extraction.requirements]
    try:
        results: list[list[ChunkResult]] = await asyncio.gather(*[retrieve(spec) for spec in specs])
    except Exception:
        log.exception("jdfit retrieval step failed")
        yield sse_format(DeltaEvent(text=_FALLBACK_MSG))
        yield sse_format(DoneEvent())
        return

    # filter by threshold
    evidence: list[list[ChunkResult]] = [
        [c for c in chunks if c.score >= _T_WEAK] for chunks in results
    ]

    # Step 3 — synthesize report
    yield sse_format(RetrievalStepEvent(step="synthesizing"))

    synth_blocks: list[str] = []
    for req, chunks in zip(extraction.requirements, evidence):
        block = f"Requirement: {req.spec} (category: {req.category})\n"
        if chunks:
            for i, c in enumerate(chunks, 1):
                block += f"  Evidence {i} [{c.title}]: {c.content[:400]}\n"
        else:
            block += "  Evidence: none found in knowledge base\n"
        synth_blocks.append(block)

    synth_content = "\n".join(synth_blocks)
    synth_messages = [{"role": "user", "content": synth_content}]

    try:
        raw_report, model_name = await _call_with_fallback(
            synth_messages, SYNTHESIZE_SYSTEM, SYNTHESIZE_TOOL, "submit_jdfit_report"
        )
        report = JdfitReport.model_validate(raw_report)
    except (LLMError, Exception):
        log.exception("jdfit synthesize step failed")
        yield sse_format(DeltaEvent(text=_FALLBACK_MSG))
        yield sse_format(DoneEvent())
        return

    yield sse_format(ModelEvent(model=model_name))

    # Render and stream markdown
    markdown = _render_markdown(report)
    yield sse_format(DeltaEvent(text=markdown))

    # Citations — dedupe all sources used across all requirements
    seen: set[str] = set()
    sources: list[CitationSource] = []
    for chunks in evidence:
        for chunk in chunks:
            if chunk.source_path not in seen:
                seen.add(chunk.source_path)
                sources.append(CitationSource(title=chunk.title))
    if sources:
        yield sse_format(CitationEvent(sources=sources))

    yield sse_format(DoneEvent())
