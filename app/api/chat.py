from __future__ import annotations

import logging
import os
from typing import AsyncGenerator, Literal

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.commands.handler import detect_command, handle_command
from app.jdfit.pipeline import run_jdfit
from app.llm.client import HAIKU, SONNET, LLMError, stream_completion
from app.models import (
    CitationEvent,
    CitationSource,
    DebugEvent,
    DeltaEvent,
    DoneEvent,
    ErrorEvent,
    ModelEvent,
    RetrievalStepEvent,
    sse_format,
)
from app.prompt.contacts import MARK_EMAIL
from app.prompt.system import build_system_prompt_grouped
from app.rag.embedding import EmbeddingError
from app.rag.query_planner import plan_queries
from app.rag.retrieval import ChunkResult, retrieve_many
from app.security.abuse import check_and_log_abuse
from app.security.rate_limit import check_rate_limit

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_T_WEAK = 0.35
_WORD_CAP = 3000

_WORD_CAP_MSG = f"You've reached the {_WORD_CAP}-word input cap. Please shorten your message."
_NO_MATCH_MSG = (
    "I don't have enough information about that in my knowledge base. "
    "Ask about Mark's projects, work, or skills — or reach him directly "
    f"at {MARK_EMAIL}."
)
_FALLBACK_LLM = (
    f"The models are taking a nap — try again in a moment. Or reach Mark directly at {MARK_EMAIL}."
)
_FALLBACK_EMBEDDING = (
    f"Voyage AI is rate-limiting me (free tier problems). "
    f"Slow down a bit and try again, or email Mark at {MARK_EMAIL}."
)
_FALLBACK_DB = (
    f"The knowledge base is temporarily unavailable. "
    f"Email Mark at {MARK_EMAIL} and he'll respond directly."
)


def _llm_err_msg(e: LLMError | None) -> str:
    code = str(e) if e else ""
    if code in ("rate_limit", "api_error"):
        return (
            "AI models are currently overloaded — try again in a moment. Or reach Mark at "
            + MARK_EMAIL
            + "."
        )
    if code == "timeout":
        return "The AI timed out — try again. Or reach Mark at " + MARK_EMAIL + "."
    return (
        "Both Haiku and Sonnet are currently unavailable. Try again later or email Mark at "
        + MARK_EMAIL
        + "."
    )


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: str | None = None


async def _chat_stream(request: ChatRequest, client_ip: str) -> AsyncGenerator[str, None]:
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        yield sse_format(ErrorEvent(code="no_user_message", message="No user message provided."))
        return

    query = user_messages[-1].content

    # 1. Word cap (applies to all input including /jdfit bodies)
    if len(query.split()) > _WORD_CAP:
        yield sse_format(DeltaEvent(text=_WORD_CAP_MSG))
        yield sse_format(DoneEvent())
        return

    # 2. Slash command detection
    cmd = detect_command(query)

    if cmd is not None:
        name, body = cmd

        # /jdfit with body — runs through abuse + rate-limit then pipeline
        if name == "jdfit" and body:
            is_abuse, abuse_msg = await check_and_log_abuse(client_ip, body)
            if is_abuse:
                yield sse_format(DeltaEvent(text=abuse_msg))
                yield sse_format(DoneEvent())
                return

            allowed, rate_msg = await check_rate_limit(client_ip)
            if not allowed:
                yield sse_format(DeltaEvent(text=rate_msg))
                yield sse_format(DoneEvent())
                return

            async for chunk in run_jdfit(body):
                yield chunk
            return

        # All other commands (incl. /jdfit with no body)
        async for chunk in handle_command(name, body):
            yield chunk
        return

    # 3. Abuse / jailbreak check
    is_abuse, abuse_msg = await check_and_log_abuse(client_ip, query)
    if is_abuse:
        yield sse_format(DeltaEvent(text=abuse_msg))
        yield sse_format(DoneEvent())
        return

    # 4. Rate limit
    allowed, rate_msg = await check_rate_limit(client_ip)
    if not allowed:
        yield sse_format(DeltaEvent(text=rate_msg))
        yield sse_format(DoneEvent())
        return

    # 5. Query planning: intent detection + query generation
    try:
        yield sse_format(RetrievalStepEvent(step="planning", detail="planning queries"))
        try:
            is_abusive, refusal_msg, queries = await plan_queries(query, client_ip)
        except LLMError as e:
            yield sse_format(DeltaEvent(text=_llm_err_msg(e)))
            yield sse_format(DoneEvent())
            return

        if is_abusive:
            yield sse_format(DeltaEvent(text=refusal_msg))
            yield sse_format(DoneEvent())
            return

        # 6. RAG retrieval (skipped for conversational messages with no queries)
        if queries:
            queries = [query] + queries  # raw message preserved as anchor query

        query_chunk_pairs: list[tuple[str, list[ChunkResult]]] = []
        if queries:
            yield sse_format(
                RetrievalStepEvent(step="retrieving", detail="searching knowledge base")
            )
            try:
                all_results = await retrieve_many(queries, top_k=3)
            except EmbeddingError:
                yield sse_format(DeltaEvent(text=_FALLBACK_EMBEDDING))
                yield sse_format(DoneEvent())
                return
            except Exception:
                log.exception("DB retrieval error for query=%r", query[:80])
                yield sse_format(DeltaEvent(text=_FALLBACK_DB))
                yield sse_format(DoneEvent())
                return

            for q, chunks in zip(queries, all_results):
                passing = [c for c in chunks if c.score >= _T_WEAK]
                if passing:
                    query_chunk_pairs.append((q, passing))

            if os.environ.get("APP_ENV") == "test":
                yield sse_format(
                    DebugEvent(
                        data={
                            "queries": queries,
                            "results": [
                                {
                                    "query": q,
                                    "chunks": [
                                        {
                                            "title": c.title,
                                            "score": round(c.score, 3),
                                            "passed": c.score >= _T_WEAK,
                                        }
                                        for c in chunks
                                    ],
                                }
                                for q, chunks in zip(queries, all_results)
                            ],
                        }
                    )
                )

            if not query_chunk_pairs:
                yield sse_format(DeltaEvent(text=_NO_MATCH_MSG))
                yield sse_format(DoneEvent())
                return

            yield sse_format(RetrievalStepEvent(step="searching", detail="ranking results"))

        system = build_system_prompt_grouped(query_chunk_pairs)

        yield sse_format(RetrievalStepEvent(step="synthesizing"))

        anthropic_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        emitted_model = False
        succeeded = False
        last_llm_err: LLMError | None = None
        _models: list[tuple[str, Literal["haiku", "sonnet"]]] = [
            (HAIKU, "haiku"),
            (SONNET, "sonnet"),
        ]
        for model_id, model_name in _models:
            try:
                async for text in stream_completion(model_id, anthropic_messages, system):
                    if not emitted_model:
                        yield sse_format(ModelEvent(model=model_name))
                        emitted_model = True
                    yield sse_format(DeltaEvent(text=text))
                succeeded = True
                break
            except LLMError as e:
                last_llm_err = e
                continue

        if not succeeded:
            yield sse_format(DeltaEvent(text=_llm_err_msg(last_llm_err)))
            yield sse_format(DoneEvent())
            return

        seen: set[str] = set()
        sources: list[CitationSource] = []
        for _, chunks in query_chunk_pairs:
            for result in chunks:
                if result.source_path not in seen:
                    seen.add(result.source_path)
                    sources.append(CitationSource(title=result.title))

        if sources:
            yield sse_format(CitationEvent(sources=sources))

        yield sse_format(DoneEvent())

    except Exception:
        log.exception("Unhandled pipeline error for query=%r", query[:80])
        yield sse_format(DeltaEvent(text=_FALLBACK_LLM))
        yield sse_format(DoneEvent())


@router.post("/chat")
async def chat(request: Request, body: ChatRequest) -> StreamingResponse:
    client_ip = request.client.host if request.client else "unknown"
    return StreamingResponse(
        _chat_stream(body, client_ip),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
