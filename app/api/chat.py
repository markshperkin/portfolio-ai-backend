from __future__ import annotations

import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.commands.handler import detect_command, handle_command
from app.llm.client import LLMError, stream_completion
from app.models import CitationEvent, CitationSource, DeltaEvent, DoneEvent, ErrorEvent, RetrievalStepEvent, sse_format
from app.prompt.contacts import MARK_EMAIL
from app.prompt.system import build_system_prompt
from app.rag.embedding import EmbeddingError
from app.rag.retrieval import retrieve
from app.security.rate_limit import check_rate_limit

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_FALLBACK_LLM = (
    f"The models are taking a nap — try again in a moment. "
    f"Or reach Mark directly at {MARK_EMAIL}."
)
_FALLBACK_EMBEDDING = (
    f"Having trouble searching the knowledge base right now. "
    f"Try again in a moment, or email Mark at {MARK_EMAIL}."
)
_FALLBACK_DB = (
    f"The knowledge base is temporarily unavailable. "
    f"Email Mark at {MARK_EMAIL} and he'll respond directly."
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

    # 1. Slash command short-circuit (before rate-limit and RAG)
    cmd = detect_command(query)
    if cmd is not None:
        async for chunk in handle_command(cmd):
            yield chunk
        return

    # 2. Rate limit
    allowed, rate_msg = await check_rate_limit(client_ip)
    if not allowed:
        yield sse_format(DeltaEvent(text=rate_msg))
        yield sse_format(DoneEvent())
        return

    # 3. RAG pipeline with typed error mapping
    try:
        yield sse_format(RetrievalStepEvent(step="retrieving", detail="searching knowledge base"))
        try:
            chunks = await retrieve(query)
        except EmbeddingError:
            yield sse_format(DeltaEvent(text=_FALLBACK_EMBEDDING))
            yield sse_format(DoneEvent())
            return
        except Exception:
            log.exception("DB retrieval error for query=%r", query[:80])
            yield sse_format(DeltaEvent(text=_FALLBACK_DB))
            yield sse_format(DoneEvent())
            return

        yield sse_format(RetrievalStepEvent(step="searching", detail="ranking results"))
        system = build_system_prompt(chunks)

        yield sse_format(RetrievalStepEvent(step="synthesizing"))

        anthropic_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        try:
            async for text in stream_completion(anthropic_messages, system):
                yield sse_format(DeltaEvent(text=text))
        except LLMError:
            yield sse_format(DeltaEvent(text=_FALLBACK_LLM))
            yield sse_format(DoneEvent())
            return

        # Citation: de-duped by source path
        seen: set[str] = set()
        sources: list[CitationSource] = []
        for chunk in chunks:
            if chunk.source_path not in seen:
                seen.add(chunk.source_path)
                sources.append(CitationSource(title=chunk.title))

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
