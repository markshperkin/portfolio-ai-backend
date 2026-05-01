from __future__ import annotations

from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.llm.client import stream_completion
from app.models import CitationEvent, CitationSource, DeltaEvent, DoneEvent, ErrorEvent, RetrievalStepEvent, sse_format
from app.prompt.system import build_system_prompt
from app.rag.retrieval import retrieve

router = APIRouter(prefix="/api")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: str | None = None


async def _rag_stream(request: ChatRequest) -> AsyncGenerator[str, None]:
    # Extract the last user message for retrieval
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        yield sse_format(ErrorEvent(code="no_user_message", message="No user message provided."))
        return

    query = user_messages[-1].content

    try:
        yield sse_format(RetrievalStepEvent(step="retrieving", detail="searching knowledge base"))
        chunks = await retrieve(query)

        yield sse_format(RetrievalStepEvent(step="searching", detail="ranking results"))
        system = build_system_prompt(chunks)

        yield sse_format(RetrievalStepEvent(step="synthesizing"))

        # Pass full conversation history to Anthropic (stateless multi-turn)
        anthropic_messages = [{"role": m.role, "content": m.content} for m in request.messages]

        async for text in stream_completion(anthropic_messages, system):
            yield sse_format(DeltaEvent(text=text))

        # Emit citation event with de-duped source titles
        seen: set[str] = set()
        sources: list[CitationSource] = []
        for chunk in chunks:
            if chunk.source_path not in seen:
                seen.add(chunk.source_path)
                sources.append(CitationSource(title=chunk.title))

        if sources:
            yield sse_format(CitationEvent(sources=sources))

        yield sse_format(DoneEvent())

    except Exception as exc:
        yield sse_format(ErrorEvent(code="pipeline_error", message="Something went wrong. Try again."))
        raise


@router.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _rag_stream(body),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
