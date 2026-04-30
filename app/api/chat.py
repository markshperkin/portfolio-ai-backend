import asyncio
from typing import AsyncGenerator
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.models import RetrievalStepEvent, DeltaEvent, DoneEvent, sse_format

router = APIRouter(prefix="/api")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    session_id: str | None = None


async def _tracer_stream() -> AsyncGenerator[str, None]:
    """Hardcoded hello-world stream: retrieval_step → deltas → done."""
    yield sse_format(RetrievalStepEvent(step="retrieving", detail="searching knowledge base"))
    await asyncio.sleep(0.05)
    yield sse_format(RetrievalStepEvent(step="searching", detail="ranking results"))
    await asyncio.sleep(0.05)
    yield sse_format(RetrievalStepEvent(step="synthesizing"))
    await asyncio.sleep(0.05)

    response = "Hello! I'm Mark's GPT, a RAG-backed assistant. Ask me about Mark's projects, experience, or skills."
    for char in response:
        yield sse_format(DeltaEvent(text=char))
        await asyncio.sleep(0.01)

    yield sse_format(DoneEvent())


@router.post("/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _tracer_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
