from typing import Literal, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# SSE event schemas (TASK-5-BE)
# Wire format: event: <type>\ndata: <json>\n\n
# ---------------------------------------------------------------------------

class RetrievalStepEvent(BaseModel):
    type: Literal["retrieval_step"] = "retrieval_step"
    step: Literal["retrieving", "searching", "synthesizing"]
    detail: Optional[str] = None


class DeltaEvent(BaseModel):
    type: Literal["delta"] = "delta"
    text: str


class CitationSource(BaseModel):
    title: str


class CitationEvent(BaseModel):
    type: Literal["citation"] = "citation"
    sources: list[CitationSource]


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


class ActionEvent(BaseModel):
    type: Literal["action"] = "action"
    action_type: Literal["download", "open"]
    url: str


SSEEvent = RetrievalStepEvent | DeltaEvent | CitationEvent | DoneEvent | ErrorEvent | ActionEvent


def sse_format(event: BaseModel) -> str:
    """Serialise a pydantic event model to SSE wire format."""
    name = event.model_fields["type"].default  # type: ignore[attr-defined]
    data = event.model_dump_json(exclude={"type"})
    return f"event: {name}\ndata: {data}\n\n"
