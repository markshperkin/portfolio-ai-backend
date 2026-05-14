import asyncio
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.llm.client import get_client
from app.rag.store import get_collection

router = APIRouter(prefix="/api")
log = logging.getLogger(__name__)

_started_at = datetime.now(timezone.utc).isoformat()


class CheckResult(BaseModel):
    status: str  # "ok" | "error"
    detail: str


class ReadinessResponse(BaseModel):
    status: str
    git_sha: str
    started_at: str
    knowledge_base: CheckResult
    model: CheckResult


async def _check_knowledge_base() -> CheckResult:
    try:
        count = get_collection().count()
        if count > 0:
            return CheckResult(status="ok", detail=str(count))
        return CheckResult(status="error", detail="0 chunks")
    except Exception as e:
        log.error("KB check failed: %s", e)
        return CheckResult(status="error", detail="unavailable")


async def _check_model() -> CheckResult:
    try:
        client = get_client()
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1,
            messages=[{"role": "user", "content": "reply: ok"}],
        )
        _ = msg  # just need no exception
        return CheckResult(status="ok", detail="ready")
    except Exception as e:
        log.error("Model check failed: %s", e)
        return CheckResult(status="error", detail="unavailable")


async def _run_checks() -> ReadinessResponse:
    kb, model = await asyncio.gather(_check_knowledge_base(), _check_model())
    overall = "ok" if kb.status == "ok" and model.status == "ok" else "degraded"
    return ReadinessResponse(
        status=overall,
        git_sha=os.environ.get("GIT_SHA", "dev"),
        started_at=_started_at,
        knowledge_base=kb,
        model=model,
    )


@router.get("/health", response_model=ReadinessResponse)
async def health() -> ReadinessResponse:
    return await _run_checks()
