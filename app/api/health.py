import os
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api")

_started_at = datetime.now(timezone.utc).isoformat()


class HealthResponse(BaseModel):
    status: str
    git_sha: str
    started_at: str


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        git_sha=os.environ.get("GIT_SHA", "dev"),
        started_at=_started_at,
    )
