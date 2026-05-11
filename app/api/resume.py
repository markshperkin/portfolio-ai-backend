from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter(prefix="/api")

_RESUME_PATH = Path(__file__).parent.parent.parent / "static" / "resume.pdf"


@router.get("/resume.pdf")
async def serve_resume() -> FileResponse:
    if not _RESUME_PATH.exists():
        raise HTTPException(status_code=404, detail="Resume not found")
    return FileResponse(
        _RESUME_PATH,
        media_type="application/pdf",
        filename="Mark_Shperkin_Resume.pdf",
    )
