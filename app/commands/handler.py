"""Slash command detection and response (TASK-15-BE + TASK-18).

Commands short-circuit before RAG, abuse classifier, and rate limiting.
Matching: full-line only (no match if embedded in longer message).
Tolerance: case-insensitive, leading/trailing whitespace, optional trailing .!?
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

from app.models import DeltaEvent, DoneEvent, sse_format

# --- Patterns (full-line, case-insensitive) ---

_FLAGS = re.IGNORECASE
_TAIL = r"[.!?]?\s*$"  # optional trailing punctuation

_WHOAMI = re.compile(r"^\s*whoami" + _TAIL, _FLAGS)
_HELP = re.compile(r"^\s*/help" + _TAIL, _FLAGS)
_HIRE = re.compile(r"^\s*sudo\s+hire-?mark" + _TAIL, _FLAGS)
_RESUME = re.compile(r"^\s*cat\s+resume\.pdf" + _TAIL, _FLAGS)

# Any standalone slash / sudo / cat that didn't match above → unknown command
_ANY_SLASH = re.compile(r"^\s*(/\S+|sudo\s+\S+|cat\s+\S.*)" + _TAIL, _FLAGS)

# --- Canned response text ---

_WHOAMI_TEXT = (
    "I'm Mark's GPT — a RAG-backed assistant trained on Mark Shperkin's actual work: "
    "projects, experience, skills, and more. Ask anything. I'll cite my sources."
)

_HELP_TEXT = """\
Available commands:
  whoami          → what I am
  /help           → all commands
  sudo hire-mark  → Mark's contact info
  cat resume.pdf  → download his résumé"""

_UNKNOWN_TEXT = "command not found — try /help"


def detect_command(text: str) -> str | None:
    """Return the canonical command key if text is a standalone command, else None."""
    t = text.strip()
    if _WHOAMI.match(t):
        return "whoami"
    if _HELP.match(t):
        return "/help"
    if _HIRE.match(t):
        return "sudo hire-mark"
    if _RESUME.match(t):
        return "cat resume.pdf"
    if _ANY_SLASH.match(t):
        return "unknown"
    return None


async def handle_command(cmd: str) -> AsyncGenerator[str, None]:
    """Yield SSE chunks for a recognized (or unknown) slash command."""
    if cmd == "whoami":
        response = _WHOAMI_TEXT
    elif cmd == "/help":
        response = _HELP_TEXT
    elif cmd == "sudo hire-mark":
        # Populated in TASK-16 (HITL) — contacts.py constants replace this
        from app.prompt import contacts as _c

        response = (
            f"Here's how to reach Mark:\n\n"
            f"  Email:    {_c.MARK_EMAIL}\n"
            f"  LinkedIn: {_c.MARK_LINKEDIN_URL}\n"
            f"  Calendly: {_c.MARK_CALENDLY_URL}"
        )
    elif cmd == "cat resume.pdf":
        # Populated in TASK-17-BE (HITL) — emits action event before done
        from app.models import ActionEvent

        yield sse_format(DeltaEvent(text="Opening résumé…"))
        yield sse_format(ActionEvent(action_type="download", url="/api/resume.pdf"))
        yield sse_format(DoneEvent())
        return
    else:
        response = _UNKNOWN_TEXT

    yield sse_format(DeltaEvent(text=response))
    yield sse_format(DoneEvent())
