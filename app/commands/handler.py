"""Slash command detection and dispatch.

Commands short-circuit before RAG, abuse classifier, and rate limiting
(except /jdfit with body — see chat.py for special handling).

Matching: full-line only, case-insensitive, leading/trailing whitespace tolerated.
All commands start with /. Any /name not in the known set returns "unknown".
"""

from __future__ import annotations

import re
from typing import AsyncGenerator

from app.models import ActionEvent, DeltaEvent, DoneEvent, sse_format

_PATTERN = re.compile(
    r"^\s*/(?P<name>[\w-]+)(?:\s+(?P<body>.+?))?\s*$",
    re.IGNORECASE | re.DOTALL,
)

_HELP_TEXT = """\
Available commands:
  /whoami         → what I am
  /help           → show this menu
  /hire-mark      → Mark's contact info
  /resume         → download his résumé
  /jdfit <jd>     → paste a job description to get a personalised fit report\
"""

_UNKNOWN_TEXT = "command not found — try /help"

_JDFIT_USAGE = "Don't forget to paste the job description after /jdfit."


def detect_command(text: str) -> tuple[str, str] | None:
    """Return (name, body) if text is a slash command, else None.

    name is lowercased. body is stripped, may be empty string.
    """
    m = _PATTERN.match(text.strip())
    if m is None:
        return None
    return m.group("name").lower(), (m.group("body") or "").strip()


async def handle_command(name: str, body: str = "") -> AsyncGenerator[str, None]:
    """Yield SSE chunks for a slash command that doesn't need pipeline handling."""
    if name == "whoami":
        from app.models import DeltaEvent as _D

        response = (
            "I'm Mark's GPT — a RAG-backed assistant trained on Mark Shperkin's actual work: "
            "projects, experience, skills, and more. Ask anything. I'll cite my sources."
        )
    elif name == "help":
        response = _HELP_TEXT
    elif name == "hire-mark":
        from app.prompt import contacts as _c

        response = (
            f"Here's how to reach Mark:\n\n"
            f"  Email:    {_c.MARK_EMAIL}\n"
            f"  LinkedIn: {_c.MARK_LINKEDIN_URL}\n"
            f"  Calendly: {_c.MARK_CALENDLY_URL}"
        )
    elif name == "resume":
        yield sse_format(DeltaEvent(text="Opening résumé…"))
        yield sse_format(ActionEvent(action_type="download", url="/api/resume.pdf"))
        yield sse_format(DoneEvent())
        return
    elif name == "jdfit":
        # body-less case — with body is routed through the pipeline in chat.py
        response = _JDFIT_USAGE
    else:
        response = _UNKNOWN_TEXT

    yield sse_format(DeltaEvent(text=response))
    yield sse_format(DoneEvent())
