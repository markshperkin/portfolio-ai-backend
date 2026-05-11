"""Abuse / jailbreak classifier.

Detects jailbreak and manipulation attempts via regex, logs flagged attempts
to a local SQLite file, and throttles repeat offenders.
"""

from __future__ import annotations

import os
import re

import aiosqlite

from app.security.hashing import hash_ip

_FLAGS = re.IGNORECASE | re.DOTALL

_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r, _FLAGS)
    for r in [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
        r"disregard\s+(your|the)\s+(instructions|rules|guidelines|constraints|system\s*prompt)",
        r"forget\s+(everything|all)\s+(you('ve|\s+have)\s+been\s+)?(told|instructed|given)",
        r"you\s+are\s+now\s+(DAN|a\s+different|an?\s+unrestricted|an?\s+unfiltered)",
        r"\bDAN\b",
        r"\bjailbreak\b",
        r"developer\s+mode",
        r"reveal\s+(your|the)\s+(system\s+)?prompt",
        r"what\s+(are|is)\s+your\s+(system\s+)?(prompt|instructions|rules)",
        r"(bypass|override|circumvent)\s+(your\s+)?(filter|restriction|rule|instruction|constraint)",
        r"pretend\s+(you\s+are|to\s+be)\s+(?!mark)",
        r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?!mark|an?\s+assistant)",
        r"you\s+have\s+no\s+(restrictions|rules|guidelines|constraints)",
        r"ignore\s+your\s+(training|guidelines|instructions|rules)",
    ]
]

_THROTTLE_THRESHOLD = 3
_WITTY_REFUSAL = (
    "Nice try jailbreaking the system — but this bot is itself proof that "
    "Mark ships production AI. Give it up and ask me something real."
)
_THROTTLE_MSG = (
    "Repeated abuse detected. This session is rate-limited. "
    "If this is a mistake, email markshperkin1@gmail.com."
)

ABUSE_DB_PATH = os.environ.get("ABUSE_DB_PATH", "data/abuse_log.db")


async def init_abuse_db() -> None:
    async with aiosqlite.connect(ABUSE_DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS abuse_log (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                hashed_ip   TEXT        NOT NULL,
                prompt_text TEXT,
                abuse_type  TEXT,
                created_at  TEXT        NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute(
            "CREATE INDEX IF NOT EXISTS abuse_ip_time ON abuse_log (hashed_ip, created_at DESC)"
        )
        await db.commit()


def is_abusive(text: str) -> bool:
    return any(p.search(text) for p in _PATTERNS)


async def check_and_log_abuse(ip: str, text: str) -> tuple[bool, str]:
    """Return (is_abusive, message). Logs to abuse_log and throttles repeat offenders."""
    if not is_abusive(text):
        return False, ""

    hashed = hash_ip(ip)

    async with aiosqlite.connect(ABUSE_DB_PATH) as db:
        await db.execute(
            "INSERT INTO abuse_log (hashed_ip, prompt_text, abuse_type) VALUES (?, ?, 'jailbreak')",
            (hashed, text[:500]),
        )
        await db.commit()

        async with db.execute(
            "SELECT COUNT(*) FROM abuse_log "
            "WHERE hashed_ip = ? AND created_at > datetime('now', '-24 hours')",
            (hashed,),
        ) as cursor:
            row = await cursor.fetchone()
            count = row[0] if row else 0

    if count >= _THROTTLE_THRESHOLD:
        return True, _THROTTLE_MSG
    return True, _WITTY_REFUSAL
