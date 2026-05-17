"""Per-IP rate limiter (TASK-22-BE).

In-process defaultdict(deque) + asyncio.Lock per ADR 008.
State lost on backend restart — acceptable.

Limits:
  - Cooldown: 5s between requests from same hashed IP
  - Daily cap:  50 requests in a rolling 24h window
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from app.prompt.contacts import MARK_EMAIL
from app.security.hashing import hash_ip

COOLDOWN_SECONDS = 5
DAILY_CAP = 50
WINDOW_SECONDS = 86400  # 24h

_lock = asyncio.Lock()
# hashed_ip → deque of timestamps (float, monotonic)
_request_log: dict[str, deque[float]] = defaultdict(deque)


async def check_rate_limit(ip: str) -> tuple[bool, str]:
    """Return (allowed, message). Message is non-empty only when blocked."""
    hashed = hash_ip(ip)
    now = time.monotonic()

    async with _lock:
        q = _request_log[hashed]

        # Purge expired timestamps outside the 24h window
        cutoff = now - WINDOW_SECONDS
        while q and q[0] < cutoff:
            q.popleft()

        # Cooldown check
        if q and (now - q[-1]) < COOLDOWN_SECONDS:
            return False, "Slow down — one message at a time. Try again in a few seconds."

        # Daily cap check
        if len(q) >= DAILY_CAP:
            return False, (
                f"You've hit the daily limit. Want to keep talking? "
                f"Reach Mark directly at {MARK_EMAIL}."
            )

        # Record this request
        q.append(now)
        return True, ""
