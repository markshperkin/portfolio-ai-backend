"""HMAC-SHA256 IP hashing (TASK-21). Plaintext IP is never persisted."""

from __future__ import annotations

import hashlib
import hmac
import os

_salt: bytes | None = None


def _get_salt() -> bytes:
    global _salt
    if _salt is None:
        raw = os.environ.get("IP_HASH_SALT", "")
        if not raw:
            raise RuntimeError("IP_HASH_SALT environment variable not set")
        _salt = raw.encode()
    return _salt


def hash_ip(ip: str) -> str:
    """Return hex digest of HMAC-SHA256(ip, IP_HASH_SALT). Never store plaintext IP."""
    return hmac.new(_get_salt(), ip.encode(), hashlib.sha256).hexdigest()
