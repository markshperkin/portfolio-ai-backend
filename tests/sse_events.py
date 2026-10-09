"""Parse the backend's SSE wire format back into event dicts for assertions."""

from __future__ import annotations

import json


def parse_sse(raw: str | bytes) -> list[dict]:
    text = raw.decode() if isinstance(raw, bytes) else raw
    events = []
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        event_type = None
        data = None
        for line in block.strip().split("\n"):
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_type and data is not None:
            events.append({"type": event_type, **data})
    return events


def text_of(events: list[dict]) -> str:
    return "".join(e["text"] for e in events if e["type"] == "delta")


def of_type(events: list[dict], event_type: str) -> list[dict]:
    return [e for e in events if e["type"] == event_type]
