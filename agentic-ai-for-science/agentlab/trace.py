"""Traces: a complete, replayable record of an agent run, one JSON event per line.

Every line passes through `redact` before it is written, so an API key can never land in a trace
even if something upstream misbehaves. (The sandbox already strips keys from model-run code.)
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

_KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")
_SECRET_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def redact(text: str) -> str:
    for name in _SECRET_ENV:
        val = os.environ.get(name)
        if val and len(val) >= 8:
            text = text.replace(val, "[REDACTED]")
    return _KEY_PATTERN.sub("[REDACTED]", text)


class Trace:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.events: list[dict] = []
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def log(self, kind: str, **data):
        event = {"t": round(time.time(), 2), "kind": kind, **data}
        line = redact(json.dumps(event, default=str))
        self.events.append(json.loads(line))
        if self.path:
            with self.path.open("a") as f:
                f.write(line + "\n")


def load_trace(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
