"""Small, dependency-free helper utilities used across the app."""
from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Anything that looks like an API key, so a failure reason shown in the UI (and
# stored in the turn log) can never carry a credential.
_SECRET_RE = re.compile(r"\b(?:sk|gsk|xai|pk)-[A-Za-z0-9_\-]{8,}", re.I)


@contextmanager
def timed() -> Iterator[dict]:
    """Context manager that records elapsed wall-clock milliseconds.

    Usage:
        with timed() as t:
            do_work()
        print(t["ms"])
    """
    bucket: dict = {"ms": 0.0}
    start = time.perf_counter()
    try:
        yield bucket
    finally:
        bucket["ms"] = round((time.perf_counter() - start) * 1000, 2)


def safe_json(value: Any) -> str:
    """Serialise anything to JSON, falling back to str() for odd types."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return json.dumps(str(value), ensure_ascii=False)


def truncate(text: str, limit: int = 280) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


def failure_reason(exc: Exception, limit: int = 140) -> str:
    """A short, secret-free description of why a call failed.

    Surfaced in the UI so 'unreachable' is diagnosable (a 401 means the key or
    base URL is wrong; a timeout means the network is). Redacted because the
    reply text is also written to the turn log.
    """
    detail = " ".join(str(exc).split())
    detail = _SECRET_RE.sub("[redacted]", detail)
    name = type(exc).__name__
    if not detail:
        return name
    return truncate(f"{name}: {detail}", limit)
