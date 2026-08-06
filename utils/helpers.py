"""Small, dependency-free helper utilities used across the app."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Any, Iterator


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
