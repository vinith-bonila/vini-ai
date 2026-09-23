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


_CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*", re.M)
_QUOTE_RE = re.compile(r"^\s{0,3}>+\s*", re.M)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+", re.M)
_ORDERED_RE = re.compile(r"^\s*\d+[.)]\s+", re.M)
_RULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$", re.M)
_TABLE_DIVIDER_RE = re.compile(r"^\s*\|?[\s:|-]{4,}\|?\s*$", re.M)
_EMPHASIS_RE = re.compile(r"(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1", re.S)
_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n{2,}")


def speech_text(markdown: str) -> str:
    """Strip Markdown so a reply is spoken, not spelled out.

    Text-to-speech reads syntax literally: a '#' heading becomes "hash", '**'
    becomes "asterisk asterisk", and a table turns into a stream of pipes.
    Structure is dropped and the words are kept.
    """
    text = markdown or ""
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _IMAGE_RE.sub(" ", text)
    text = _LINK_RE.sub(r"\1", text)        # keep the label, drop the URL
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _RULE_RE.sub(" ", text)
    text = _TABLE_DIVIDER_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _ORDERED_RE.sub("", text)
    text = _EMPHASIS_RE.sub(r"\2", text)
    text = text.replace("|", " ")            # table cells
    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANKS_RE.sub("\n", text)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


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
