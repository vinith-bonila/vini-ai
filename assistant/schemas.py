"""Data structures for the assistant/skill layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Link:
    label: str
    url: str


@dataclass
class SkillResponse:
    """What a skill hands back to the dispatcher.

    `speech` is spoken aloud / shown as the primary reply. `links` render as
    clickable buttons in the UI (used instead of server-side browser control,
    which is meaningless on a hosted deployment). `data` carries any structured
    payload (e.g. a weather table) for richer rendering.
    """

    speech: str
    links: list[Link] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    success: bool = True

    @classmethod
    def error(cls, message: str) -> "SkillResponse":
        return cls(speech=message, success=False)
