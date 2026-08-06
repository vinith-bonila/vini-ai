"""Typed result objects passed between the NLP layer and the assistant layer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Entity:
    text: str
    label: str
    start: int = -1
    end: int = -1


@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    tag: str
    dep: str
    head: str


@dataclass
class IntentScore:
    intent: str
    score: float


@dataclass
class NLUResult:
    """The complete linguistic analysis of a single user utterance."""

    text: str
    intent: str
    confidence: float
    entities: list[Entity] = field(default_factory=list)
    tokens: list[Token] = field(default_factory=list)
    sentiment_label: str = "NEUTRAL"
    sentiment_score: float = 0.0
    language: str = "en"
    language_confidence: float = 0.0
    intent_ranking: list[IntentScore] = field(default_factory=list)
    engine: str = "tfidf"  # which classifier produced the intent

    def entities_by_label(self, label: str) -> list[str]:
        return [e.text for e in self.entities if e.label == label]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
