"""The dispatcher must hand the classifier's confidence to the router.

Without it the router cannot tell a confident skill match from a misroute, and
the low-confidence fallback silently stops working.

Skipped where the NLP stack is not installed; runs in a full environment.
"""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="NLP stack not installed")
pytest.importorskip("spacy", reason="NLP stack not installed")

from assistant import dispatcher  # noqa: E402
from assistant.conversation import Conversation  # noqa: E402
from assistant.schemas import SkillResponse  # noqa: E402
from nlp.schemas import NLUResult  # noqa: E402


@pytest.fixture(autouse=True)
def no_analytics_writes(monkeypatch):
    """Never touch the real turn log from a test."""
    monkeypatch.setattr(dispatcher.history, "record", lambda **kw: None)


def test_confidence_reaches_the_router(monkeypatch):
    captured = {}

    def fake_route(intent, text, entities, context, confidence=1.0):
        captured.update(intent=intent, text=text, confidence=confidence)
        return SkillResponse(speech="ok")

    monkeypatch.setattr(
        dispatcher, "analyze",
        lambda text, *a, **k: NLUResult(text=text, intent="TRANSLATE", confidence=0.41),
    )
    monkeypatch.setattr(dispatcher, "route", fake_route)

    dispatcher.handle("How do we extract crude oil?", Conversation())

    assert captured["confidence"] == pytest.approx(0.41)
    assert captured["intent"] == "TRANSLATE"


def test_analytics_failure_never_breaks_the_reply(monkeypatch):
    monkeypatch.setattr(
        dispatcher, "analyze",
        lambda text, *a, **k: NLUResult(text=text, intent="GREETING", confidence=0.9),
    )
    monkeypatch.setattr(dispatcher, "route", lambda *a, **k: SkillResponse(speech="hello"))

    def boom(**kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(dispatcher.history, "record", boom)

    result = dispatcher.handle("hi", Conversation())
    assert result.response.speech == "hello"
