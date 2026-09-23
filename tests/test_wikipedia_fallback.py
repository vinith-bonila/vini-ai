"""A Wikipedia miss must reach a grounded source, not a model guess.

"Tell me about IIPE" hit Wikipedia's disambiguation page, which offers
chemistry and medical terms and not the Indian Institute of Petroleum and
Energy at all. That reply was marked successful, so nothing rescued it.
"""
from __future__ import annotations

import pytest
import wikipedia

from assistant import router
from assistant.schemas import SkillResponse
from services import wikipedia_service


@pytest.fixture
def wiki_disambiguation(monkeypatch):
    """Wikipedia returning the real IIPE disambiguation options."""
    options = [
        "ethyl eicosapentaenoic acid", "diisopropyl ether", "Handroanthus",
        "iris pigment epithelium", "swimming-induced pulmonary edema",
    ]

    def raise_disambiguation(*a, **k):
        raise wikipedia.DisambiguationError("IIPE", options)

    monkeypatch.setattr(wikipedia, "summary", raise_disambiguation)
    monkeypatch.setattr(wikipedia, "page", raise_disambiguation)
    return options


def test_disambiguation_is_not_reported_as_success(wiki_disambiguation):
    result = wikipedia_service.run("Tell me about IIPE", [])
    assert result.success is False, "a useless disambiguation must not look successful"
    assert result.data.get("disambiguation") is True


def test_disambiguation_still_shows_the_options(wiki_disambiguation):
    result = wikipedia_service.run("Tell me about IIPE", [])
    assert "diisopropyl ether" in result.speech


def test_wikipedia_miss_falls_back_to_grounded_search(monkeypatch, wiki_disambiguation):
    """Grounded search, not the bare model, so the answer has sources."""
    called = {}

    def fake_search(text, entities=None, context=None):
        called["search"] = text
        return SkillResponse(speech="IIPE is in Visakhapatnam, Andhra Pradesh.")

    def fake_chat(text, entities=None, context=None):
        called["chat"] = text
        return SkillResponse(speech="a model guess")

    monkeypatch.setitem(router._FALLBACK_HANDLER, "WIKIPEDIA", fake_search)
    monkeypatch.setattr(router.llm_service, "chat", fake_chat)

    result = router.route("WIKIPEDIA", "Tell me about IIPE", [], None, confidence=0.50)

    assert "Visakhapatnam" in result.speech
    assert "search" in called, "grounded search was not used"
    assert "chat" not in called, "fell back to the model instead of a grounded source"


def test_wikipedia_is_mapped_to_search_not_the_model():
    """Pins the routing decision itself rather than a monkeypatched stand-in."""
    from services import search_service

    assert router._FALLBACK_HANDLER["WIKIPEDIA"] is search_service.run


def test_a_failing_fallback_keeps_the_original_message(monkeypatch, wiki_disambiguation):
    def boom(*a, **k):
        raise ConnectionError("search down")

    monkeypatch.setitem(router._FALLBACK_HANDLER, "WIKIPEDIA", boom)
    result = router.route("WIKIPEDIA", "Tell me about IIPE", [], None, confidence=0.50)
    assert result.success is False
    assert "could mean several things" in result.speech


def test_successful_lookup_is_untouched(monkeypatch):
    monkeypatch.setattr(wikipedia, "summary", lambda *a, **k: "Alan Turing was a mathematician.")
    monkeypatch.setattr(
        wikipedia, "page",
        lambda *a, **k: type("P", (), {"url": "http://w/Alan", "title": "Alan Turing"})(),
    )
    result = wikipedia_service.run("who is Alan Turing", [])
    assert result.success is True
    assert "mathematician" in result.speech
