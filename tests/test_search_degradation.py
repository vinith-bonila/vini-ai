"""Live search being down must not dead-end an answerable question.

"What are the main concepts in Chemical Engineering?" routes here at 91%
confidence, so the router's low-confidence fallback cannot rescue it - the
skill has to degrade on its own. A model-only answer is always labelled,
because currency is this skill's entire purpose.
"""
from __future__ import annotations

import pytest

from assistant.schemas import SkillResponse
from services import search_service

QUESTION = "What are the main concepts in Chemical Engineering?"
MODEL_ANSWER = "Mass and energy balances, transport phenomena, thermodynamics..."


@pytest.fixture
def model(monkeypatch):
    calls = []

    def fake_chat(text, entities=None, context=None):
        calls.append(text)
        return SkillResponse(speech=MODEL_ANSWER)

    monkeypatch.setattr(search_service.llm_service, "chat", fake_chat)
    return calls


def _search_raises(*a, **k):
    raise ConnectionError("getaddrinfo failed")


def test_search_outage_answers_from_the_model(monkeypatch, model):
    monkeypatch.setattr(search_service, "_search", _search_raises)
    result = search_service.run(QUESTION)
    assert MODEL_ANSWER in result.speech
    assert result.success is True
    assert model, "the model was never consulted"


def test_degraded_answer_is_labelled_not_live(monkeypatch, model):
    monkeypatch.setattr(search_service, "_search", _search_raises)
    result = search_service.run(QUESTION)
    assert "may be out of date" in result.speech
    assert result.data.get("live_sources") is False


def test_empty_results_also_fall_back(monkeypatch, model):
    monkeypatch.setattr(search_service, "_search", lambda *a, **k: [])
    result = search_service.run(QUESTION)
    assert MODEL_ANSWER in result.speech


def test_without_a_model_it_reports_the_real_failure(monkeypatch):
    monkeypatch.setattr(search_service, "_search", _search_raises)
    monkeypatch.setattr(
        search_service.llm_service, "chat",
        lambda *a, **k: SkillResponse.error("no key configured"),
    )
    result = search_service.run(QUESTION)
    assert result.success is False
    assert "ConnectionError" in result.speech


def test_successful_search_is_never_labelled_degraded(monkeypatch, model):
    rows = [{"title": "Chem Eng", "body": "Mass balances.", "url": "http://x"}]
    monkeypatch.setattr(search_service, "_search", lambda *a, **k: rows)
    monkeypatch.setattr(search_service, "_synthesize", lambda *a, **k: "Live answer.")
    result = search_service.run(QUESTION)
    assert result.speech == "Live answer."
    assert "may be out of date" not in result.speech
    assert result.data.get("live_sources") is not False
    assert not model, "the live path must not call the model directly"
