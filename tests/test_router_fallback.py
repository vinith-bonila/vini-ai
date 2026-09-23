"""A weak classification must not dead-end on a skill's error message.

Both cases here are taken from real misroutes:
  * "How do we extract crude oil ...?"  -> TRANSLATE  @ 41%
  * "tell me about GPT-6 Astra"         -> WIKIPEDIA  @ 50%
Each produced a useless reply even though the LLM could answer.
"""
from __future__ import annotations

import pytest

from assistant import router
from assistant.schemas import SkillResponse

LLM_ANSWER = "Crude oil is brought to surface by drilling a wellbore..."


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """No test may reach the real web-search backend."""
    def blocked(*a, **k):
        raise AssertionError("a test tried to hit live web search")

    monkeypatch.setattr(router.search_service, "run", blocked)


@pytest.fixture
def llm(monkeypatch):
    """Record LLM calls and return a successful answer."""
    calls = []

    def fake_chat(text, entities=None, context=None):
        calls.append(text)
        return SkillResponse(speech=LLM_ANSWER)

    monkeypatch.setattr(router.llm_service, "chat", fake_chat)
    return calls


def fail_with(message):
    def handler(text, entities, context):
        return SkillResponse.error(message)
    return handler


# --------------------------------------------------------------------------- #
# The two reported misroutes
# --------------------------------------------------------------------------- #
def test_low_confidence_translate_miss_defers_to_llm(monkeypatch, llm):
    monkeypatch.setitem(
        router.ROUTES, "TRANSLATE",
        fail_with("Tell me what to translate and into which language."),
    )
    result = router.route(
        "TRANSLATE",
        "How do we extract crude oil from a subsurface to the surface?",
        [], None, confidence=0.41,
    )
    assert result.speech == LLM_ANSWER
    assert result.success is True
    assert llm, "the LLM was never consulted"


def test_low_confidence_wikipedia_miss_is_rescued(monkeypatch, llm):
    """WIKIPEDIA rescues via grounded search - see test_wikipedia_fallback.py."""
    grounded = "GPT-6 Astra does not appear in any source."
    monkeypatch.setitem(
        router.ROUTES, "WIKIPEDIA", fail_with("Wikipedia lookup failed. Please try again."),
    )
    monkeypatch.setitem(
        router._FALLBACK_HANDLER, "WIKIPEDIA",
        lambda t, e=None, c=None: SkillResponse(speech=grounded),
    )
    result = router.route("WIKIPEDIA", "tell me about GPT-6 Astra", [], None, confidence=0.50)
    assert result.speech == grounded
    assert not llm, "factual lookups must prefer a grounded source over the model"


# --------------------------------------------------------------------------- #
# ... without trampling legitimate clarifications
# --------------------------------------------------------------------------- #
def test_confident_skill_keeps_its_own_message(monkeypatch, llm):
    """A real 'translate this' with no target language still asks for one."""
    monkeypatch.setitem(
        router.ROUTES, "TRANSLATE",
        fail_with("Tell me what to translate and into which language."),
    )
    result = router.route("TRANSLATE", "translate this", [], None, confidence=0.95)
    assert "into which language" in result.speech
    assert not llm, "a confident skill must not be overridden"


def test_successful_skill_is_never_second_guessed(monkeypatch, llm):
    monkeypatch.setitem(
        router.ROUTES, "WEATHER",
        lambda t, e, c: SkillResponse(speech="It is 26 degrees in Mumbai."),
    )
    result = router.route("WEATHER", "weather in mumbai", [], None, confidence=0.30)
    assert "26 degrees" in result.speech
    assert not llm


@pytest.mark.parametrize("intent", sorted(router._NO_LLM_FALLBACK))
def test_excluded_intents_never_fall_back(monkeypatch, llm, intent):
    """Their own failure text is the better answer, or the LLM already ran."""
    monkeypatch.setitem(router.ROUTES, intent, fail_with("specific failure text"))
    result = router.route(intent, "anything", [], None, confidence=0.10)
    assert result.speech == "specific failure text"
    assert not llm


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def test_crashing_skill_also_falls_back(monkeypatch, llm):
    def boom(text, entities, context):
        raise RuntimeError("skill exploded")

    monkeypatch.setitem(router.ROUTES, "DICTIONARY", boom)
    result = router.route("DICTIONARY", "define x", [], None, confidence=0.4)
    assert result.speech == LLM_ANSWER


def test_crashing_skill_without_fallback_still_returns_a_reply(monkeypatch):
    """No LLM available: the turn must still end with a message, not an exception."""
    def boom(text, entities, context):
        raise RuntimeError("skill exploded")

    monkeypatch.setitem(router.ROUTES, "DICTIONARY", boom)
    monkeypatch.setattr(
        router.llm_service, "chat",
        lambda *a, **k: SkillResponse.error("no key configured"),
    )
    result = router.route("DICTIONARY", "define x", [], None, confidence=0.4)
    assert result.success is False
    assert result.speech


def test_skill_message_wins_when_fallback_also_fails(monkeypatch):
    """The skill's error is more specific than 'model unreachable'."""
    monkeypatch.setitem(router.ROUTES, "DICTIONARY", fail_with("No definition found."))
    monkeypatch.setattr(
        router.llm_service, "chat",
        lambda *a, **k: SkillResponse.error("My language model is unreachable."),
    )
    result = router.route("DICTIONARY", "define x", [], None, confidence=0.4)
    assert result.speech == "No definition found."


def test_default_confidence_preserves_old_behaviour(monkeypatch, llm):
    """Callers that pass no confidence get no surprise fallback."""
    monkeypatch.setitem(router.ROUTES, "WIKIPEDIA", fail_with("nope"))
    result = router.route("WIKIPEDIA", "x", [], None)
    assert result.speech == "nope"
    assert not llm


def test_unknown_intent_goes_straight_to_the_llm(llm):
    result = router.route("NOT_A_REAL_INTENT", "hello there", [], None, confidence=0.9)
    assert result.speech == LLM_ANSWER
