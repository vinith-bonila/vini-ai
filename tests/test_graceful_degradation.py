"""Missing credentials must degrade, never crash.

Constructing an OpenAI client with no key raises, so every provider call site
has to check first. These tests hold that guard in place.

`Settings` is a frozen dataclass and each service binds it at import time, so
overrides are built with `dataclasses.replace` and patched onto the module
under test rather than mutated in place.
"""
from __future__ import annotations

import dataclasses

import pytest

import config


def settings_with(**overrides):
    return dataclasses.replace(config.settings, **overrides)


@pytest.fixture
def keyless(monkeypatch):
    """Patch a no-key Settings onto every module that talks to the provider."""
    from services import llm_service, search_service
    from speech import speech_to_text

    fake = settings_with(openai_api_key="")
    for module in (llm_service, search_service, speech_to_text):
        monkeypatch.setattr(module, "settings", fake, raising=False)
    return fake


def test_settings_replace_gives_a_keyless_instance():
    assert settings_with(openai_api_key="").openai_enabled is False
    assert settings_with(openai_api_key="k").openai_enabled is True


def test_chat_without_key_returns_message_not_exception(keyless):
    from services import llm_service

    result = llm_service.chat("explain diffusion")
    assert result.success is False
    assert result.speech
    # The provider may be Groq now, so the copy must not name OpenAI.
    assert "OpenAI API key" not in result.speech


def test_hosted_stt_without_key_raises_typed_error(monkeypatch):
    from speech import speech_to_text
    from speech.speech_to_text import TranscriptionError

    monkeypatch.setattr(
        speech_to_text, "settings",
        settings_with(openai_api_key="", stt_backend="openai"),
    )
    with pytest.raises(TranscriptionError):
        speech_to_text.transcribe(b"\x00\x01\x02")


def test_empty_audio_is_not_an_error(keyless):
    from speech.speech_to_text import transcribe

    assert transcribe(b"") == ""


def test_tts_default_backend_does_not_need_the_provider():
    """Both edge and gtts are keyless; only the openai backend needs one."""
    assert config.settings.tts_backend in {"edge", "gtts"}


def test_search_without_key_still_returns_snippets(monkeypatch):
    """No LLM to summarise with: hand back what search found, don't raise."""
    from services import search_service

    monkeypatch.setattr(search_service, "settings", settings_with(openai_api_key=""))
    rows = [{"title": "Alan Turing", "body": "British mathematician.", "url": "http://x"}]
    monkeypatch.setattr(search_service, "_search", lambda *a, **k: rows)

    result = search_service.run("who is alan turing")
    assert "Alan Turing" in result.speech


def test_search_survives_a_failing_summariser(monkeypatch):
    """The LLM call sat outside the try block and could crash the whole turn."""
    from services import search_service

    rows = [{"title": "Alan Turing", "body": "British mathematician.", "url": "http://x"}]
    monkeypatch.setattr(search_service, "_search", lambda *a, **k: rows)

    def boom(*a, **k):
        raise RuntimeError("Error code: 401 - invalid api key sk-abcdefgh12345678")

    monkeypatch.setattr(search_service, "_synthesize", boom)

    result = search_service.run("who is alan turing")
    assert result.success is False
    assert "Alan Turing" in result.speech             # results survive the outage
    assert result.links                                # sources still offered
    assert "sk-abcdefgh12345678" not in result.speech  # and the key is redacted


def test_search_reports_why_it_failed(monkeypatch):
    from services import search_service

    def boom(*a, **k):
        raise ConnectionError("getaddrinfo failed")

    monkeypatch.setattr(search_service, "_search", boom)
    result = search_service.run("anything")
    assert result.success is False
    assert "ConnectionError" in result.speech
