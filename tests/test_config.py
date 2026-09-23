"""Provider-configuration invariants.

These pin the behaviour that a misconfiguration previously broke silently: a
key sent to the wrong endpoint just returns 401, which surfaces only as
"my language model is unreachable".
"""
from __future__ import annotations

import importlib
import os

import pytest

PROVIDER_VARS = (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_CHAT_MODEL",
    "OPENAI_STT_MODEL",
    "OPENAI_TTS_MODEL",
    "TTS_BACKEND",
)


def load_config(monkeypatch, **env):
    """Re-import config with a controlled environment.

    Settings are frozen at import time, so each case needs a fresh module.
    `.env` is neutralised too, otherwise a developer's own file would decide
    the result.
    """
    import config as config_module

    monkeypatch.setattr(config_module, "load_dotenv", lambda *a, **k: False, raising=False)
    for var in PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    # dotenv is called at import, so patch it at source for the reload.
    import dotenv

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr("config.load_dotenv", lambda *a, **k: False, raising=False)
    return importlib.reload(config_module)


# --------------------------------------------------------------------------- #
# A fresh install with no .env must match the documented Groq defaults
# --------------------------------------------------------------------------- #
def test_fresh_install_resolves_to_groq(monkeypatch):
    cfg = load_config(monkeypatch)
    s = cfg.settings
    assert s.openai_base_url == "https://api.groq.com/openai/v1"
    assert s.openai_chat_model == "openai/gpt-oss-20b"
    assert s.openai_stt_model == "whisper-large-v3"


def test_fresh_install_keeps_gtts(monkeypatch):
    cfg = load_config(monkeypatch)
    assert cfg.settings.tts_backend == "gtts"


def test_no_api_key_is_ever_hardcoded(monkeypatch):
    cfg = load_config(monkeypatch)
    assert cfg.settings.openai_api_key == ""
    assert cfg.settings.openai_enabled is False


# --------------------------------------------------------------------------- #
# Blank / whitespace must not become an empty base URL
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("blank", ["", "   ", "\t", "\n  "])
def test_blank_base_url_falls_back_to_default(monkeypatch, blank):
    cfg = load_config(monkeypatch, OPENAI_BASE_URL=blank)
    assert cfg.settings.openai_base_url == cfg.DEFAULT_BASE_URL
    assert cfg.settings.openai_client_kwargs["base_url"] == cfg.DEFAULT_BASE_URL


@pytest.mark.parametrize("blank", ["", "   "])
def test_blank_model_vars_fall_back(monkeypatch, blank):
    cfg = load_config(monkeypatch, OPENAI_CHAT_MODEL=blank, OPENAI_STT_MODEL=blank)
    assert cfg.settings.openai_chat_model == cfg.DEFAULT_CHAT_MODEL
    assert cfg.settings.openai_stt_model == cfg.DEFAULT_STT_MODEL


def test_base_url_is_always_passed_explicitly(monkeypatch):
    """The SDK reads OPENAI_BASE_URL itself and takes a blank value literally."""
    cfg = load_config(monkeypatch, OPENAI_BASE_URL="")
    assert cfg.settings.openai_client_kwargs["base_url"]


def test_surrounding_whitespace_is_stripped(monkeypatch):
    cfg = load_config(monkeypatch, OPENAI_BASE_URL="  https://api.groq.com/openai/v1  ")
    assert cfg.settings.openai_base_url == "https://api.groq.com/openai/v1"


# --------------------------------------------------------------------------- #
# Switching provider stays possible
# --------------------------------------------------------------------------- #
def test_can_switch_back_to_openai(monkeypatch):
    cfg = load_config(
        monkeypatch,
        OPENAI_BASE_URL="https://api.openai.com/v1",
        OPENAI_CHAT_MODEL="gpt-4o-mini",
        OPENAI_STT_MODEL="whisper-1",
    )
    s = cfg.settings
    assert s.openai_base_url == cfg.OPENAI_BASE_URL
    assert s.openai_chat_model == "gpt-4o-mini"
    assert s.openai_client_kwargs["base_url"] == "https://api.openai.com/v1"


def test_can_point_at_a_local_server(monkeypatch):
    cfg = load_config(monkeypatch, OPENAI_BASE_URL="http://localhost:11434/v1")
    assert cfg.settings.openai_client_kwargs["base_url"] == "http://localhost:11434/v1"


def test_env_overrides_every_provider_default(monkeypatch):
    cfg = load_config(
        monkeypatch,
        OPENAI_API_KEY="test-key",
        OPENAI_BASE_URL="https://example.test/v1",
        OPENAI_CHAT_MODEL="custom-chat",
        OPENAI_STT_MODEL="custom-stt",
        OPENAI_TTS_MODEL="custom-tts",
    )
    s = cfg.settings
    assert (s.openai_base_url, s.openai_chat_model) == ("https://example.test/v1", "custom-chat")
    assert (s.openai_stt_model, s.openai_tts_model) == ("custom-stt", "custom-tts")
    assert s.openai_enabled is True


# --------------------------------------------------------------------------- #
# Client kwargs
# --------------------------------------------------------------------------- #
def test_client_kwargs_shape(monkeypatch):
    cfg = load_config(monkeypatch, OPENAI_API_KEY="k")
    assert set(cfg.settings.openai_client_kwargs) == {"api_key", "base_url"}


def test_client_kwargs_reach_the_sdk(monkeypatch):
    """Guards the original bug: base_url silently not reaching the client."""
    openai = pytest.importorskip("openai")
    cfg = load_config(monkeypatch, OPENAI_API_KEY="k")
    client = openai.OpenAI(**cfg.settings.openai_client_kwargs)
    assert "api.groq.com" in str(client.base_url)


def test_env_var_does_not_leak_into_sdk_default(monkeypatch):
    """A blank env var must not reach the SDK's own OPENAI_BASE_URL lookup."""
    openai = pytest.importorskip("openai")
    cfg = load_config(monkeypatch, OPENAI_API_KEY="k", OPENAI_BASE_URL="")
    assert os.environ.get("OPENAI_BASE_URL") == ""
    client = openai.OpenAI(**cfg.settings.openai_client_kwargs)
    assert "api.groq.com" in str(client.base_url)
