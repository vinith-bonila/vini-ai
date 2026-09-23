"""
Central configuration for VINI AI.

All tunable behaviour lives here and is overridable through environment
variables (loaded from a local .env in development). Nothing else in the
codebase should read os.environ directly - they import `settings` instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Provider defaults
#
# The app ships Groq-first: free tier and fast enough inference to keep the
# voice loop responsive. These are the values a fresh checkout with no .env
# resolves to, so the running app matches what the README documents.
#
# To use OpenAI (or any other compatible provider) set OPENAI_BASE_URL and the
# model variables explicitly - see .env.example.
# --------------------------------------------------------------------------- #
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"

DEFAULT_BASE_URL = GROQ_BASE_URL
DEFAULT_CHAT_MODEL = "openai/gpt-oss-20b"
DEFAULT_STT_MODEL = "whisper-large-v3"
DEFAULT_TTS_MODEL = "tts-1"  # only used when TTS_BACKEND=openai


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_str(key: str, default: str) -> str:
    """Env value with blank and whitespace-only treated as unset.

    A commented-out `.env` line left as `OPENAI_BASE_URL=` must not resolve to
    an empty base URL - the SDK would take that literally and every call would
    fail against it.
    """
    return (os.getenv(key) or "").strip() or default


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved at import time."""

    # --- Identity -------------------------------------------------------
    app_name: str = "VINI AI"
    tagline: str = "An NLP-first voice assistant"
    author: str = "Vinith Bonila"
    version: str = "1.0.0"

    # --- LLM provider (any OpenAI-compatible endpoint) -------------------
    # Never defaulted: a key only ever comes from the environment.
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: _get_str("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    openai_chat_model: str = field(default_factory=lambda: _get_str("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL))
    openai_tts_model: str = field(default_factory=lambda: _get_str("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL))
    openai_stt_model: str = field(default_factory=lambda: _get_str("OPENAI_STT_MODEL", DEFAULT_STT_MODEL))

    # --- NLP models -----------------------------------------------------
    spacy_model: str = field(default_factory=lambda: os.getenv("SPACY_MODEL", "en_core_web_sm"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    use_embeddings: bool = field(default_factory=lambda: _get_bool("USE_EMBEDDINGS", False))
    use_hf_sentiment: bool = field(default_factory=lambda: _get_bool("USE_HF_SENTIMENT", False))
    intent_confidence_floor: float = field(
        default_factory=lambda: float(os.getenv("INTENT_CONFIDENCE_FLOOR", "0.28"))
    )
    # When a deterministic skill fails and the classifier was not confident,
    # the request was probably misrouted - hand it to the LLM rather than
    # dead-ending on the skill's own error.
    skill_fallback_confidence: float = field(
        default_factory=lambda: float(os.getenv("SKILL_FALLBACK_CONFIDENCE", "0.60"))
    )

    # --- Speech ---------------------------------------------------------
    # "openai" uses the hosted Whisper/TTS API; "local" uses on-device libs.
    stt_backend: str = field(default_factory=lambda: os.getenv("STT_BACKEND", "openai"))
    tts_backend: str = field(default_factory=lambda: os.getenv("TTS_BACKEND", "gtts"))
    default_tts_language: str = field(default_factory=lambda: os.getenv("DEFAULT_TTS_LANGUAGE", "en"))
    voice_speed: float = field(default_factory=lambda: float(os.getenv("VOICE_SPEED", "1.0")))

    # --- Skills / services ---------------------------------------------
    # Desktop-only skills (opening local apps) are meaningless on a hosted
    # server, so they are disabled by default and gated behind this flag.
    allow_local_system_skills: bool = field(
        default_factory=lambda: _get_bool("ALLOW_LOCAL_SYSTEM_SKILLS", False)
    )
    weather_default_city: str = field(default_factory=lambda: os.getenv("WEATHER_DEFAULT_CITY", "Visakhapatnam"))
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))

    # --- Persistence ----------------------------------------------------
    db_path: str = field(default_factory=lambda: os.getenv("DB_PATH", str(DATA_DIR / "vini_ai.db")))

    # --- Logging --------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @property
    def openai_enabled(self) -> bool:
        return bool(self.openai_api_key)

    @property
    def openai_client_kwargs(self) -> dict[str, str]:
        """Constructor kwargs for an OpenAI SDK client.

        Passing `base_url` explicitly is what lets a non-OpenAI provider work:
        a Groq key against api.openai.com just returns 401, which surfaces as
        the model being 'unreachable'.
        """
        # base_url is always passed explicitly. Left to itself the SDK reads the
        # OPENAI_BASE_URL environment variable, where a blank value would be
        # taken literally as an empty base URL; `openai_base_url` has already
        # resolved blanks to the default provider.
        return {
            "api_key": self.openai_api_key,
            "base_url": self.openai_base_url,
        }


settings = Settings()
