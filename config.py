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


def _get_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings resolved at import time."""

    # --- Identity -------------------------------------------------------
    app_name: str = "VINI AI"
    tagline: str = "An NLP-first voice assistant"
    author: str = "Vinith Bonila"
    version: str = "1.0.0"

    # --- LLM / OpenAI ---------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    openai_tts_model: str = field(default_factory=lambda: os.getenv("OPENAI_TTS_MODEL", "tts-1"))
    openai_stt_model: str = field(default_factory=lambda: os.getenv("OPENAI_STT_MODEL", "whisper-1"))

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


settings = Settings()
