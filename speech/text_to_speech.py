"""
Text-to-speech.

Backends (config TTS_BACKEND):
  * "gtts"   - Google TTS, free, returns MP3 bytes. Great for hosted apps: the
    browser plays the bytes via st.audio(). Default.
  * "openai" - OpenAI TTS (higher quality). Needs an API key.
  * "pyttsx3"- offline system voice. Local desktop use only (no audio device on
    a hosted server), so it writes to a temp file and returns its bytes.

Every backend returns MP3/audio bytes so the caller renders them uniformly.
"""
from __future__ import annotations

import io
import tempfile

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _gtts(text: str, lang: str) -> bytes:
    from gtts import gTTS

    buffer = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buffer)
    return buffer.getvalue()


def _openai_tts(text: str) -> bytes:
    from openai import OpenAI

    client = OpenAI(**settings.openai_client_kwargs)
    resp = client.audio.speech.create(model=settings.openai_tts_model, voice="alloy", input=text)
    return resp.read()


def _pyttsx3(text: str) -> bytes:
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", int(175 * settings.voice_speed))
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        engine.save_to_file(text, tmp.name)
        engine.runAndWait()
        tmp.seek(0)
        return open(tmp.name, "rb").read()


def synthesize(text: str, lang: str | None = None) -> bytes | None:
    """Return spoken-audio bytes for `text`, or None on failure."""
    text = (text or "").strip()
    if not text:
        return None
    lang = lang or settings.default_tts_language
    backend = settings.tts_backend.lower()
    try:
        if backend == "openai" and settings.openai_enabled:
            return _openai_tts(text)
        if backend == "pyttsx3":
            return _pyttsx3(text)
        return _gtts(text, lang)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS failed (%s): %s", backend, exc)
        try:
            return _gtts(text, "en")
        except Exception:  # noqa: BLE001
            return None
