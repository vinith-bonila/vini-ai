"""
Text-to-speech.

Backends (config TTS_BACKEND):
  * "edge"   - Microsoft Edge neural voices. Free, no API key, and markedly
    more natural than the others - this is the default assistant voice.
  * "gtts"   - Google Translate TTS. Free fallback; robotic but very reliable.
  * "openai" - OpenAI TTS. Needs a key on an endpoint that serves it (Groq
    does not).
  * "pyttsx3"- offline system voice. Local desktop use only (no audio device on
    a hosted server), so it writes to a temp file and returns its bytes.

Every backend returns MP3/audio bytes so the caller renders them uniformly.
Input is Markdown from the LLM, so it is stripped first - otherwise the voice
reads '#' and '**' aloud.
"""
from __future__ import annotations

import asyncio
import io
import tempfile

from config import settings
from utils.helpers import speech_text
from utils.logger import get_logger

logger = get_logger(__name__)

# gTTS accent, so English comes out in the configured region rather than US.
_GTTS_TLD = {"en": "co.in"}


def _edge(text: str, lang: str) -> bytes:
    """Microsoft neural voice - the closest free match to an assistant voice."""
    import edge_tts

    voice = settings.edge_voice

    async def run() -> bytes:
        buffer = io.BytesIO()
        rate = f"{int((settings.voice_speed - 1) * 100):+d}%"
        async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()

    return asyncio.run(run())


def _gtts(text: str, lang: str) -> bytes:
    from gtts import gTTS

    buffer = io.BytesIO()
    gTTS(text=text, lang=lang, tld=_GTTS_TLD.get(lang, "com")).write_to_fp(buffer)
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
    """Return spoken-audio bytes for `text`, or None on failure.

    `text` may be Markdown; only its words are spoken.
    """
    text = speech_text(text)
    if not text:
        return None
    lang = lang or settings.default_tts_language
    backend = settings.tts_backend.lower()
    try:
        if backend == "edge":
            return _edge(text, lang)
        if backend == "openai" and settings.openai_enabled:
            return _openai_tts(text)
        if backend == "pyttsx3":
            return _pyttsx3(text)
        return _gtts(text, lang)
    except Exception as exc:  # noqa: BLE001
        # gTTS is the safety net: no key, no extra service, always available.
        logger.warning("TTS failed (%s): %s", backend, exc)
        try:
            return _gtts(text, lang if lang in _GTTS_TLD else "en")
        except Exception:  # noqa: BLE001
            return None
