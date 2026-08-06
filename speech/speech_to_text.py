"""
Speech-to-text.

Backends (config STT_BACKEND):
  * "openai" - hosted Whisper (whisper-1). Recommended for deployment: no heavy
    local model, low memory. Needs an API key.
  * "local"  - faster-whisper on-device. No network, but heavier.

Input is raw audio bytes (e.g. from Streamlit's st.audio_input), so it works
in the browser without server-side microphone access.
"""
from __future__ import annotations

import io
import tempfile
from functools import lru_cache

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


def _transcribe_openai(audio_bytes: bytes) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    buffer = io.BytesIO(audio_bytes)
    buffer.name = "audio.wav"
    result = client.audio.transcriptions.create(model=settings.openai_stt_model, file=buffer)
    return result.text.strip()


@lru_cache(maxsize=1)
def _local_model():
    from faster_whisper import WhisperModel

    return WhisperModel("base", device="cpu", compute_type="int8")


def _transcribe_local(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        segments, _ = _local_model().transcribe(tmp.name)
        return " ".join(seg.text for seg in segments).strip()


def transcribe(audio_bytes: bytes) -> str:
    """Convert audio bytes to text. Returns '' on failure."""
    if not audio_bytes:
        return ""
    backend = settings.stt_backend.lower()
    try:
        if backend == "openai" and settings.openai_enabled:
            return _transcribe_openai(audio_bytes)
        if backend == "openai" and not settings.openai_enabled:
            logger.warning("STT backend 'openai' set but no API key; trying local.")
        return _transcribe_local(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Transcription failed (%s): %s", backend, exc)
        return ""
