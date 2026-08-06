"""
Speech-to-text.

Backends (config STT_BACKEND):
  * "openai" - hosted transcription (Whisper / gpt-4o-transcribe family).
    Recommended for deployment: no heavy local model, low memory. Needs a key.
  * "local"  - faster-whisper on-device. No network, but heavier.

Input is raw audio bytes (e.g. from Streamlit's st.audio_input, which records
WAV in the browser), so it works without server-side microphone access.

Errors are raised (not silently swallowed) so the UI can show the real reason.
"""
from __future__ import annotations

import importlib
import tempfile
from functools import lru_cache

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class TranscriptionError(RuntimeError):
    """Raised when speech-to-text fails, carrying a human-readable reason."""


def _transcribe_openai(audio_bytes: bytes) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    # Pass an explicit (filename, bytes, mimetype) tuple - the most reliable way
    # to tell the API the format. st.audio_input records WAV.
    file_tuple = ("speech.wav", audio_bytes, "audio/wav")
    try:
        result = client.audio.transcriptions.create(
            model=settings.openai_stt_model,
            file=file_tuple,
            response_format="text",
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriptionError(str(exc)) from exc
    # With response_format="text" the SDK returns a plain string.
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return (text or "").strip()


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
    """Convert audio bytes to text.

    Raises TranscriptionError with the real reason on failure so the caller can
    display it, rather than returning a silent empty string.
    """
    if not audio_bytes:
        return ""

    backend = settings.stt_backend.lower()
    if backend == "openai":
        if not settings.openai_enabled:
            raise TranscriptionError(
                "No OpenAI API key is set. Add one in Settings, or switch the "
                "STT backend to 'local'."
            )
        return _transcribe_openai(audio_bytes)

    try:
        return _transcribe_local(audio_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Local transcription failed: %s", exc)
        raise TranscriptionError(str(exc)) from exc