"""Language identification with a confidence estimate."""
from __future__ import annotations

from utils.logger import get_logger

logger = get_logger(__name__)

# Deterministic output across runs.
try:
    from langdetect import DetectorFactory, detect_langs

    DetectorFactory.seed = 0
    _AVAILABLE = True
except Exception:  # noqa: BLE001
    _AVAILABLE = False


def detect_language(text: str) -> tuple[str, float]:
    """Return (iso_639_1_code, confidence in 0..1). Defaults to English."""
    text = (text or "").strip()
    if not text or not _AVAILABLE:
        return "en", 0.0
    try:
        best = detect_langs(text)[0]
        return best.lang, round(float(best.prob), 4)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Language detection failed: %s", exc)
        return "en", 0.0
