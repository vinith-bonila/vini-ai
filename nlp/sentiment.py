"""Sentiment analysis: VADER by default, optional Hugging Face transformer."""
from __future__ import annotations

from functools import lru_cache

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def _hf_pipeline():
    from transformers import pipeline  # type: ignore

    return pipeline("sentiment-analysis")


def analyze_sentiment(text: str) -> tuple[str, float]:
    """Return (label, score) where label is POSITIVE / NEGATIVE / NEUTRAL.

    score is the signed compound polarity in [-1, 1].
    """
    text = (text or "").strip()
    if not text:
        return "NEUTRAL", 0.0

    if settings.use_hf_sentiment:
        try:
            result = _hf_pipeline()(text[:512])[0]
            label = result["label"].upper()
            signed = result["score"] if label == "POSITIVE" else -result["score"]
            return label, round(float(signed), 4)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HF sentiment unavailable (%s); using VADER.", exc)

    compound = _vader().polarity_scores(text)["compound"]
    if compound >= 0.05:
        label = "POSITIVE"
    elif compound <= -0.05:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return label, round(float(compound), 4)
