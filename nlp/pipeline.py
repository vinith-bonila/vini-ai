"""
The NLU pipeline.

Runs a raw utterance through the full stack and returns one `NLUResult`:
    text -> language detection
         -> intent classification (+ ranking, confidence)
         -> entity / slot extraction (intent-aware)
         -> POS + dependency parse
         -> sentiment
This single object is everything the assistant layer needs to act.
"""
from __future__ import annotations

from nlp.intent_classifier import get_classifier
from nlp.language_detector import detect_language
from nlp.ner import extract_entities
from nlp.parser import parse
from nlp.schemas import NLUResult
from nlp.sentiment import analyze_sentiment
from utils.logger import get_logger

logger = get_logger(__name__)


def analyze(text: str, parse_syntax: bool = True) -> NLUResult:
    """Full linguistic analysis of `text`.

    `parse_syntax=False` skips POS/dependency parsing for a faster path when
    the caller only needs intent + entities.
    """
    text = (text or "").strip()

    language, lang_conf = detect_language(text)
    intent, confidence, ranking = get_classifier().classify(text)
    entities = extract_entities(text, intent)
    sentiment_label, sentiment_score = analyze_sentiment(text)
    tokens = parse(text) if parse_syntax else []

    result = NLUResult(
        text=text,
        intent=intent,
        confidence=confidence,
        entities=entities,
        tokens=tokens,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        language=language,
        language_confidence=lang_conf,
        intent_ranking=ranking,
        engine=get_classifier().engine,
    )
    logger.debug("NLU: intent=%s conf=%.2f entities=%d", intent, confidence, len(entities))
    return result
