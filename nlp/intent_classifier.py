"""
Intent classification.

Design goals:
  * NLP-first: classify by semantic similarity to labelled examples, not by
    hard-coded keyword matching.
  * Robust: three interchangeable backends with automatic graceful
    degradation, so the app runs on a 1 GB free-tier server or a laptop:
        1. sentence-transformer embeddings  (best accuracy, optional download)
        2. TF-IDF + cosine                  (no downloads, always available)
        3. rapidfuzz token similarity       (last-resort fallback)
  * Extensible: new intents come from `nlp/intents.py` only.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import settings
from nlp.intents import INTENT_EXAMPLES
from nlp.schemas import IntentScore
from utils.logger import get_logger

logger = get_logger(__name__)


class IntentClassifier:
    """Classifies an utterance into one of the catalog intents with a score."""

    def __init__(self) -> None:
        self._labels: list[str] = []
        self._corpus: list[str] = []
        for intent, examples in INTENT_EXAMPLES.items():
            for ex in examples:
                self._labels.append(intent)
                self._corpus.append(ex)

        self.engine = "tfidf"
        self._encode: Callable[[list[str]], np.ndarray] | None = None
        self._example_matrix = None

        if settings.use_embeddings:
            self._try_init_embeddings()
        if self._encode is None:
            self._init_tfidf()

    # ------------------------------------------------------------------ #
    # Backend initialisation
    # ------------------------------------------------------------------ #
    def _try_init_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(settings.embedding_model)

            def _encode(texts: list[str]) -> np.ndarray:
                return model.encode(texts, normalize_embeddings=True)

            self._encode = _encode
            self._example_matrix = _encode(self._corpus)
            self.engine = "embeddings"
            logger.info("Intent classifier using embeddings: %s", settings.embedding_model)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.warning("Embedding backend unavailable (%s); falling back to TF-IDF.", exc)
            self._encode = None

    def _init_tfidf(self) -> None:
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        self._example_matrix = self._vectorizer.fit_transform(self._corpus)
        self.engine = "tfidf"
        logger.info("Intent classifier using TF-IDF backend.")

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def _similarities(self, text: str) -> np.ndarray:
        if self.engine == "embeddings" and self._encode is not None:
            query_vec = self._encode([text])
            return cosine_similarity(query_vec, self._example_matrix)[0]
        query_vec = self._vectorizer.transform([text])
        return cosine_similarity(query_vec, self._example_matrix)[0]

    def _fuzzy_scores(self, text: str) -> dict[str, float]:
        from rapidfuzz import fuzz

        best: dict[str, float] = {}
        for label, example in zip(self._labels, self._corpus, strict=True):
            score = fuzz.token_set_ratio(text, example) / 100.0
            if score > best.get(label, 0.0):
                best[label] = score
        return best

    def classify(self, text: str, top_k: int = 5) -> tuple[str, float, list[IntentScore]]:
        """Return (intent, confidence, ranked_scores).

        `confidence` is a softmax probability over per-intent best scores.
        If the strongest structured match is weak, we defer to GENERAL_CHAT so
        the LLM can handle open-ended requests.
        """
        text = (text or "").strip()
        if not text:
            return "GENERAL_CHAT", 0.0, []

        try:
            sims = self._similarities(text)
            per_intent: dict[str, float] = {}
            for label, sim in zip(self._labels, sims, strict=True):
                if sim > per_intent.get(label, -1.0):
                    per_intent[label] = float(sim)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Primary similarity failed (%s); using fuzzy fallback.", exc)
            per_intent = self._fuzzy_scores(text)

        ranked = sorted(per_intent.items(), key=lambda kv: kv[1], reverse=True)
        top_intent, top_raw = ranked[0]

        # Softmax confidence over the top matches (temperature-scaled).
        top_scores = [s for _, s in ranked[:top_k]]
        temp = 0.12
        exps = [math.exp(s / temp) for s in top_scores]
        denom = sum(exps) or 1.0
        confidence = exps[0] / denom

        ranking = [IntentScore(intent=i, score=round(s, 4)) for i, s in ranked[:top_k]]

        # Weak structured signal -> hand to the conversational LLM.
        if top_raw < settings.intent_confidence_floor and top_intent not in {"GREETING", "GOODBYE"}:
            logger.debug("Top intent %s below floor (%.3f); deferring to GENERAL_CHAT.", top_intent, top_raw)
            return "GENERAL_CHAT", round(confidence, 4), ranking

        return top_intent, round(confidence, 4), ranking


_classifier: IntentClassifier | None = None


def get_classifier() -> IntentClassifier:
    """Process-wide singleton (fitting the vectoriser once is enough)."""
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
