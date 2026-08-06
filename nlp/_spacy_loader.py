"""Single cached spaCy model shared by the parser and NER modules."""
from __future__ import annotations

from functools import lru_cache

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_nlp():
    """Load the spaCy pipeline once per process."""
    import spacy

    try:
        nlp = spacy.load(settings.spacy_model)
    except OSError:
        logger.warning("spaCy model '%s' not found; downloading...", settings.spacy_model)
        from spacy.cli import download

        download(settings.spacy_model)
        nlp = spacy.load(settings.spacy_model)
    logger.info("Loaded spaCy model: %s", settings.spacy_model)
    return nlp
