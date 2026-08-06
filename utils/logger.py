"""Centralised, idempotent logging setup."""
from __future__ import annotations

import logging
import sys
from functools import lru_cache

from config import settings


@lru_cache(maxsize=None)
def get_logger(name: str = "vini_ai") -> logging.Logger:
    """Return a configured logger. Cached so handlers attach only once."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(settings.log_level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
