"""
Named Entity Recognition and slot extraction.

Two layers:
  1. General NER from spaCy (PERSON, GPE, DATE, ...).
  2. Intent-specific slot extractors (SONG/ARTIST for PLAY_MUSIC,
     TEXT/TARGET_LANG for TRANSLATE, EXPRESSION for CALCULATOR, ...) using
     lightweight regex patterns. These give the router clean, typed slots.
"""
from __future__ import annotations

import re

from nlp._spacy_loader import get_nlp
from nlp.schemas import Entity

# --------------------------------------------------------------------------- #
# Regex slot patterns per intent
# --------------------------------------------------------------------------- #
_PLAY_BY = re.compile(r"play\s+(?P<song>.+?)\s+by\s+(?P<artist>.+?)(?:\s+on\s+youtube)?$", re.I)
_PLAY_ONLY = re.compile(r"play\s+(?P<song>.+?)(?:\s+on\s+youtube)?$", re.I)
_TRANSLATE = re.compile(
    r"(?:translate|say|convert)\s+(?:this\s+)?(?:sentence\s+)?['\"]?(?P<text>.+?)['\"]?\s+"
    r"(?:in|into|to)\s+(?P<lang>[a-zA-Z]+)",
    re.I,
)
_TRANSLATE_HOWTO = re.compile(
    r"how\s+do\s+you\s+say\s+(?P<text>.+?)\s+in\s+(?P<lang>[a-zA-Z]+)", re.I
)
_DEFINE = re.compile(r"(?:define|meaning of|definition of|what does)\s+(?P<term>[a-zA-Z\- ]+?)(?:\s+mean)?$", re.I)
_MATH = re.compile(r"[-+/*^%().\d\s]|times|plus|minus|divided|power|percent|of|sqrt|square", re.I)


def _slot_entities(text: str, intent: str) -> list[Entity]:
    slots: list[Entity] = []
    t = text.strip()

    if intent == "PLAY_MUSIC":
        m = _PLAY_BY.search(t)
        if m:
            slots.append(Entity(m.group("song").strip(), "SONG"))
            slots.append(Entity(m.group("artist").strip(), "ARTIST"))
        else:
            m = _PLAY_ONLY.search(t)
            if m and m.group("song").strip() not in {"music", "a song", "some music"}:
                slots.append(Entity(m.group("song").strip(), "SONG"))

    elif intent == "TRANSLATE":
        m = _TRANSLATE.search(t) or _TRANSLATE_HOWTO.search(t)
        if m:
            slots.append(Entity(m.group("text").strip(), "TEXT"))
            slots.append(Entity(m.group("lang").strip().lower(), "TARGET_LANG"))

    elif intent == "DICTIONARY":
        m = _DEFINE.search(t)
        if m:
            slots.append(Entity(m.group("term").strip(), "TERM"))

    elif intent in {"CALCULATOR", "UNIT_CONVERSION"}:
        slots.append(Entity(t, "EXPRESSION"))

    return slots


def extract_entities(text: str, intent: str = "") -> list[Entity]:
    """Return general spaCy entities plus intent-specific slots (deduplicated)."""
    doc = get_nlp()(text or "")
    entities = [
        Entity(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents
    ]
    entities.extend(_slot_entities(text, intent))

    seen: set[tuple[str, str]] = set()
    unique: list[Entity] = []
    for e in entities:
        key = (e.text.lower(), e.label)
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return unique
