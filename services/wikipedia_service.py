"""Wikipedia summary lookup."""
from __future__ import annotations

import wikipedia

from assistant.schemas import Link, SkillResponse
from utils.logger import get_logger

logger = get_logger(__name__)


def _clean_query(text: str, entities) -> str:
    people = [e.text for e in entities if e.label in {"PERSON", "ORG", "GPE", "WORK_OF_ART", "EVENT"}]
    if people:
        return people[0]
    for stop in ("who is", "what is", "tell me about", "search wikipedia for",
                 "give me a summary of", "wikipedia"):
        text = text.lower().replace(stop, "")
    return text.strip(" ?.")


def run(text: str, entities, context=None) -> SkillResponse:
    query = _clean_query(text, entities)
    if not query:
        return SkillResponse.error("What would you like me to look up on Wikipedia?")
    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True, redirect=True)
        page = wikipedia.page(query, auto_suggest=True)
        return SkillResponse(
            speech=summary,
            links=[Link("Read on Wikipedia", page.url)],
            data={"title": page.title, "query": query},
        )
    except wikipedia.DisambiguationError as exc:
        options = ", ".join(exc.options[:5])
        return SkillResponse(speech=f"That could mean several things: {options}. Which did you mean?")
    except wikipedia.PageError:
        return SkillResponse.error(f"I couldn't find a Wikipedia article for '{query}'.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Wikipedia error: %s", exc)
        return SkillResponse.error("Wikipedia lookup failed. Please try again.")
