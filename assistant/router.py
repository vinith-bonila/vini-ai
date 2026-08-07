"""
Command routing.

Maps every intent to a handler with a plain registry - no if/elif ladder.
Registering a new skill is one line here plus its entry in `nlp/intents.py`.
Each handler has the signature: handler(text, entities, context) -> SkillResponse
"""
from __future__ import annotations

from typing import Callable

from assistant.schemas import SkillResponse
from services import (
    browser_service,
    calculator_service,
    llm_service,
    search_service,
    translation_service,
    utility_service,
    weather_service,
    wikipedia_service,
)
from utils.logger import get_logger

logger = get_logger(__name__)

Handler = Callable[..., SkillResponse]

ROUTES: dict[str, Handler] = {
    "PLAY_MUSIC": browser_service.play_music,
    "WIKIPEDIA": wikipedia_service.run,
    "WEB_SEARCH": browser_service.web_search,
    "YOUTUBE_SEARCH": browser_service.youtube_search,
    "TRANSLATE": translation_service.run,
    "WEATHER": weather_service.run,
    "NEWS": utility_service.news_headlines,
    "WEB_ANSWER": search_service.run,
    "TIME": utility_service.tell_time,
    "DATE": utility_service.tell_date,
    "CALCULATOR": calculator_service.calculate,
    "UNIT_CONVERSION": calculator_service.convert_units,
    "DICTIONARY": utility_service.define_word,
    "JOKE": utility_service.tell_joke,
    "OPEN_APP": utility_service.open_app,
    "GENERAL_CHAT": llm_service.chat,
    # GREETING / GOODBYE are handled inline below for a warmer touch.
}


def _greeting(text, entities, context) -> SkillResponse:
    return SkillResponse(speech="Hello! I'm VINI AI. How can I help you today?")


def _goodbye(text, entities, context) -> SkillResponse:
    return SkillResponse(speech="Goodbye! Have a great day.", data={"end_session": True})


ROUTES["GREETING"] = _greeting
ROUTES["GOODBYE"] = _goodbye


def route(intent: str, text: str, entities, context) -> SkillResponse:
    handler = ROUTES.get(intent, llm_service.chat)
    try:
        return handler(text, entities, context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Handler for %s failed: %s", intent, exc)
        return SkillResponse.error("Something went wrong handling that request.")
