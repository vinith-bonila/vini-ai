"""
Command routing.

Maps every intent to a handler with a plain registry - no if/elif ladder.
Registering a new skill is one line here plus its entry in `nlp/intents.py`.
Each handler has the signature: handler(text, entities, context) -> SkillResponse
"""
from __future__ import annotations

from collections.abc import Callable

from assistant.schemas import SkillResponse
from config import settings
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


# Intents whose own failure message is already the right answer, so replacing
# it with an LLM guess would be worse:
#   GENERAL_CHAT / WEB_ANSWER - the LLM has already been tried
#   OPEN_APP                  - "disabled on a hosted server" is the answer
#   GREETING / GOODBYE        - cannot fail
_NO_LLM_FALLBACK = {"GENERAL_CHAT", "WEB_ANSWER", "OPEN_APP", "GREETING", "GOODBYE"}


def route(intent: str, text: str, entities, context, confidence: float = 1.0) -> SkillResponse:
    """Dispatch to a skill, falling back to the LLM on a low-confidence miss.

    A weak classification that lands on a deterministic skill used to dead-end
    on that skill's error ("tell me what to translate...") even when the user
    had asked a perfectly answerable question. When the skill fails and the
    classifier was unsure, the intent is treated as a misroute.
    """
    handler = ROUTES.get(intent, llm_service.chat)
    try:
        response = handler(text, entities, context)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Handler for %s failed: %s", intent, exc)
        response = SkillResponse.error("Something went wrong handling that request.")

    if (
        not response.success
        and intent not in _NO_LLM_FALLBACK
        and confidence < settings.skill_fallback_confidence
    ):
        logger.info(
            "Skill %s failed at %.0f%% confidence; deferring to the LLM.",
            intent, confidence * 100,
        )
        fallback = llm_service.chat(text, entities, context)
        # Keep the skill's own message if the LLM cannot answer either - it is
        # the more specific of the two.
        if fallback.success:
            return fallback

    return response
