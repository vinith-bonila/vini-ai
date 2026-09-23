"""
The dispatcher - the single entry point the UI calls.

For each user utterance it: runs the NLU pipeline, resolves follow-ups against
conversation context, resolves the final intent (fixing classifier false
positives), routes to the right skill, records the turn, persists an analytics
row, and returns a combined result the UI can render.

Intent resolution policy (layered on top of the NLP classifier):
  * TIME / DATE only fire when the question genuinely asks for the current
    clock time or today's date - never because the word "time" happens to
    appear (e.g. "how much time does it take to travel to the moon").
  * Questions needing current, real-world facts ("who is the current PM",
    "latest ...", "price of ...") go to live web search.
  * Everything else that isn't a deterministic skill is answered by the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from assistant.conversation import Conversation
from assistant.router import route
from assistant.schemas import SkillResponse
from database import history
from nlp.pipeline import analyze
from nlp.schemas import NLUResult
from utils.helpers import timed
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AssistantResult:
    nlu: NLUResult
    response: SkillResponse
    response_time_ms: float

    @property
    def end_session(self) -> bool:
        return bool(self.response.data.get("end_session"))


# --------------------------------------------------------------------------- #
# Intent resolution rules
# --------------------------------------------------------------------------- #

# A genuine request for the current clock time.
_TIME_REQUEST = re.compile(
    r"(what(?:'?s| is)?\s+the\s+time"
    r"|what\s+time\s+is\s+it"
    r"|current\s+time"
    r"|time\s+(?:right\s+)?now"
    r"|tell\s+me\s+the\s+time"
    r"|give\s+me\s+the\s+time)"
    r"|^\s*time\s*\??\s*$",
    re.I,
)

# A genuine request for today's date (not "the date of <some event>").
_DATE_REQUEST = re.compile(
    r"(what(?:'?s| is)?\s+the\s+date(?!\s+of)"
    r"|what(?:'?s| is)?\s+date"
    r"|today'?s\s+date"
    r"|current\s+date"
    r"|date\s+today"
    r"|what\s+day\s+is\s+(?:it|today))"
    r"|^\s*date\s*\??\s*$",
    re.I,
)

# Words signalling a question needs current, real-world facts.
_FRESH_MARKERS = (
    "current", "currently", "latest", "today", "right now", "this year",
    "recent", "recently", "nowadays", "at present", "these days",
    "2024", "2025", "2026", "who won", "price of", "stock price", "as of",
    "up to date", "up-to-date",
)

# Deterministic skills that own their queries and must not be hijacked.
# (Wikipedia and News are intentionally NOT here: a "current/latest" question
#  should be answered from a live web search, not a static summary or a link.)
_PROTECTED = {
    "WEATHER", "CALCULATOR", "UNIT_CONVERSION", "PLAY_MUSIC", "TRANSLATE",
    "JOKE", "DICTIONARY", "GREETING", "GOODBYE", "WEB_SEARCH", "YOUTUBE_SEARCH",
}


def _needs_fresh_info(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _FRESH_MARKERS)


def resolve_intent(nlu: NLUResult) -> str:
    """Decide the final intent, correcting common classifier mistakes."""
    text = nlu.text
    intent = nlu.intent

    # 1) Genuine current-time / current-date question -> the right skill,
    #    even if the classifier missed it.
    if _TIME_REQUEST.search(text):
        return "TIME"
    if _DATE_REQUEST.search(text):
        return "DATE"

    # 2) Classifier said TIME/DATE but it isn't actually asking for the clock
    #    or today's date (e.g. "how much time to travel", "give me ...").
    if intent in {"TIME", "DATE"}:
        return "WEB_ANSWER" if _needs_fresh_info(text) else "GENERAL_CHAT"

    # 3) A current-facts question that isn't a deterministic skill -> web search.
    if intent not in _PROTECTED and _needs_fresh_info(text):
        return "WEB_ANSWER"

    # 4) Otherwise trust the classifier (skills, Wikipedia, general chat, ...).
    return intent


def handle(text: str, conversation: Conversation) -> AssistantResult:
    """Process one utterance end-to-end."""
    with timed() as t:
        nlu = analyze(text)
        nlu = conversation.resolve_followups(nlu)
        final_intent = resolve_intent(nlu)
        if final_intent != nlu.intent:
            logger.debug("Intent override: %s -> %s (%r)", nlu.intent, final_intent, text)
            nlu.intent = final_intent
        response = route(nlu.intent, nlu.text, nlu.entities, conversation, nlu.confidence)

    result = AssistantResult(nlu=nlu, response=response, response_time_ms=t["ms"])
    conversation.add_turn(text, nlu, response.speech)

    try:
        history.record(
            session_id=conversation.session_id,
            user_text=text,
            nlu=nlu,
            response_text=response.speech,
            response_time_ms=t["ms"],
            success=response.success,
        )
    except Exception as exc:  # noqa: BLE001 - analytics must never break the reply
        logger.warning("Failed to persist turn: %s", exc)

    return result
