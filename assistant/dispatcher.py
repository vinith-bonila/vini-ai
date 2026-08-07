"""
The dispatcher - the single entry point the UI calls.

For each user utterance it: runs the NLU pipeline, resolves follow-ups against
conversation context, routes to the right skill, records the turn, persists an
analytics row, and returns a combined result the UI can render.
"""
from __future__ import annotations

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


# Words that signal a question needs current, real-world facts.
_FRESH_MARKERS = (
    "current", "currently", "latest", "today", "right now", "this year",
    "recent", "recently", "nowadays", "at present", "these days",
    "2024", "2025", "2026", "who won", "price of", "stock price", "as of",
)
# Skills that own their queries and must never be hijacked by web search.
_PROTECTED = {
    "WEATHER", "CALCULATOR", "UNIT_CONVERSION", "PLAY_MUSIC", "TRANSLATE",
    "JOKE", "DICTIONARY", "GREETING", "GOODBYE",
}
# Words that mark a genuine time/date request (so those stay with TIME/DATE).
_TIME_DATE_WORDS = ("time", "clock", "o'clock", "date", "day is", "what day")


def _needs_fresh_info(text: str, intent: str) -> bool:
    """Should this question be answered with a live web search instead?"""
    low = text.lower()
    if not any(marker in low for marker in _FRESH_MARKERS):
        return False
    if intent in _PROTECTED:
        return False
    if intent in {"TIME", "DATE"}:
        # A real time/date question keeps its skill; a misrouted one
        # (e.g. "who is the current PM") falls through to web search.
        return not any(word in low for word in _TIME_DATE_WORDS)
    return True


def handle(text: str, conversation: Conversation) -> AssistantResult:
    """Process one utterance end-to-end."""
    with timed() as t:
        nlu = analyze(text)
        nlu = conversation.resolve_followups(nlu)
        if _needs_fresh_info(nlu.text, nlu.intent):
            logger.debug("Freshness override: %s -> WEB_ANSWER", nlu.intent)
            nlu.intent = "WEB_ANSWER"
        response = route(nlu.intent, nlu.text, nlu.entities, conversation)

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