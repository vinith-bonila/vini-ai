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


def handle(text: str, conversation: Conversation) -> AssistantResult:
    """Process one utterance end-to-end."""
    with timed() as t:
        nlu = analyze(text)
        nlu = conversation.resolve_followups(nlu)
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
