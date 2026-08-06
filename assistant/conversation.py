"""
Conversation / context management.

Holds the running dialogue for a session and provides two things the rest of
the assistant needs:
  * `history_for_llm()` - recent turns formatted for the chat model, giving
    multi-turn memory.
  * `resolve_followups()` - fills slots omitted in elliptical follow-ups
    ("and in Delhi?", "what about tomorrow") from the previous turn, giving the
    assistant genuine context awareness rather than treating each turn cold.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from nlp.schemas import Entity, NLUResult


@dataclass
class Turn:
    user_text: str
    nlu: NLUResult
    assistant_text: str


@dataclass
class Conversation:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    turns: list[Turn] = field(default_factory=list)
    max_llm_turns: int = 6

    # ------------------------------------------------------------------ #
    @property
    def last_turn(self) -> Turn | None:
        return self.turns[-1] if self.turns else None

    def add_turn(self, user_text: str, nlu: NLUResult, assistant_text: str) -> None:
        self.turns.append(Turn(user_text, nlu, assistant_text))

    def clear(self) -> None:
        self.turns.clear()

    # ------------------------------------------------------------------ #
    def history_for_llm(self) -> list[dict]:
        msgs: list[dict] = []
        for turn in self.turns[-self.max_llm_turns:]:
            msgs.append({"role": "user", "content": turn.user_text})
            msgs.append({"role": "assistant", "content": turn.assistant_text})
        return msgs

    def resolve_followups(self, nlu: NLUResult) -> NLUResult:
        """Inherit missing slots from the previous turn for elliptical inputs."""
        prev = self.last_turn
        if prev is None:
            return nlu

        # Weather: "what about tomorrow / and in Delhi" keeps the earlier city.
        if nlu.intent == "WEATHER" and not nlu.entities_by_label("GPE"):
            prev_city = prev.nlu.entities_by_label("GPE")
            if prev_city and prev.nlu.intent == "WEATHER":
                nlu.entities.append(Entity(prev_city[0], "GPE"))

        # Translate: "and in Spanish?" reuses the previous text to translate.
        if nlu.intent == "TRANSLATE" and not nlu.entities_by_label("TEXT"):
            prev_text = prev.nlu.entities_by_label("TEXT")
            if prev_text:
                nlu.entities.append(Entity(prev_text[0], "TEXT"))

        return nlu
