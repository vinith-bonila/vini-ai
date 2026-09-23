"""
General conversation via an OpenAI-compatible Chat API (Groq by default).

Used for the GENERAL_CHAT intent and any request the deterministic skills don't
cover. Degrades gracefully to a helpful message when no API key is configured,
so the whole app still runs offline for the structured skills.
"""
from __future__ import annotations

from functools import lru_cache

from assistant.schemas import SkillResponse
from config import settings
from utils.helpers import failure_reason
from utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "You are VINI AI, a university-level educational assistant. First identify "
    "the user's intent (explain, compare, define, list, calculate, code, "
    "summarise, etc.), then answer like an experienced professor. Use well "
    "structured Markdown: headings, tables, numbered and bulleted lists, code "
    "blocks, and clear ASCII diagrams or flowcharts where a process, structure "
    "or comparison is involved. For 'difference between' questions, lead with a "
    "comparison table across multiple technical parameters, then explain each "
    "item and conclude with which to use where. For engineering or scientific "
    "topics, explain formulae, define every variable, and note real industrial "
    "applications. Prioritise accuracy over completeness, and stay on topic. "
    "Where you are confident, be thorough rather than shallow, and end with a "
    "short Summary and a References section (standard textbooks, NPTEL, MIT "
    "OCW, IEEE, SPE, API, or official docs).\n\n"
    "Accuracy rules, which override the style guidance above:\n"
    "- If you are unsure of a specific fact - a name, date, location, figure, "
    "acronym or organisation - say so plainly instead of guessing. Asked about "
    "an acronym or institution you do not recognise, say you are not sure and "
    "ask the user to confirm what they mean. Do not infer details from a name "
    "that merely looks familiar.\n"
    "- Never invent citations, statistics or founding dates. Cite only sources "
    "you are certain exist.\n"
    "- A short honest answer is better than a long confident wrong one. Do not "
    "pad an answer to look complete."
)


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(**settings.openai_client_kwargs)


def chat(text: str, entities=None, context=None) -> SkillResponse:
    if not settings.openai_enabled:
        return SkillResponse(
            speech="I can handle that once an API key is set in Settings. "
                   "Meanwhile I can still play music, translate, calculate, check the "
                   "weather, search Wikipedia and more.",
            success=False,
        )

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if context is not None:
        messages.extend(context.history_for_llm())
    messages.append({"role": "user", "content": text})

    try:
        resp = _client().chat.completions.create(
            model=settings.openai_chat_model,
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
        )
        return SkillResponse(speech=resp.choices[0].message.content.strip())
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM chat failed (model=%s base_url=%s): %s",
                       settings.openai_chat_model, settings.openai_base_url or "default", exc)
        return SkillResponse.error(
            f"My language model is unreachable right now - {failure_reason(exc)}. "
            f"Check OPENAI_API_KEY, OPENAI_BASE_URL and OPENAI_CHAT_MODEL "
            f"(currently '{settings.openai_chat_model}')."
        )
