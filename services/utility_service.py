"""Small deterministic skills: time, date, jokes, dictionary, news, open-app."""
from __future__ import annotations

import datetime as _dt
import random

import requests

from assistant.schemas import Link, SkillResponse
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_JOKES = [
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "I told my computer I needed a break, and it said 'No problem, I'll go to sleep.'",
    "Why did the developer go broke? Because he used up all his cache.",
    "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
]


def tell_time(text="", entities=None, context=None) -> SkillResponse:
    now = _dt.datetime.now().strftime("%I:%M %p")
    return SkillResponse(speech=f"It's {now}.", data={"time": now})


def tell_date(text="", entities=None, context=None) -> SkillResponse:
    today = _dt.date.today().strftime("%A, %d %B %Y")
    return SkillResponse(speech=f"Today is {today}.", data={"date": today})


def tell_joke(text="", entities=None, context=None) -> SkillResponse:
    return SkillResponse(speech=random.choice(_JOKES))


def define_word(text="", entities=None, context=None) -> SkillResponse:
    term = next((e.text for e in (entities or []) if e.label == "TERM"), "")
    if not term:
        return SkillResponse.error("Which word should I define?")
    try:
        data = requests.get(
            f"https://api.dictionaryapi.dev/api/v2/entries/en/{term}", timeout=8
        ).json()
        meaning = data[0]["meanings"][0]
        definition = meaning["definitions"][0]["definition"]
        pos = meaning.get("partOfSpeech", "")
        return SkillResponse(speech=f"{term} ({pos}): {definition}",
                             data={"term": term, "part_of_speech": pos, "definition": definition})
    except Exception as exc:  # noqa: BLE001
        logger.debug("Dictionary error: %s", exc)
        return SkillResponse.error(f"I couldn't find a definition for '{term}'.")


def news_headlines(text="", entities=None, context=None) -> SkillResponse:
    if not settings.news_api_key:
        return SkillResponse(
            speech="News needs a NewsAPI key. You can read top headlines here instead.",
            links=[Link("Google News", "https://news.google.com")],
        )
    try:
        data = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "in", "pageSize": 5, "apiKey": settings.news_api_key},
            timeout=8,
        ).json()
        titles = [a["title"] for a in data.get("articles", [])][:5]
        if not titles:
            return SkillResponse.error("No headlines available right now.")
        spoken = "Here are the top headlines. " + " ... ".join(titles)
        return SkillResponse(speech=spoken, data={"headlines": titles})
    except Exception as exc:  # noqa: BLE001
        logger.warning("News error: %s", exc)
        return SkillResponse.error("I couldn't fetch the news right now.")


# App names -> OS command (local mode only).
_APPS = {"notepad": "notepad.exe", "calculator": "calc.exe", "calc": "calc.exe",
         "file explorer": "explorer.exe", "browser": "start chrome"}


def open_app(text="", entities=None, context=None) -> SkillResponse:
    low = (text or "").lower()
    target = next((name for name in _APPS if name in low), None)
    if not settings.allow_local_system_skills:
        return SkillResponse(
            speech="Opening desktop apps only works when VINI AI runs on your own machine. "
                   "It's disabled in the hosted version for safety.",
            success=False,
        )
    if not target:
        return SkillResponse.error("Which application should I open?")
    import os
    os.system(_APPS[target])  # noqa: S605 - gated behind explicit local flag
    return SkillResponse(speech=f"Opening {target}.")
