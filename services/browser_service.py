"""Web / YouTube search. Returns clickable links instead of opening a browser
on the server, which is the correct behaviour for a hosted deployment."""
from __future__ import annotations

from urllib.parse import quote_plus

from assistant.schemas import Link, SkillResponse

_STOP = ("search the web for", "search for", "google", "look up", "find information about",
         "find", "search youtube for", "youtube", "show me", "search")


def _clean(text: str) -> str:
    low = text.lower()
    for s in _STOP:
        low = low.replace(s, "")
    return low.strip(" ?.")


def web_search(text: str, entities=None, context=None) -> SkillResponse:
    q = _clean(text) or text
    url = f"https://www.google.com/search?q={quote_plus(q)}"
    return SkillResponse(speech=f"Here are Google results for '{q}'.",
                         links=[Link("Open Google results", url)], data={"query": q})


def youtube_search(text: str, entities=None, context=None) -> SkillResponse:
    q = _clean(text) or text
    url = f"https://www.youtube.com/results?search_query={quote_plus(q)}"
    return SkillResponse(speech=f"Here are YouTube results for '{q}'.",
                         links=[Link("Open on YouTube", url)], data={"query": q})


def play_music(text: str, entities, context=None) -> SkillResponse:
    song = next((e.text for e in entities if e.label == "SONG"), "")
    artist = next((e.text for e in entities if e.label == "ARTIST"), "")
    query = " ".join(p for p in [song, artist] if p) or _clean(text)
    if not query:
        return SkillResponse.error("What would you like me to play?")
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    label = f"{song} by {artist}" if artist else (song or query)
    return SkillResponse(speech=f"Playing {label} on YouTube.",
                         links=[Link(f"Play {label}", url)], data={"song": song, "artist": artist})
