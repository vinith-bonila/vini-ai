"""Weather via the free Open-Meteo API (no key required)."""
from __future__ import annotations

import requests

from assistant.schemas import SkillResponse
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "rime fog", 51: "light drizzle", 61: "light rain",
    63: "moderate rain", 65: "heavy rain", 71: "light snow", 80: "rain showers",
    95: "thunderstorm", 96: "thunderstorm with hail",
}


def _city_from_entities(entities) -> str:
    for e in entities:
        if e.label in {"GPE", "LOC"}:
            return e.text
    return settings.weather_default_city


def run(text: str, entities, context=None) -> SkillResponse:
    city = _city_from_entities(entities)
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=8,
        ).json()
        if not geo.get("results"):
            return SkillResponse.error(f"I couldn't find a place called '{city}'.")
        loc = geo["results"][0]
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current": "temperature_2m,weather_code,relative_humidity_2m"},
            timeout=8,
        ).json()["current"]
        desc = _CODES.get(wx["weather_code"], "unknown conditions")
        temp = wx["temperature_2m"]
        humidity = wx["relative_humidity_2m"]
        speech = f"It's {temp}\u00b0C and {desc} in {loc['name']}, with {humidity}% humidity."
        return SkillResponse(speech=speech, data={"city": loc["name"], "temp_c": temp,
                                                  "conditions": desc, "humidity": humidity})
    except Exception as exc:  # noqa: BLE001
        logger.warning("Weather error: %s", exc)
        return SkillResponse.error("I couldn't fetch the weather right now.")
