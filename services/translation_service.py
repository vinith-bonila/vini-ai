"""Translation via deep-translator (Google backend, no API key)."""
from __future__ import annotations

from deep_translator import GoogleTranslator

from assistant.schemas import SkillResponse
from utils.logger import get_logger

logger = get_logger(__name__)

# Friendly names -> ISO codes deep-translator accepts directly, but we map the
# common ones so "french" etc. always resolve.
_LANG = {
    "english": "en", "french": "fr", "spanish": "es", "german": "de",
    "italian": "it", "portuguese": "pt", "hindi": "hi", "telugu": "te",
    "tamil": "ta", "japanese": "ja", "korean": "ko", "chinese": "zh-CN",
    "arabic": "ar", "russian": "ru", "dutch": "nl", "bengali": "bn",
}


def run(text: str, entities, context=None) -> SkillResponse:
    to_translate = next((e.text for e in entities if e.label == "TEXT"), "")
    target_raw = next((e.text for e in entities if e.label == "TARGET_LANG"), "")
    if not to_translate or not target_raw:
        return SkillResponse.error("Tell me what to translate and into which language.")
    target = _LANG.get(target_raw.lower(), target_raw.lower())
    try:
        translated = GoogleTranslator(source="auto", target=target).translate(to_translate)
        return SkillResponse(
            speech=translated,
            data={"source_text": to_translate, "target_language": target_raw, "translated": translated},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Translation error: %s", exc)
        return SkillResponse.error(f"I couldn't translate into '{target_raw}'.")
