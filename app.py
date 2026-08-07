"""
VINI AI - Streamlit entry point (home / assistant page).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib
import re

import streamlit as st

from assistant.conversation import Conversation
from assistant.dispatcher import handle
from config import settings
from speech.speech_to_text import TranscriptionError, transcribe
from speech.text_to_speech import synthesize
from ui import brand_header, inject_css, render_nlu_panels, status_pill

st.set_page_config(page_title="VINI AI", page_icon="\U0001F399\uFE0F", layout="wide")
inject_css()

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
_defaults = {
    "conversation": None,
    "last_result": None,
    "status": "idle",
    "last_audio_hash": None,
    "speak_replies": True,
    "pending_text": "",
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
if st.session_state.conversation is None:
    st.session_state.conversation = Conversation()

conv: Conversation = st.session_state.conversation


def process(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    st.session_state.status = "thinking"
    result = handle(text, conv)
    st.session_state.last_result = result
    st.session_state.status = "idle"


def _submit_text() -> None:
    """Enter in the text box: queue the message and clear the field."""
    typed = st.session_state.get("text_box", "")
    if typed.strip():
        st.session_state.pending_text = typed
    st.session_state.text_box = ""


def _speech_lead(text: str, limit: int = 320) -> str:
    """Speak only a short lead for long answers so voice stays pleasant."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    lead = " ".join(sentences[:2]).strip()
    return (lead or text)[:limit]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    brand_header()
    st.divider()
    st.markdown("**Assistant status**")
    st.markdown(status_pill(st.session_state.status), unsafe_allow_html=True)
    st.divider()

    st.session_state.speak_replies = st.toggle("Speak replies aloud", value=st.session_state.speak_replies)
    st.caption(
        f"LLM: {'connected' if settings.openai_enabled else 'offline (structured skills only)'}"
    )
    st.caption(f"STT: {settings.stt_backend}  \u00b7  TTS: {settings.tts_backend}")

    st.divider()
    if st.button("Clear conversation", width='stretch'):
        conv.clear()
        st.session_state.last_result = None
        st.rerun()
    st.caption(f"Session `{conv.session_id}` \u00b7 {len(conv.turns)} turns")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
brand_header()
st.write(
    "Ask by text or voice. VINI AI understands intent through NLP - not keyword "
    "matching - and shows its full analysis for every request."
)

# --------------------------------------------------------------------------- #
# Input row: text field + voice button, side by side
# --------------------------------------------------------------------------- #
text_col, mic_col = st.columns([4, 1.3], vertical_alignment="bottom")
with text_col:
    st.text_input(
        "Message",
        key="text_box",
        placeholder="Type a message to VINI AI...",
        label_visibility="collapsed",
        on_change=_submit_text,
    )
with mic_col:
    audio = st.audio_input("Voice", label_visibility="collapsed") if hasattr(st, "audio_input") else None

st.caption(
    "Try: *play Believer by Imagine Dragons* \u00b7 *translate good morning to French* "
    "\u00b7 *what's 18% of 4500* \u00b7 *weather in Mumbai* \u00b7 *who is Alan Turing*"
)

# Handle a submitted text message.
if st.session_state.pending_text:
    queued = st.session_state.pending_text
    st.session_state.pending_text = ""
    process(queued)

# Handle a new voice recording.
if audio is not None:
    audio_bytes = audio.getvalue()
    digest = hashlib.md5(audio_bytes).hexdigest()
    if digest != st.session_state.last_audio_hash:
        st.session_state.last_audio_hash = digest
        st.session_state.status = "listening"
        spoken, error = "", None
        try:
            spoken = transcribe(audio_bytes)
        except TranscriptionError as exc:
            error = str(exc)
        if spoken:
            st.toast(f"Heard: {spoken}")
            process(spoken)
        elif error:
            st.error(f"Transcription failed: {error}")
        else:
            st.warning("I heard silence - try speaking a little longer.")

# --------------------------------------------------------------------------- #
# Latest response + NLP transparency panels
# --------------------------------------------------------------------------- #
result = st.session_state.last_result
if result is not None:
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Response")
        with st.chat_message("assistant"):
            st.write(result.response.speech)
            for link in result.response.links:
                st.link_button(link.label, link.url)
        if st.session_state.speak_replies and result.response.success:
            # Always speak English by default; only a translation is spoken in
            # its target language, so replies never come out in a misdetected voice.
            tts_lang = "en"
            if result.nlu.intent == "TRANSLATE":
                tts_lang = result.response.data.get("target_code", "en")
            audio_bytes = synthesize(_speech_lead(result.response.speech), lang=tts_lang)
            if audio_bytes:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)
    with right:
        st.markdown("#### NLP analysis")
        render_nlu_panels(result)

# --------------------------------------------------------------------------- #
# Conversation history
# --------------------------------------------------------------------------- #
if conv.turns:
    st.divider()
    st.markdown("#### Conversation")
    for turn in conv.turns:
        with st.chat_message("user"):
            st.write(turn.user_text)
        with st.chat_message("assistant"):
            st.write(turn.assistant_text)
