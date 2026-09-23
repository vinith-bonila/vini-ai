"""
VINI AI - Streamlit entry point (home / assistant page).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import re

import streamlit as st

from assistant.conversation import Conversation
from assistant.dispatcher import handle
from speech.speech_to_text import TranscriptionError, transcribe
from speech.text_to_speech import synthesize
from ui import (
    about_section,
    assistant_bubble,
    brand_header,
    hero_headline,
    inject_css,
    render_nlu_panels,
    render_voice_core,
    sidebar_nav,
    user_bubble,
    voice_recorder,
)

st.set_page_config(page_title="VINI AI", page_icon="\U0001F399️", layout="wide")
inject_css()

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
_defaults = {
    "conversation": None,
    "last_result": None,
    "status": "idle",
    "last_recording_id": None,
    "speak_replies": True,
    "_pending_audio": None,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val
if st.session_state.conversation is None:
    st.session_state.conversation = Conversation()

conv: Conversation = st.session_state.conversation

SUGGESTIONS = [
    "Play Believer by Imagine Dragons",
    "Translate good morning to French",
    "What is 18% of 4500?",
    "Weather in Mumbai",
    "Who is Alan Turing?",
]


def _speech_lead(text: str, limit: int = 320) -> str:
    """Speak only a short lead for long answers so voice stays pleasant."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text)
    lead = " ".join(sentences[:2]).strip()
    return (lead or text)[:limit]


def process(text: str, core_ph, response_ph) -> None:
    """Run one utterance through the real pipeline.

    The Voice Intelligence Core is redrawn into `core_ph` right before each
    phase's actual work starts, so what the UI shows always matches the work
    genuinely in flight - never a faked or pre-scripted animation.
    """
    text = (text or "").strip()
    if not text:
        return

    # Drop the previous answer before anything else. Its <audio> element would
    # otherwise keep playing over the new turn while this one is processed.
    response_ph.empty()
    st.session_state["_pending_audio"] = None

    st.session_state.status = "understanding"
    render_voice_core(core_ph, "understanding")

    st.session_state.status = "thinking"
    render_voice_core(core_ph, "thinking")
    result = handle(text, conv)
    st.session_state.last_result = result

    audio_bytes = None
    if st.session_state.speak_replies and result.response.success:
        # Always speak English by default; only a translation is spoken in
        # its target language, so replies never come out in a misdetected voice.
        tts_lang = "en"
        if result.nlu.intent == "TRANSLATE":
            tts_lang = result.response.data.get("target_code", "en")
        st.session_state.status = "speaking"
        render_voice_core(core_ph, "speaking")
        audio_bytes = synthesize(_speech_lead(result.response.speech), lang=tts_lang)

    st.session_state.status = "idle"
    if not audio_bytes:
        render_voice_core(core_ph, "idle")
    st.session_state["_pending_audio"] = audio_bytes


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
sidebar_nav()
with st.sidebar:
    st.divider()
    st.session_state.speak_replies = st.toggle("Speak replies aloud", value=st.session_state.speak_replies)
    st.divider()

    # Filled at the end of the run: the sidebar renders before this turn is
    # processed, so counting here would always lag one turn behind.
    session_info = st.empty()
    if st.button("Clear conversation", width="stretch"):
        conv.clear()
        st.session_state.last_result = None
        st.session_state["_pending_audio"] = None
        st.rerun()

# --------------------------------------------------------------------------- #
# Hero + Voice Intelligence Core
# --------------------------------------------------------------------------- #
brand_header()
hero_headline()

core_ph = st.empty()
render_voice_core(core_ph, st.session_state.status)

# --------------------------------------------------------------------------- #
# Mic input - one tap, then it stops itself when you stop speaking
# --------------------------------------------------------------------------- #
recording = voice_recorder(silence_ms=1500, max_ms=15000)

# --------------------------------------------------------------------------- #
# Command bar (secondary text input)
# --------------------------------------------------------------------------- #
with st.container(key="command_bar"):
    with st.form(key="command_bar_form", clear_on_submit=True, border=False):
        col_text, col_send = st.columns([6, 1], vertical_alignment="bottom")
        with col_text:
            typed = st.text_input(
                "Message",
                placeholder="Type a message to VINI.AI...",
                label_visibility="collapsed",
            )
        with col_send:
            submitted = st.form_submit_button("Send", width="stretch")

# --------------------------------------------------------------------------- #
# Suggestions
# --------------------------------------------------------------------------- #
with st.container(key="suggestions"):
    clicked_suggestion = None
    for suggestion in SUGGESTIONS:
        if st.button(suggestion, key=f"sugg_{suggestion}"):
            clicked_suggestion = suggestion

# Declared before routing so a new turn can clear the previous answer - and
# with it the audio element still playing from that answer.
response_ph = st.empty()

# --------------------------------------------------------------------------- #
# Route whichever input fired this run through the real pipeline
# --------------------------------------------------------------------------- #
if submitted and typed.strip():
    process(typed, core_ph, response_ph)
elif clicked_suggestion:
    process(clicked_suggestion, core_ph, response_ph)
elif recording is not None:
    audio_bytes, recording_id = recording
    if recording_id != st.session_state.last_recording_id:
        st.session_state.last_recording_id = recording_id
        st.session_state.status = "understanding"
        render_voice_core(core_ph, "understanding")
        spoken, error = "", None
        try:
            spoken = transcribe(audio_bytes)
        except TranscriptionError as exc:
            error = str(exc)
        if spoken:
            st.toast(f"Heard: {spoken}")
            process(spoken, core_ph, response_ph)
        else:
            st.session_state.status = "idle"
            render_voice_core(core_ph, "idle")
            if error:
                st.error(f"Transcription failed: {error}")
            else:
                st.warning("I heard silence - try speaking a little longer.")

# --------------------------------------------------------------------------- #
# Latest response + NLP transparency panels
# --------------------------------------------------------------------------- #
result = st.session_state.last_result
if result is not None:
    # Rendered inside the placeholder so the next turn can remove it wholesale,
    # which is what actually stops the previous reply's audio.
    with response_ph.container():
        st.divider()
        left, right = st.columns([1.3, 1])
        with left:
            st.markdown("#### Response")
            last_turn = conv.last_turn
            if last_turn is not None:
                user_bubble(last_turn.user_text)
            assistant_bubble(result.response.speech)
            for link in result.response.links:
                st.link_button(link.label, link.url)

            pending_audio = st.session_state.get("_pending_audio")
            if pending_audio:
                st.audio(pending_audio, format="audio/mp3", autoplay=True)
            st.session_state["_pending_audio"] = None
        with right:
            st.markdown("#### Intent Analysis")
            render_nlu_panels(result)

# --------------------------------------------------------------------------- #
# Conversation history
# --------------------------------------------------------------------------- #
if conv.turns:
    st.divider()
    st.markdown("#### Conversation")
    for turn in conv.turns:
        user_bubble(turn.user_text)
        assistant_bubble(turn.assistant_text)

# --------------------------------------------------------------------------- #
# Explainer - last, so the core and mic own the top of the screen
# --------------------------------------------------------------------------- #
st.divider()
about_section()

turns = len(conv.turns)
session_info.caption(f"Current session · {turns} turn{'' if turns == 1 else 's'}")
