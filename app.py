"""
VINI AI - Streamlit entry point (home / assistant page).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import hashlib

import streamlit as st

from assistant.conversation import Conversation
from assistant.dispatcher import handle
from config import settings
from speech.speech_to_text import transcribe
from speech.text_to_speech import synthesize
from ui import brand_header, confidence_meter, inject_css, render_nlu_panels, status_pill

st.set_page_config(page_title="VINI AI", page_icon="\U0001F399\uFE0F", layout="wide")
inject_css()

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "conversation" not in st.session_state:
    st.session_state.conversation = Conversation()
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "status" not in st.session_state:
    st.session_state.status = "idle"
if "last_audio_hash" not in st.session_state:
    st.session_state.last_audio_hash = None
if "speak_replies" not in st.session_state:
    st.session_state.speak_replies = True

conv: Conversation = st.session_state.conversation


def process(text: str) -> None:
    text = (text or "").strip()
    if not text:
        return
    st.session_state.status = "thinking"
    result = handle(text, conv)
    st.session_state.last_result = result
    st.session_state.status = "idle"


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
# Input row: push-to-talk (voice) + text
# --------------------------------------------------------------------------- #
col_voice, col_hint = st.columns([1, 2])
with col_voice:
    if hasattr(st, "audio_input"):
        audio = st.audio_input("Push to talk")
        if audio is not None:
            audio_bytes = audio.getvalue()
            digest = hashlib.md5(audio_bytes).hexdigest()
            if digest != st.session_state.last_audio_hash:
                st.session_state.last_audio_hash = digest
                st.session_state.status = "listening"
                spoken = transcribe(audio_bytes)
                if spoken:
                    st.toast(f"Heard: {spoken}")
                    process(spoken)
                else:
                    st.warning("I couldn't transcribe that. Check the STT backend in Settings.")
    else:
        st.info("Upgrade Streamlit to enable in-browser voice input.")
with col_hint:
    st.caption(
        "Try: *play Believer by Imagine Dragons* \u00b7 *translate good morning to French* "
        "\u00b7 *what's 18% of 4500* \u00b7 *weather in Mumbai* \u00b7 *who is Alan Turing*"
    )

text_in = st.chat_input("Type a message to VINI AI...")
if text_in:
    process(text_in)

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
            audio_bytes = synthesize(result.response.speech, lang=result.nlu.language)
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
