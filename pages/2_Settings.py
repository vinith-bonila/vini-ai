"""Settings page - runtime configuration surfaced from config.py.

Values entered here are applied to the running session via environment
variables. For permanent changes, edit your .env file (see .env.example).
"""
from __future__ import annotations

import os

import streamlit as st

from config import settings
from ui import brand_header, connection_status, inject_css, sidebar_nav

st.set_page_config(page_title="VINI AI · Settings", page_icon="⚙️", layout="wide")
inject_css()
sidebar_nav()
brand_header()

st.markdown(
    "<div class='hero-headline' style='font-size:1.35rem;text-align:left;margin-top:0.8rem'>Settings</div>"
    "<div class='hero-sub' style='text-align:left;margin:0'>"
    "Session-level overrides. Persist changes permanently in your .env file.</div>",
    unsafe_allow_html=True,
)
st.write("")

with st.form("settings"):
    st.markdown("<div class='vcard-title'>VOICE</div>", unsafe_allow_html=True)
    col = st.columns(3)
    stt = col[0].selectbox(
        "Speech-to-text", ["openai", "local"],
        index=["openai", "local"].index(settings.stt_backend),
    )
    tts = col[1].selectbox(
        "Text-to-speech", ["gtts", "openai", "pyttsx3"],
        index=["gtts", "openai", "pyttsx3"].index(settings.tts_backend),
    )
    lang = col[2].text_input("Default language", value=settings.default_tts_language)
    speed = st.slider("Voice speed", 0.5, 2.0, settings.voice_speed, 0.1)

    st.divider()
    st.markdown("<div class='vcard-title'>AI</div>", unsafe_allow_html=True)
    api_key = st.text_input(
        "OpenAI API key", value="", type="password",
        placeholder="sk-... (leave blank to keep current)",
    )
    chat_model = st.text_input("Chat model", value=settings.openai_chat_model)
    col2 = st.columns(2)
    use_emb = col2[0].toggle("Use embedding intent model", value=settings.use_embeddings)
    use_hf = col2[1].toggle("Use HF sentiment model", value=settings.use_hf_sentiment)
    floor = st.slider("Intent confidence floor", 0.0, 0.6, settings.intent_confidence_floor, 0.02)

    st.divider()
    st.markdown("<div class='vcard-title'>SYSTEM</div>", unsafe_allow_html=True)
    log_level = st.selectbox(
        "Logging level", ["DEBUG", "INFO", "WARNING", "ERROR"],
        index=["DEBUG", "INFO", "WARNING", "ERROR"].index(settings.log_level.upper()),
    )

    if st.form_submit_button("Apply for this session", width="stretch"):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_CHAT_MODEL"] = chat_model
        os.environ["STT_BACKEND"] = stt
        os.environ["TTS_BACKEND"] = tts
        os.environ["DEFAULT_TTS_LANGUAGE"] = lang
        os.environ["VOICE_SPEED"] = str(speed)
        os.environ["USE_EMBEDDINGS"] = "true" if use_emb else "false"
        os.environ["USE_HF_SENTIMENT"] = "true" if use_hf else "false"
        os.environ["INTENT_CONFIDENCE_FLOOR"] = str(floor)
        os.environ["LOG_LEVEL"] = log_level
        st.success("Applied. Some model changes take effect on the next app restart.")
        st.info("Note: config.py reads env at import; restart the app to fully re-resolve models.")

st.write("")
st.markdown("<div class='vcard-title'>SYSTEM STATUS</div>", unsafe_allow_html=True)
st.markdown("<div class='vcard'>", unsafe_allow_html=True)
connection_status()
st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.markdown("<div class='vcard-title'>SESSION</div>", unsafe_allow_html=True)
if st.button("Clear conversation"):
    if st.session_state.get("conversation") is not None:
        st.session_state.conversation.clear()
        st.session_state.last_result = None
        st.session_state["_pending_audio"] = None
        st.success("Conversation cleared. Return to the Assistant page to see it.")
    else:
        st.info("No active conversation yet - visit the Assistant page first.")
