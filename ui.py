"""
Reusable Streamlit UI helpers shared by the home page and sub-pages.

Keeps all presentation logic in one place so pages stay declarative. Visual
identity: a dark "AI command center" theme built around a single animated
Voice Intelligence Core, with NLP internals rendered as instrument-style
readouts so the app's real pipeline stays visible, not hidden behind a
generic chat box.

Nothing here fabricates data - every value rendered comes from the actual
`AssistantResult` / `NLUResult` / `config.settings` the backend produced.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

import streamlit as st

from assistant.dispatcher import AssistantResult
from config import settings

_CSS = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")

# --------------------------------------------------------------------------- #
# Voice-core state vocabulary
# --------------------------------------------------------------------------- #
# The five real processing phases a turn passes through. "understanding" and
# "thinking" are both surfaced distinctly even though they happen inside one
# synchronous call, because the UI transitions the core through each phase
# *before* the work for it starts - so the phase shown always matches the
# work actually in flight at that point in the script.
CORE_STATUS = {
    "idle": "READY TO LISTEN",
    "listening": "LISTENING",
    "understanding": "UNDERSTANDING...",
    "thinking": "THINKING...",
    "speaking": "SPEAKING",
}

MIC_STATUS = {
    "idle": ("○", "Tap to speak"),
    "listening": ("●", "Listening..."),
    "understanding": ("◌", "Understanding..."),
    "thinking": ("◌", "Thinking..."),
    "speaking": ("◉", "Speaking..."),
}

# (active step index, indices already "done") for the live 5-node stepper.
_STEP_LABELS = ["VOICE", "TRANSCRIBE", "UNDERSTAND", "REASON", "RESPOND"]
_STEP_STATE = {
    "idle": (-1, []),
    "listening": (0, []),
    "understanding": (2, [0, 1]),
    "thinking": (3, [0, 1, 2]),
    "speaking": (4, [0, 1, 2, 3]),
}


def inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def brand_header() -> None:
    st.markdown(
        "<div class='vini-brand'>VINI<span class='dot'>.</span>AI</div>"
        "<div class='vini-tag'>NLP-FIRST VOICE ASSISTANT</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Hero / "how it works"
# --------------------------------------------------------------------------- #
def hero_headline() -> None:
    """One compact line above the core.

    Deliberately short: the core and the mic have to stay above the fold, so
    the full explanation lives in `about_section()` below the input area.
    """
    st.markdown(
        "<div class='hero-wrap'>"
        "<div class='hero-headline'>Speak naturally. Let AI understand the intent.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def about_section() -> None:
    """The explainer - rendered *below* the interaction area, not above it."""
    st.markdown(
        "<div class='hero-sub'>VINI.AI combines speech recognition, natural-language "
        "understanding, LLM reasoning, and voice synthesis to turn conversations into "
        "intelligent actions and responses.</div>",
        unsafe_allow_html=True,
    )
    steps = ["VOICE / TEXT", "STT", "NLP / INTENT", "LLM", "RESPONSE", "TTS"]
    flow = "<span class='arrow'>&rarr;</span>".join(f"<span class='step'>{s}</span>" for s in steps)
    st.markdown(f"<div class='pipeline-flow'>{flow}</div>", unsafe_allow_html=True)

    with st.expander("What is VINI.AI?"):
        st.write(
            "VINI.AI is an NLP-first voice assistant that combines speech recognition, "
            "natural-language understanding, LLM reasoning, and text-to-speech to process "
            "natural conversations. Every request runs through a real intent classifier, "
            "entity extractor, and sentiment model before a skill or the LLM ever answers - "
            "and that analysis is shown, not hidden, in the Intent Analysis panel."
        )
    st.markdown(
        "<div class='disclaimer-note'>AI responses may be imperfect. "
        "Verify important information before relying on it.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Voice Intelligence Core
# --------------------------------------------------------------------------- #
def _voice_core_html(state: str) -> str:
    state = state if state in CORE_STATUS else "idle"
    status_text = CORE_STATUS[state]
    active_idx, done_idx = _STEP_STATE[state]

    nodes = []
    for i, label in enumerate(_STEP_LABELS):
        cls = "node"
        if i == active_idx:
            cls += " active"
        elif i in done_idx:
            cls += " done"
        nodes.append(f"<span class='{cls}'>{label}</span>")
    steps_html = "<span class='sep'>&rsaquo;</span>".join(nodes)

    wave_bars = "".join("<span></span>" for _ in range(5))

    return f"""
<div class="vc-stage">
  <div class="vc" data-state="{state}">
    <div class="vc-ring vc-ring-1"></div>
    <div class="vc-ring vc-ring-2"></div>
    <div class="vc-ring vc-ring-3"></div>
    <div class="vc-particles">
      <div class="orbit o1"><i></i></div>
      <div class="orbit o2"><i></i></div>
      <div class="orbit o3"><i></i></div>
    </div>
    <div class="vc-orb">
      <div class="vc-wave">{wave_bars}</div>
    </div>
    <div class="vc-label">VINI<span class="dot">.</span>AI</div>
  </div>
  <div class="vc-status">{status_text}</div>
  <div class="vc-steps">{steps_html}</div>
</div>
"""


def render_voice_core(target, state: str) -> None:
    """Render (or update) the Voice Intelligence Core into a placeholder.

    `target` is anything with a `.markdown()` method - typically `st` itself
    or an `st.empty()` placeholder. Passing a placeholder lets the caller
    redraw the core in place as the turn moves through real processing
    phases, instead of only ever showing the final state.
    """
    target.markdown(_voice_core_html(state), unsafe_allow_html=True)


def mic_caption(state: str) -> None:
    glyph, label = MIC_STATUS.get(state, MIC_STATUS["idle"])
    st.markdown(
        f"<div class='mic-caption'><span class='glyph'>{glyph}</span>{label}</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar status
# --------------------------------------------------------------------------- #
def sidebar_nav() -> None:
    """Brand lockup + page navigation, shared by every page.

    The default Streamlit page nav is hidden in CSS, so this is what makes the
    app navigable - it must be rendered on every page.
    """
    with st.sidebar:
        brand_header()
        st.divider()
        st.page_link("app.py", label="Assistant", icon="\U0001F399️")
        st.page_link("pages/1_Analytics.py", label="Analytics", icon="\U0001F4CA")
        st.page_link("pages/2_Settings.py", label="Settings", icon="⚙️")
        st.divider()
        st.caption("SYSTEM STATUS")
        connection_status()


def connection_status() -> None:
    """Real backend connection state - never fabricated."""
    rows = [
        ("LLM", "connected" if settings.openai_enabled else "offline", settings.openai_chat_model if settings.openai_enabled else "structured skills only"),
        ("STT", "connected" if (settings.stt_backend != "openai" or settings.openai_enabled) else "offline", settings.stt_backend),
        ("TTS", "connected", settings.tts_backend),
    ]
    for name, state, detail in rows:
        dot = "on" if state == "connected" else "off"
        st.markdown(
            f"<div class='status-row'><span class='status-dot {dot}'></span>"
            f"<span>{name}</span><span class='val'>&middot; {detail}</span></div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------- #
# Confidence meter
# --------------------------------------------------------------------------- #
def confidence_meter(label: str, value: float) -> None:
    pct = max(0.0, min(1.0, value)) * 100
    st.markdown(
        f"<div class='meter-label'>{label}</div>"
        f"<div class='meter'><span style='width:{pct:.0f}%'></span></div>"
        f"<div class='meter-pct'>{pct:.0f}%</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Conversation bubbles
# --------------------------------------------------------------------------- #
def user_bubble(text: str) -> None:
    st.markdown(
        "<div class='msg-row user'>"
        "<div class='msg-avatar user'>\U0001F3A4</div>"
        "<div class='msg-body'>"
        f"<div class='msg-name'>YOU</div><div class='msg-bubble'>{escape(text)}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


def assistant_bubble(text: str) -> None:
    st.markdown(
        "<div class='msg-row assistant'>"
        "<div class='msg-avatar assistant'>✨</div>"
        "<div class='msg-body'>"
        f"<div class='msg-name'>VINI.AI</div><div class='msg-bubble'>{escape(text)}</div>"
        "</div></div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# NLP transparency panels
# --------------------------------------------------------------------------- #
def render_nlu_panels(result: AssistantResult) -> None:
    """The transparency panels - every NLP result in expandable readouts."""
    nlu = result.nlu

    stats = [
        ("CONFIDENCE", f"{nlu.confidence*100:.0f}%"),
        ("SENTIMENT", f"{nlu.sentiment_label.title()} {nlu.sentiment_score:+.2f}"),
        ("LATENCY", f"{result.response_time_ms:.0f} ms"),
    ]
    stat_html = "".join(
        f"<div class='stat'><div class='stat-label'>{label}</div>"
        f"<div class='stat-value'>{value}</div></div>"
        for label, value in stats
    )
    chips = "".join(
        f"<span class='chip'>{escape(e.text)} &middot; {escape(e.label)}</span>" for e in nlu.entities
    )
    chips_html = f"<div class='chip-row'>{chips}</div>" if chips else ""

    st.markdown(
        "<div class='vcard'>"
        f"<div class='intent-head'><span class='pill pill-intent'>{escape(nlu.intent)}</span>"
        f"<span class='intent-engine'>engine: {escape(nlu.engine)}</span></div>"
        f"<div class='stat-grid'>{stat_html}</div>"
        f"{chips_html}"
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Intent classification"):
        confidence_meter("Top intent confidence", nlu.confidence)
        st.dataframe(
            {"intent": [s.intent for s in nlu.intent_ranking],
             "score": [s.score for s in nlu.intent_ranking]},
            hide_index=True, width="stretch",
        )

    with st.expander("Entities (NER + slots)"):
        if nlu.entities:
            st.dataframe(
                {"text": [e.text for e in nlu.entities],
                 "label": [e.label for e in nlu.entities]},
                hide_index=True, width="stretch",
            )
        else:
            st.caption("No entities detected.")

    with st.expander("Language & sentiment"):
        c1, c2 = st.columns(2)
        c1.write(f"**Language:** `{nlu.language}` ({nlu.language_confidence*100:.0f}% conf.)")
        c2.write(f"**Sentiment:** `{nlu.sentiment_label}` ({nlu.sentiment_score:+.2f})")

    with st.expander("POS tags & dependency parse"):
        if nlu.tokens:
            st.dataframe(
                {"token": [t.text for t in nlu.tokens],
                 "lemma": [t.lemma for t in nlu.tokens],
                 "POS": [t.pos for t in nlu.tokens],
                 "tag": [t.tag for t in nlu.tokens],
                 "dep": [t.dep for t in nlu.tokens],
                 "head": [t.head for t in nlu.tokens]},
                hide_index=True, width="stretch",
            )
        else:
            st.caption("Parsing skipped.")
