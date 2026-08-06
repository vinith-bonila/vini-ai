"""
Reusable Streamlit UI helpers shared by the home page and sub-pages.

Keeps all presentation logic in one place so pages stay declarative. The visual
identity is deliberately restrained: one violet->teal accent, machine-readout
mono type for the NLP internals (the app's signature - you can watch it think).
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from assistant.dispatcher import AssistantResult
from config import settings

_CSS = (Path(__file__).parent / "assets" / "style.css").read_text(encoding="utf-8")


def inject_css() -> None:
    st.markdown(f"<style>{_CSS}</style>", unsafe_allow_html=True)


def brand_header() -> None:
    st.markdown(
        f"<div class='vini-brand'>VINI<span class='dot'>.</span>AI</div>"
        f"<div class='vini-tag'>{settings.tagline} \u00b7 by {settings.author}</div>",
        unsafe_allow_html=True,
    )


def status_pill(state: str) -> str:
    labels = {
        "idle": ("pill-idle", "\u25CB IDLE"),
        "listening": ("pill-live", "\u25CF LISTENING"),
        "thinking": ("pill-intent", "\u25D0 THINKING"),
        "speaking": ("pill-live", "\u25B6 SPEAKING"),
    }
    cls, text = labels.get(state, labels["idle"])
    return f"<span class='pill {cls}'>{text}</span>"


def confidence_meter(label: str, value: float) -> None:
    pct = max(0.0, min(1.0, value)) * 100
    st.markdown(
        f"<div style='font-size:0.8rem;color:#8A93AB'>{label}</div>"
        f"<div class='meter'><span style='width:{pct:.0f}%'></span></div>"
        f"<div style='text-align:right;font-family:JetBrains Mono;font-size:0.8rem'>{pct:.0f}%</div>",
        unsafe_allow_html=True,
    )


def render_nlu_panels(result: AssistantResult) -> None:
    """The transparency panels - every NLP result in expandable readouts."""
    nlu = result.nlu

    top = st.columns([1.4, 1, 1, 1])
    with top[0]:
        st.markdown(
            f"<span class='pill pill-intent'>{nlu.intent}</span>", unsafe_allow_html=True
        )
        st.caption(f"engine: {nlu.engine}")
    top[1].metric("Confidence", f"{nlu.confidence*100:.0f}%")
    top[2].metric("Sentiment", nlu.sentiment_label.title(), f"{nlu.sentiment_score:+.2f}")
    top[3].metric("Latency", f"{result.response_time_ms:.0f} ms")

    with st.expander("Intent classification"):
        confidence_meter("Top intent confidence", nlu.confidence)
        st.dataframe(
            {"intent": [s.intent for s in nlu.intent_ranking],
             "score": [s.score for s in nlu.intent_ranking]},
            hide_index=True, width='stretch',
        )

    with st.expander("Entities (NER + slots)"):
        if nlu.entities:
            st.dataframe(
                {"text": [e.text for e in nlu.entities],
                 "label": [e.label for e in nlu.entities]},
                hide_index=True, width='stretch',
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
                hide_index=True, width='stretch',
            )
        else:
            st.caption("Parsing skipped.")
