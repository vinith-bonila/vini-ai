"""Analytics dashboard - all charts derived from the SQLite turn log."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database import history
from ui import brand_header, inject_css, sidebar_nav

st.set_page_config(page_title="VINI AI · Analytics", page_icon="\U0001F4CA", layout="wide")
inject_css()
sidebar_nav()
brand_header()

st.markdown(
    "<div class='hero-headline' style='font-size:1.35rem;text-align:left;margin-top:0.8rem'>Analytics</div>"
    "<div class='hero-sub' style='text-align:left;margin:0'>"
    "Real usage derived from every logged turn - nothing here is simulated.</div>",
    unsafe_allow_html=True,
)
st.write("")

_ACCENT = ["#6366F1", "#22D3EE", "#8B5CF6", "#D946EF", "#34D399", "#94A3B8"]
_PAPER = "rgba(0,0,0,0)"
_TEXT = "#94A3B8"


def _themed(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=_PAPER,
        plot_bgcolor=_PAPER,
        font=dict(color=_TEXT, family="Inter, sans-serif"),
        margin=dict(t=30, b=30, l=10, r=10),
        legend=dict(bgcolor=_PAPER),
    )
    fig.update_xaxes(gridcolor="rgba(248,250,252,0.06)", zerolinecolor="rgba(248,250,252,0.06)")
    fig.update_yaxes(gridcolor="rgba(248,250,252,0.06)", zerolinecolor="rgba(248,250,252,0.06)")
    return fig


stats = history.summary_stats()


def _duration(ms: float) -> str:
    """Keep the metric card readable: switch to seconds past 1000 ms."""
    return f"{ms/1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


c = st.columns(5)
c[0].metric("Conversations", stats["total_sessions"])
c[1].metric("Voice/text commands", stats["total_turns"])
c[2].metric("Avg response", _duration(stats["avg_response_ms"]))
c[3].metric("Avg confidence", f"{stats['avg_confidence']*100:.0f}%")
c[4].metric("Success rate", f"{stats['success_rate']:.0f}%")

df = history.load_dataframe()
if df.empty:
    st.info("No data yet. Talk to VINI AI on the Assistant page and come back.")
    st.stop()

st.write("")
left, right = st.columns(2)
with left:
    st.markdown("<div class='vcard-title'>Most common intents</div>", unsafe_allow_html=True)
    counts = df["intent"].value_counts().reset_index()
    counts.columns = ["intent", "count"]
    fig = px.bar(counts, x="count", y="intent", orientation="h", color_discrete_sequence=_ACCENT)
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(_themed(fig), width="stretch")
with right:
    st.markdown("<div class='vcard-title'>Sentiment distribution</div>", unsafe_allow_html=True)
    sent = df["sentiment_label"].value_counts().reset_index()
    sent.columns = ["sentiment", "count"]
    fig = px.pie(sent, names="sentiment", values="count", hole=0.62, color_discrete_sequence=_ACCENT)
    fig.update_traces(marker=dict(line=dict(color="#07080D", width=2)))
    st.plotly_chart(_themed(fig), width="stretch")

left2, right2 = st.columns(2)
with left2:
    st.markdown("<div class='vcard-title'>Language usage</div>", unsafe_allow_html=True)
    lang = df["language"].value_counts().reset_index()
    lang.columns = ["language", "count"]
    fig = px.bar(lang, x="language", y="count", color_discrete_sequence=_ACCENT)
    st.plotly_chart(_themed(fig), width="stretch")
with right2:
    st.markdown("<div class='vcard-title'>Confidence by intent</div>", unsafe_allow_html=True)
    conf = df.groupby("intent")["confidence"].mean().reset_index()
    fig = px.bar(conf, x="intent", y="confidence", color_discrete_sequence=_ACCENT)
    fig.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(_themed(fig), width="stretch")

st.markdown("<div class='vcard-title'>Daily usage trend</div>", unsafe_allow_html=True)
daily = df.set_index("created_at").resample("D").size().reset_index(name="commands")
fig = px.line(daily, x="created_at", y="commands", markers=True, color_discrete_sequence=_ACCENT)
fig.update_traces(line=dict(width=2.5))
st.plotly_chart(_themed(fig), width="stretch")

st.markdown("<div class='vcard-title'>Response time distribution</div>", unsafe_allow_html=True)
fig = px.histogram(df, x="response_time_ms", nbins=30, color_discrete_sequence=_ACCENT)
st.plotly_chart(_themed(fig), width="stretch")

with st.expander("Raw turn log"):
    st.dataframe(
        df[["created_at", "user_text", "intent", "confidence",
            "sentiment_label", "language", "response_time_ms", "success"]],
        hide_index=True, width="stretch",
    )
    if st.button("Clear all analytics data"):
        history.clear_all()
        st.rerun()
