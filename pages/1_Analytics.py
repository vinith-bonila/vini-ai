"""Analytics dashboard - all charts derived from the SQLite turn log."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from database import history
from ui import brand_header, inject_css

st.set_page_config(page_title="VINI AI \u00b7 Analytics", page_icon="\U0001F4CA", layout="wide")
inject_css()
brand_header()
st.markdown("## Analytics")

df = history.load_dataframe()
stats = history.summary_stats()

c = st.columns(5)
c[0].metric("Conversations", stats["total_sessions"])
c[1].metric("Voice/text commands", stats["total_turns"])
c[2].metric("Avg response", f"{stats['avg_response_ms']:.0f} ms")
c[3].metric("Avg confidence", f"{stats['avg_confidence']*100:.0f}%")
c[4].metric("Success rate", f"{stats['success_rate']:.0f}%")

if df.empty:
    st.info("No data yet. Talk to VINI AI on the Home page and come back.")
    st.stop()

_ACCENT = ["#7C5CFF", "#38E1C6", "#B7A6FF", "#F2C14E", "#EF6F6C", "#6C8AE4"]

left, right = st.columns(2)
with left:
    st.markdown("#### Most common intents")
    counts = df["intent"].value_counts().reset_index()
    counts.columns = ["intent", "count"]
    st.plotly_chart(
        px.bar(counts, x="count", y="intent", orientation="h",
               color_discrete_sequence=_ACCENT).update_layout(yaxis={"categoryorder": "total ascending"}),
        width='stretch',
    )
with right:
    st.markdown("#### Sentiment distribution")
    sent = df["sentiment_label"].value_counts().reset_index()
    sent.columns = ["sentiment", "count"]
    st.plotly_chart(
        px.pie(sent, names="sentiment", values="count", hole=0.5,
               color_discrete_sequence=_ACCENT),
        width='stretch',
    )

left2, right2 = st.columns(2)
with left2:
    st.markdown("#### Language usage")
    lang = df["language"].value_counts().reset_index()
    lang.columns = ["language", "count"]
    st.plotly_chart(px.bar(lang, x="language", y="count",
                           color_discrete_sequence=_ACCENT), width='stretch')
with right2:
    st.markdown("#### Confidence by intent")
    conf = df.groupby("intent")["confidence"].mean().reset_index()
    st.plotly_chart(px.bar(conf, x="intent", y="confidence",
                           color_discrete_sequence=_ACCENT).update_layout(xaxis_tickangle=-40),
                    width='stretch')

st.markdown("#### Daily usage trend")
daily = df.set_index("created_at").resample("D").size().reset_index(name="commands")
st.plotly_chart(px.line(daily, x="created_at", y="commands", markers=True,
                        color_discrete_sequence=_ACCENT), width='stretch')

st.markdown("#### Response time distribution")
st.plotly_chart(px.histogram(df, x="response_time_ms", nbins=30,
                             color_discrete_sequence=_ACCENT), width='stretch')

with st.expander("Raw turn log"):
    st.dataframe(df[["created_at", "user_text", "intent", "confidence",
                     "sentiment_label", "language", "response_time_ms", "success"]],
                 hide_index=True, width='stretch')
    if st.button("Clear all analytics data"):
        history.clear_all()
        st.rerun()
