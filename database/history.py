"""
SQLite persistence for conversation turns plus analytics aggregates.

One row per turn. All analytics on the dashboard are derived from this table,
so the app has a real data layer rather than in-memory-only state.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

import pandas as pd

from config import settings
from nlp.schemas import NLUResult
from utils.helpers import safe_json
from utils.logger import get_logger

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       TEXT    NOT NULL,
    created_at       TEXT    NOT NULL,
    user_text        TEXT    NOT NULL,
    intent           TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    entities         TEXT,
    sentiment_label  TEXT,
    sentiment_score  REAL,
    language         TEXT,
    engine           TEXT,
    response_text    TEXT,
    response_time_ms REAL,
    success          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_turns_created ON turns(created_at);
CREATE INDEX IF NOT EXISTS idx_turns_intent  ON turns(intent);
"""


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _conn() as conn:
        conn.executescript(_SCHEMA)


def record(
    *,
    session_id: str,
    user_text: str,
    nlu: NLUResult,
    response_text: str,
    response_time_ms: float,
    success: bool,
) -> None:
    entities = safe_json([{"text": e.text, "label": e.label} for e in nlu.entities])
    with _conn() as conn:
        conn.execute(
            """INSERT INTO turns
               (session_id, created_at, user_text, intent, confidence, entities,
                sentiment_label, sentiment_score, language, engine,
                response_text, response_time_ms, success)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                session_id, datetime.utcnow().isoformat(timespec="seconds"), user_text,
                nlu.intent, nlu.confidence, entities, nlu.sentiment_label,
                nlu.sentiment_score, nlu.language, nlu.engine, response_text,
                response_time_ms, int(success),
            ),
        )


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def load_dataframe() -> pd.DataFrame:
    with _conn() as conn:
        df = pd.read_sql_query("SELECT * FROM turns ORDER BY created_at DESC", conn)
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def summary_stats() -> dict:
    df = load_dataframe()
    if df.empty:
        return {
            "total_turns": 0, "total_sessions": 0, "avg_response_ms": 0.0,
            "avg_confidence": 0.0, "success_rate": 0.0,
        }
    return {
        "total_turns": int(len(df)),
        "total_sessions": int(df["session_id"].nunique()),
        "avg_response_ms": round(float(df["response_time_ms"].mean()), 1),
        "avg_confidence": round(float(df["confidence"].mean()), 3),
        "success_rate": round(float(df["success"].mean()) * 100, 1),
    }


def clear_all() -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM turns")


# Ensure the table exists on import.
init_db()
