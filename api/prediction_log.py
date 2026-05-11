"""
SQLite-backed prediction logger.

Every call to /predict writes one row to prediction_log.db. The drift
detector and Streamlit dashboard both read from this file.

Schema:
    predictions (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id    TEXT NOT NULL,
        timestamp_iso TEXT NOT NULL,
        observation   TEXT NOT NULL,   -- JSON array of floats
        action        INTEGER NOT NULL,
        action_kind   TEXT NOT NULL,
        model_name    TEXT NOT NULL,
        model_version TEXT NOT NULL,
        latency_ms    REAL NOT NULL
    )
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

# Default DB path; override with FOOD_RESCUE_LOG_DB env var
_DEFAULT_DB = Path("experiments/prediction_log.db")


def _get_db_path() -> Path:
    return Path(os.environ.get("FOOD_RESCUE_LOG_DB", str(_DEFAULT_DB)))


def _get_conn() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id    TEXT NOT NULL,
            timestamp_iso TEXT NOT NULL,
            observation   TEXT NOT NULL,
            action        INTEGER NOT NULL,
            action_kind   TEXT NOT NULL,
            model_name    TEXT NOT NULL,
            model_version TEXT NOT NULL,
            latency_ms    REAL NOT NULL
        )
    """)
    conn.commit()


def log_prediction(request, response, latency_ms: float) -> None:
    """Write one prediction row to the SQLite log."""
    conn = _get_conn()
    try:
        _ensure_table(conn)
        conn.execute(
            """
            INSERT INTO predictions
                (request_id, timestamp_iso, observation, action,
                 action_kind, model_name, model_version, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                response.request_id,
                response.timestamp_iso,
                json.dumps(request.observation),
                response.action,
                response.action_kind,
                response.model_name,
                response.model_version,
                round(latency_ms, 3),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_recent(n: int = 500) -> list[dict]:
    """Return the n most recent prediction rows as dicts (newest first)."""
    db_path = _get_db_path()
    if not db_path.exists():
        return []
    conn = _get_conn()
    try:
        _ensure_table(conn)
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def fetch_observations(n: int = 500) -> list[list[float]]:
    """Return raw observation vectors from the n most recent predictions."""
    rows = fetch_recent(n)
    return [json.loads(r["observation"]) for r in rows]
