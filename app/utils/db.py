"""
app/utils/db.py — SQLite persistence layer for prediction history.

The database is auto-created at data/predictions.db on first import.

Tables:
    predictions (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp       TEXT,
        student_name    TEXT,
        risk_level      TEXT,
        gpa_prediction  REAL,
        prob_high       REAL,
        prob_medium     REAL,
        prob_low        REAL,
        input_json      TEXT    -- full input dict serialized as JSON
    )
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "predictions.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    student_name    TEXT,
    risk_level      TEXT    NOT NULL,
    gpa_prediction  REAL    NOT NULL,
    prob_high       REAL,
    prob_medium     REAL,
    prob_low        REAL,
    input_json      TEXT
);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the predictions table if it does not exist."""
    with _connect() as conn:
        conn.execute(_CREATE_SQL)
        conn.commit()


def insert_prediction(
    risk_level: str,
    gpa_prediction: float,
    probabilities: dict,
    input_dict: dict,
    student_name: Optional[str] = None,
):
    """Insert one prediction record into the database."""
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO predictions
                (timestamp, student_name, risk_level, gpa_prediction,
                 prob_high, prob_medium, prob_low, input_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                student_name or "Anonymous",
                risk_level,
                round(gpa_prediction, 2),
                round(probabilities.get("High", 0.0), 3),
                round(probabilities.get("Medium", 0.0), 3),
                round(probabilities.get("Low", 0.0), 3),
                json.dumps(input_dict),
            ),
        )
        conn.commit()


def get_all_predictions(limit: int = 500) -> list[dict]:
    """Return all predictions ordered by most recent first."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def clear_predictions():
    """Delete all records from the predictions table."""
    with _connect() as conn:
        conn.execute("DELETE FROM predictions")
        conn.commit()


def prediction_count() -> int:
    init_db()
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
