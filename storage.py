"""
SQLite-backed storage for classification results.

One flat table, deliberately not normalized -- see design note Section 4
for the reasoning. Uses stdlib sqlite3 directly rather than an ORM: the
schema is small and stable enough that an ORM would add indirection without
buying much, and it keeps the dependency list short for an offline deploy.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS classification_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tile_id TEXT NOT NULL UNIQUE,
    source_filename TEXT NOT NULL,
    predicted_label TEXT NOT NULL,
    confidence REAL NOT NULL,
    class_probabilities TEXT NOT NULL,  -- JSON blob: {"Forest": 0.87, ...}
    model_version TEXT NOT NULL,
    review_flag INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predicted_label ON classification_results(predicted_label);
CREATE INDEX IF NOT EXISTS idx_confidence ON classification_results(confidence);
CREATE INDEX IF NOT EXISTS idx_created_at ON classification_results(created_at);
"""

# Below this confidence, a result is auto-flagged for analyst review.
# Tunable; see design note Section 5 on why this lives at storage/query
# time rather than being baked into the classifier itself.
REVIEW_THRESHOLD = 0.6


@contextmanager
def get_connection(db_path: str):
    # A new connection per call is simple and safe for SQLite, but under
    # concurrent requests (multiple in-flight /classify calls each writing
    # a row) the default behavior is to fail fast with "database is locked"
    # rather than wait. WAL mode lets readers and a writer coexist without
    # blocking each other, and busy_timeout makes a writer wait (up to 5s)
    # instead of erroring immediately if another write is briefly in flight.
    conn = sqlite3.connect(db_path, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def insert_result(
    db_path: str,
    tile_id: str,
    source_filename: str,
    predicted_label: str,
    confidence: float,
    class_probabilities: Dict[str, float],
    model_version: str,
) -> Dict[str, Any]:
    review_flag = int(confidence < REVIEW_THRESHOLD)
    created_at = datetime.now(timezone.utc).isoformat()

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO classification_results
                (tile_id, source_filename, predicted_label, confidence,
                 class_probabilities, model_version, review_flag, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(tile_id) DO UPDATE SET
                source_filename=excluded.source_filename,
                predicted_label=excluded.predicted_label,
                confidence=excluded.confidence,
                class_probabilities=excluded.class_probabilities,
                model_version=excluded.model_version,
                review_flag=excluded.review_flag,
                created_at=excluded.created_at
            """,
            (
                tile_id,
                source_filename,
                predicted_label,
                confidence,
                json.dumps(class_probabilities),
                model_version,
                review_flag,
                created_at,
            ),
        )
        conn.commit()

    return get_result(db_path, tile_id)


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["class_probabilities"] = json.loads(d["class_probabilities"])
    d["review_flag"] = bool(d["review_flag"])
    return d


def get_result(db_path: str, tile_id: str) -> Optional[Dict[str, Any]]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM classification_results WHERE tile_id = ?", (tile_id,)
        ).fetchone()
    return _row_to_dict(row) if row else None


def list_results(
    db_path: str,
    label: Optional[str] = None,
    min_confidence: Optional[float] = None,
    max_confidence: Optional[float] = None,
    model_version: Optional[str] = None,
    review_flag: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Filtered listing -- the "analyst queries results" path from the design
    note (Section 6, point 2). Point lookup is get_result(); aggregate/
    monitoring queries (point 3) are intentionally not implemented here --
    stubbed per the design note, since they're a monitoring concern rather
    than core to the classify-and-store slice.
    """
    clauses, params = [], []

    if label is not None:
        clauses.append("predicted_label = ?")
        params.append(label)
    if min_confidence is not None:
        clauses.append("confidence >= ?")
        params.append(min_confidence)
    if max_confidence is not None:
        clauses.append("confidence <= ?")
        params.append(max_confidence)
    if model_version is not None:
        clauses.append("model_version = ?")
        params.append(model_version)
    if review_flag is not None:
        clauses.append("review_flag = ?")
        params.append(int(review_flag))

    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    query = f"""
        SELECT * FROM classification_results
        {where}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """
    params.extend([limit, offset])

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(r) for r in rows]


# --- Aggregate helpers, used by the dashboard (not the API) ---
#
# The design note (Section 6) deliberately leaves aggregate/monitoring
# queries out of the serving API's scope -- these exist for the analyst
# dashboard, which reads this SQLite file directly rather than round-
# tripping through HTTP, so adding them here doesn't change that.


def get_label_counts(db_path: str) -> Dict[str, int]:
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT predicted_label, COUNT(*) as n FROM classification_results GROUP BY predicted_label"
        ).fetchall()
    return {r["predicted_label"]: r["n"] for r in rows}


def get_distinct_model_versions(db_path: str) -> List[str]:
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT model_version FROM classification_results").fetchall()
    return [r["model_version"] for r in rows]


def get_summary_stats(db_path: str) -> Dict[str, Any]:
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) as total,
                   AVG(confidence) as avg_confidence,
                   SUM(review_flag) as flagged_count
            FROM classification_results
            """
        ).fetchone()
    return {
        "total": row["total"] or 0,
        "avg_confidence": round(row["avg_confidence"], 4) if row["avg_confidence"] is not None else None,
        "flagged_count": row["flagged_count"] or 0,
    }
