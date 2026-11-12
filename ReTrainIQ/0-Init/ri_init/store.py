"""store.py
SQLite database wrapper for ReTrainIQ.
DB file default at WORKSPACE / 'retrainiq.db'
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from ri_init.config import WORKSPACE

DB_PATH = WORKSPACE / "retrainiq.db"

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    target = db_path or DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(target))
    conn.row_factory = sqlite3.Row
    return conn

def init_db(db_path: Optional[Path] = None) -> None:
    conn = get_connection(db_path)
    with conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            source TEXT,
            profile_json TEXT,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS model_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT,
            version TEXT,
            path TEXT,
            trained_on TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT
        );

        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT,
            file TEXT,
            n_rows INTEGER,
            n_labelled INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT,
            batch_id INTEGER,
            drift_json TEXT,
            cara_json TEXT,
            quality_json TEXT,
            score_json TEXT,
            zone TEXT,
            summary TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id INTEGER,
            action TEXT,
            by TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
    conn.close()

def add_project(project_id: str, source: str, profile_json: str, status: str = "active", db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO projects (id, source, profile_json, status) VALUES (?, ?, ?, ?)",
            (project_id, source, profile_json, status)
        )
    conn.close()

def get_project(project_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM projects WHERE id = ?", (project_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def add_batch(project_id: str, file: str, n_rows: int, n_labelled: int, db_path: Optional[Path] = None) -> int:
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        cursor = conn.execute(
            "INSERT INTO batches (project_id, file, n_rows, n_labelled) VALUES (?, ?, ?, ?)",
            (project_id, file, n_rows, n_labelled)
        )
        batch_id = cursor.lastrowid
    conn.close()
    return batch_id

def get_batch(batch_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM batches WHERE id = ?", (batch_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_batches(project_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return a project's batches in insertion order."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM batches WHERE project_id = ? ORDER BY id ASC", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def update_batch(
    batch_id: int,
    n_rows: int,
    n_labelled: int,
    db_path: Optional[Path] = None,
) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            updated = conn.execute(
                "UPDATE batches SET n_rows = ?, n_labelled = ? WHERE id = ?",
                (n_rows, n_labelled, batch_id),
            )
            if updated.rowcount == 0:
                raise ValueError(f"Batch {batch_id} was not found.")
    finally:
        conn.close()

def add_report(project_id: str, batch_id: int, drift_json: str, cara_json: str, quality_json: str, score_json: str, zone: str, summary: str, db_path: Optional[Path] = None) -> int:
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        cursor = conn.execute(
            """INSERT INTO reports (project_id, batch_id, drift_json, cara_json, quality_json, score_json, zone, summary)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (project_id, batch_id, drift_json, cara_json, quality_json, score_json, zone, summary)
        )
        report_id = cursor.lastrowid
    conn.close()
    return report_id

def update_report(report_id: int, db_path: Optional[Path] = None, **fields: Any) -> None:
    if not fields:
        return
    init_db(db_path)
    conn = get_connection(db_path)
    set_clauses = [f"{k} = ?" for k in fields.keys()]
    values = list(fields.values()) + [report_id]
    query = f"UPDATE reports SET {', '.join(set_clauses)} WHERE id = ?"
    with conn:
        conn.execute(query, values)
    conn.close()

def get_report(report_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_reports(project_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE project_id = ? ORDER BY id ASC", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_decision(report_id: Optional[int], action: str, by: str, note: str, db_path: Optional[Path] = None) -> int:
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        cursor = conn.execute(
            "INSERT INTO decisions (report_id, action, by, note) VALUES (?, ?, ?, ?)",
            (report_id, action, by, note)
        )
        decision_id = cursor.lastrowid
    conn.close()
    return decision_id

def add_version(project_id: str, version: str, path: str, trained_on: str, status: str = "candidate", db_path: Optional[Path] = None) -> int:
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        existing = conn.execute(
            "SELECT id FROM model_versions WHERE project_id = ? AND version = ? ORDER BY id DESC LIMIT 1",
            (project_id, version),
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE model_versions SET path = ?, trained_on = ?, status = ? WHERE id = ?",
                (path, trained_on, status, existing["id"]),
            )
            v_id = existing["id"]
        else:
            cursor = conn.execute(
                "INSERT INTO model_versions (project_id, version, path, trained_on, status) VALUES (?, ?, ?, ?, ?)",
                (project_id, version, path, trained_on, status),
            )
            v_id = cursor.lastrowid
    conn.close()
    return v_id

def set_active_version(project_id: str, version: str, db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            target = conn.execute(
                """SELECT id FROM model_versions WHERE project_id = ? AND version = ?
                   ORDER BY id DESC LIMIT 1""",
                (project_id, version),
            ).fetchone()
            if not target:
                raise ValueError(
                    f"Model version {version!r} is not registered for project {project_id!r}"
                )
            conn.execute(
                "UPDATE model_versions SET status = 'previous' WHERE project_id = ? AND status = 'active'",
                (project_id,)
            )
            conn.execute(
                "UPDATE model_versions SET status = 'active' WHERE id = ?",
                (target["id"],)
            )
    finally:
        conn.close()

def get_versions(project_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return the latest registration for each model version in registration order."""
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """SELECT version_row.* FROM model_versions AS version_row
           WHERE version_row.project_id = ?
             AND version_row.id = (
                 SELECT MAX(latest.id) FROM model_versions AS latest
                 WHERE latest.project_id = version_row.project_id
                   AND latest.version = version_row.version
             )
           ORDER BY version_row.id ASC""",
        (project_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_active_version(project_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM model_versions WHERE project_id = ? AND status = 'active' ORDER BY id DESC LIMIT 1",
        (project_id,)
    )
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_decisions(project_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """All decisions of a project (joined through its reports), oldest first."""
    init_db(db_path)
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT d.* FROM decisions d JOIN reports r ON r.id = d.report_id "
        "WHERE r.project_id = ? ORDER BY d.id ASC",
        (project_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_llm_log(model: str, system: str, user: str, response: str, db_path: Optional[Path] = None) -> int:
    """Record one LLM call (model, prompt and response) for auditing."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_log (id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT, "
            "system TEXT, user TEXT, response TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        cursor = conn.execute(
            "INSERT INTO llm_log (model, system, user, response) VALUES (?, ?, ?, ?)",
            (model, system, user, response),
        )
        log_id = cursor.lastrowid
    conn.close()
    return int(log_id)


def delete_project(project_id: str, db_path: Optional[Path] = None) -> None:
    """Remove every stored row of a project (used by reset and demo mode)."""
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "DELETE FROM decisions WHERE report_id IN (SELECT id FROM reports WHERE project_id = ?)",
            (project_id,),
        )
        conn.execute("DELETE FROM decisions WHERE report_id IS NULL")
        for table in ("reports", "batches", "model_versions"):
            conn.execute(f"DELETE FROM {table} WHERE project_id = ?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    conn.close()
