"""store.py
SQLite database wrapper for ReTrainIQ.
DB file default at WORKSPACE / 'retrainiq.db'
"""
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

def get_reports(project_id: str, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    init_db(db_path)
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE project_id = ? ORDER BY id ASC", (project_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def add_decision(report_id: int, action: str, by: str, note: str, db_path: Optional[Path] = None) -> int:
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
        cursor = conn.execute(
            "INSERT INTO model_versions (project_id, version, path, trained_on, status) VALUES (?, ?, ?, ?, ?)",
            (project_id, version, path, trained_on, status)
        )
        v_id = cursor.lastrowid
    conn.close()
    return v_id

def set_active_version(project_id: str, version: str, db_path: Optional[Path] = None) -> None:
    init_db(db_path)
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE model_versions SET status = 'inactive' WHERE project_id = ? AND status = 'active'",
            (project_id,)
        )
        conn.execute(
            "UPDATE model_versions SET status = 'active' WHERE project_id = ? AND version = ?",
            (project_id, version)
        )
    conn.close()

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
