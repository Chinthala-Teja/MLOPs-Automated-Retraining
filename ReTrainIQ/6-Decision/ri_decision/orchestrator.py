"""orchestrator.py stub for ri_decision"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path

IS_STUB = True

def setup_project(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "STUB",
        "project_id": profile.get("project_id", "demo"),
        "initialized": True,
    }

def on_new_data(project_id: str, file_path: Path, labels_path: Optional[Path] = None) -> Dict[str, Any]:
    if labels_path is None:
        return {
            "status": "STUB",
            "project_id": project_id,
            "processed_file": str(file_path),
            "zone": "HUMAN",
            "alert": "request labels",
            "proxy_error": None,
        }
    return {
        "status": "STUB",
        "project_id": project_id,
        "processed_file": str(file_path),
        "zone": "KEEP",
        "proxy_error": 0.005,
    }
