from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT / "workspace"

# Default decision settings
DEFAULT_CONFIG = {
    "b": 0.0,
    "a1": 1.5,
    "a2": 1.0,
    "a3": 0.5,
    "retrain_limit": 0.65,
    "keep_limit": 0.35,
    "z_clip": 3.0,
    "s_floor": 1e-6,
    "auto_rollback": False,
}


def load_config(project_id: str | None = None) -> Dict[str, Any]:
    """DEFAULT_CONFIG overlaid with workspace/<project_id>/config.json (e.g. fitted coefficients)."""
    cfg: Dict[str, Any] = dict(DEFAULT_CONFIG)
    if project_id:
        path = WORKSPACE / project_id / "config.json"
        if path.is_file():
            cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg
