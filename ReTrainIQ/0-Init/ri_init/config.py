from __future__ import annotations

from pathlib import Path

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
