"""proxy_eval.py stub for ri_cara"""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd

IS_STUB = True

def run_proxy_eval(profile: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"batch_id": 0, "spearman": 0.85, "mae": 0.02, "status": "STUB"}
    ])

