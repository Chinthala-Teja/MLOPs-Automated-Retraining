"""retrain.py stub for ri_decision"""
from typing import Any, Dict
from pathlib import Path
import pandas as pd

IS_STUB = True

def train_candidate(profile: Dict[str, Any], train_df: pd.DataFrame, out_path: Path) -> Dict[str, Any]:
    return {
        "status": "STUB",
        "candidate_path": str(out_path),
        "trained_rows": len(train_df) if train_df is not None else 0,
    }

def verify(candidate_path: Path, holdout_df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "STUB",
        "verified": True,
        "metric_score": 0.92,
    }

def promote(candidate_path: Path, active_path: Path) -> Dict[str, Any]:
    return {
        "status": "STUB",
        "promoted": True,
        "active_path": str(active_path),
    }

def rollback(previous_path: Path, active_path: Path) -> Dict[str, Any]:
    return {
        "status": "STUB",
        "rolled_back": True,
        "active_path": str(active_path),
    }

