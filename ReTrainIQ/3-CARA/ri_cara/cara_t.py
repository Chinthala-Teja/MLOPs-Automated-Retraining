"""cara_t.py stub for ri_cara"""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd

IS_STUB = True

def calibrate(model_path: str, history_df: pd.DataFrame, profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module": "cara_t",
        "status": "STUB",
        "staleness": 0.05,
        "tau_star": 0.10,
        "kappa": 0.8,
        "s": 0.05,
        "decision": "KEEP",
        "n_queries": 100,
        "n_points": len(history_df) if history_df is not None else 0,
        "basis": "measured",
        "staleness_est": None,
        "staleness_est_lo": None,
        "decision_est": None,
        "proxy": {
            "conf_cal": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "map": {"x": [0.0, 1.0], "y": [0.0, 1.0], "resid_q90": 0.05},
            "spearman": 0.85,
        },
    }

def online_staleness(model_path: str, new_df: pd.DataFrame, train_sample_df: pd.DataFrame, calib: Dict[str, Any], profile: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    return {
        "module": "cara_t",
        "status": "STUB",
        "staleness": 0.04,
        "tau_star": 0.10,
        "kappa": 0.8,
        "s": 0.04,
        "decision": "KEEP",
        "n_queries": 50,
        "n_points": len(new_df) if new_df is not None else 0,
        "basis": "measured",
        "staleness_est": None,
        "staleness_est_lo": None,
        "decision_est": None,
    }

def estimate_staleness(model_path: str, new_df: pd.DataFrame, train_sample_df: pd.DataFrame, calib: Dict[str, Any], profile: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    return {
        "module": "cara_t",
        "status": "ESTIMATED",
        "staleness": 0.04,
        "tau_star": 0.10,
        "kappa": 0.8,
        "s": 0.04,
        "decision": "KEEP",
        "n_queries": 0,
        "n_points": len(new_df) if new_df is not None else 0,
        "basis": "estimated",
        "staleness_est": 0.045,
        "staleness_est_lo": 0.035,
        "decision_est": "KEEP",
    }
