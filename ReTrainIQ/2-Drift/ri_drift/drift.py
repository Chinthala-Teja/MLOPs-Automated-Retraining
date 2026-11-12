"""drift.py stub for ri_drift"""
from typing import Any, Dict
import pandas as pd

IS_STUB = True

def run_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "module": "drift",
        "status": "STUB",
        "drift_share": 0.0,
        "per_feature": [
            {
                "feature": col,
                "type": "numeric" if col in profile.get("features", {}).get("numeric", []) else "categorical",
                "psi": 0.01,
                "ks_stat": 0.02,
                "p_value": 0.95,
                "flag": False,
            }
            for col in profile.get("features", {}).get("numeric", []) + profile.get("features", {}).get("categorical", [])
        ],
        "n_ref": len(reference_df) if reference_df is not None else 0,
        "n_new": len(new_df) if new_df is not None else 0,
    }

def validate_schema(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {"module": "drift", "status": "STUB", "valid": True, "missing_columns": []}

def fit_reference(reference_df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    return {"module": "drift", "status": "STUB", "fitted": True}

def prediction_shift(ref_pred: Any, new_pred: Any, ref_proba: Any = None, new_proba: Any = None) -> Dict[str, Any]:
    return {"module": "drift", "status": "STUB", "shift_detected": False, "p_value": 1.0}

