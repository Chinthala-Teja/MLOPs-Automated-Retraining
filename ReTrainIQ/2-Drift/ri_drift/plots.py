"""plots.py stub for ri_drift"""
from typing import Any, Dict, List
import pandas as pd

IS_STUB = True

def drift_bar(result: Dict[str, Any]) -> Any:
    return {"status": "STUB", "plot": "drift_bar"}

def distribution_overlay(reference_df: pd.DataFrame, new_df: pd.DataFrame, feature: str, ftype: str) -> Any:
    return {"status": "STUB", "plot": "distribution_overlay", "feature": feature, "ftype": ftype}

def top_drifted(result: Dict[str, Any], k: int = 3) -> List[Dict[str, Any]]:
    per_feat = result.get("per_feature", [])
    return per_feat[:k]

