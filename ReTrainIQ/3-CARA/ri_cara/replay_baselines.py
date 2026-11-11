"""replay_baselines.py stub for ri_cara"""
from typing import Any, Dict, List
import pandas as pd

IS_STUB = True

def run_baselines(batches: List[pd.DataFrame], calib: Dict[str, Any], profile: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([
        {"batch_id": i, "policy": "cara_t", "retrain": False, "cost": 0.0}
        for i in range(len(batches))
    ])

