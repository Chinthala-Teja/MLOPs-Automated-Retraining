"""metrics.py stub for ri_metrics"""
from typing import Any, Dict, List
import pandas as pd

IS_STUB = True

REGISTRY: Dict[str, Any] = {
    "accuracy": "Accuracy score",
    "recall": "Recall score",
    "precision": "Precision score",
    "f1": "F1 score",
    "roc_auc": "ROC AUC score",
    "mse": "Mean Squared Error",
}

def default_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": "STUB",
        "primary_metric": profile.get("primary_metric", "recall"),
        "secondary_metrics": ["accuracy", "precision"],
    }

def validate_plan(plan: Dict[str, Any]) -> bool:
    return True

def run_metrics(model_path: str, df_labelled: pd.DataFrame, profile: Dict[str, Any], plan: Dict[str, Any], n_boot: int = 200, seed: int = 42) -> Dict[str, Any]:
    primary = plan.get("primary_metric", profile.get("primary_metric", "recall"))
    return {
        "module": "quality",
        "status": "STUB",
        "n": len(df_labelled) if df_labelled is not None else 0,
        "metrics": {
            primary: {"value": 0.88, "se": 0.02},
            "accuracy": {"value": 0.90, "se": 0.015},
        },
        "curves": {},
        "z_m": 0.1,
        "degraded": False,
        "table": [
            {"metric": primary, "reference": 0.90, "current": 0.88, "delta": -0.02},
            {"metric": "accuracy", "reference": 0.91, "current": 0.90, "delta": -0.01},
        ],
    }

def compute_baseline(model_path: str, holdout_df: pd.DataFrame, profile: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
    primary = plan.get("primary_metric", profile.get("primary_metric", "recall"))
    return {
        "module": "quality",
        "status": "STUB",
        "n": len(holdout_df) if holdout_df is not None else 0,
        "metrics": {
            primary: {"value": 0.90, "se": 0.01},
            "accuracy": {"value": 0.91, "se": 0.01},
        },
        "curves": {},
        "z_m": 0.0,
        "degraded": False,
        "table": [
            {"metric": primary, "reference": 0.90, "current": 0.90, "delta": 0.0},
        ],
    }

def compare(baseline: Dict[str, Any], current: Dict[str, Any], primary_metric: str) -> Dict[str, Any]:
    return {
        "module": "quality",
        "status": "STUB",
        "primary_metric": primary_metric,
        "degraded": False,
        "delta": 0.0,
    }

