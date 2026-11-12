"""metrics.py - Module 4: quality metrics from a vetted registry.

The LLM (or the default plan) may only choose metric NAMES from REGISTRY; code
computes every value, with bootstrap standard errors. No supervised metric is
ever computed without labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics as skm

from ri_init.config import ROOT

IS_STUB = False

Z_DEGRADED = 1.64


def _binary(fn: Callable[..., float]) -> Callable[..., float]:
    def inner(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
        return float(fn(y, pred, pos_label=ctx["positive_class"], zero_division=0))
    return inner


def _averaged(fn: Callable[..., float], average: str) -> Callable[..., float]:
    def inner(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
        return float(fn(y, pred, average=average, zero_division=0))
    return inner


def _accuracy(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
    return float(skm.accuracy_score(y, pred))


def _roc_auc(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
    return float(skm.roc_auc_score(y == ctx["positive_class"], score))


def _pr_auc(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
    return float(skm.average_precision_score(y == ctx["positive_class"], score))


def _log_loss(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
    return float(skm.log_loss(y == ctx["positive_class"], np.clip(score, 1e-7, 1 - 1e-7), labels=[False, True]))


def _mae(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
    return float(skm.mean_absolute_error(y, pred))


def _rmse(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
    return float(np.sqrt(skm.mean_squared_error(y, pred)))


def _r2(y: np.ndarray, pred: np.ndarray, score: Any, ctx: Dict[str, Any]) -> float:
    return float(skm.r2_score(y, pred))


def _entry(fn: Callable[..., float], applies_to: List[str], needs_proba: bool = False,
           higher_is_better: bool = True, description: str = "") -> Dict[str, Any]:
    return {"fn": fn, "applies_to": applies_to, "needs_proba": needs_proba,
            "higher_is_better": higher_is_better, "description": description}


REGISTRY: Dict[str, Dict[str, Any]] = {
    "accuracy": _entry(_accuracy, ["binary", "multiclass"], description="share of correct predictions"),
    "precision": _entry(_binary(skm.precision_score), ["binary"], description="positive predictions that are right"),
    "recall": _entry(_binary(skm.recall_score), ["binary"], description="actual positives that are found"),
    "f1": _entry(_binary(skm.f1_score), ["binary"], description="harmonic mean of precision and recall"),
    "roc_auc": _entry(_roc_auc, ["binary"], needs_proba=True, description="ranking quality over all thresholds"),
    "pr_auc": _entry(_pr_auc, ["binary"], needs_proba=True, description="average precision"),
    "log_loss": _entry(_log_loss, ["binary"], needs_proba=True, higher_is_better=False,
                       description="penalty for confident wrong probabilities"),
    "macro_precision": _entry(_averaged(skm.precision_score, "macro"), ["multiclass"]),
    "macro_recall": _entry(_averaged(skm.recall_score, "macro"), ["multiclass"]),
    "macro_f1": _entry(_averaged(skm.f1_score, "macro"), ["multiclass"]),
    "weighted_precision": _entry(_averaged(skm.precision_score, "weighted"), ["multiclass"]),
    "weighted_recall": _entry(_averaged(skm.recall_score, "weighted"), ["multiclass"]),
    "weighted_f1": _entry(_averaged(skm.f1_score, "weighted"), ["multiclass"]),
    "mae": _entry(_mae, ["regression"], higher_is_better=False, description="mean absolute error"),
    "rmse": _entry(_rmse, ["regression"], higher_is_better=False, description="root mean squared error"),
    "r2": _entry(_r2, ["regression"], description="explained variance share"),
}


def _kind(profile: Dict[str, Any]) -> str:
    task = profile.get("task", {})
    if task.get("type") == "regression":
        return "regression"
    return "multiclass" if task.get("subtype") == "multiclass" else "binary"


def default_plan(profile: Dict[str, Any]) -> List[str]:
    """Every registry metric that applies to the task, primary metric first."""
    kind = _kind(profile)
    names = [n for n, e in REGISTRY.items() if kind in e["applies_to"]]
    primary = str(profile.get("primary_metric", "")).lower()
    if primary in names:
        names.remove(primary)
        names.insert(0, primary)
    return names


def validate_plan(plan: Any, profile: Optional[Dict[str, Any]] = None) -> List[str]:
    """Return the cleaned plan; raise ValueError for names outside the registry."""
    if isinstance(plan, dict):  # legacy shape: {'primary_metric': ..., 'secondary_metrics': [...]}
        plan = [plan.get("primary_metric"), *plan.get("secondary_metrics", [])]
    names = [str(n).lower() for n in (plan or []) if n]
    unknown = [n for n in names if n not in REGISTRY]
    if unknown:
        raise ValueError(f"Metrics not in the registry: {unknown}")
    if profile is not None:
        kind = _kind(profile)
        wrong = [n for n in names if kind not in REGISTRY[n]["applies_to"]]
        if wrong:
            raise ValueError(f"Metrics that do not apply to a {kind} task: {wrong}")
    return list(dict.fromkeys(names))


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _curve(x: np.ndarray, y: np.ndarray, n: int = 60) -> Dict[str, List[float]]:
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).astype(int))
    return {"x": [float(v) for v in x[idx]], "y": [float(v) for v in y[idx]]}


def run_metrics(model_path: str, df_labelled: pd.DataFrame, profile: Dict[str, Any], plan: Any, n_boot: int = 200, seed: int = 42) -> Dict[str, Any]:
    """Compute the planned metrics on labelled rows, with bootstrap standard errors."""
    target = profile.get("task", {}).get("target")
    pending = {"module": "quality", "status": "PENDING", "n": 0, "metrics": {}, "curves": {},
               "z_m": 0.0, "degraded": False, "table": [], "label_coverage": 0.0}
    if df_labelled is None or target not in df_labelled.columns or len(df_labelled) == 0:
        return pending
    lab = df_labelled.loc[df_labelled[target].notna()]
    coverage = float(len(lab) / len(df_labelled))
    if len(lab) == 0:
        return pending

    names = validate_plan(plan, profile) or default_plan(profile)
    kind = _kind(profile)
    feats = profile.get("features", {})
    cols = list(feats.get("numeric", [])) + list(feats.get("categorical", []))
    model = joblib.load(_resolve(model_path))
    y = lab[target].to_numpy()
    pred = np.asarray(model.predict(lab[cols]))
    ctx = {"positive_class": profile.get("task", {}).get("positive_class", 1)}

    score: Optional[np.ndarray] = None
    if kind == "binary" and hasattr(model, "predict_proba"):
        classes = list(getattr(model, "classes_", []))
        if ctx["positive_class"] in classes:
            score = np.asarray(model.predict_proba(lab[cols]))[:, classes.index(ctx["positive_class"])]
    skipped = [n for n in names if REGISTRY[n]["needs_proba"] and score is None]
    names = [n for n in names if n not in skipped]

    def compute(idx: Optional[np.ndarray]) -> Dict[str, float]:
        yy, pp = (y, pred) if idx is None else (y[idx], pred[idx])
        ss = score if idx is None or score is None else score[idx]
        out: Dict[str, float] = {}
        one_class = len(np.unique(yy)) < 2
        for name in names:
            if one_class and REGISTRY[name]["needs_proba"]:
                continue
            out[name] = REGISTRY[name]["fn"](yy, pp, ss, ctx)
        return out

    point = compute(None)
    rng = np.random.default_rng(seed)
    boots: Dict[str, List[float]] = {n: [] for n in names}
    for _ in range(int(n_boot)):
        for name, value in compute(rng.integers(0, len(y), size=len(y))).items():
            boots[name].append(value)
    metrics_out = {
        name: {"value": float(point[name]),
               "se": float(np.std(boots[name], ddof=1)) if len(boots[name]) > 1 else 0.0}
        for name in names if name in point
    }

    curves: Dict[str, Any] = {}
    confusion: Optional[List[List[int]]] = None
    labels_out: Optional[List[Any]] = None
    if kind != "regression":
        labels = sorted(pd.unique(np.concatenate([y, pred])).tolist())
        confusion = skm.confusion_matrix(y, pred, labels=labels).astype(int).tolist()
        labels_out = [v.item() if hasattr(v, "item") else v for v in labels]
    per_class_recall: Optional[Dict[str, float]] = None
    if kind == "multiclass":
        recalls = skm.recall_score(y, pred, labels=labels, average=None, zero_division=0)
        per_class_recall = {str(label): float(r) for label, r in zip(labels_out or [], recalls)}
    if kind == "binary" and score is not None and len(np.unique(y)) == 2:
        fpr, tpr, _ = skm.roc_curve(y == ctx["positive_class"], score)
        prec, rec, _ = skm.precision_recall_curve(y == ctx["positive_class"], score)
        curves = {"roc": _curve(fpr, tpr), "pr": _curve(rec[::-1], prec[::-1])}

    return {
        "module": "quality", "status": "OK", "n": int(len(lab)), "label_coverage": coverage,
        "metrics": metrics_out, "curves": curves, "confusion_matrix": confusion, "labels": labels_out,
        "per_class_recall": per_class_recall,
        "skipped": skipped, "z_m": 0.0, "degraded": False, "table": [],
    }


def compute_baseline(model_path: str, holdout_df: pd.DataFrame, profile: Dict[str, Any], plan: Any) -> Dict[str, Any]:
    """Baseline quality of a model on data it has never seen (the holdout)."""
    return run_metrics(model_path, holdout_df, profile, plan,
                       n_boot=200, seed=int(profile.get("seed", 42)))


def compare(baseline: Dict[str, Any], current: Dict[str, Any], primary_metric: str) -> Dict[str, Any]:
    """Add z_m, the degraded flag and the comparison table to the current result.

    z_m = (m_ref - m_new) / sqrt(se_ref^2 + se_new^2) for higher-is-better
    metrics (sign flipped otherwise); positive means the model got worse.
    """
    out = dict(current)
    if current.get("status") != "OK":
        return out
    out["primary_metric"] = primary_metric
    if not baseline or baseline.get("status") != "OK":
        out.update({"status": "NO_BASELINE", "z_m": 0.0, "degraded": False, "table": [],
                    "note": "no holdout baseline: quality comparison is switched off"})
        return out
    table = []
    for name, cur in current["metrics"].items():
        ref = baseline["metrics"].get(name)
        if ref is None:
            continue
        table.append({"metric": name, "reference": float(ref["value"]), "current": float(cur["value"]),
                      "delta": float(cur["value"] - ref["value"])})
    out["table"] = table
    primary = str(primary_metric).lower()
    ref, cur = baseline["metrics"].get(primary), current["metrics"].get(primary)
    if ref is None or cur is None:
        out.update({"z_m": 0.0, "degraded": False,
                    "note": f"primary metric '{primary}' unavailable; z_m set to 0"})
        return out
    denom = max(float(np.sqrt(ref["se"] ** 2 + cur["se"] ** 2)), 1e-9)
    diff = ref["value"] - cur["value"]
    if not REGISTRY[primary]["higher_is_better"]:
        diff = -diff
    out["z_m"] = float(diff / denom)
    out["degraded"] = bool(out["z_m"] >= Z_DEGRADED)
    return out
