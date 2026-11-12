"""drift.py - Module 2: data drift and schema validation.

Numeric features: PSI on 10 reference-quantile bins plus a two-sample KS test.
Categorical features: PSI on category shares (NaN is its own category, unseen
categories are merged into '__other__').
Drift is a warning signal only; it never triggers retraining by itself.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ri_init.config import WORKSPACE

IS_STUB = False

PSI_FLAG = 0.2
KS_P_FLAG = 0.01
KS_STAT_FLAG = 0.1
MAX_ROWS = 50_000
EPS = 1e-4
NAN_KEY = "__nan__"
OTHER_KEY = "__other__"


def _features(profile: Dict[str, Any]) -> tuple[List[str], List[str]]:
    feats = profile.get("features", {})
    return list(feats.get("numeric", [])), list(feats.get("categorical", []))


def _cap(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(df) > MAX_ROWS:
        return df.sample(n=MAX_ROWS, random_state=seed)
    return df


def _psi(p: np.ndarray, q: np.ndarray) -> float:
    """PSI = sum((p - q) * ln(p / q)); zero shares are replaced by EPS."""
    p = np.where(p <= 0, EPS, p)
    q = np.where(q <= 0, EPS, q)
    return float(np.sum((p - q) * np.log(p / q)))


def _numeric_edges(values: np.ndarray) -> List[float]:
    """Inner bin edges from reference deciles (deduplicated); outer bins are open."""
    if len(values) == 0:
        return []
    qs = np.quantile(values, np.linspace(0.1, 0.9, 9))
    return [float(v) for v in np.unique(qs)]


def _bin_shares(values: np.ndarray, edges: List[float]) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(len(edges) + 1)
    idx = np.searchsorted(np.asarray(edges), values, side="right")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    return counts / counts.sum()


def _cat_keys(series: pd.Series) -> pd.Series:
    return series.astype("object").where(series.notna(), NAN_KEY).astype(str)


def fit_reference(reference_df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Summarise the reference data: bins, shares, 1st-99th percentile bounds, missing rates."""
    numeric, categorical = _features(profile)
    seed = int(profile.get("seed", 42))
    ref = _cap(reference_df, seed)
    out: Dict[str, Any] = {
        "module": "drift",
        "status": "OK",
        "n_ref": int(len(ref)),
        "numeric": {},
        "categorical": {},
    }
    for col in numeric:
        if col not in ref.columns:
            continue
        raw = pd.to_numeric(ref[col], errors="coerce")
        vals = raw.dropna().to_numpy(dtype=float)
        edges = _numeric_edges(vals)
        out["numeric"][col] = {
            "edges": edges,
            "shares": _bin_shares(vals, edges).tolist(),
            "p01": float(np.quantile(vals, 0.01)) if len(vals) else None,
            "p99": float(np.quantile(vals, 0.99)) if len(vals) else None,
            "missing_rate": float(raw.isna().mean()) if len(raw) else 0.0,
        }
    for col in categorical:
        if col not in ref.columns:
            continue
        shares = _cat_keys(ref[col]).value_counts(normalize=True)
        out["categorical"][col] = {
            "shares": {str(k): float(v) for k, v in shares.items()},
            "missing_rate": float(ref[col].isna().mean()) if len(ref) else 0.0,
        }
    return out


def run_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Compare a new batch with the reference data, feature by feature."""
    numeric, categorical = _features(profile)
    seed = int(profile.get("seed", 42))
    ref = _cap(reference_df, seed)
    new = _cap(new_df, seed)
    per_feature: List[Dict[str, Any]] = []
    missing_rate: Dict[str, Dict[str, float]] = {}

    for col in numeric:
        if col not in ref.columns or col not in new.columns:
            continue
        r_raw = pd.to_numeric(ref[col], errors="coerce")
        n_raw = pd.to_numeric(new[col], errors="coerce")
        r = r_raw.dropna().to_numpy(dtype=float)
        n = n_raw.dropna().to_numpy(dtype=float)
        missing_rate[col] = {"ref": float(r_raw.isna().mean()), "new": float(n_raw.isna().mean())}
        if len(r) == 0 or len(n) == 0:
            per_feature.append({"feature": col, "type": "numeric", "psi": 0.0,
                                "ks_stat": None, "p_value": None, "flag": False})
            continue
        edges = _numeric_edges(r)
        psi = _psi(_bin_shares(n, edges), _bin_shares(r, edges))
        ks = stats.ks_2samp(r, n)
        ks_stat, p_value = float(ks.statistic), float(ks.pvalue)
        per_feature.append({
            "feature": col, "type": "numeric", "psi": psi,
            "ks_stat": ks_stat, "p_value": p_value,
            "flag": bool(p_value < KS_P_FLAG and ks_stat >= KS_STAT_FLAG),
        })

    for col in categorical:
        if col not in ref.columns or col not in new.columns:
            continue
        r_keys, n_keys = _cat_keys(ref[col]), _cat_keys(new[col])
        missing_rate[col] = {"ref": float(ref[col].isna().mean()), "new": float(new[col].isna().mean())}
        known = set(r_keys.unique())
        n_keys = n_keys.where(n_keys.isin(known), OTHER_KEY)
        cats = sorted(known | {OTHER_KEY})
        r_sh = r_keys.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
        n_sh = n_keys.value_counts(normalize=True).reindex(cats, fill_value=0.0).to_numpy()
        if n_sh[cats.index(OTHER_KEY)] == 0 and r_sh[cats.index(OTHER_KEY)] == 0:
            keep = [i for i, c in enumerate(cats) if c != OTHER_KEY]
            r_sh, n_sh = r_sh[keep], n_sh[keep]
        psi = _psi(n_sh, r_sh)
        per_feature.append({
            "feature": col, "type": "categorical", "psi": psi,
            "ks_stat": None, "p_value": None, "flag": bool(psi >= PSI_FLAG),
        })

    n_feat = len(per_feature)
    share = float(sum(f["flag"] for f in per_feature) / n_feat) if n_feat else 0.0
    return {
        "module": "drift",
        "status": "DRIFT" if share > 0 else "STABLE",
        "drift_share": share,
        "per_feature": per_feature,
        "n_ref": int(len(ref)),
        "n_new": int(len(new)),
        "missing_rate": missing_rate,
    }


def _stored_reference(profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    path = WORKSPACE / str(profile.get("project_id", "")) / "reference_stats.json"
    if path.is_file():
        try:
            stats_dict = json.loads(Path(path).read_text(encoding="utf-8"))
            if "numeric" in stats_dict:
                return stats_dict
        except (OSError, ValueError):
            return None
    return None


def validate_schema(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    reference: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Check an upload. BLOCKED if a feature column is missing or a numeric column cannot be cast.

    Unseen categories and out-of-range counts (outside the reference 1st-99th
    percentile) are reported as information; they do not block the upload.
    """
    numeric, categorical = _features(profile)
    if reference is None:
        reference = _stored_reference(profile)
    issues: List[str] = []
    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        issues.append(f"missing feature columns: {missing}")
    if len(df) == 0:
        issues.append("file has no rows")

    out_of_range: Dict[str, int] = {}
    unseen: Dict[str, List[str]] = {}
    for col in numeric:
        if col not in df.columns:
            continue
        cast = pd.to_numeric(df[col], errors="coerce")
        bad = int((cast.isna() & df[col].notna()).sum())
        if bad:
            issues.append(f"numeric column '{col}' has {bad} values that cannot be cast to a number")
        ref_stats = (reference or {}).get("numeric", {}).get(col)
        if ref_stats and ref_stats.get("p01") is not None:
            count = int(((cast < ref_stats["p01"]) | (cast > ref_stats["p99"])).sum())
            if count:
                out_of_range[col] = count
    for col in categorical:
        if col not in df.columns:
            continue
        ref_stats = (reference or {}).get("categorical", {}).get(col)
        if ref_stats:
            new_vals = sorted(set(_cat_keys(df[col]).unique()) - set(ref_stats["shares"]))
            if new_vals:
                unseen[col] = new_vals[:20]

    blocked = bool(issues)
    return {
        "module": "drift",
        "status": "BLOCKED" if blocked else "VALID",
        "valid": not blocked,
        "issues": issues,
        "missing_columns": missing,
        "unseen_categories": unseen,
        "out_of_range": out_of_range,
    }


def prediction_shift(ref_pred: Any, new_pred: Any, ref_proba: Any = None, new_proba: Any = None) -> Dict[str, Any]:
    """Change in positive-prediction rate and in mean confidence between two prediction sets."""
    ref_p = np.asarray(ref_pred, dtype=float)
    new_p = np.asarray(new_pred, dtype=float)
    rate_shift = float(new_p.mean() - ref_p.mean()) if len(ref_p) and len(new_p) else 0.0

    def _conf(proba: Any) -> Optional[float]:
        if proba is None:
            return None
        arr = np.asarray(proba, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = np.maximum(arr, 1.0 - arr)
        else:
            arr = arr.max(axis=1)
        return float(arr.mean())

    ref_c, new_c = _conf(ref_proba), _conf(new_proba)
    return {
        "module": "drift",
        "status": "OK",
        "positive_rate_shift": rate_shift,
        "mean_confidence_shift": float(new_c - ref_c) if ref_c is not None and new_c is not None else 0.0,
    }
