"""scenarios.py - simulated drift scenarios for the demo and the evaluation.

Everything produced here is a SIMULATION. Never report it as real drift.

Covariate shift is injected by resampling rows (importance resampling), so the
feature distribution P(x) changes while every row keeps its own label and
P(y | x) is untouched. Concept shift flips labels inside a region, so P(y | x)
changes while the features stay as they were.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

IS_STUB = False

SCENARIOS = ("stable", "covariate", "concept", "invalid", "delayed_labels")


def _seed(profile: Dict[str, Any]) -> int:
    return int(profile.get("seed", 42))


def inject_covariate_shift(
    df: pd.DataFrame, profile: Dict[str, Any], feature: str, magnitude: float, start_row: int
) -> pd.DataFrame:
    """Resample rows from start_row onward so that `feature` shifts.

    Numeric: exponential tilting with weights exp(magnitude * z); for a roughly
    normal feature the mean moves by about `magnitude` standard deviations.
    Categorical: the rarest category is over-sampled by a factor (1 + |magnitude|) * 3.
    """
    out = df.copy().reset_index(drop=True)
    tail = out.iloc[start_row:]
    if tail.empty or feature not in out.columns or magnitude == 0:
        return out
    rng = np.random.default_rng(_seed(profile))
    if feature in profile.get("features", {}).get("numeric", []):
        vals = pd.to_numeric(tail[feature], errors="coerce").to_numpy(dtype=float)
        std = np.nanstd(vals) or 1.0
        z = np.nan_to_num((vals - np.nanmean(vals)) / std)
        weights = np.exp(np.clip(magnitude * z, -30, 30))
    else:
        keys = tail[feature].astype(str)
        rare = keys.value_counts().idxmin()
        weights = np.where(keys == rare, (1.0 + abs(magnitude)) * 3.0, 1.0).astype(float)
    weights = weights / weights.sum()
    picked = rng.choice(len(tail), size=len(tail), replace=True, p=weights)
    out.iloc[start_row:] = tail.iloc[picked].to_numpy()
    return out.astype(df.dtypes.to_dict())


def inject_concept_shift(
    df: pd.DataFrame, profile: Dict[str, Any], region_rule: Any, flip_fraction: float, start_row: int
) -> pd.DataFrame:
    """Flip the binary target for a fraction of the rows that match a pandas query string."""
    out = df.copy().reset_index(drop=True)
    target = profile.get("task", {}).get("target")
    if target not in out.columns:
        return out
    tail = out.iloc[start_row:]
    region = tail.query(region_rule).index if region_rule else tail.index
    region = [i for i in region if pd.notna(out.at[i, target])]
    rng = np.random.default_rng(_seed(profile))
    n_flip = int(round(len(region) * float(flip_fraction)))
    if n_flip == 0:
        return out
    flip = rng.choice(np.asarray(region), size=n_flip, replace=False)
    classes = sorted(pd.unique(out[target].dropna()))
    if len(classes) != 2:
        raise ValueError("inject_concept_shift supports binary targets only.")
    low, high = classes
    out.loc[flip, target] = np.where(out.loc[flip, target] == low, high, low)
    return out


def make_invalid_upload(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Break an upload on purpose: 'missing_column', 'bad_dtype' or 'out_of_range'."""
    out = df.copy()
    numeric_cols = list(out.select_dtypes(include="number").columns)
    if kind == "missing_column":
        return out.drop(columns=[out.columns[0]])
    if kind == "bad_dtype":
        col = numeric_cols[0]
        out[col] = out[col].astype(object)
        out.iloc[: max(1, len(out) // 10), out.columns.get_loc(col)] = "not-a-number"
        return out
    if kind == "out_of_range":
        col = numeric_cols[0]
        out[col] = out[col].astype(float) * 1000.0 + 1e6
        return out
    raise ValueError(f"Unknown invalid-upload kind: {kind!r}")


def hold_back_labels(df: pd.DataFrame, fraction: float, target: str | None = None, seed: int = 42) -> pd.DataFrame:
    """Set the target to NaN for a fraction of rows (the target defaults to the last column)."""
    out = df.copy()
    col = target or out.columns[-1]
    rng = np.random.default_rng(seed)
    n_hide = int(round(len(out) * float(fraction)))
    hide = rng.choice(len(out), size=n_hide, replace=False)
    out[col] = out[col].astype(float)
    out.iloc[hide, out.columns.get_loc(col)] = np.nan
    return out


def build_scenario_stream(
    batches: List[pd.DataFrame],
    scenario_name: str,
    profile: Dict[str, Any] | None = None,
    onset: int | None = None,
    magnitude: float = 1.5,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """Return (batches, info) for a named scenario; info records the true onset batch."""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario_name!r}; expected one of {SCENARIOS}")
    profile = profile or {}
    out = [b.copy() for b in batches]
    onset = len(out) // 2 if onset is None else onset
    info: Dict[str, Any] = {"scenario": scenario_name, "onset_batch": None, "simulated": True}
    numeric = profile.get("features", {}).get("numeric", [])
    target = profile.get("task", {}).get("target")
    if scenario_name == "stable":
        return out, info
    info["onset_batch"] = onset
    for i in range(onset, len(out)):
        if scenario_name == "covariate":
            feature = numeric[0] if numeric else out[i].columns[0]
            out[i] = inject_covariate_shift(out[i], profile, feature, magnitude, 0)
            info["feature"] = feature
        elif scenario_name == "concept":
            out[i] = inject_concept_shift(out[i], profile, None, 0.5, 0)
        elif scenario_name == "invalid" and i == onset:
            out[i] = make_invalid_upload(out[i], "missing_column")
        elif scenario_name == "delayed_labels":
            out[i] = hold_back_labels(out[i], 1.0, target=target, seed=_seed(profile) + i)
    return out, info
