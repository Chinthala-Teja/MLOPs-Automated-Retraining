"""proxy_eval.py - evaluation of CARA-T-L (label-free staleness) under label delay. SIMULATION only.

For each scenario the stream is replayed twice per batch: with labels (measured
staleness, which in reality only becomes available `lag` batches later) and
without labels (the confidence-based estimate, available at once).

Scenarios (all injected, all simulated):
  stable              no change
  covariate_tilt      rows resampled so the first numeric feature shifts (P(y|x) unchanged)
  covariate_uncertain rows resampled towards inputs the model is unsure about (P(y|x) unchanged)
  concept             labels flipped for half the rows (P(y|x) changes, inputs do not)

The concept scenario is the known failure case: the model stays confident while
being wrong, so a confidence-based estimate cannot see it. It is reported, not hidden.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from ri_cara import cara_t
from ri_drift.drift import run_drift
from ri_drift.scenarios import inject_concept_shift, inject_covariate_shift
from ri_init.config import WORKSPACE

IS_STUB = False

SCENARIOS = ("stable", "covariate_tilt", "covariate_uncertain", "concept")


def _uncertain_resample(df: pd.DataFrame, model: Any, profile: Dict[str, Any], strength: float, seed: int) -> pd.DataFrame:
    conf = cara_t.confidence(model, df, profile)
    weights = np.exp(strength * (1.0 - conf) * 10.0)
    weights = weights / weights.sum()
    picked = np.random.default_rng(seed).choice(len(df), size=len(df), replace=True, p=weights)
    return df.iloc[picked].reset_index(drop=True)


def _first(flags: List[bool], start: int) -> Optional[int]:
    return next((i for i in range(start, len(flags)) if flags[i]), None)


def run_proxy_eval(profile: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    """One row per (scenario, label lag): detection batch of each signal, false alarms, tracking error."""
    cfg = cfg or {}
    proj = WORKSPACE / str(profile.get("project_id"))
    bundle = proj / "models" / "v1"
    calib = cfg.get("calib") or json.loads((bundle / "calib.json").read_text(encoding="utf-8"))
    train_sample = cfg.get("train_sample_df")
    if train_sample is None:
        train_sample = pd.read_csv(bundle / "train_sample.csv")
    model_path = cfg.get("model_path") or str(bundle / "model.joblib")
    if not calib.get("proxy"):
        return pd.DataFrame([{"scenario": "n/a", "note": "label-free estimate unavailable (no proxy calibration)"}])
    batches = cfg.get("batches")
    if batches is None:
        folder = Path(cfg.get("batches_dir") or proj / "data" / "stream")
        files = sorted(folder.glob("batch_*.csv"))[: int(cfg.get("n_batches", 16))]
        batches = [pd.read_csv(f) for f in files]
    n = len(batches)
    onset = int(cfg.get("onset", n // 2))
    lags = list(cfg.get("lags", (0, 2, 5)))
    magnitude = float(cfg.get("magnitude", 1.0))
    seed = int(profile.get("seed", 42))
    target = profile["task"]["target"]
    numeric = profile.get("features", {}).get("numeric", [])
    model = joblib.load(cara_t.resolve(model_path))
    tau = float(calib["tau_star"])

    rows: List[Dict[str, Any]] = []
    for scenario in cfg.get("scenarios", SCENARIOS):
        measured, measured_lo, est, est_lo, drift_flag = [], [], [], [], []
        for t, batch in enumerate(batches):
            frame = batch.copy()
            if t >= onset:
                if scenario == "covariate_tilt" and numeric:
                    frame = inject_covariate_shift(frame, {**profile, "seed": seed + t}, numeric[0], magnitude, 0)
                elif scenario == "covariate_uncertain":
                    frame = _uncertain_resample(frame, model, profile, magnitude, seed + t)
                elif scenario == "concept":
                    frame = inject_concept_shift(frame, {**profile, "seed": seed + t}, None, 0.5, 0)
            m = cara_t.online_staleness(model_path, frame, train_sample, calib, profile, seed)
            e = cara_t.estimate_staleness(model_path, frame.drop(columns=[target]), train_sample, calib, profile, seed)
            measured.append(float(m["staleness"]))
            measured_lo.append(float(m["staleness_lo"]))
            est.append(float(e["staleness_est"]))
            est_lo.append(float(e["staleness_est_lo"]))
            drift_flag.append(run_drift(train_sample, frame, profile)["drift_share"] > 0)
        meas_alert = [v >= tau for v in measured_lo]  # same rule the decision score uses
        point_alert = [v >= tau for v in est]
        lo_alert = [v >= tau for v in est_lo]
        rho = stats.spearmanr(est, measured).statistic if len(set(est)) > 1 else float("nan")
        pre = max(onset, 1) if scenario != "stable" else n
        start = onset if scenario != "stable" else n
        for lag in lags:
            d_meas = _first(meas_alert, start)
            rows.append({
                "scenario": scenario, "label_lag": lag, "onset_batch": None if scenario == "stable" else onset,
                "measured_actionable_at": None if d_meas is None else d_meas + lag,
                "estimate_alert_at": _first(point_alert, start),
                "estimate_lower_bound_alert_at": _first(lo_alert, start),
                "drift_alert_at": _first(drift_flag, start),
                "false_alarm_rate_measured": float(np.mean(meas_alert[:pre])),
                "false_alarm_rate_estimate": float(np.mean(point_alert[:pre])),
                "false_alarm_rate_lower_bound": float(np.mean(lo_alert[:pre])),
                "false_alarm_rate_drift": float(np.mean(drift_flag[:pre])),
                "spearman_estimate_vs_measured": None if not np.isfinite(rho) else float(rho),
                "mean_abs_error": float(np.mean(np.abs(np.array(est) - np.array(measured)))),
                "mean_measured_after_onset": float(np.mean(measured[start:])) if start < n else None,
                "mean_estimate_after_onset": float(np.mean(est[start:])) if start < n else None,
                "tau_star": tau, "simulated": True,
            })
    return pd.DataFrame(rows)
