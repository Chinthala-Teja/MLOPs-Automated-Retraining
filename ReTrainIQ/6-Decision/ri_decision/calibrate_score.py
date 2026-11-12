"""calibrate_score.py - is the decision score a probability? Fit and test it out of time.

Decision points come from a model bank (ri_cara.replay_baselines.build_bank):
every (serving model, batch) pair gives the inputs (zc, zm, d) and a label
'retraining helped' = a model trained at that batch has a lower mean 0-1 loss
than the serving model on the next H batches by more than `margin`.
A logistic regression is fitted on the earlier batches and tested on the later
ones. The score may be called a probability only if the out-of-time Brier score
beats both the starting coefficients and the base rate.

Caveat: decision points from one replay overlap heavily and are autocorrelated,
so the effective sample is far smaller than the row count.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ri_cara import replay_baselines as rb
from ri_init.config import DEFAULT_CONFIG, WORKSPACE, load_config

IS_STUB = False


def load_project(project_id: str, batches_dir: Optional[str | Path] = None, limit: Optional[int] = None) -> Dict[str, Any]:
    """Profile, v1 calibration and the ordered replay batches of a project."""
    proj = WORKSPACE / project_id
    profile = json.loads((proj / "profile.json").read_text(encoding="utf-8"))
    calib = json.loads((proj / "models" / "v1" / "calib.json").read_text(encoding="utf-8"))
    profile["model"]["artifact"] = str(proj / "models" / "v1" / "model.joblib")
    files = sorted(Path(batches_dir or proj / "data" / "stream").glob("batch_*.csv"))[:limit]
    return {"profile": profile, "calib": calib, "batches": [pd.read_csv(f) for f in files]}


def decision_points(bank: Dict[str, Any], calib: Dict[str, Any], profile: Dict[str, Any], cfg: Dict[str, Any],
                    horizon: int = 3, margin: float = 0.01) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n, err = bank["n"], bank["err"]
    for t in range(n - horizon):
        if not bank["valid"][t + 1]:
            continue
        for cur in range(0, t + 1):
            if not np.isfinite(bank["stal"][cur, t]):
                continue
            future = slice(t + 1, t + 1 + horizon)
            gain = float(np.mean(err[cur, future]) - np.mean(err[t + 1, future]))
            rows.append({"t": t, "model": cur, **rb.score_inputs(bank, calib, profile, t, cur, cfg),
                         "gain": gain, "helped": int(gain > margin)})
    return pd.DataFrame(rows)


def _prob(points: pd.DataFrame, coef: Dict[str, float]) -> np.ndarray:
    logit = coef["b"] + coef["a1"] * points["zc"] + coef["a2"] * points["zm"] + coef["a3"] * points["d"]
    return 1.0 / (1.0 + np.exp(-logit.to_numpy()))


def calibrate_score(project_id: str = "demo", batches_dir: Optional[str | Path] = None, horizon: int = 3,
                    margin: float = 0.01, train_share: float = 0.6, write: bool = False,
                    limit: Optional[int] = None, bank: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Fit the score coefficients on earlier decision points and test them on later ones."""
    data = load_project(project_id, batches_dir, limit)
    profile, calib = data["profile"], data["calib"]
    cfg = load_config(None)
    bank = bank or rb.build_bank(data["batches"], calib, profile, cfg)
    points = decision_points(bank, calib, profile, cfg, horizon, margin)
    cut = int(points["t"].max() * train_share)
    train, test = points[points["t"] <= cut], points[points["t"] > cut]
    start = {k: float(DEFAULT_CONFIG[k]) for k in ("b", "a1", "a2", "a3")}
    out: Dict[str, Any] = {"status": "OK", "n_points": int(len(points)), "n_train": int(len(train)),
                           "n_test": int(len(test)), "base_rate_train": float(train["helped"].mean()),
                           "base_rate_test": float(test["helped"].mean()) if len(test) else None,
                           "horizon": horizon, "margin": margin, "start_coefficients": start, "simulated": True}
    if train["helped"].nunique() < 2 or len(test) == 0:
        out.update({"status": "NOT_FITTED", "calibrated": False,
                    "note": "the training part contains only one outcome class; coefficients left unchanged"})
        return out
    model = LogisticRegression(C=1.0, max_iter=1000).fit(train[["zc", "zm", "d"]], train["helped"])
    fitted = {"b": float(model.intercept_[0]), "a1": float(model.coef_[0][0]),
              "a2": float(model.coef_[0][1]), "a3": float(model.coef_[0][2])}
    guard = math.log(DEFAULT_CONFIG["retrain_limit"] / (1 - DEFAULT_CONFIG["retrain_limit"]))
    notes: List[str] = []
    if fitted["a3"] >= guard:
        fitted["a3"] = round(guard - 0.01, 4)
        notes.append("a3 capped so that drift alone cannot reach the automatic-retrain line")
    y = test["helped"].to_numpy()
    p_fit, p_start = _prob(test, fitted), _prob(test, start)
    brier = {"fitted": float(np.mean((p_fit - y) ** 2)), "start": float(np.mean((p_start - y) ** 2)),
             "base_rate": float(np.mean((train["helped"].mean() - y) ** 2))}
    bins = np.clip((p_fit * 5).astype(int), 0, 4)
    reliability = [{"bin": f"{b / 5:.1f}-{(b + 1) / 5:.1f}", "n": int((bins == b).sum()),
                    "mean_score": float(p_fit[bins == b].mean()), "share_helped": float(y[bins == b].mean())}
                   for b in range(5) if (bins == b).any()]
    calibrated = brier["fitted"] < brier["start"] and brier["fitted"] < brier["base_rate"]
    out.update({"fitted_coefficients": fitted, "brier": brier, "reliability": reliability,
                "calibrated": bool(calibrated), "notes": notes})
    if write and calibrated:
        path = WORKSPACE / project_id / "config.json"
        current = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        path.write_text(json.dumps({**current, **fitted}, indent=2), encoding="utf-8")
        out["written_to"] = str(path)
    return out
