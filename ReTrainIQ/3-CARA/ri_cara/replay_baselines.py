"""replay_baselines.py - compare retraining strategies on a replayed stream (a SIMULATION).

A "bank" is built once: the champion plus one model per batch (same training
recipe as online: last W batches). Every strategy is then a set of look-ups, so
all strategies see exactly the same models and the same costs.

Order of events (test-then-train): the current model serves batch t and pays
its relative staleness PsiBar(t, current); a RETRAIN decided at t costs kappa
and the model trained on batch t serves from t + 1.
    J = sum of staleness + kappa * number of retrains
The Oracle (dynamic programme over the whole stream) uses future batches and is
a retrospective lower bound only, never a deployable strategy.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from ri_cara import cara_t
from ri_drift.drift import run_drift
from ri_init.config import WORKSPACE, load_config

IS_STUB = False


def _train_sample(profile: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    if cfg.get("train_sample_df") is not None:
        return cfg["train_sample_df"]
    v1 = WORKSPACE / str(profile.get("project_id")) / "models" / "v1" / "train_sample.csv"
    path = v1 if v1.is_file() else WORKSPACE / str(profile.get("project_id")) / "data" / "train_sample.csv"
    return pd.read_csv(path)


def build_bank(batches: List[pd.DataFrame], calib: Dict[str, Any], profile: Dict[str, Any],
               cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Models, staleness matrix and quality matrices. Model 0 is the champion; model j+1 is trained on batch j."""
    cfg = cfg or {}
    seed = int(profile.get("seed", 42))
    rng = np.random.default_rng(seed)
    target = profile["task"]["target"]
    cols = cara_t.feature_cols(profile)
    positive = profile.get("task", {}).get("positive_class", 1)
    window = max(1, int(profile.get("W", 1)))
    champion = joblib.load(cara_t.resolve(cfg.get("model_path") or profile["model"]["artifact"]))
    mean = np.asarray(calib["scaler_mean"], dtype=np.float32)
    scale = np.asarray(calib["scaler_scale"], dtype=np.float32)
    gamma, y_scale = float(calib["gamma"]), float(calib.get("y_scale", 1.0))
    batches = [cara_t.labelled(b, profile).reset_index(drop=True) for b in batches]
    n = len(batches)

    def emb(frame: pd.DataFrame) -> np.ndarray:
        return (cara_t.embed(champion, frame, profile) - mean) / scale

    train_sample = cara_t.labelled(_train_sample(profile, cfg), profile)
    models: List[Any] = [champion]
    for j in range(n):
        train = pd.concat(batches[max(0, j - window + 1): j + 1], ignore_index=True)
        ok = (not cara_t.is_classification(profile)) or train[target].nunique() > 1
        models.append(clone(champion).fit(train[cols], train[target]) if ok else None)

    point_idx = [np.sort(rng.choice(len(b), size=min(cara_t.D_NOM, len(b)), replace=False)) for b in batches]
    query_idx = [np.sort(rng.choice(len(b), size=min(cara_t.Q_NOM, len(b)), replace=False)) for b in batches]
    z_points = [emb(b.iloc[i]) for b, i in zip(batches, point_idx)]
    z_queries = [emb(b.iloc[i]) for b, i in zip(batches, query_idx)]
    z_train0 = emb(cara_t._sample(train_sample, cara_t.D_NOM, rng))
    loss_train0 = cara_t.row_loss(champion, cara_t._sample(train_sample, cara_t.D_NOM, np.random.default_rng(seed)),
                                  profile, y_scale)
    z_train0 = emb(cara_t._sample(train_sample, cara_t.D_NOM, np.random.default_rng(seed)))

    stal = np.full((n + 1, n), np.nan)
    acc = np.full((n + 1, n), np.nan)
    rec = np.full((n + 1, n), np.nan)
    err = np.full((n + 1, n), np.nan)
    n_pos = np.array([int((b[target] == positive).sum()) for b in batches])
    loss_pts: Dict[tuple, np.ndarray] = {}
    for m, model in enumerate(models):
        if model is None:
            continue
        for t in range(max(0, m - 1), n):
            full = cara_t.row_loss(model, batches[t], profile, y_scale)
            loss_pts[(m, t)] = full[point_idx[t]]
            err[m, t] = float(full.mean())
            acc[m, t] = 1.0 - err[m, t]
            pos_mask = (batches[t][target] == positive).to_numpy()
            rec[m, t] = float(1.0 - full[pos_mask].mean()) if pos_mask.any() else np.nan
    for t in range(n):
        w_tt = cara_t.query_weights(z_queries[t], z_points[t], gamma)
        w_t0 = cara_t.query_weights(z_queries[t], z_train0, gamma)
        stal[0, t] = float(w_tt @ loss_pts[(0, t)] - w_t0 @ loss_train0)
        for m in range(1, t + 1):  # model m was trained on batch m-1 and may serve batches >= m
            if models[m] is None:
                continue
            w_tj = cara_t.query_weights(z_queries[t], z_points[m - 1], gamma)
            stal[m, t] = float(w_tt @ loss_pts[(m, t)] - w_tj @ loss_pts[(m, m - 1)])
    return {"n": n, "stal": stal, "acc": acc, "rec": rec, "err": err, "n_pos": n_pos,
            "sizes": np.array([len(b) for b in batches]), "batches": batches,
            "train_sample": train_sample, "valid": [m is not None for m in models]}


def simulate(bank: Dict[str, Any], kappa: float, rule: Callable[[int, int, float, int], bool]) -> Dict[str, Any]:
    """Run one strategy. rule(t, current_model, staleness, batches_since_retrain) -> retrain?"""
    cur, since, staleness_cost, retrains = 0, 0, 0.0, 0
    accs, recs, schedule = [], [], []
    for t in range(bank["n"]):
        st = float(bank["stal"][cur, t])
        staleness_cost += st
        accs.append(bank["acc"][cur, t])
        recs.append(bank["rec"][cur, t])
        if rule(t, cur, st, since) and bank["valid"][t + 1]:
            cur, since, retrains = t + 1, 0, retrains + 1
            schedule.append(t)
        else:
            since += 1
    return {"retrains": retrains, "staleness_cost": staleness_cost, "total_cost_J": staleness_cost + kappa * retrains,
            "mean_accuracy": float(np.nanmean(accs)), "mean_recall": float(np.nanmean(recs)), "retrain_at": schedule}


def oracle(bank: Dict[str, Any], kappa: float) -> Dict[str, Any]:
    """Minimum-cost schedule by dynamic programming (uses the future; reference only)."""
    n, stal = bank["n"], bank["stal"]
    best = np.zeros((n + 2, n + 1))
    choice = np.zeros((n + 1, n + 1), dtype=bool)
    for t in range(n - 1, -1, -1):
        for cur in range(0, t + 1):
            if not np.isfinite(stal[cur, t]):
                best[t, cur] = np.inf
                continue
            keep = best[t + 1, cur]
            switch = kappa + best[t + 1, t + 1] if bank["valid"][t + 1] else np.inf
            choice[t, cur] = switch < keep
            best[t, cur] = stal[cur, t] + min(keep, switch)
    plan = {(t, cur): bool(choice[t, cur]) for t in range(n) for cur in range(t + 1)}
    return simulate(bank, kappa, lambda t, cur, st, since: plan[(t, cur)])


def score_inputs(bank: Dict[str, Any], calib: Dict[str, Any], profile: Dict[str, Any], t: int, cur: int,
                 cfg: Dict[str, Any]) -> Dict[str, float]:
    """(zc, zm, d) for the decision score at batch t with model `cur` serving.

    Approximations used in bank replays: zc uses the point staleness (no bootstrap
    bound); zm is a z-score of the primary metric (recall or accuracy) against the
    model's first served batch with a binomial standard error.
    """
    clip = float(cfg.get("z_clip", 3.0))
    s = max(float(calib["s"]), float(cfg.get("s_floor", 1e-6)))
    zc = float(np.clip((bank["stal"][cur, t] - float(calib["tau_star"])) / s, -clip, clip))
    use_recall = str(profile.get("primary_metric", "")).lower() == "recall"
    mat, counts = (bank["rec"], bank["n_pos"]) if use_recall else (bank["acc"], bank["sizes"])
    ref_t = max(cur, 0) if cur > 0 else None
    if ref_t is None:
        ref, n_ref = cfg.get("baseline_value"), cfg.get("baseline_n")
        if ref is None:
            ref, n_ref = float(mat[0, 0]), int(counts[0])
    else:
        ref, n_ref = float(mat[cur, ref_t]), int(counts[ref_t])
    new, n_new = float(mat[cur, t]), int(counts[t])
    se = np.sqrt(max(ref * (1 - ref), 1e-9) / max(n_ref, 1) + max(new * (1 - new), 1e-9) / max(n_new, 1))
    zm = float(np.clip((ref - new) / se, -clip, clip)) if np.isfinite(new) else 0.0
    reference = bank["train_sample"] if cur == 0 else bank["batches"][cur - 1]
    d = float(run_drift(reference, bank["batches"][t], profile)["drift_share"])
    return {"zc": zc, "zm": zm, "d": d}


def score_rule(bank: Dict[str, Any], calib: Dict[str, Any], profile: Dict[str, Any], cfg: Dict[str, Any],
               log: Optional[List[Dict[str, Any]]] = None) -> Callable[[int, int, float, int], bool]:
    """ReTrainIQ decision score, automatic zone only (HUMAN counts as keep; no verification step)."""
    from ri_decision.decision import decide

    def rule(t: int, cur: int, st: float, since: int) -> bool:
        z = score_inputs(bank, calib, profile, t, cur, cfg)
        res = decide({"status": "STABLE", "drift_share": z["d"]},
                     {"status": "OK", "basis": "measured", "staleness": st,
                      "tau_star": calib["tau_star"], "s": calib["s"]},
                     {"status": "OK", "z_m": z["zm"]}, cfg)
        if log is not None:
            log.append({"t": t, "model": cur, "P": res["P"], "zone": res["zone"], **z})
        cooldown = int(profile.get("W", 1))
        return res["zone"] == "RETRAIN" and (cur == 0 or since >= cooldown)
    return rule


def run_baselines(batches: List[pd.DataFrame], calib: Dict[str, Any], profile: Dict[str, Any], cfg: Dict[str, Any]) -> pd.DataFrame:
    """One row per strategy: retrains, costs, mean accuracy and recall, and % cost above the Oracle."""
    cfg = {**load_config(profile.get("project_id")), **(cfg or {})}
    bank = cfg.get("bank") or build_bank(batches, calib, profile, cfg)
    kappa, tau = float(calib["kappa"]), float(calib["tau_star"])
    period = int(cfg.get("period", 5))
    baseline_path = WORKSPACE / str(profile.get("project_id")) / "models" / "v1" / "baseline.json"
    if "baseline_value" not in cfg and baseline_path.is_file():
        base = json.loads(baseline_path.read_text(encoding="utf-8"))
        metric = base.get("metrics", {}).get(str(profile.get("primary_metric", "")).lower())
        if metric and str(profile.get("primary_metric", "")).lower() in ("recall", "accuracy"):
            cfg["baseline_value"] = metric["value"]
            cfg["baseline_n"] = int(bank["n_pos"].mean()) if profile.get("primary_metric") == "recall" else base.get("n")
    strategies: Dict[str, Callable[[int, int, float, int], bool]] = {
        "Never": lambda t, cur, st, since: False,
        "Always": lambda t, cur, st, since: True,
        f"Periodic({period})": lambda t, cur, st, since: (t + 1) % period == 0,
        "Markov (staleness >= kappa)": lambda t, cur, st, since: st >= kappa,
        "CARA-T (staleness >= tau*)": lambda t, cur, st, since: st >= tau,
        "ReTrainIQ score (auto zone, no verification)": score_rule(bank, calib, profile, cfg),
    }
    rows = [{"strategy": name, **simulate(bank, kappa, rule)} for name, rule in strategies.items()]
    orc = oracle(bank, kappa)
    rows.append({"strategy": "Oracle (retrospective)", **orc})
    out = pd.DataFrame(rows)
    out["pct_above_oracle"] = 100.0 * (out["total_cost_J"] - orc["total_cost_J"]) / orc["total_cost_J"]
    out["retrain_at"] = out["retrain_at"].apply(lambda v: " ".join(str(i) for i in v))
    out.attrs["kappa"], out.attrs["tau_star"], out.attrs["simulated"] = kappa, tau, True
    return out
