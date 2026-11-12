"""cara_t.py - Module 3: CARA-T staleness (Mahadevan and Mathioudakis, 2024) plus CARA-T-L.

Measured staleness (needs labels), as in the paper:
    psi(q, D, M)   = (1/|D|) * sum_{(x,y) in D} sim(q, x) * loss(M, x, y)
    Psi(Q, D, M)   = sum_{q in Q} psi(q, D, M)
    PsiBar(t, t')  = Psi(Q_t, D_t, M_t') - Psi(Q_t, D_t', M_t')      (same queries Q_t)
    sim(q, x)      = exp(-gamma * ||q - x||^2),  loss = 0-1 loss
    RETRAIN if PsiBar >= tau*, where tau* minimises the offline strategy cost.

CARA-T-L (label-free estimate, used while labels are pending):
    the 0-1 loss is replaced by an expected loss 1 - g(conf(x)), where g is an
    isotonic map from model confidence to P(correct) fitted on held-out data.
    The proxy is mapped onto the measured scale with a second isotonic map
    fitted offline, and a lower bound (estimate - q90 of offline residuals) is
    compared with tau*.

Adaptation notes (this is not an exact reproduction of the paper):
  * similarity uses the champion's fitted preprocessing output, standardised;
  * gamma = 1 / median squared pairwise distance of 1,000 sampled points;
  * Psi is scaled to a nominal 500 queries so batches of any size are comparable;
  * tau* is found by exhaustive search over the distinct offline values;
  * regression uses |error| / std(y) capped at 1 as the loss; the label-free
    estimate needs predict_proba, so it is classification only.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression

from ri_init.config import ROOT

IS_STUB = False

Q_NOM = 500
D_NOM = 2000
MIN_BATCH_ROWS = 500
MIN_BATCHES = 10


# ----------------------------------------------------------------------------
# helpers (also used by replay_baselines, proxy_eval and calibrate_score)
# ----------------------------------------------------------------------------
def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def feature_cols(profile: Dict[str, Any]) -> List[str]:
    feats = profile.get("features", {})
    return list(feats.get("numeric", [])) + list(feats.get("categorical", []))


def is_classification(profile: Dict[str, Any]) -> bool:
    return profile.get("task", {}).get("type", "classification") == "classification"


def labelled(df: pd.DataFrame, profile: Dict[str, Any]) -> pd.DataFrame:
    target = profile.get("task", {}).get("target")
    if df is None or target not in df.columns:
        return pd.DataFrame(columns=list(df.columns) if df is not None else [])
    return df.loc[df[target].notna()]


def embed(model: Any, frame: pd.DataFrame, profile: Dict[str, Any]) -> np.ndarray:
    """Feature space for the similarity kernel: the pipeline's preprocessing output."""
    cols = feature_cols(profile)
    if hasattr(model, "steps") and len(model.steps) > 1:
        z = model[:-1].transform(frame[cols])
    else:
        z = frame[profile.get("features", {}).get("numeric", [])].to_numpy()
    if sparse.issparse(z):
        z = z.toarray()
    return np.nan_to_num(np.asarray(z, dtype=np.float32))


def fit_scaler(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = z.mean(axis=0)
    scale = z.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def median_gamma(z: np.ndarray, rng: np.random.Generator) -> float:
    idx = rng.choice(len(z), size=min(1000, len(z)), replace=False)
    d2 = sq_dists(z[idx], z[idx])
    off = d2[np.triu_indices_from(d2, k=1)]
    med = float(np.median(off)) if len(off) else 0.0
    return 1.0 / med if med > 0 else 1.0


def query_weights(zq: np.ndarray, zd: np.ndarray, gamma: float) -> np.ndarray:
    """w such that Psi(Q, D, M) = w . loss(M, D); scaled to Q_NOM queries."""
    kernel = np.exp(-gamma * sq_dists(zq, zd))
    return kernel.sum(axis=0) / max(len(zd), 1) * (Q_NOM / max(len(zq), 1))


def bootstrap_se(w_new: np.ndarray, loss_new: np.ndarray, w_train: np.ndarray, loss_train: np.ndarray,
                 seed: int, n_boot: int = 200) -> float:
    """Standard error of relative staleness from resampling the data points of both terms."""
    rng = np.random.default_rng(seed + 1)
    a, b = w_new * loss_new, w_train * loss_train
    reps = [a[rng.integers(0, len(a), len(a))].sum() - b[rng.integers(0, len(b), len(b))].sum()
            for _ in range(n_boot)]
    return float(np.std(reps, ddof=1))


def row_loss(model: Any, frame: pd.DataFrame, profile: Dict[str, Any], y_scale: float = 1.0) -> np.ndarray:
    target = profile["task"]["target"]
    pred = np.asarray(model.predict(frame[feature_cols(profile)]))
    truth = frame[target].to_numpy()
    if is_classification(profile):
        return (pred != truth).astype(np.float64)
    return np.minimum(np.abs(pred.astype(float) - truth.astype(float)) / max(y_scale, 1e-12), 1.0)


def confidence(model: Any, frame: pd.DataFrame, profile: Dict[str, Any]) -> np.ndarray:
    return np.asarray(model.predict_proba(frame[feature_cols(profile)])).max(axis=1)


def fit_conf_cal(model: Any, frame: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, List[float]]:
    """Isotonic map from confidence to P(prediction correct), fitted on labelled held-out rows."""
    conf = confidence(model, frame, profile)
    correct = 1.0 - row_loss(model, frame, profile)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(conf, correct)
    return {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]}


def expected_loss(model: Any, frame: pd.DataFrame, profile: Dict[str, Any], conf_cal: Dict[str, Any]) -> np.ndarray:
    conf = confidence(model, frame, profile)
    return 1.0 - np.interp(conf, conf_cal["x"], conf_cal["y"])


def apply_map(proxy: float, mapping: Dict[str, Any]) -> float:
    """Isotonic map inside the calibrated range; slope-1 extrapolation outside it.

    The proxy is in the same units as measured staleness when the confidence
    calibration holds, so identity extrapolation is the neutral choice.
    """
    xs, ys = mapping["x"], mapping["y"]
    if proxy < xs[0]:
        return float(ys[0] + (proxy - xs[0]))
    if proxy > xs[-1]:
        return float(ys[-1] + (proxy - xs[-1]))
    return float(np.interp(proxy, xs, ys))


def select_tau(cost: np.ndarray, kappa: float, s: float) -> Dict[str, Any]:
    """Simulate Algorithm 1 on the offline cost matrix for every candidate threshold.

    cost[j, t] = PsiBar(t, j) for j < t (NaN where unavailable). The candidate
    'max + s' stands in for +infinity: no offline value reaches it, so its cost
    equals never retraining, and it stays usable (finite) online.
    """
    n = cost.shape[0]
    vals = cost[np.triu_indices(n, k=1)]
    vals = np.unique(vals[np.isfinite(vals)])
    never_tau = float(vals.max() + max(s, 1e-9))
    # Every threshold between two neighbouring offline values behaves the same
    # offline, so each such plateau is represented by its midpoint.
    mids = [float((a + b) / 2.0) for a, b in zip(vals[:-1], vals[1:])]
    candidates = [float(vals.min() - max(s, 1e-9))] + mids + [never_tau]
    curve: List[List[float]] = []
    for tau in candidates:
        cur, total, retrains = 0, 0.0, 0
        for t in range(1, n):
            c = cost[cur, t]
            can_switch = t == n - 1 or bool(np.isfinite(cost[t, t + 1:]).any())
            if np.isfinite(c) and c >= tau and can_switch:
                cur, total, retrains = t, total + kappa, retrains + 1
            else:
                total += float(c) if np.isfinite(c) else 0.0
        curve.append([tau, total, retrains])
    best = min(c[1] for c in curve)
    tau_star = max(c[0] for c in curve if c[1] <= best + 1e-12)
    return {
        "tau_star": float(tau_star),
        "tau_extrapolated": bool(tau_star == never_tau),
        "cost_curve": curve,
        "best_cost": float(best),
    }


def tau_for_kappa(calib: Dict[str, Any], kappa_multiplier: float) -> Dict[str, Any]:
    """Re-derive tau* for another kappa multiplier from the stored offline cost matrix."""
    cost = np.array([[np.nan if v is None else v for v in row] for row in calib["cost_matrix"]], dtype=float)
    kappa = float(calib["kappa_base"]) * float(kappa_multiplier)
    sel = select_tau(cost, kappa, float(calib["s"]))
    return {"kappa": kappa, "tau_star": sel["tau_star"], "tau_extrapolated": sel["tau_extrapolated"]}


def _result(status: str, calib: Dict[str, Any], **kw: Any) -> Dict[str, Any]:
    base = {
        "module": "cara_t",
        "status": status,
        "staleness": 0.0,
        "tau_star": float(calib.get("tau_star", 0.0) or 0.0),
        "kappa": float(calib.get("kappa", 0.0) or 0.0),
        "s": float(calib.get("s", 0.0) or 0.0),
        "decision": "KEEP",
        "n_queries": 0,
        "n_points": 0,
        "basis": "none",
        "staleness_est": None,
        "staleness_est_lo": None,
        "decision_est": None,
    }
    base.update(kw)
    return base


def _sample(frame: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= n:
        return frame
    return frame.iloc[np.sort(rng.choice(len(frame), size=n, replace=False))]


def plan_batches(n_rows: int, n_requested: int) -> Tuple[Optional[int], int, List[str]]:
    """Choose the number of offline batches N and their size."""
    warnings: List[str] = []
    n = int(n_requested)
    if n_rows // n < MIN_BATCH_ROWS:
        n = n_rows // MIN_BATCH_ROWS
        warnings.append(f"offline batches reduced to {n} so each has at least {MIN_BATCH_ROWS} rows")
    if n < MIN_BATCHES:
        return None, 0, [f"history has {n_rows} labelled rows; at least {MIN_BATCHES * MIN_BATCH_ROWS} are needed"]
    return n, n_rows // n, warnings


# ----------------------------------------------------------------------------
# offline calibration
# ----------------------------------------------------------------------------
def calibrate(model_path: str, history_df: pd.DataFrame, profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Offline CARA-T step: cost matrix, kappa, tau*, s and the label-free proxy maps."""
    cfg = cfg or {}
    seed = int(profile.get("seed", 42))
    rng = np.random.default_rng(seed)
    hist = labelled(history_df, profile).reset_index(drop=True)
    n, bsz, warnings = plan_batches(len(hist), int(profile.get("offline_batches_N", 25)))
    if n is None:
        return _result("INSUFFICIENT_HISTORY", {}, warnings=warnings, proxy=None)
    if not profile.get("data", {}).get("ordered", True):
        warnings.append("data is not time-ordered; offline batches are arbitrary slices")

    model = joblib.load(resolve(model_path))
    target = profile["task"]["target"]
    cols = feature_cols(profile)
    classification = is_classification(profile)
    hist = hist.iloc[len(hist) - n * bsz:].reset_index(drop=True)
    batches = [hist.iloc[i * bsz:(i + 1) * bsz] for i in range(n)]
    y_scale = 1.0 if classification else float(hist[target].astype(float).std() or 1.0)

    mean, scale = fit_scaler(embed(model, _sample(hist, 5000, rng), profile))

    def emb(frame: pd.DataFrame) -> np.ndarray:
        return (embed(model, frame, profile) - mean) / scale

    gamma = float(cfg.get("gamma") or median_gamma(emb(_sample(hist, 1000, rng)), rng))
    window = max(1, int(profile.get("W", 1)))

    # one model per offline batch, trained with the same recipe used online
    models: List[Any] = []
    for t in range(n):
        train = pd.concat(batches[max(0, t - window + 1): t + 1], ignore_index=True)
        if classification and train[target].nunique() < 2:
            models.append(None)
            continue
        models.append(clone(model).fit(train[cols], train[target]))

    points = [_sample(b, D_NOM, rng) for b in batches]
    queries = [_sample(b, Q_NOM, rng) for b in batches]
    z_points = [emb(p) for p in points]
    z_queries = [emb(q) for q in queries]

    has_proba = classification and hasattr(model, "predict_proba")
    loss = [[None] * n for _ in range(n)]
    exp_loss = [[None] * n for _ in range(n)]
    for j in range(n):
        if models[j] is None:
            continue
        cal_j = fit_conf_cal(models[j], batches[j + 1], profile) if has_proba and j + 1 < n else None
        for t in range(j, n):
            loss[j][t] = row_loss(models[j], points[t], profile, y_scale)
            if cal_j is not None:
                exp_loss[j][t] = expected_loss(models[j], points[t], profile, cal_j)

    cost = np.full((n, n), np.nan)
    proxy = np.full((n, n), np.nan)
    for t in range(1, n):
        w_tt = query_weights(z_queries[t], z_points[t], gamma)
        for j in range(t):
            if models[j] is None:
                continue
            w_tj = query_weights(z_queries[t], z_points[j], gamma)
            cost[j, t] = float(w_tt @ loss[j][t] - w_tj @ loss[j][j])
            if exp_loss[j][t] is not None:
                proxy[j, t] = float(w_tt @ exp_loss[j][t] - w_tj @ exp_loss[j][j])

    upper = cost[np.triu_indices(n, k=1)]
    upper = upper[np.isfinite(upper)]
    s = float(upper.std())
    positive = upper[upper > 0]
    multiplier = float(cfg.get("kappa_multiplier", 1.0))
    out_common = {
        "gamma": gamma, "N": n, "batch_size": bsz, "W": window, "y_scale": y_scale,
        "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist(),
        "offline_psibar": [float(v) for v in upper], "warnings": warnings,
        "cost_matrix": [[None if not np.isfinite(v) else float(v) for v in row] for row in cost],
    }
    if cfg.get("kappa") is not None:
        kappa_base = float(cfg["kappa"])
    elif len(positive):
        kappa_base = float(np.median(positive))
    else:
        return _result("NEEDS_KAPPA", {"s": s}, proxy=None, **out_common)
    kappa = kappa_base * multiplier
    sel = select_tau(cost, kappa, s)
    if sel["tau_extrapolated"]:
        warnings.append("never retraining was cheapest offline; tau* is set one standard deviation "
                        "above the largest offline staleness")

    proxy_out: Optional[Dict[str, Any]] = None
    if has_proba:
        pairs = [(proxy[j, t], cost[j, t]) for j in range(n) for t in range(j + 2, n)
                 if np.isfinite(proxy[j, t]) and np.isfinite(cost[j, t])]
        holdout_path = profile.get("data", {}).get("holdout_path")
        holdout = None
        if cfg.get("holdout_df") is not None:
            holdout = cfg["holdout_df"]
        elif holdout_path and resolve(holdout_path).is_file():
            hp = resolve(holdout_path)
            holdout = pd.read_parquet(hp) if hp.suffix.lower() == ".parquet" else pd.read_csv(hp)
        if holdout is None:
            warnings.append("no holdout: label-free staleness estimate is disabled")
        elif len(pairs) < 10:
            warnings.append("too few offline pairs: label-free staleness estimate is disabled")
        else:
            px = np.array([p[0] for p in pairs])
            py = np.array([p[1] for p in pairs])
            iso = IsotonicRegression(out_of_bounds="clip").fit(px, py)
            resid = np.abs(iso.predict(px) - py)
            rho = stats.spearmanr(px, py).statistic if len(np.unique(px)) > 1 else float("nan")
            proxy_out = {
                "conf_cal": fit_conf_cal(model, labelled(holdout, profile), profile),
                "map": {"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_],
                        "resid_q90": float(np.quantile(resid, 0.9))},
                "spearman": None if not np.isfinite(rho) else float(rho),
                "n_pairs": len(pairs),
            }

    return _result(
        "OK", {"tau_star": sel["tau_star"], "kappa": kappa, "s": s},
        basis="measured", n_queries=int(len(queries[0])), n_points=int(len(points[0])),
        staleness=float(np.mean(upper)), kappa_base=kappa_base, kappa_multiplier=multiplier,
        tau_extrapolated=sel["tau_extrapolated"], cost_curve=sel["cost_curve"],
        offline_cost=sel["best_cost"], proxy=proxy_out, seed=seed, **out_common,
    )


# ----------------------------------------------------------------------------
# online
# ----------------------------------------------------------------------------
def _online_parts(
    model_path: str, new_df: pd.DataFrame, train_sample_df: pd.DataFrame,
    calib: Dict[str, Any], profile: Dict[str, Any], seed: int, points_new: pd.DataFrame,
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    model = joblib.load(resolve(model_path))
    mean = np.asarray(calib["scaler_mean"], dtype=np.float32)
    scale = np.asarray(calib["scaler_scale"], dtype=np.float32)
    gamma = float(calib["gamma"])
    queries = _sample(new_df, Q_NOM, rng)
    d_new = _sample(points_new, D_NOM, rng)
    d_train = _sample(labelled(train_sample_df, profile), D_NOM, rng)
    zq = (embed(model, queries, profile) - mean) / scale
    w_new = query_weights(zq, (embed(model, d_new, profile) - mean) / scale, gamma)
    w_train = query_weights(zq, (embed(model, d_train, profile) - mean) / scale, gamma)
    return model, d_new, d_train, w_new, w_train, len(queries)


def online_staleness(model_path: str, new_df: pd.DataFrame, train_sample_df: pd.DataFrame, calib: Dict[str, Any], profile: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """Measured relative staleness of the active model on a labelled batch."""
    if calib.get("status") != "OK":
        return _result("PENDING", calib, reason=f"calibration status {calib.get('status')}")
    lab = labelled(new_df, profile)
    need = max(30, int(0.9 * min(int(profile.get("batch_size_B", len(new_df))), len(new_df))))
    if len(lab) < need or len(labelled(train_sample_df, profile)) == 0:
        return _result("PENDING", calib, reason=f"{len(lab)} labelled rows; {need} needed")
    model, d_new, d_train, w_new, w_train, nq = _online_parts(
        model_path, new_df, train_sample_df, calib, profile, seed, lab)
    y_scale = float(calib.get("y_scale", 1.0))
    loss_new = row_loss(model, d_new, profile, y_scale)
    loss_train = row_loss(model, d_train, profile, y_scale)
    current, reference = float(w_new @ loss_new), float(w_train @ loss_train)
    staleness = current - reference
    se = bootstrap_se(w_new, loss_new, w_train, loss_train, seed)
    return _result(
        "OK", calib, staleness=staleness, basis="measured",
        staleness_se=se, staleness_lo=staleness - 1.645 * se,
        decision="RETRAIN" if staleness >= float(calib["tau_star"]) else "KEEP",
        n_queries=nq, n_points=int(len(d_new)), psi_current=current, psi_reference=reference,
    )


def estimate_staleness(model_path: str, new_df: pd.DataFrame, train_sample_df: pd.DataFrame, calib: Dict[str, Any], profile: Dict[str, Any], seed: int = 42) -> Dict[str, Any]:
    """CARA-T-L: label-free staleness estimate from calibrated model confidence."""
    proxy_cal = calib.get("proxy")
    if calib.get("status") != "OK" or not proxy_cal or len(new_df) == 0:
        return _result("PENDING", calib, reason="label-free estimate unavailable for this project")
    model, d_new, d_train, w_new, w_train, nq = _online_parts(
        model_path, new_df, train_sample_df, calib, profile, seed, new_df)
    raw = float(w_new @ expected_loss(model, d_new, profile, proxy_cal["conf_cal"])
                - w_train @ expected_loss(model, d_train, profile, proxy_cal["conf_cal"]))
    est = apply_map(raw, proxy_cal["map"])
    lo = est - float(proxy_cal["map"]["resid_q90"])
    tau = float(calib["tau_star"])
    return _result(
        "ESTIMATED", calib, staleness=est, basis="estimated",
        staleness_est=est, staleness_est_lo=lo, proxy_raw=raw,
        decision="RETRAIN" if est >= tau else "KEEP",
        decision_est="RETRAIN" if lo >= tau else "KEEP",
        n_queries=nq, n_points=int(len(d_new)),
    )
