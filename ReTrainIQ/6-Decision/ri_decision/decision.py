"""decision.py
Stage 2 Decision engine: computes decision score P, determines zone (KEEP, HUMAN, RETRAIN),
analyzes drivers, and performs what-if analysis.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from ri_init.config import DEFAULT_CONFIG

IS_STUB = False


def decide(
    drift: Dict[str, Any],
    cara: Dict[str, Any],
    quality: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute decision score P and classify zone.
    Returns:
      {'P', 'zone', 'zc', 'zm', 'd', 'reason', 'basis', 'terms', 'flags'}
    """
    # Merge config
    config = dict(DEFAULT_CONFIG)
    if cfg:
        config.update(cfg)

    b = float(config["b"])
    a1 = float(config["a1"])
    a2 = float(config["a2"])
    a3 = float(config["a3"])
    retrain_limit = float(config["retrain_limit"])
    keep_limit = float(config["keep_limit"])
    z_clip = float(config["z_clip"])
    s_floor = float(config["s_floor"])

    # Safety guard checks
    if keep_limit >= retrain_limit:
        raise ValueError(f"keep_limit ({keep_limit}) must be less than retrain_limit ({retrain_limit})")

    # Safety guard: drift alone must never reach retrain limit
    # Max logit difference achievable by drift alone is a3 * 1.0 (since max d is 1.0)
    # Target logit for retrain_limit: log(retrain_limit / (1 - retrain_limit))
    retrain_logit = math.log(retrain_limit / (1.0 - retrain_limit))
    if a3 >= retrain_logit:
        raise ValueError(
            f"Safety guard violated: a3 ({a3}) >= log(retrain_limit/(1-retrain_limit)) ({retrain_logit:.4f}). "
            f"Drift alone must never be able to reach the retrain limit."
        )

    flags: List[str] = []

    # Check for stub status in inputs
    if any(m.get("status") == "STUB" for m in (drift, cara, quality)):
        flags.append("stub inputs")

    # 1. Drift evidence: d
    drift_status = drift.get("status")
    if drift_status in ("STABLE", "DRIFT", "STUB"):
        d = float(drift.get("drift_share", 0.0))
    else:
        d = 0.0

    # 2. CARA evidence: zc and basis
    cara_status = cara.get("status")
    cara_basis = cara.get("basis", "measured")

    if cara_status in ("OK", "STUB") and cara_basis == "measured":
        staleness = float(cara.get("staleness", 0.0))
        tau_star = float(cara.get("tau_star", 0.0))
        s = float(cara.get("s", 0.0))
        denom = max(s, s_floor)
        raw_zc = (staleness - tau_star) / denom
        zc = max(-z_clip, min(z_clip, raw_zc))
        basis = "measured"
    elif cara_status == "ESTIMATED" or cara_basis == "estimated":
        staleness_est_lo = float(cara.get("staleness_est_lo", 0.0) or 0.0)
        tau_star = float(cara.get("tau_star", 0.0))
        s = float(cara.get("s", 0.0))
        denom = max(s, s_floor)
        raw_zc = (staleness_est_lo - tau_star) / denom
        zc = max(-z_clip, min(z_clip, raw_zc))
        basis = "estimated"
    else:  # 'PENDING', 'NEEDS_KAPPA', missing or unrecognised
        zc = 0.0
        basis = "none"

    # 3. Quality evidence: zm
    quality_status = quality.get("status")
    if quality_status in ("OK", "STUB"):
        raw_zm = float(quality.get("z_m", 0.0))
        zm = max(-z_clip, min(z_clip, raw_zm))
    else:
        zm = 0.0
        flags.append("quality comparison unavailable")

    # Terms
    term_b = b
    term_zc = a1 * zc
    term_zm = a2 * zm
    term_d = a3 * d
    terms = {
        "b": term_b,
        "a1_zc": term_zc,
        "a2_zm": term_zm,
        "a3_d": term_d,
    }

    # Logit and P
    logit = term_b + term_zc + term_zm + term_d
    # Numerically stable sigmoid
    if logit >= 0:
        P = 1.0 / (1.0 + math.exp(-logit))
    else:
        exp_z = math.exp(logit)
        P = exp_z / (1.0 + exp_z)

    # Determine raw zone from P
    if P >= retrain_limit:
        raw_zone = "RETRAIN"
    elif P <= keep_limit:
        raw_zone = "KEEP"
    else:
        raw_zone = "HUMAN"

    # Special rules for basis
    if basis == "none":
        zone = "HUMAN"
        reason = "awaiting labels"
    elif basis == "estimated":
        # Cap zone at HUMAN
        zone = "HUMAN" if raw_zone == "RETRAIN" else raw_zone
        reason = "estimated from model confidence, labels pending"
    else:
        zone = raw_zone
        # Driver analysis: name term with largest absolute contribution among a1_zc, a2_zm, a3_d
        driver_map = {
            "staleness (CARA-T)": abs(term_zc),
            "performance degradation": abs(term_zm),
            "data drift": abs(term_d),
        }
        main_driver = max(driver_map.items(), key=lambda x: x[1])[0]
        reason = f"Decision is {zone} primarily driven by {main_driver}."

    return {
        "P": P,
        "zone": zone,
        "zc": zc,
        "zm": zm,
        "d": d,
        "reason": reason,
        "basis": basis,
        "terms": terms,
        "flags": flags,
    }


def what_if(
    drift: Dict[str, Any],
    cara: Dict[str, Any],
    quality: Dict[str, Any],
    cfg_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate decision under hypothetical configuration or parameter overrides.
    Inputs are deep-copied / never mutated.
    Overrides may contain 'tau_star' to replace cara['tau_star'].
    """
    drift_copy = dict(drift)
    cara_copy = dict(cara)
    quality_copy = dict(quality)

    cfg = dict(cfg_overrides or {})
    if "tau_star" in cfg:
        cara_copy["tau_star"] = cfg.pop("tau_star")

    return decide(drift_copy, cara_copy, quality_copy, cfg=cfg)
