from __future__ import annotations

import copy
import pytest
from ri_decision.decision import decide, what_if


def test_decision_stable_measured_gives_keep():
    drift = {"module": "drift", "status": "STABLE", "drift_share": 0.05}
    cara = {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 0.02, "tau_star": 0.10, "s": 0.05}
    quality = {"module": "quality", "status": "OK", "z_m": -1.0}

    res = decide(drift, cara, quality)
    assert res["zone"] == "KEEP"
    assert res["basis"] == "measured"
    assert res["P"] < 0.35


def test_decision_measured_staleness_well_above_tau_star_gives_retrain():
    drift = {"module": "drift", "status": "DRIFT", "drift_share": 0.2}
    cara = {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 0.25, "tau_star": 0.10, "s": 0.05}
    quality = {"module": "quality", "status": "OK", "z_m": 0.5}

    res = decide(drift, cara, quality)
    assert res["zone"] == "RETRAIN"
    assert res["basis"] == "measured"
    assert res["P"] >= 0.65
    assert "staleness" in res["reason"]


def test_decision_drift_alone_no_labels_does_not_give_retrain():
    # Drift share 1.0, but no labels (cara status PENDING, basis none)
    drift = {"module": "drift", "status": "DRIFT", "drift_share": 1.0}
    cara = {"module": "cara_t", "status": "PENDING", "basis": "none"}
    quality = {"module": "quality", "status": "PENDING"}

    res = decide(drift, cara, quality)
    assert res["zone"] == "HUMAN"
    assert res["zone"] != "RETRAIN"
    assert res["reason"] == "awaiting labels"
    assert res["basis"] == "none"
    assert res["P"] < 0.65


def test_decision_pending_gives_human_awaiting_labels():
    drift = {"module": "drift", "status": "STABLE", "drift_share": 0.0}
    cara = {"module": "cara_t", "status": "PENDING"}
    quality = {"module": "quality", "status": "PENDING"}

    res = decide(drift, cara, quality)
    assert res["zone"] == "HUMAN"
    assert res["reason"] == "awaiting labels"
    assert "quality comparison unavailable" in res["flags"]


def test_decision_estimated_high_staleness_capped_at_human():
    # Very high staleness_est_lo (e.g. 1.0 with tau_star 0.1) would give high P >= 0.65, but zone must be capped at HUMAN
    drift = {"module": "drift", "status": "DRIFT", "drift_share": 0.5}
    cara = {
        "module": "cara_t",
        "status": "ESTIMATED",
        "basis": "estimated",
        "staleness_est": 0.8,
        "staleness_est_lo": 0.7,
        "tau_star": 0.10,
        "s": 0.05,
    }
    quality = {"module": "quality", "status": "PENDING"}

    res = decide(drift, cara, quality)
    assert res["P"] >= 0.65
    assert res["zone"] == "HUMAN"
    assert "estimated from model confidence, labels pending" in res["reason"]
    assert res["basis"] == "estimated"


def test_decision_clipping_bounds():
    drift = {"module": "drift", "status": "STABLE", "drift_share": 0.0}
    # Extreme positive and negative zc and zm
    cara_hi = {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 100.0, "tau_star": 0.0, "s": 0.1}
    quality_hi = {"module": "quality", "status": "OK", "z_m": 50.0}

    res_hi = decide(drift, cara_hi, quality_hi)
    assert res_hi["zc"] == 3.0
    assert res_hi["zm"] == 3.0

    cara_lo = {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 0.0, "tau_star": 100.0, "s": 0.1}
    quality_lo = {"module": "quality", "status": "OK", "z_m": -50.0}

    res_lo = decide(drift, cara_lo, quality_lo)
    assert res_lo["zc"] == -3.0
    assert res_lo["zm"] == -3.0


def test_decision_monotonicity_property():
    # P is non-decreasing in zc, zm and d
    # Grid check
    d_vals = [0.0, 0.2, 0.5, 0.8, 1.0]
    zc_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]
    zm_vals = [-2.0, -1.0, 0.0, 1.0, 2.0]

    for zc in zc_vals:
        for zm in zm_vals:
            for i in range(len(d_vals) - 1):
                d1 = d_vals[i]
                d2 = d_vals[i + 1]
                p1 = decide(
                    {"module": "drift", "status": "STABLE", "drift_share": d1},
                    {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 0.1 + zc * 0.1, "tau_star": 0.1, "s": 0.1},
                    {"module": "quality", "status": "OK", "z_m": zm},
                )["P"]
                p2 = decide(
                    {"module": "drift", "status": "STABLE", "drift_share": d2},
                    {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 0.1 + zc * 0.1, "tau_star": 0.1, "s": 0.1},
                    {"module": "quality", "status": "OK", "z_m": zm},
                )["P"]
                assert p2 >= p1 - 1e-9


def test_decision_safety_guard_raises():
    drift = {"module": "drift", "status": "STABLE", "drift_share": 0.1}
    cara = {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 0.1, "tau_star": 0.1, "s": 0.1}
    quality = {"module": "quality", "status": "OK", "z_m": 0.0}

    # a3 = 1.0 with retrain_limit=0.65 -> retrain_logit = log(0.65/0.35) = 0.619 -> a3 (1.0) >= 0.619 -> ValueError
    with pytest.raises(ValueError) as exc:
        decide(drift, cara, quality, cfg={"a3": 1.0})
    assert "Safety guard violated" in str(exc.value)

    # keep_limit >= retrain_limit
    with pytest.raises(ValueError):
        decide(drift, cara, quality, cfg={"keep_limit": 0.70, "retrain_limit": 0.65})


def test_decision_what_if_and_no_mutation():
    drift = {"module": "drift", "status": "STABLE", "drift_share": 0.1}
    # In default config: b=0, a1=1.5, a2=1.0, a3=0.5, retrain_limit=0.65
    # Let zc = 0.0 (staleness == tau_star), zm = 0.0 -> logit = 0.5 * 0.1 = 0.05 -> P ~ 0.512 -> HUMAN
    cara = {"module": "cara_t", "status": "OK", "basis": "measured", "staleness": 0.10, "tau_star": 0.10, "s": 0.05}
    quality = {"module": "quality", "status": "OK", "z_m": 0.0}

    drift_orig = copy.deepcopy(drift)
    cara_orig = copy.deepcopy(cara)
    quality_orig = copy.deepcopy(quality)

    res_default = decide(drift, cara, quality)
    assert res_default["zone"] == "HUMAN"

    # With what_if lowering tau_star to 0.01:
    # zc = (0.10 - 0.01) / 0.05 = 1.8 -> logit = 1.5 * 1.8 + 0.05 = 2.75 -> P = 0.94 -> RETRAIN
    res_what_if = what_if(drift, cara, quality, cfg_overrides={"tau_star": 0.01})
    assert res_what_if["zone"] == "RETRAIN"

    # Inputs must not be mutated
    assert drift == drift_orig
    assert cara == cara_orig
    assert quality == quality_orig
