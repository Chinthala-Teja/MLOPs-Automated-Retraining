from __future__ import annotations

import pytest

def test_imports():
    import importlib
    modules = [
        "ri_init.config",
        "ri_init.setup_env",
        "ri_init.store",
        "ri_init.worker",
        "ri_init.make_demo_project",
        "ri_profile.understand",
        "ri_drift.drift",
        "ri_drift.plots",
        "ri_drift.scenarios",
        "ri_cara.cara_t",
        "ri_cara.replay_baselines",
        "ri_cara.proxy_eval",
        "ri_metrics.metrics",
        "ri_analysis.llm",
        "ri_analysis.summary",
        "ri_analysis.chat",
        "ri_decision.decision",
        "ri_decision.retrain",
        "ri_decision.orchestrator",
        "ri_decision.cli",
        "ri_decision.calibrate_score",
        "ri_decision.sensitivity",
    ]
    for m in modules:
        mod = importlib.import_module(m)
        assert mod is not None

def test_no_module_is_a_stub():
    """Every module in the contract is implemented (IS_STUB is False everywhere)."""
    import importlib

    for name in [
        "ri_drift.drift", "ri_drift.plots", "ri_drift.scenarios", "ri_cara.cara_t",
        "ri_cara.replay_baselines", "ri_cara.proxy_eval", "ri_metrics.metrics",
        "ri_analysis.llm", "ri_analysis.summary", "ri_analysis.chat", "ri_decision.decision",
        "ri_decision.retrain", "ri_decision.orchestrator", "ri_decision.cli",
        "ri_decision.calibrate_score", "ri_decision.sensitivity",
    ]:
        assert getattr(importlib.import_module(name), "IS_STUB") is False, name


def test_estimated_basis_is_capped_at_human():
    import ri_decision.decision as decision_mod

    cara_est = {"module": "cara_t", "status": "ESTIMATED", "basis": "estimated", "staleness": 1.0,
                "staleness_est": 1.0, "staleness_est_lo": 0.9, "tau_star": 0.10, "s": 0.04}
    res = decision_mod.decide({"status": "STABLE", "drift_share": 0.0}, cara_est,
                              {"status": "PENDING", "z_m": 0.0}, {})
    assert res["zone"] == "HUMAN"
    assert "estimated from model confidence, labels pending" in res["reason"]
    assert pytest.approx(res["zc"]) == 3.0
