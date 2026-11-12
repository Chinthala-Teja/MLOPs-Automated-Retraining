from __future__ import annotations

import pytest
import pandas as pd

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

def test_contract_stubs():
    import ri_drift.drift as drift_mod
    import ri_cara.cara_t as cara_mod
    import ri_cara.proxy_eval as proxy_eval_mod
    import ri_metrics.metrics as metrics_mod
    import ri_decision.decision as decision_mod

    profile = {
        "project_id": "test_proj",
        "problem_statement": "Test problem",
        "task": {
            "type": "classification",
            "subtype": "binary",
            "target": "y",
            "positive_class": 1,
        },
        "features": {
            "numeric": ["f1", "f2"],
            "categorical": ["c1"],
        },
        "model": {
            "framework": "sklearn",
            "artifact": "model.joblib",
            "algorithm": "RandomForestClassifier",
        },
        "data": {
            "reference_path": "ref.csv",
            "holdout_path": None,
            "id_column": None,
            "ordered": True,
        },
        "primary_metric": "recall",
        "batch_size_B": 100,
        "offline_batches_N": 25,
        "W": 1,
        "seed": 42,
    }

    ref_df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "c1": ["a", "b"], "y": [0, 1]})
    new_df = pd.DataFrame({"f1": [2, 3], "f2": [4, 5], "c1": ["b", "a"], "y": [1, 1]})

    # 1. Drift
    drift_res = drift_mod.run_drift(ref_df, new_df, profile)
    assert drift_res["module"] == "drift"
    assert drift_res["status"] in ["STABLE", "DRIFT", "STUB"]
    for k in ["drift_share", "per_feature", "n_ref", "n_new"]:
        assert k in drift_res

    # 2. CARA-T: calibrate
    calib = cara_mod.calibrate("model.joblib", ref_df, profile, {})
    assert calib["module"] == "cara_t"
    assert calib["status"] in ["OK", "PENDING", "STUB"]
    for k in ["staleness", "tau_star", "kappa", "s", "decision", "n_queries", "n_points", "basis", "staleness_est", "staleness_est_lo", "decision_est"]:
        assert k in calib
    assert calib["basis"] == "measured"
    assert calib["staleness_est"] is None
    assert "proxy" in calib
    for pk in ["conf_cal", "map", "spearman"]:
        assert pk in calib["proxy"]

    # CARA-T: online_staleness (measured)
    cara_measured = cara_mod.online_staleness("model.joblib", new_df, ref_df, calib, profile)
    assert cara_measured["module"] == "cara_t"
    assert cara_measured["basis"] == "measured"
    assert cara_measured["staleness_est"] is None

    # CARA-T: estimate_staleness (estimated)
    cara_est = cara_mod.estimate_staleness("model.joblib", new_df, ref_df, calib, profile)
    assert cara_est["module"] == "cara_t"
    assert cara_est["basis"] == "estimated"
    assert cara_est["status"] == "ESTIMATED"
    assert cara_est["staleness_est"] is not None
    assert cara_est["staleness_est_lo"] is not None
    assert cara_est["decision_est"] in ["KEEP", "RETRAIN"]

    # CARA-T: proxy_eval
    eval_df = proxy_eval_mod.run_proxy_eval(profile, {})
    assert isinstance(eval_df, pd.DataFrame)
    assert len(eval_df) > 0

    # 3. Quality / Metrics
    plan = metrics_mod.default_plan(profile)
    quality_res = metrics_mod.run_metrics("model.joblib", ref_df, profile, plan)
    assert quality_res["module"] == "quality"
    assert quality_res["status"] in ["OK", "PENDING", "STUB"]
    for k in ["n", "metrics", "curves", "z_m", "degraded", "table"]:
        assert k in quality_res

    # 4. Decision: measured
    dec_res = decision_mod.decide(drift_res, calib, quality_res, {})
    assert dec_res["zone"] in ["RETRAIN", "HUMAN", "KEEP"]
    for k in ["P", "zone", "zc", "zm", "d", "reason"]:
        assert k in dec_res

    # 5. Decision: estimated cara
    dec_est_res = decision_mod.decide(drift_res, cara_est, quality_res, {})
    assert dec_est_res["zone"] in ["KEEP", "HUMAN"]
    assert "estimated from model confidence, labels pending" in dec_est_res["reason"]
    # Verify zc was computed using staleness_est_lo: (0.035 - 0.10) / 0.04 = -1.625
    expected_zc = (cara_est["staleness_est_lo"] - cara_est["tau_star"]) / cara_est["s"]
    assert pytest.approx(dec_est_res["zc"], 1e-4) == expected_zc
