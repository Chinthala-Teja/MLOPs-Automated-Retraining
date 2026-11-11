from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

def test_imports():
    import ri_init.config
    import ri_init.setup_env
    import ri_init.store
    import ri_init.worker
    import ri_init.make_demo_project

    import ri_profile.understand

    import ri_drift.drift
    import ri_drift.plots
    import ri_drift.scenarios

    import ri_cara.cara_t
    import ri_cara.replay_baselines
    import ri_cara.proxy_eval

    import ri_metrics.metrics

    import ri_analysis.llm
    import ri_analysis.summary
    import ri_analysis.chat

    import ri_decision.decision
    import ri_decision.retrain
    import ri_decision.orchestrator
    import ri_decision.cli
    import ri_decision.calibrate_score
    import ri_decision.sensitivity

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
    assert dec_est_res["zc"] == cara_est["staleness_est_lo"]
