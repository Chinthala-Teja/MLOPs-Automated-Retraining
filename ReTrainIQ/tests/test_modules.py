"""Tests for Modules 2, 3, 4, the analysis layer and the evaluation code (real implementations)."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from sklearn.tree import DecisionTreeClassifier

import ri_cara.cara_t as cara
import ri_cara.replay_baselines as rb
import ri_decision.decision as decision
import ri_drift.drift as drift
import ri_drift.plots as plots
import ri_drift.scenarios as scenarios
import ri_metrics.metrics as metrics
from ri_analysis.chat import chat
from ri_analysis.llm import MockLLM, propose_metric_plan
from ri_analysis.summary import summarize_with_meta, template_summary, validate_summary
from ri_decision.cli import parse_interval
from ri_init.config import WORKSPACE

DEMO = WORKSPACE / "demo"
PROFILE = {
    "project_id": "unit", "task": {"type": "classification", "subtype": "binary", "target": "y", "positive_class": 1},
    "features": {"numeric": ["a", "b"], "categorical": ["c"]}, "primary_metric": "accuracy",
    "batch_size_B": 500, "seed": 42,
}


def _frame(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(shift, 1.0, n)
    return pd.DataFrame({"a": a, "b": rng.uniform(0, 1, n), "c": rng.choice(["x", "y", "z"], n),
                         "y": (a + rng.normal(0, 0.5, n) > 0).astype(int)})


# ------------------------------------------------------------------ Module 2
def test_drift_identical_samples_are_stable():
    df = _frame(3000, 1)
    res = drift.run_drift(df, df.copy(), PROFILE)
    assert res["status"] == "STABLE" and res["drift_share"] == 0.0
    assert {f["feature"] for f in res["per_feature"]} == {"a", "b", "c"}


def test_drift_flags_mean_shift_and_new_category():
    ref, new = _frame(3000, 1), _frame(3000, 2, shift=1.0)
    new.loc[: len(new) // 2, "c"] = "brand_new"
    flags = {f["feature"]: f["flag"] for f in drift.run_drift(ref, new, PROFILE)["per_feature"]}
    assert flags == {"a": True, "b": False, "c": True}


def test_validate_schema_blocks_bad_uploads():
    df = _frame(200, 3)
    assert drift.validate_schema(df, PROFILE)["status"] == "VALID"
    assert drift.validate_schema(scenarios.make_invalid_upload(df, "missing_column"), PROFILE)["status"] == "BLOCKED"
    assert drift.validate_schema(scenarios.make_invalid_upload(df, "bad_dtype"), PROFILE)["status"] == "BLOCKED"
    ref_stats = drift.fit_reference(df, PROFILE)
    wild = drift.validate_schema(scenarios.make_invalid_upload(df, "out_of_range"), PROFILE, ref_stats)
    assert wild["status"] == "VALID" and wild["out_of_range"]["a"] == len(df)


def test_scenarios_change_only_what_they_claim():
    df = _frame(2000, 4)
    cov = scenarios.inject_covariate_shift(df, PROFILE, "a", 1.0, 1000)
    assert cov.iloc[:1000].equals(df.iloc[:1000])
    assert cov["a"].iloc[1000:].mean() > df["a"].iloc[1000:].mean() + 0.5
    original_rows = set(map(tuple, df.round(9).to_numpy().tolist()))
    assert all(tuple(r) in original_rows for r in cov.round(9).to_numpy().tolist())  # labels travel with rows
    con = scenarios.inject_concept_shift(df, PROFILE, "a > 0", 1.0, 0)
    assert con[["a", "b", "c"]].equals(df[["a", "b", "c"]])
    assert (con.loc[df["a"] > 0, "y"] != df.loc[df["a"] > 0, "y"]).all()
    assert (con.loc[df["a"] <= 0, "y"] == df.loc[df["a"] <= 0, "y"]).all()
    assert scenarios.hold_back_labels(df, 0.25, target="y")["y"].isna().mean() == pytest.approx(0.25)
    stream, info = scenarios.build_scenario_stream([df, df, df, df], "concept", PROFILE)
    assert info["onset_batch"] == 2 and stream[0].equals(df) and not stream[3].equals(df)


def test_plots_return_figures():
    ref, new = _frame(500, 1), _frame(500, 2, shift=1.0)
    res = drift.run_drift(ref, new, PROFILE)
    assert isinstance(plots.drift_bar(res), go.Figure)
    assert isinstance(plots.distribution_overlay(ref, new, "a", "numeric"), go.Figure)
    assert isinstance(plots.distribution_overlay(ref, new, "c", "categorical"), go.Figure)
    assert plots.top_drifted(res, 1)[0]["feature"] == "a"


# ------------------------------------------------------------------ Module 4
def test_metrics_perfect_model_and_registry(tmp_path: Path):
    df = _frame(600, 5)
    prof = {**PROFILE, "features": {"numeric": ["a", "b"], "categorical": []}}
    path = tmp_path / "tree.joblib"
    joblib.dump(DecisionTreeClassifier(random_state=0).fit(df[["a", "b"]], df["y"]), path)
    res = metrics.run_metrics(str(path), df, prof, metrics.default_plan(prof), n_boot=50)
    assert res["status"] == "OK" and res["metrics"]["accuracy"]["value"] == 1.0
    assert res["metrics"]["accuracy"]["se"] == pytest.approx(0.0)
    assert metrics.run_metrics(str(path), df.drop(columns=["y"]), prof, ["accuracy"])["status"] == "PENDING"
    with pytest.raises(ValueError):
        metrics.validate_plan(["accuracy", "made_up_metric"])
    with pytest.raises(ValueError):
        metrics.validate_plan(["rmse"], prof)
    llm = MockLLM(json_response={"metrics": ["f1", "invented"]})
    assert propose_metric_plan(prof, llm) == metrics.default_plan(prof)  # bad plan -> default plan
    assert propose_metric_plan(prof, MockLLM(json_response={"metrics": ["f1"]})) == ["accuracy", "f1"]


def test_metrics_compare_detects_degradation():
    base = {"status": "OK", "metrics": {"accuracy": {"value": 0.90, "se": 0.01}, "log_loss": {"value": 0.3, "se": 0.01}}}
    cur = {"status": "OK", "metrics": {"accuracy": {"value": 0.80, "se": 0.01}, "log_loss": {"value": 0.5, "se": 0.01}}}
    worse = metrics.compare(base, cur, "accuracy")
    assert worse["z_m"] > 5 and worse["degraded"] and worse["table"][0]["delta"] == pytest.approx(-0.10)
    assert metrics.compare(base, cur, "log_loss")["z_m"] > 5  # lower-is-better metric: sign flipped
    assert metrics.compare({"status": "PENDING"}, cur, "accuracy")["status"] == "NO_BASELINE"


# ------------------------------------------------------------------ Module 3 (demo project)
@pytest.fixture(scope="module")
def demo():
    profile = json.loads((DEMO / "profile.json").read_text(encoding="utf-8"))
    profile["model"]["artifact"] = str(DEMO / "models" / "v1" / "model.joblib")
    small = {**profile, "offline_batches_N": 10}
    history = pd.read_csv(DEMO / "data" / "reference.csv").tail(10_000)
    calib = cara.calibrate(profile["model"]["artifact"], history, small, {})
    return {"profile": small, "history": history, "calib": calib,
            "train": pd.read_csv(DEMO / "data" / "reference.csv").sample(2000, random_state=42),
            "stable": pd.read_csv(DEMO / "data" / "stream" / "batch_000.csv"),
            "shifted": pd.read_csv(DEMO / "data" / "stream" / "batch_030.csv")}


def test_cara_calibration_is_finite_and_reproducible(demo):
    calib = demo["calib"]
    assert calib["status"] == "OK" and np.isfinite(calib["tau_star"]) and calib["kappa"] > 0 and calib["s"] > 0
    assert calib["proxy"] and {"conf_cal", "map", "spearman"} <= set(calib["proxy"])
    json.dumps(calib)  # must be JSON-serialisable
    again = cara.calibrate(demo["profile"]["model"]["artifact"], demo["history"], demo["profile"], {})
    assert again["tau_star"] == calib["tau_star"] and again["kappa"] == calib["kappa"]
    assert cara.tau_for_kappa(calib, 1.0)["tau_star"] == pytest.approx(calib["tau_star"])
    assert cara.tau_for_kappa(calib, 4.0)["tau_star"] >= calib["tau_star"]


def test_cara_too_little_history_is_reported(demo):
    res = cara.calibrate(demo["profile"]["model"]["artifact"], demo["history"].head(900), demo["profile"], {})
    assert res["status"] == "INSUFFICIENT_HISTORY"
    assert cara.online_staleness("x", demo["stable"], demo["train"], res, demo["profile"])["status"] == "PENDING"


def test_staleness_rises_after_concept_shift_and_decisions_follow(demo):
    path, prof, calib = demo["profile"]["model"]["artifact"], demo["profile"], demo["calib"]
    stable = cara.online_staleness(path, demo["stable"], demo["train"], calib, prof)
    shifted = cara.online_staleness(path, demo["shifted"], demo["train"], calib, prof)
    assert stable["status"] == shifted["status"] == "OK" and stable["basis"] == "measured"
    assert shifted["staleness"] > stable["staleness"] + 5 * calib["s"]
    assert stable["staleness_lo"] < stable["staleness"] and shifted["decision"] == "RETRAIN"
    quiet = {"status": "STABLE", "drift_share": 0.0}
    no_change = {"status": "OK", "z_m": 0.0}
    assert decision.decide(quiet, shifted, no_change)["zone"] == "RETRAIN"
    assert decision.decide(quiet, stable, no_change)["zone"] in ("KEEP", "HUMAN")

    unlabelled = demo["shifted"].drop(columns=["delayed"])
    assert cara.online_staleness(path, unlabelled, demo["train"], calib, prof)["status"] == "PENDING"
    est = cara.estimate_staleness(path, unlabelled, demo["train"], calib, prof)
    assert est["status"] == "ESTIMATED" and est["basis"] == "estimated"
    assert est["staleness_est_lo"] < est["staleness_est"] and est["decision_est"] in ("KEEP", "RETRAIN")
    assert decision.decide(quiet, est, {"status": "PENDING"})["zone"] in ("KEEP", "HUMAN")  # never automatic


def test_baseline_strategies_on_a_small_bank(demo):
    files = sorted((DEMO / "data" / "stream").glob("batch_*.csv"))[22:30]
    batches = [pd.read_csv(f) for f in files]
    table = rb.run_baselines(batches, demo["calib"], demo["profile"], {"train_sample_df": demo["train"]})
    by = table.set_index("strategy")
    assert by.loc["Never", "retrains"] == 0 and by.loc["Always", "retrains"] == len(batches)
    assert by.loc["Oracle (retrospective)", "total_cost_J"] <= table["total_cost_J"].min() + 1e-6
    assert (table["pct_above_oracle"] >= -1e-6).all()


# ------------------------------------------------------------------ analysis layer
REPORT = {
    "report_id": 7, "zone": "HUMAN", "action": "pending",
    "drift": {"status": "DRIFT", "drift_share": 0.2, "per_feature": [
        {"feature": "a", "type": "numeric", "psi": 0.412, "ks_stat": 0.3, "p_value": 0.0001, "flag": True},
        {"feature": "b", "type": "numeric", "psi": 0.01, "ks_stat": 0.02, "p_value": 0.7, "flag": False}]},
    "cara": {"status": "OK", "basis": "measured", "staleness": 70.25, "tau_star": 69.5, "kappa": 67.0, "s": 2.4,
             "decision": "RETRAIN"},
    "quality": {"status": "OK", "primary_metric": "recall", "z_m": 0.8, "degraded": False,
                "metrics": {"recall": {"value": 0.61, "se": 0.02}},
                "table": [{"metric": "recall", "reference": 0.64, "current": 0.61, "delta": -0.03}]},
    "decision": {"P": 0.58, "zone": "HUMAN", "basis": "measured", "reason": "Decision is HUMAN."},
}


def test_template_summary_passes_its_own_number_check():
    text = template_summary(REPORT)
    assert "Recommended action: HUMAN REVIEW" in text and "0.412" in text
    assert validate_summary(text, REPORT) == []


def test_validate_summary_catches_invented_numbers_and_wrong_advice():
    assert any("93.7" in p for p in validate_summary("Recall fell to 93.7%.\nRecommended action: HUMAN REVIEW", REPORT))
    assert any("does not match" in p for p in validate_summary("Staleness is 70.25.\nRecommended action: RETRAIN", REPORT))
    assert any("does not record" in p for p in validate_summary(
        "The model was retrained.\nRecommended action: HUMAN REVIEW", REPORT))
    assert validate_summary("Score 58% and staleness 70.3 against 69.5.\nRecommended action: HUMAN REVIEW", REPORT) == []


def test_llm_summary_is_used_only_when_it_passes_the_check():
    good = "Staleness is 70.25 against 69.5; the score is 0.58.\nRecommended action: HUMAN REVIEW"
    assert summarize_with_meta(REPORT, MockLLM(text_response=good))["source"] == "llm"
    bad = MockLLM(text_response="Accuracy is 99.9 percent.\nRecommended action: RETRAIN")
    res = summarize_with_meta(REPORT, bad)
    assert res["source"].startswith("template") and res["text"] == template_summary(REPORT)
    assert len(bad.calls) == 2  # one retry before falling back
    assert summarize_with_meta(REPORT, None, use_env_client=False)["source"] == "template"


def test_chat_answers_from_tools_only():
    tools = {
        "get_report": lambda check_id=None: REPORT,
        "get_history": lambda n=10: [{"check_id": 7, "zone": "HUMAN", "P": 0.58}],
        "get_drift_table": lambda check_id=None: REPORT["drift"]["per_feature"],
        "get_versions": lambda: [{"version": "v1", "status": "active", "trained_on": "reference"}],
        "what_if": lambda overrides, check_id=None: {"P": 0.9, "zone": "RETRAIN", "overrides": overrides},
    }
    assert "a (PSI 0.412)" in chat("which features drifted?", "unit", tools, use_env_client=False)["answer"]
    assert "RETRAIN" in chat("what if kappa is 0.5?", "unit", tools, use_env_client=False)["answer"]
    assert "v1" in chat("which versions exist?", "unit", tools, use_env_client=False)["answer"]
    lying = MockLLM(text_response="The score is 0.99.")
    assert chat("why?", "unit", tools, llm=lying)["mode"] == "rules (llm failed number check)"
    honest = MockLLM(text_response="The score is 0.58 and the zone is HUMAN.")
    assert chat("why?", "unit", tools, llm=honest)["mode"] == "llm"
    empty = {**tools, "get_report": lambda check_id=None: None}
    assert "No checks" in chat("why?", "unit", empty, use_env_client=False)["answer"]


def test_parse_interval():
    assert (parse_interval("5h"), parse_interval("1d"), parse_interval("1w"), parse_interval("90")) == (
        18000, 86400, 604800, 90)


def test_frontend_renders_without_errors():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "frontend" / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    for page in ("Monitor", "Report", "Chat"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
