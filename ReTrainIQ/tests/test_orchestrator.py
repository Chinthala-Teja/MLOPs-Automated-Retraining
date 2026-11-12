from __future__ import annotations

import io
from contextlib import redirect_stdout
import json
import tempfile
from pathlib import Path
import pandas as pd

from ri_init.config import WORKSPACE
import ri_init.store as store
import ri_decision.orchestrator as orch
import ri_decision.cli as cli_mod


def test_orchestrator_setup_and_replay_10_batches():
    demo_dir = WORKSPACE / "demo"
    assert (demo_dir / "profile.json").exists()

    with open(demo_dir / "profile.json", "r", encoding="utf-8") as f:
        profile = json.load(f)

    # 1. Setup
    setup_res = orch.setup_project(profile)
    assert setup_res["status"] == "OK"

    assert (demo_dir / "state.json").exists()
    assert (demo_dir / "baseline.json").exists()
    assert (demo_dir / "calib.json").exists()
    assert (demo_dir / "data" / "train_sample.csv").exists()

    # 2. Replay 10 batches
    stream_dir = demo_dir / "data" / "stream"
    batch_files = sorted(list(stream_dir.glob("batch_*.csv")))[:10]
    assert len(batch_files) == 10

    total_batches = 0
    for f in batch_files:
        res = orch.on_new_data("demo", f)
        assert res["status"] in ("OK", "DUPLICATE")
        total_batches += res["batches_processed"]

    reports = store.get_reports("demo")
    assert len(reports) >= 10


def test_buffer_two_half_batches_produce_one_batch():
    demo_dir = WORKSPACE / "demo"

    # Clean buffer and state
    state = orch._load_state("demo")
    state["buffer_rows"] = 0
    state["uploaded_hashes"] = []
    orch._save_state("demo", state)
    (demo_dir / "data" / "buffer.csv").unlink(missing_ok=True)

    # Create two 1000-row files with distinct timestamps/noise to avoid identical hash
    sample_df = pd.read_csv(demo_dir / "data" / "stream" / "batch_000.csv")
    half1 = sample_df.iloc[:1000].copy()
    half2 = sample_df.iloc[1000:2000].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = Path(tmpdir) / "half1.csv"
        p2 = Path(tmpdir) / "half2.csv"
        half1.to_csv(p1, index=False)
        half2.to_csv(p2, index=False)

        res1 = orch.on_new_data("demo", p1)
        assert res1["batches_processed"] == 0
        assert res1["buffer_rows"] == 1000

        res2 = orch.on_new_data("demo", p2)
        assert res2["batches_processed"] == 1
        assert res2["buffer_rows"] == 0


def test_unlabelled_batch_ends_in_human_and_request_labels():
    demo_dir = WORKSPACE / "demo"
    with open(demo_dir / "profile.json", "r", encoding="utf-8") as f:
        profile = json.load(f)
    target_col = profile["task"]["target"]

    # Clear buffer and hashes
    (demo_dir / "data" / "buffer.csv").unlink(missing_ok=True)
    state = orch._load_state("demo")
    state["buffer_rows"] = 0
    state["uploaded_hashes"] = []
    orch._save_state("demo", state)

    batch_df = pd.read_csv(demo_dir / "data" / "stream" / "batch_001.csv").drop(columns=[target_col])
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir) / "unlab.csv"
        batch_df.to_csv(p, index=False)

        res = orch.on_new_data("demo", p)
        assert res["batches_processed"] == 1
        rep = res["reports"][0]
        assert rep["zone"] == "HUMAN"
        assert rep["labelled"] is False
        assert rep["decision"].get("alert") == "request labels"


def test_attach_labels_stores_proxy_error_and_is_idempotent():
    demo_dir = WORKSPACE / "demo"
    with open(demo_dir / "profile.json", "r", encoding="utf-8") as f:
        profile = json.load(f)
    target_col = profile["task"]["target"]

    # Clear buffer and hashes
    (demo_dir / "data" / "buffer.csv").unlink(missing_ok=True)
    state = orch._load_state("demo")
    state["buffer_rows"] = 0
    state["uploaded_hashes"] = []
    orch._save_state("demo", state)

    full_df = pd.read_csv(demo_dir / "data" / "stream" / "batch_002.csv")
    unlab_df = full_df.drop(columns=[target_col])
    labels_df = full_df[[target_col]]

    with tempfile.TemporaryDirectory() as tmpdir:
        p_data = Path(tmpdir) / "unlab2.csv"
        p_labels = Path(tmpdir) / "labels2.csv"
        unlab_df.to_csv(p_data, index=False)
        labels_df.to_csv(p_labels, index=False)

        res = orch.on_new_data("demo", p_data)
        rep = res["reports"][0]
        batch_id = rep["batch_id"]

        # 1. Attach labels
        attach_res = orch.attach_labels("demo", batch_id, p_labels)
        assert attach_res["status"] == "OK"
        assert "proxy_error" in attach_res
        assert store.get_batch(batch_id)["n_labelled"] == len(full_df)

        # Verify update in store
        updated_rep = store.get_report(rep["report_id"])
        score = json.loads(updated_rep["score_json"])
        assert "proxy_error" in score

        # 2. Idempotency: calling twice does not create new decisions
        attach_res2 = orch.attach_labels("demo", batch_id, p_labels)
        assert attach_res2["status"] == "ALREADY_ATTACHED"


def test_duplicate_upload_deduplication():
    demo_dir = WORKSPACE / "demo"
    state = orch._load_state("demo")
    state["uploaded_hashes"] = []
    orch._save_state("demo", state)

    batch_path = demo_dir / "data" / "stream" / "batch_003.csv"
    res1 = orch.on_new_data("demo", batch_path)
    assert res1["status"] == "OK"

    res2 = orch.on_new_data("demo", batch_path)
    assert res2["status"] == "DUPLICATE"
    assert res2["batches_processed"] == 0


def test_missing_feature_column_returns_blocked():
    demo_dir = WORKSPACE / "demo"
    bad_df = pd.read_csv(demo_dir / "data" / "stream" / "batch_000.csv").drop(columns=["dep_delay"])

    # Temporarily make drift.validate_schema return BLOCKED
    import ri_drift.drift as drift_mod
    orig_validate = drift_mod.validate_schema
    drift_mod.validate_schema = lambda df, prof: {"status": "BLOCKED", "valid": False, "missing_columns": ["dep_delay"]}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "bad.csv"
            bad_df.to_csv(p, index=False)
            res = orch.on_new_data("demo", p)
            assert res["status"] == "BLOCKED"
    finally:
        drift_mod.validate_schema = orig_validate


def test_cooldown_downgrades_retrain_to_human():
    demo_dir = WORKSPACE / "demo"
    state = orch._load_state("demo")
    state["cooldown_left"] = 2
    state["buffer_rows"] = 0
    state["uploaded_hashes"] = []
    orch._save_state("demo", state)
    (demo_dir / "data" / "buffer.csv").unlink(missing_ok=True)

    # Force decision to return RETRAIN
    import ri_decision.decision as decision_mod
    orig_decide = decision_mod.decide
    decision_mod.decide = lambda d, c, q, cfg=None: {
        "P": 0.90,
        "zone": "RETRAIN",
        "zc": 2.0,
        "zm": 1.0,
        "d": 0.5,
        "reason": "Forced retrain",
    }

    try:
        sample_path = demo_dir / "data" / "stream" / "batch_004.csv"
        res = orch.on_new_data("demo", sample_path)
        assert res["batches_processed"] == 1
        rep = res["reports"][0]
        assert rep["zone"] == "HUMAN"
        assert rep["decision"]["reason"] == "cooldown"
    finally:
        decision_mod.decide = orig_decide


def test_resolve_human_writes_decision():
    reports = store.get_reports("demo")
    assert len(reports) > 0
    rep_id = reports[0]["id"]

    res = orch.resolve_human("demo", rep_id, "KEEP")
    assert res["status"] == "OK"
    assert res["choice"] == "KEEP"

    updated = store.get_report(rep_id)
    assert updated["zone"] == "KEEP"


def test_cli_replay_label_lag_attachments():
    demo_dir = WORKSPACE / "demo"
    with open(demo_dir / "profile.json", "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Re-setup to have clean state
    orch.setup_project(profile)
    stream_dir = demo_dir / "data" / "stream"

    # Capture output of 6-batch replay with label-lag 2
    f_out = io.StringIO()
    with redirect_stdout(f_out):
        cli_mod.run_replay("demo", stream_dir, limit=6, label_lag=2)
    output = f_out.getvalue()

    # 6 batches sent with lag 2:
    # batch 0, batch 1 -> in queue (len 2)
    # batch 2 -> queue len 3 > 2 -> attaches batch 0 (1st attachment)
    # batch 3 -> attaches batch 1 (2nd attachment)
    # batch 4 -> attaches batch 2 (3rd attachment)
    # batch 5 -> attaches batch 3 (4th attachment)
    # Total attachments: 4
    assert "labels attached: 4" in output
    assert "batches sent: 6" in output
