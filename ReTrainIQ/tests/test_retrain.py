from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from ri_init.config import WORKSPACE
import ri_init.store as store
import ri_decision.orchestrator as orchestrator
import ri_decision.retrain as retrain
import ri_decision.cli as cli
import ri_decision.decision as decision


@pytest.fixture
def retrain_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    source_dir = WORKSPACE / "demo"
    project_dir = tmp_path / "workspace" / "demo"
    data_dir = project_dir / "data"
    model_dir = project_dir / "models" / "v1"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    reference = pd.read_csv(source_dir / "data" / "reference.csv")
    holdout = pd.read_csv(source_dir / "data" / "holdout.csv")
    reference_path = data_dir / "reference.csv"
    holdout_path = data_dir / "holdout.csv"
    reference.to_csv(reference_path, index=False)
    holdout.to_csv(holdout_path, index=False)

    with open(source_dir / "profile.json", "r", encoding="utf-8") as file:
        profile = json.load(file)
    target = profile["task"]["target"]
    champion = clone(joblib.load(source_dir / "models" / "v1" / "model.joblib"))
    champion.fit(
        reference.iloc[:500].drop(columns=[target]),
        reference.iloc[:500][target],
    )
    champion_path = model_dir / "model.joblib"
    joblib.dump(champion, champion_path)

    profile["model"]["artifact"] = str(champion_path)
    profile["data"]["reference_path"] = str(reference_path)
    profile["data"]["holdout_path"] = str(holdout_path)
    profile["W"] = 1
    profile["offline_batches_N"] = 10
    profile_path = project_dir / "profile.json"
    with open(profile_path, "w", encoding="utf-8") as file:
        json.dump(profile, file, indent=2)

    stream_dir = data_dir / "stream"
    stream_dir.mkdir()
    for batch_name in ("batch_025.csv", "batch_030.csv"):
        shutil.copy2(source_dir / "data" / "stream" / batch_name, stream_dir / batch_name)

    monkeypatch.setattr(store, "DB_PATH", tmp_path / "retrain-test.db")
    monkeypatch.setattr(retrain, "WORKSPACE", tmp_path / "workspace")
    monkeypatch.setattr(orchestrator, "WORKSPACE", tmp_path / "workspace")
    setup = orchestrator.setup_project(profile)
    assert setup["status"] == "OK"
    return {
        "root": tmp_path,
        "project_dir": project_dir,
        "profile": profile,
        "reference": reference,
        "holdout": holdout,
        "stream_dir": stream_dir,
    }


def register_batch(
    project: dict[str, object],
    frame: pd.DataFrame,
    filename: str = "batch_025.csv",
) -> tuple[int, Path]:
    project_dir = project["project_dir"]
    assert isinstance(project_dir, Path)
    target = project["profile"]["task"]["target"]
    batch_path = project_dir / "data" / "batches" / filename
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(batch_path, index=False)
    batch_id = store.add_batch(
        "demo",
        str(batch_path.resolve()),
        len(frame),
        int(frame[target].notna().sum()),
    )
    return batch_id, batch_path


def demo_batch(project: dict[str, object], batch_name: str = "batch_025.csv") -> pd.DataFrame:
    stream_dir = project["stream_dir"]
    assert isinstance(stream_dir, Path)
    return pd.read_csv(stream_dir / batch_name)


def test_qualifying_demo_batch_promotes_v2_and_sets_cooldown(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    register_batch(project, demo_batch(project))

    result = retrain.run_retrain_cycle("demo", trigger="automatic")

    assert result["status"] == "PROMOTED"
    assert result["version"] == "v2"
    assert result["verify"]["verified"] is True
    assert "kept unchanged" in result["warnings"][0]
    assert store.get_active_version("demo")["version"] == "v2"
    state = json.loads((project["project_dir"] / "state.json").read_text(encoding="utf-8"))
    assert state["cooldown_left"] == project["profile"]["W"]


def test_postshift_candidate_passes_because_outdated_holdout_is_waived(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    profile = project["profile"]
    batch = demo_batch(project, "batch_030.csv")
    batch_id, _ = register_batch(project, batch, "postshift_030.csv")
    training = retrain.build_training_data("demo", profile)
    assert training["batch_ids"] == [batch_id]

    candidate_path = project["project_dir"] / "models" / "postshift" / "model.joblib"
    retrain.train_candidate(
        profile,
        training["train_df"],
        candidate_path,
        seed=profile["seed"],
    )
    result = retrain.verify(
        candidate_path,
        project["holdout"],
        profile,
        champion_path=profile["model"]["artifact"],
        v_df=training["v_df"],
    )

    # After a concept shift the champion is clearly worse on the newest slice than
    # on the reference holdout, so the holdout no longer describes current data.
    # The gate is waived (and reported); the candidate must still win on the slice.
    assert result["holdout_drop"] > 0.02
    assert result["holdout_gate"] == "waived"
    assert result["gain_metric"] == "accuracy" and result["gain_lb"] > 0
    assert result["verified"] is True

    # On a stable batch the holdout still applies.
    stable = demo_batch(project, "batch_025.csv")
    stable_result = retrain.verify(
        candidate_path,
        project["holdout"],
        profile,
        champion_path=profile["model"]["artifact"],
        v_df=stable.iloc[1600:],
    )
    assert stable_result["holdout_gate"] == "applied"
    assert stable_result["verified"] is False


def test_shuffled_labels_rejected_and_training_seconds_recorded(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    shuffled = demo_batch(project)
    target = project["profile"]["task"]["target"]
    shuffled[target] = shuffled[target].sample(frac=1, random_state=7).to_numpy()
    register_batch(project, shuffled, "shuffled.csv")

    result = retrain.run_retrain_cycle("demo", trigger="automatic")

    assert result["status"] == "VERIFY_FAILED"
    rejected = [v for v in store.get_versions("demo") if v["version"] == result["version"]]
    assert rejected[-1]["status"] == "rejected"
    rejected_model_dir = project["project_dir"] / "models" / result["version"]
    assert (rejected_model_dir / "calib.json").is_file()
    assert (rejected_model_dir / "baseline.json").is_file()
    assert (rejected_model_dir / "train_sample.csv").is_file()
    assert (rejected_model_dir / "reference_stats.json").is_file()
    connection = store.get_connection()
    try:
        rows = connection.execute(
            "SELECT note FROM decisions WHERE action = 'VERIFY_FAILED'"
        ).fetchall()
    finally:
        connection.close()
    assert rows
    assert f"training_seconds={result['seconds']:.6f}" in rows[-1]["note"]


def test_atomic_promote_failure_preserves_existing_active_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate.joblib"
    active = tmp_path / "active" / "model.joblib"
    candidate.write_bytes(b"candidate")
    active.parent.mkdir()
    active.write_bytes(b"old-model")

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("simulated copy failure")

    monkeypatch.setattr(retrain.shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="simulated copy failure"):
        retrain.promote(candidate, active)

    assert active.read_bytes() == b"old-model"


def test_two_retrain_cycles_register_v2_then_v3(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    batch = demo_batch(project)
    register_batch(project, batch, "first.csv")
    first = retrain.run_retrain_cycle("demo", trigger="automatic")
    register_batch(project, batch, "second.csv")
    second = retrain.run_retrain_cycle("demo", trigger="automatic")

    assert first["status"] == second["status"] == "PROMOTED"
    assert first["version"] == "v2"
    assert second["version"] == "v3"


def test_calibration_uses_last_ten_labelled_batches_and_saves_bundle(
    retrain_project: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = retrain_project
    batch = demo_batch(project)
    for index in range(12):
        marked = batch.copy()
        marked["history_marker"] = index
        register_batch(project, marked, f"history_{index:02d}.csv")

    original_run = retrain.run_in_worker
    calibration_history: dict[str, pd.DataFrame] = {}

    def capture_calibration(
        callable_path: str,
        kwargs: dict[str, object],
        timeout: int = 900,
    ) -> dict[str, object]:
        if callable_path == "ri_cara.cara_t:calibrate":
            calibration_history["frame"] = kwargs["history_df"]
        return original_run(callable_path, kwargs, timeout)

    monkeypatch.setattr(retrain, "run_in_worker", capture_calibration)
    result = retrain.run_retrain_cycle("demo", trigger="automatic")

    assert result["status"] == "PROMOTED"
    history = calibration_history["frame"]
    assert len(history) == 20_000
    assert history["history_marker"].unique().tolist() == list(range(2, 12))
    bundle = project["project_dir"] / "models" / "v2"
    for asset in (
        "model.joblib",
        "calib.json",
        "baseline.json",
        "train_sample.csv",
        "reference_stats.json",
    ):
        assert (bundle / asset).is_file()


def test_human_retrain_uses_stored_batch_path_and_registers_active_version(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    batch_id, _ = register_batch(project, demo_batch(project), "batch_099.csv")
    report_id = store.add_report(
        "demo",
        batch_id,
        "{}",
        "{}",
        "{}",
        "{}",
        "HUMAN",
        "Human retrain requested.",
    )

    result = orchestrator.resolve_human("demo", report_id, "RETRAIN")

    assert result["status"] == "OK"
    assert result["action"] == "PROMOTED"
    assert store.get_active_version("demo")["version"] == "v2"
    versions = {version["version"]: version for version in store.get_versions("demo")}
    assert versions["v2"]["status"] == "active"
    assert Path(versions["v2"]["path"]).name == "model.joblib"


def test_automatic_retrain_and_cli_versions_rollback(
    retrain_project: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = retrain_project
    incoming = project["project_dir"] / "data" / "incoming.csv"
    demo_batch(project).to_csv(incoming, index=False)
    monkeypatch.setattr(
        decision,
        "decide",
        lambda drift, cara, quality, cfg=None: {
            "P": 0.9,
            "zone": "RETRAIN",
            "zc": 2.0,
            "zm": 1.0,
            "d": 0.5,
            "reason": "test retrain",
        },
    )
    monkeypatch.setattr(
        retrain,
        "shadow_check",
        lambda project_id, batch_df: {"rollback_suggested": True},
    )

    result = orchestrator.on_new_data("demo", incoming)
    assert result["reports"][0]["action"] == "PROMOTED"
    assert result["reports"][0]["decision"]["alert"] == "rollback suggested"
    assert store.get_active_version("demo")["version"] == "v2"

    monkeypatch.setattr(sys, "argv", ["ri_decision.cli", "versions", "--project", "demo"])
    cli.main()
    versions_output = capsys.readouterr().out
    assert "Version" in versions_output
    assert "v2" in versions_output

    monkeypatch.setattr(sys, "argv", ["ri_decision.cli", "rollback", "--project", "demo"])
    cli.main()
    assert "active_version=v1" in capsys.readouterr().out
    assert store.get_active_version("demo")["version"] == "v1"


def test_shadow_check_scores_three_labelled_batches(
    retrain_project: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = retrain_project
    register_batch(project, demo_batch(project))
    result = retrain.run_retrain_cycle("demo", trigger="automatic")
    assert result["status"] == "PROMOTED"
    original_run = retrain.run_in_worker
    checked: list[str] = []

    def compare_models(
        callable_path: str,
        kwargs: dict[str, object],
        timeout: int = 900,
    ) -> dict[str, object]:
        if callable_path == "ri_decision.retrain:_shadow_worker":
            checked.append(callable_path)
            return {"active_metric": 0.5, "previous_metric": 0.6}
        return original_run(callable_path, kwargs, timeout)

    monkeypatch.setattr(retrain, "run_in_worker", compare_models)
    batch = demo_batch(project)
    first = retrain.shadow_check("demo", batch)
    second = retrain.shadow_check("demo", batch)
    third = retrain.shadow_check("demo", batch)

    assert len(checked) == 3
    assert first["rollback_suggested"] is True
    assert second["rollback_suggested"] is True
    assert third["rollback_suggested"] is True
    assert third["shadow_left"] == 0


def test_rollback_restores_previous_model_and_bundle(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    project_dir = project["project_dir"]
    v1_dir = project_dir / "models" / "v1"
    assets = {
        "calib.json": {"bundle": "v1-calibration"},
        "baseline.json": {"bundle": "v1-baseline"},
        "reference_stats.json": {"bundle": "v1-reference"},
    }
    for filename, value in assets.items():
        (project_dir / filename).write_text(json.dumps(value), encoding="utf-8")
    pd.DataFrame({"bundle": ["v1-training-sample"]}).to_csv(
        project_dir / "data" / "train_sample.csv",
        index=False,
    )
    retrain.save_bundle("demo", "v1")

    for index in range(10):
        register_batch(project, demo_batch(project), f"history_{index:02d}.csv")
    promoted = retrain.run_retrain_cycle("demo", trigger="automatic")
    assert promoted["status"] == "PROMOTED"
    assert json.loads((project_dir / "calib.json").read_text(encoding="utf-8")) != assets[
        "calib.json"
    ]

    result = orchestrator.rollback("demo")

    assert result["status"] == "ROLLED_BACK"
    assert store.get_active_version("demo")["version"] == "v1"
    for filename, expected in assets.items():
        assert json.loads((project_dir / filename).read_text(encoding="utf-8")) == expected
        assert json.loads((v1_dir / filename).read_text(encoding="utf-8")) == expected
    assert pd.read_csv(project_dir / "data" / "train_sample.csv").iloc[0, 0] == (
        "v1-training-sample"
    )
    connection = store.get_connection()
    try:
        rollback_row = connection.execute(
            "SELECT \"by\" FROM decisions WHERE action = 'ROLLBACK'"
        ).fetchone()
    finally:
        connection.close()
    assert rollback_row["by"] == "human"


def test_tiny_history_returns_insufficient_data(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    tiny = demo_batch(project).iloc[:30].copy()
    register_batch(project, tiny, "tiny.csv")

    result = retrain.build_training_data("demo", project["profile"])

    assert result["status"] == "INSUFFICIENT_DATA"
    assert "500" in result["reason"]


def test_same_seed_produces_same_candidate_predictions(
    retrain_project: dict[str, object],
) -> None:
    project = retrain_project
    profile = project["profile"]
    training = demo_batch(project).iloc[:1600].copy()
    first_path = project["project_dir"] / "models" / "seed-a.joblib"
    second_path = project["project_dir"] / "models" / "seed-b.joblib"

    first = retrain.train_candidate(profile, training, first_path, seed=13)
    second = retrain.train_candidate(profile, training, second_path, seed=13)

    assert first["status"] == second["status"] == "OK"
    model_a = joblib.load(first_path)
    model_b = joblib.load(second_path)
    features = training.drop(columns=[profile["task"]["target"]])
    np.testing.assert_array_equal(model_a.predict(features), model_b.predict(features))
