"""Candidate training, verification, promotion, and rollback."""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics
from sklearn.base import clone

from ri_init.config import ROOT, WORKSPACE
import ri_init.store as store
from ri_init.worker import run_in_worker
import ri_drift.drift as drift_mod
import ri_metrics.metrics as metrics_mod

IS_STUB = False

_BUNDLE_FILES = {
    "calib.json": "calib.json",
    "baseline.json": "baseline.json",
    "train_sample.csv": "data/train_sample.csv",
    "reference_stats.json": "reference_stats.json",
}


def _resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _read_frame(path: str | Path) -> pd.DataFrame:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Data file not found: {resolved}")
    if resolved.suffix.lower() == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved)


def _feature_columns(profile: Dict[str, Any]) -> List[str]:
    features = profile.get("features", {})
    columns = list(features.get("numeric", [])) + list(features.get("categorical", []))
    if not columns:
        raise ValueError("Profile must define at least one feature column.")
    return columns


def _labelled_rows(
    frame: pd.DataFrame,
    profile: Dict[str, Any],
) -> tuple[pd.DataFrame, pd.Series]:
    target = profile.get("task", {}).get("target")
    if not target or target not in frame.columns:
        raise ValueError("Training or evaluation data does not contain the target column.")
    columns = _feature_columns(profile)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Data is missing feature columns: {missing}")
    labelled = frame.loc[frame[target].notna()]
    if labelled.empty:
        raise ValueError("No labelled rows are available.")
    return labelled[columns], labelled[target]


def _prediction_scores(model: Any, features: pd.DataFrame, profile: Dict[str, Any]) -> np.ndarray:
    positive_class = profile.get("task", {}).get("positive_class", 1)
    classes = list(getattr(model, "classes_", []))
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features))
        if probabilities.ndim == 1:
            return probabilities
        if len(classes) == 2:
            if positive_class not in classes:
                raise ValueError(f"Positive class {positive_class!r} is not in model classes.")
            return probabilities[:, classes.index(positive_class)]
        return probabilities
    if hasattr(model, "decision_function"):
        values = np.asarray(model.decision_function(features))
        if values.ndim == 1 and len(classes) == 2 and classes[1] != positive_class:
            return -values
        return values
    raise ValueError("ROC AUC verification requires predict_proba or decision_function.")


def _score_predictions(
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray],
    metric_name: str,
    profile: Dict[str, Any],
    classes: Optional[List[Any]] = None,
) -> float:
    positive_class = profile.get("task", {}).get("positive_class", 1)
    binary = profile.get("task", {}).get("subtype") == "binary"
    if metric_name == "accuracy":
        score = metrics.accuracy_score(truth, predictions)
    elif metric_name == "precision":
        score = metrics.precision_score(
            truth,
            predictions,
            pos_label=positive_class,
            average="binary" if binary else "weighted",
            zero_division=0,
        )
    elif metric_name == "recall":
        score = metrics.recall_score(
            truth,
            predictions,
            pos_label=positive_class,
            average="binary" if binary else "weighted",
            zero_division=0,
        )
    elif metric_name == "f1":
        score = metrics.f1_score(
            truth,
            predictions,
            pos_label=positive_class,
            average="binary" if binary else "weighted",
            zero_division=0,
        )
    elif metric_name == "roc_auc":
        if probabilities is None:
            raise ValueError("ROC AUC verification requires model scores.")
        if binary:
            score = metrics.roc_auc_score(
                np.asarray(truth) == positive_class,
                probabilities,
            )
        else:
            score = metrics.roc_auc_score(
                truth,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="weighted",
            )
    else:
        raise ValueError(
            f"Unsupported primary metric {metric_name!r}; expected accuracy, "
            "precision, recall, f1, or roc_auc."
        )
    return float(score)


def _evaluate_model(
    model: Any,
    frame: pd.DataFrame,
    profile: Dict[str, Any],
    metric_name: str,
) -> tuple[float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    features, truth_series = _labelled_rows(frame, profile)
    truth = truth_series.to_numpy()
    predictions = np.asarray(model.predict(features))
    probabilities = (
        _prediction_scores(model, features, profile)
        if metric_name == "roc_auc"
        else None
    )
    score = _score_predictions(
        truth,
        predictions,
        probabilities,
        metric_name,
        profile,
        list(getattr(model, "classes_", [])),
    )
    return score, truth, predictions, probabilities


def _train_worker(
    profile: Dict[str, Any],
    train_df: pd.DataFrame,
    out_path: str,
    seed: Optional[int] = None,
    champion_path: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()
    model_path = champion_path or profile.get("model", {}).get("artifact")
    if not model_path:
        raise ValueError("No champion model artifact was provided.")
    champion = joblib.load(_resolve_path(model_path))
    pipeline = clone(champion)
    if seed is not None:
        random_state_params = {
            key: seed
            for key in pipeline.get_params(deep=True)
            if key == "random_state" or key.endswith("__random_state")
        }
        if random_state_params:
            pipeline.set_params(**random_state_params)
    features, labels = _labelled_rows(train_df, profile)
    if len(features) == 0:
        raise ValueError("Candidate training data contains no labelled rows.")
    pipeline.fit(features, labels)
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    return {
        "status": "OK",
        "candidate_path": str(destination),
        "trained_rows": len(features),
        "seconds": float(time.perf_counter() - started),
        "sklearn_version": sklearn.__version__,
    }


def train_candidate(
    profile: Dict[str, Any],
    train_df: pd.DataFrame,
    out_path: str | Path,
    seed: Optional[int] = None,
    *,
    champion_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Clone and fit the champion inside the isolated worker process."""
    return run_in_worker(
        "ri_decision.retrain:_train_worker",
        {
            "profile": profile,
            "train_df": train_df,
            "out_path": str(out_path),
            "seed": seed,
            "champion_path": str(champion_path) if champion_path else None,
        },
    )


def build_training_data(project_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
    """Build training and newest-batch verification slices from labelled batches."""
    window = int(profile.get("W", 1))
    if window < 1:
        raise ValueError("Profile W must be at least one.")
    target = profile.get("task", {}).get("target")
    if not target:
        raise ValueError("Profile must define task.target.")

    labelled_records = [
        batch
        for batch in store.get_batches(project_id)
        if int(batch.get("n_rows") or 0) > 0
        and int(batch.get("n_labelled") or 0) / int(batch["n_rows"]) >= 0.9
    ]
    selected = labelled_records[-window:]
    if not selected:
        return {
            "status": "INSUFFICIENT_DATA",
            "reason": "No batches meet the 90% labelled threshold.",
            "train_df": pd.DataFrame(),
            "v_df": pd.DataFrame(),
            "batch_ids": [],
        }

    frames: List[pd.DataFrame] = []
    for batch in selected:
        file_path = batch.get("file")
        if not file_path:
            raise ValueError(f"Batch {batch['id']} has no stored file path.")
        frames.append(_read_frame(file_path))

    newest = frames[-1]
    split_at = int(len(newest) * 0.8)
    if split_at <= 0 or split_at >= len(newest):
        train_latest = newest.iloc[:0]
        verify_df = newest
    else:
        train_latest = newest.iloc[:split_at]
        verify_df = newest.iloc[split_at:]
    train_parts = [*frames[:-1], train_latest]
    train_df = pd.concat(train_parts, ignore_index=True)
    train_df = train_df.loc[train_df[target].notna()].reset_index(drop=True)
    verify_df = verify_df.loc[verify_df[target].notna()].reset_index(drop=True)

    reason = None
    if len(train_df) < 500:
        reason = f"Training data has {len(train_df)} labelled rows; at least 500 are required."
    elif len(verify_df) < 200:
        reason = (
            f"Verification slice has {len(verify_df)} labelled rows; "
            "at least 200 are required."
        )
    return {
        "status": "INSUFFICIENT_DATA" if reason else "OK",
        "reason": reason,
        "train_df": train_df,
        "v_df": verify_df,
        "batch_ids": [int(batch["id"]) for batch in selected],
        "training_rows": len(train_df),
        "verification_rows": len(verify_df),
    }


def _verify_worker(
    candidate_path: str,
    holdout_df: pd.DataFrame,
    profile: Dict[str, Any],
    champion_path: Optional[str] = None,
    v_df: Optional[pd.DataFrame] = None,
    margin: float = 0.0,
    tol: float = 0.02,
    recall_guard: float = 0.02,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    if n_boot < 1:
        raise ValueError("n_boot must be at least one.")
    metric_name = str(profile.get("primary_metric", "recall")).lower()
    candidate = joblib.load(_resolve_path(candidate_path))
    candidate_holdout, _, _, _ = _evaluate_model(
        candidate, holdout_df, profile, metric_name
    )
    reasons: List[str] = []
    if not champion_path or v_df is None:
        if not math.isfinite(candidate_holdout):
            raise ValueError("Candidate holdout metric is not finite.")
        reasons.append("no champion comparison")
        return {
            "status": "OK",
            "verified": True,
            "gain": 0.0,
            "gain_lb": 0.0,
            "holdout_drop": 0.0,
            "recall_drop": 0.0,
            "candidate_v": candidate_holdout,
            "champion_v": None,
            "reasons": reasons,
        }

    champion = joblib.load(_resolve_path(champion_path))
    candidate_v, truth, candidate_predictions, candidate_scores = _evaluate_model(
        candidate, v_df, profile, metric_name
    )
    champion_v, champion_truth, champion_predictions, champion_scores = _evaluate_model(
        champion, v_df, profile, metric_name
    )
    if not np.array_equal(truth, champion_truth):
        raise ValueError("Candidate and champion verification rows are not aligned.")
    champion_holdout, _, _, _ = _evaluate_model(
        champion, holdout_df, profile, metric_name
    )

    # Gain is measured on 0-1 loss (accuracy): CARA-T staleness, which triggers
    # retraining, is loss-based, and a single metric such as recall can be raised
    # by a worse model that simply predicts the positive class more often.
    candidate_correct = (candidate_predictions == truth).astype(float)
    champion_correct = (champion_predictions == truth).astype(float)
    paired = candidate_correct - champion_correct
    rng = np.random.default_rng(seed)
    n_rows = len(truth)
    gains = [float(paired[rng.integers(0, n_rows, size=n_rows)].mean()) for _ in range(n_boot)]
    gain = float(paired.mean())
    gain_lb = float(np.percentile(gains, 5))
    primary_drop = float(champion_v - candidate_v)

    candidate_holdout_acc = float(np.mean(
        np.asarray(candidate.predict(_labelled_rows(holdout_df, profile)[0]))
        == _labelled_rows(holdout_df, profile)[1].to_numpy()
    ))
    champion_holdout_acc = float(np.mean(
        np.asarray(champion.predict(_labelled_rows(holdout_df, profile)[0]))
        == _labelled_rows(holdout_df, profile)[1].to_numpy()
    ))
    holdout_drop = champion_holdout - candidate_holdout
    holdout_acc_drop = champion_holdout_acc - candidate_holdout_acc
    # The reference holdout only guards against regressions while it still
    # describes current data. If the champion itself is clearly worse on the
    # newest slice than on the holdout, the holdout is out of date and the gate
    # is waived (otherwise no model could ever be promoted after a concept shift).
    champion_v_acc = float(champion_correct.mean())
    se = math.sqrt(max(champion_holdout_acc * (1.0 - champion_holdout_acc), 1e-12) / max(n_rows, 1))
    holdout_gate = "waived" if (champion_holdout_acc - champion_v_acc) > max(tol, 3.0 * se) else "applied"

    recall_drop = 0.0
    if profile.get("task", {}).get("subtype") == "binary":
        positive = profile.get("task", {}).get("positive_class", 1)
        recall_drop = float(
            metrics.recall_score(truth, champion_predictions, pos_label=positive, zero_division=0)
            - metrics.recall_score(truth, candidate_predictions, pos_label=positive, zero_division=0)
        )

    if gain_lb < margin:
        reasons.append(f"verification gain lower bound {gain_lb:.6f} is below {margin:.6f}")
    if holdout_gate == "applied":
        if primary_drop > tol:
            reasons.append(f"{metric_name} drop on the newest slice {primary_drop:.6f} exceeds tolerance {tol:.6f}")
        if max(holdout_drop, holdout_acc_drop) > tol:
            reasons.append(
                f"holdout drop {max(holdout_drop, holdout_acc_drop):.6f} exceeds tolerance {tol:.6f}"
            )
        if recall_drop > recall_guard:
            reasons.append(
                f"positive-class recall drop {recall_drop:.6f} exceeds guard {recall_guard:.6f}"
            )
    return {
        "status": "OK",
        "verified": not reasons,
        "gain": gain,
        "gain_lb": gain_lb,
        "gain_metric": "accuracy",
        "holdout_drop": float(holdout_drop),
        "holdout_acc_drop": float(holdout_acc_drop),
        "holdout_gate": holdout_gate,
        "primary_drop": primary_drop,
        "recall_drop": recall_drop,
        "candidate_v": float(candidate_v),
        "champion_v": float(champion_v),
        "candidate_v_accuracy": float(candidate_correct.mean()),
        "champion_v_accuracy": champion_v_acc,
        "reasons": reasons,
    }


def verify(
    candidate_path: str | Path,
    holdout_df: pd.DataFrame,
    profile: Dict[str, Any],
    champion_path: Optional[str | Path] = None,
    v_df: Optional[pd.DataFrame] = None,
    margin: float = 0.0,
    tol: float = 0.02,
    recall_guard: float = 0.02,
    n_boot: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare a candidate against its champion with paired bootstrap verification."""
    return run_in_worker(
        "ri_decision.retrain:_verify_worker",
        {
            "candidate_path": str(candidate_path),
            "holdout_df": holdout_df,
            "profile": profile,
            "champion_path": str(champion_path) if champion_path else None,
            "v_df": v_df,
            "margin": margin,
            "tol": tol,
            "recall_guard": recall_guard,
            "n_boot": n_boot,
            "seed": seed,
        },
    )


def _atomic_copy(source_path: str | Path, destination_path: str | Path) -> None:
    source = Path(source_path)
    destination = Path(destination_path)
    if not source.is_file():
        raise FileNotFoundError(f"Model artifact not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def promote(
    candidate_path: str | Path,
    active_path: str | Path,
) -> Dict[str, Any]:
    """Atomically replace the active model with a verified candidate."""
    _atomic_copy(candidate_path, active_path)
    return {"status": "OK", "promoted": True, "active_path": str(active_path)}


def rollback(
    previous_path: str | Path,
    active_path: str | Path,
) -> Dict[str, Any]:
    """Atomically restore a previous model artifact."""
    _atomic_copy(previous_path, active_path)
    return {"status": "OK", "rolled_back": True, "active_path": str(active_path)}


def save_bundle(project_id: str, version: str) -> Dict[str, Any]:
    """Store the model and active calibration assets beside a model version."""
    project_dir = WORKSPACE / project_id
    version_dir = project_dir / "models" / version
    model_path = version_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Version model artifact not found: {model_path}")
    for bundle_name, project_relative in _BUNDLE_FILES.items():
        source = project_dir / project_relative
        if not source.is_file():
            raise FileNotFoundError(f"Bundle input not found: {source}")
        _atomic_copy(source, version_dir / bundle_name)
    return {"status": "OK", "version": version, "bundle_path": str(version_dir)}


def load_bundle(project_id: str, version: str) -> Dict[str, Any]:
    """Restore a version's model and calibration assets as the project's active bundle."""
    project_dir = WORKSPACE / project_id
    version_dir = project_dir / "models" / version
    model_path = version_dir / "model.joblib"
    if not model_path.is_file():
        raise FileNotFoundError(f"Version model artifact not found: {model_path}")
    for bundle_name, project_relative in _BUNDLE_FILES.items():
        source = version_dir / bundle_name
        if not source.is_file():
            raise FileNotFoundError(f"Version bundle asset not found: {source}")
        _atomic_copy(source, project_dir / project_relative)
    active_path = project_dir / "models" / "active" / "model.joblib"
    _atomic_copy(model_path, active_path)
    return {"status": "OK", "version": version, "active_path": str(active_path)}


def current_holdout_path(project_id: str, profile: Dict[str, Any]) -> Optional[str]:
    """Holdout that goes with the active version (the profile's holdout unless it was retired)."""
    state = _read_state(project_id)
    mapped = state.get("holdout_by_version", {}).get(state.get("active_version"))
    return mapped or profile.get("data", {}).get("holdout_path")


def _read_profile(project_id: str) -> Dict[str, Any]:
    profile_path = WORKSPACE / project_id / "profile.json"
    if profile_path.is_file():
        with open(profile_path, "r", encoding="utf-8") as file:
            return json.load(file)
    project = store.get_project(project_id)
    if project and project.get("profile_json"):
        return json.loads(project["profile_json"])
    raise FileNotFoundError(f"Project profile not found for {project_id!r}.")


def _read_state(project_id: str) -> Dict[str, Any]:
    state_path = WORKSPACE / project_id / "state.json"
    if not state_path.is_file():
        return {"active_version": "v1", "cooldown_left": 0, "shadow_left": 0}
    with open(state_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _write_state(project_id: str, state: Dict[str, Any]) -> None:
    state_path = WORKSPACE / project_id / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=state_path.parent,
            prefix=f".{state_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(state, file, indent=2)
        os.replace(temporary_path, state_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _labelled_batches(project_id: str) -> List[Dict[str, Any]]:
    return [
        batch
        for batch in store.get_batches(project_id)
        if int(batch.get("n_rows") or 0) > 0
        and int(batch.get("n_labelled") or 0) / int(batch["n_rows"]) >= 0.9
    ]


def _batch_frames(batches: List[Dict[str, Any]]) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for batch in batches:
        file_path = batch.get("file")
        if not file_path:
            raise ValueError(f"Batch {batch['id']} has no stored file path.")
        frames.append(_read_frame(file_path))
    return frames


def _version_number(version: str) -> int:
    match = re.fullmatch(r"v(\d+)", version)
    return int(match.group(1)) if match else 0


def _save_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(value, file, indent=2)


def run_retrain_cycle(
    project_id: str,
    trigger: str,
    report_id: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Train, verify, promote, and refresh a project's retraining bundle."""
    profile = _read_profile(project_id)
    data_result = build_training_data(project_id, profile)
    if data_result["status"] != "OK":
        return {
            "status": "INSUFFICIENT_DATA",
            "version": None,
            "verify": None,
            "seconds": 0.0,
            "reason": data_result.get("reason"),
        }

    versions = store.get_versions(project_id)
    next_number = max(
        (_version_number(version["version"]) for version in versions),
        default=0,
    ) + 1
    version = f"v{next_number}"
    project_dir = WORKSPACE / project_id
    version_dir = project_dir / "models" / version
    candidate_path = version_dir / "model.joblib"
    version_dir.mkdir(parents=True, exist_ok=True)

    active_record = store.get_active_version(project_id)
    champion_path = (
        _resolve_path(active_record["path"])
        if active_record
        else _resolve_path(profile["model"]["artifact"])
    )
    train_result = train_candidate(
        profile,
        data_result["train_df"],
        candidate_path,
        seed=profile.get("seed", 42),
        champion_path=champion_path,
    )
    seconds = float(train_result["seconds"])

    holdout_path = current_holdout_path(project_id, profile)
    if not holdout_path:
        raise ValueError("Retraining verification requires a reference holdout path.")
    holdout_df = _read_frame(holdout_path)
    verification = verify(
        candidate_path,
        holdout_df,
        profile,
        champion_path=champion_path,
        v_df=data_result["v_df"],
    )
    trained_on = ",".join(str(batch_id) for batch_id in data_result["batch_ids"])
    try:
        version_path = str(candidate_path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        version_path = str(candidate_path.resolve())
    decision_by = "human" if trigger.lower() == "human" else "system"
    if not verification["verified"] and not force:
        save_bundle(project_id, version)
        store.add_version(
            project_id,
            version,
            version_path,
            trained_on,
            status="rejected",
        )
        note = (
            f"version={version}; training_seconds={seconds:.6f}; "
            f"verification_failed={'; '.join(verification['reasons'])}"
        )
        store.add_decision(report_id, "VERIFY_FAILED", decision_by, note)
        return {
            "status": "VERIFY_FAILED",
            "version": version,
            "verify": verification,
            "seconds": seconds,
        }

    # If the holdout gate was waived the old holdout no longer describes current
    # data: retire it for this version and use the verification slice instead.
    holdout_retired = verification.get("holdout_gate") == "waived"
    baseline_df = data_result["v_df"] if holdout_retired else holdout_df
    history_batches = _labelled_batches(project_id)
    requested_history = int(profile.get("offline_batches_N", 25))
    warnings: List[str] = []
    new_calibration: Optional[Dict[str, Any]] = None
    if len(history_batches) >= 10:
        history_count = min(max(requested_history, 10), len(history_batches))
        history_frames = _batch_frames(history_batches[-history_count:])
        history_df = pd.concat(history_frames, ignore_index=True)
        new_calibration = run_in_worker(
            "ri_cara.cara_t:calibrate",
            {
                "model_path": str(candidate_path),
                "history_df": history_df,
                "profile": profile,
                "cfg": {"holdout_df": baseline_df},
            },
        )
    else:
        warnings.append(
            f"Calibration kept unchanged: only {len(history_batches)} labelled batches; "
            "at least 10 are required."
        )

    sample_count = min(2000, len(data_result["train_df"]))
    train_sample = data_result["train_df"].sample(
        n=sample_count,
        random_state=int(profile.get("seed", 42)),
    )
    plan = metrics_mod.default_plan(profile)
    new_baseline = run_in_worker(
        "ri_metrics.metrics:compute_baseline",
        {
            "model_path": str(candidate_path),
            "holdout_df": baseline_df,
            "profile": profile,
            "plan": plan,
        },
    )
    new_reference = drift_mod.fit_reference(data_result["train_df"], profile)

    active_path = project_dir / "models" / "active" / "model.joblib"
    promote(candidate_path, active_path)
    store.add_version(
        project_id,
        version,
        version_path,
        trained_on,
        status="candidate",
    )
    store.set_active_version(project_id, version)

    if new_calibration is not None:
        _save_json(project_dir / "calib.json", new_calibration)
    _save_json(project_dir / "baseline.json", new_baseline)
    train_sample.to_csv(project_dir / "data" / "train_sample.csv", index=False)
    _save_json(project_dir / "reference_stats.json", new_reference)
    save_bundle(project_id, version)

    state = _read_state(project_id)
    previous_version = active_record["version"] if active_record else None
    holdout_map = dict(state.get("holdout_by_version", {}))
    if holdout_retired:
        new_holdout = project_dir / "data" / f"holdout_{version}.csv"
        data_result["v_df"].to_csv(new_holdout, index=False)
        holdout_map[version] = str(new_holdout.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
        warnings.append("Reference holdout retired (out of date); the verification slice is the new baseline.")
    elif previous_version in holdout_map:
        holdout_map[version] = holdout_map[previous_version]
    state["holdout_by_version"] = holdout_map
    state.update(
        {
            "active_version": version,
            "cooldown_left": int(profile.get("W", 1)),
            "shadow_left": 3,
            "shadow_drops": [],
            "previous_version": previous_version,
        }
    )
    _write_state(project_id, state)

    note = f"version={version}; training_seconds={seconds:.6f}"
    if force and not verification["verified"]:
        note += "; verification overridden by force"
    if warnings:
        note += "; " + " ".join(warnings)
    store.add_decision(report_id, "PROMOTED", decision_by, note)
    return {
        "status": "PROMOTED",
        "version": version,
        "verify": verification,
        "seconds": seconds,
        "warnings": warnings,
    }


def _shadow_worker(
    active_path: str,
    previous_path: str,
    batch_df: pd.DataFrame,
    profile: Dict[str, Any],
) -> Dict[str, float]:
    metric_name = str(profile.get("primary_metric", "recall")).lower()
    active_model = joblib.load(_resolve_path(active_path))
    previous_model = joblib.load(_resolve_path(previous_path))
    active_score, _, _, _ = _evaluate_model(
        active_model, batch_df, profile, metric_name
    )
    previous_score, _, _, _ = _evaluate_model(
        previous_model, batch_df, profile, metric_name
    )
    return {"active_metric": active_score, "previous_metric": previous_score}


def shadow_check(project_id: str, batch_df: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the active model against its previous version during shadowing."""
    state = _read_state(project_id)
    shadow_left = int(state.get("shadow_left", 0))
    if shadow_left <= 0:
        return {"rollback_suggested": False, "shadow_left": 0}
    target = _read_profile(project_id).get("task", {}).get("target")
    if not target or target not in batch_df.columns:
        return {"rollback_suggested": False, "shadow_left": shadow_left}
    if batch_df[target].notna().sum() / max(len(batch_df), 1) < 0.9:
        return {"rollback_suggested": False, "shadow_left": shadow_left}

    previous_version = state.get("previous_version")
    active_record = store.get_active_version(project_id)
    if not previous_version or not active_record:
        state["shadow_left"] = 0
        _write_state(project_id, state)
        return {"rollback_suggested": False, "shadow_left": 0}

    previous_record = next(
        (
            version
            for version in store.get_versions(project_id)
            if version["version"] == previous_version
        ),
        None,
    )
    if not previous_record:
        raise ValueError(f"Previous model version {previous_version!r} is not registered.")
    profile = _read_profile(project_id)
    result = run_in_worker(
        "ri_decision.retrain:_shadow_worker",
        {
            "active_path": active_record["path"],
            "previous_path": previous_record["path"],
            "batch_df": batch_df,
            "profile": profile,
        },
    )
    active_worse = float(result["previous_metric"] - result["active_metric"])
    drops = list(state.get("shadow_drops", []))
    drops.append(active_worse)
    state["shadow_drops"] = drops
    state["shadow_left"] = shadow_left - 1
    average_drop = float(np.mean(drops))
    _write_state(project_id, state)
    return {
        "rollback_suggested": average_drop > 0.02,
        "active_metric": float(result["active_metric"]),
        "previous_metric": float(result["previous_metric"]),
        "average_drop": average_drop,
        "shadow_left": shadow_left - 1,
    }


def rollback_project(project_id: str) -> Dict[str, Any]:
    """Restore the previous model bundle and register a human rollback decision."""
    state = _read_state(project_id)
    current_version = state.get("active_version")
    previous_version = state.get("previous_version")
    if not previous_version:
        previous = next(
            (
                version
                for version in reversed(store.get_versions(project_id))
                if version["status"] == "previous"
            ),
            None,
        )
        previous_version = previous["version"] if previous else None
    if not previous_version:
        raise ValueError(f"Project {project_id!r} has no previous model version to restore.")

    previous_record = next(
        (
            version
            for version in reversed(store.get_versions(project_id))
            if version["version"] == previous_version
        ),
        None,
    )
    if not previous_record:
        raise ValueError(f"Previous model version {previous_version!r} is not registered.")
    load_bundle(project_id, previous_version)
    store.set_active_version(project_id, previous_version)
    state.update(
        {
            "active_version": previous_version,
            "previous_version": current_version,
            "cooldown_left": 0,
            "shadow_left": 0,
            "shadow_drops": [],
        }
    )
    _write_state(project_id, state)
    store.add_decision(
        None,
        "ROLLBACK",
        "human",
        f"Restored {previous_version} from {current_version or 'unknown'}",
    )
    return {
        "status": "ROLLED_BACK",
        "active_version": previous_version,
        "previous_version": current_version,
    }
