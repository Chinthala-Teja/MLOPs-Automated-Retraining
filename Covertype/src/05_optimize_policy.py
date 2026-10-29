import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from covtype_pipeline import (
    CARA_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    VALIDATION_BATCHES,
    absolute_cost,
    batch_bounds,
    classification_metrics,
    dense_rows,
    load_processed_data,
    sample_batch_for_cost,
    sample_reference,
    train_replacement_model,
)


KAPPA_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
DEFAULT_KAPPA_MULTIPLIER = 1.0
THRESHOLD_QUANTILES = 7
RETRAIN_EPOCHS = 15
RETRAIN_BATCH_SIZE = 1024


def build_thresholds(staleness: np.ndarray) -> np.ndarray:
    if staleness.size == 0 or not np.isfinite(staleness).all():
        raise ValueError("Validation staleness must be non-empty and finite.")
    return np.unique(
        np.concatenate(
            [
                [float("-inf")],
                np.quantile(
                    staleness,
                    np.linspace(0, 1, THRESHOLD_QUANTILES),
                ),
                [float("inf")],
            ]
        )
    )


def sequential_validation_replay(
    threshold: float,
    kappa: float,
    features,
    labels: np.ndarray,
    manifest: pd.DataFrame,
    gamma: float,
    initial_reference_data: np.ndarray,
    initial_reference_labels: np.ndarray,
    initial_reference_queries: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

    active_model = keras.models.load_model(MODEL_PATH)
    reference_data = initial_reference_data.copy()
    reference_labels = initial_reference_labels.copy()
    reference_queries = initial_reference_queries.copy()
    reference_cost = absolute_cost(
        active_model,
        reference_queries,
        reference_data,
        reference_labels,
        gamma,
    )
    logs = []
    elapsed_training_seconds = 0.0

    for batch_id in VALIDATION_BATCHES:
        start, end = batch_bounds(manifest, batch_id)
        X_batch = dense_rows(features, start, end)
        y_batch = labels[start:end]
        probabilities = active_model.predict(
            X_batch, batch_size=2048, verbose=0
        ).ravel()
        metrics = classification_metrics(y_batch, probabilities)
        query_X, data_X, data_y = sample_batch_for_cost(
            X_batch,
            y_batch,
            seed=RANDOM_SEED + batch_id,
        )
        batch_cost = absolute_cost(active_model, query_X, data_X, data_y, gamma)
        reference_cost_before = reference_cost
        relative_staleness = batch_cost - reference_cost_before
        should_retrain = bool(relative_staleness >= threshold)
        decision = "RETRAIN" if should_retrain else "KEEP"
        decision_cost = float(kappa if should_retrain else batch_cost)
        version_before = len(
            [row for row in logs if row["decision"] == "RETRAIN"]
        ) + 1

        logs.append(
            {
                "batch_id": int(batch_id),
                "model_version_before_decision": int(version_before),
                **metrics,
                "absolute_cost": batch_cost,
                "reference_cost_before_decision": float(reference_cost_before),
                "relative_staleness": float(relative_staleness),
                "threshold": float(threshold),
                "kappa": float(kappa),
                "decision": decision,
                "decision_cost": decision_cost,
            }
        )

        if should_retrain:
            active_model, training_seconds = train_replacement_model(
                X_batch,
                y_batch.astype(np.float32),
                seed=RANDOM_SEED + int(batch_id),
                epochs=RETRAIN_EPOCHS,
                batch_size=RETRAIN_BATCH_SIZE,
            )
            elapsed_training_seconds += training_seconds
            reference_data, reference_labels, reference_queries = sample_reference(
                features[start:end],
                y_batch,
                seed=RANDOM_SEED + int(batch_id),
            )
            reference_cost = absolute_cost(
                active_model,
                reference_queries,
                reference_data,
                reference_labels,
                gamma,
            )

    frame = pd.DataFrame(logs)
    if frame.empty or frame["batch_id"].tolist() != list(VALIDATION_BATCHES):
        raise ValueError("Sequential validation replay missed one or more batches.")
    return frame, elapsed_training_seconds


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Baseline model not found: {MODEL_PATH}. "
            "Run 03_train_baseline.py first."
        )
    staleness_path = CARA_DIR / "validation_staleness.csv"
    reference_state_path = CARA_DIR / "reference_state.npz"
    staleness_metadata_path = CARA_DIR / "staleness_metadata.json"
    for path in (staleness_path, reference_state_path, staleness_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required validation artifact is missing: {path}. "
                "Run 04_compute_staleness.py first."
            )

    features, labels, manifest, _ = load_processed_data()
    validation = pd.read_csv(staleness_path)
    if validation["batch_id"].astype(int).tolist() != list(VALIDATION_BATCHES):
        raise ValueError("Validation staleness must cover batches 70 through 84.")
    fixed_v1_staleness = validation["relative_staleness"].to_numpy(dtype=float)
    thresholds = build_thresholds(fixed_v1_staleness)
    positive_costs = validation.loc[
        validation["batch_absolute_cost"] > 0, "batch_absolute_cost"
    ].to_numpy(dtype=float)
    if positive_costs.size == 0:
        raise ValueError(
            "All validation proxy costs are zero; cannot derive retraining-cost sensitivity."
        )

    reference_kappa = float(np.median(positive_costs))
    with staleness_metadata_path.open("r", encoding="utf-8") as file:
        staleness_metadata = json.load(file)
    gamma = float(staleness_metadata["gamma"])
    with np.load(reference_state_path, allow_pickle=False) as reference_state:
        initial_reference_data = reference_state["reference_data"].astype(
            np.float32, copy=True
        )
        initial_reference_labels = reference_state["reference_labels"].astype(
            np.int8, copy=True
        )
        initial_reference_queries = reference_state["reference_queries"].astype(
            np.float32, copy=True
        )

    all_results = []
    best_logs = []
    best_rows = []
    for multiplier in KAPPA_MULTIPLIERS:
        kappa = reference_kappa * multiplier
        candidate_runs = []
        print(
            f"\nSequential validation policy search: "
            f"kappa multiplier={multiplier:g}, kappa={kappa:.8g}"
        )
        for index, threshold in enumerate(thresholds, start=1):
            logs, elapsed = sequential_validation_replay(
                threshold=threshold,
                kappa=kappa,
                features=features,
                labels=labels,
                manifest=manifest,
                gamma=gamma,
                initial_reference_data=initial_reference_data,
                initial_reference_labels=initial_reference_labels,
                initial_reference_queries=initial_reference_queries,
            )
            candidate_runs.append((threshold, logs, elapsed))
            print(
                f"  threshold {index}/{len(thresholds)}={threshold}: "
                f"cost={logs['decision_cost'].sum():.8g}, "
                f"retrain={int((logs['decision'] == 'RETRAIN').sum())}"
            )

        for threshold, logs, elapsed in candidate_runs:
            all_results.append(
                {
                    "kappa_multiplier": multiplier,
                    "kappa": kappa,
                    "threshold": (
                        "-inf"
                        if np.isneginf(threshold)
                        else "inf"
                        if np.isposinf(threshold)
                        else float(threshold)
                    ),
                    "total_cost": float(logs["decision_cost"].sum()),
                    "mean_cost": float(logs["decision_cost"].mean()),
                    "keep_count": int((logs["decision"] == "KEEP").sum()),
                    "retrain_count": int(
                        (logs["decision"] == "RETRAIN").sum()
                    ),
                    "mean_error": float(logs["error_rate"].mean()),
                    "retraining_seconds": float(elapsed),
                }
            )

        best_threshold, best_log, best_elapsed = min(
            candidate_runs,
            key=lambda run: (
                float(run[1]["decision_cost"].sum()),
                int((run[1]["decision"] == "RETRAIN").sum()),
                float(run[0]),
            ),
        )
        label = (
            "-inf"
            if np.isneginf(best_threshold)
            else "inf"
            if np.isposinf(best_threshold)
            else float(best_threshold)
        )
        best_path = CARA_DIR / "cost_sensitivity" / f"best_replay_kappa_{multiplier:g}.csv"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        best_log.to_csv(best_path, index=False)
        best_logs.append(best_log.assign(kappa_multiplier=multiplier))
        best_rows.append(
            {
                "kappa_multiplier": multiplier,
                "kappa": kappa,
                "selected_threshold": label,
                "total_cost": float(best_log["decision_cost"].sum()),
                "mean_cost": float(best_log["decision_cost"].mean()),
                "keep_count": int((best_log["decision"] == "KEEP").sum()),
                "retrain_count": int(
                    (best_log["decision"] == "RETRAIN").sum()
                ),
                "mean_error": float(best_log["error_rate"].mean()),
                "retraining_seconds": float(best_elapsed),
            }
        )

    search = pd.DataFrame(all_results)
    best = pd.DataFrame(best_rows)
    sensitivity_dir = CARA_DIR / "cost_sensitivity"
    sensitivity_dir.mkdir(parents=True, exist_ok=True)
    search_path = sensitivity_dir / "all_threshold_results.csv"
    best_path = sensitivity_dir / "best_policy_per_kappa.csv"
    logs_path = sensitivity_dir / "best_policy_batch_logs.csv"
    search.to_csv(search_path, index=False)
    search.to_csv(CARA_DIR / "threshold_search.csv", index=False)
    best.to_csv(best_path, index=False)
    pd.concat(best_logs, ignore_index=True).to_csv(logs_path, index=False)
    best.to_csv(CARA_DIR / "best_thresholds.csv", index=False)

    default = best.loc[
        np.isclose(best["kappa_multiplier"], DEFAULT_KAPPA_MULTIPLIER)
    ]
    if len(default) != 1:
        raise ValueError("Could not resolve the default kappa policy.")
    default_row = default.iloc[0]
    metadata = {
        "method": "sequential validation replay with dynamic model updates and kappa sensitivity",
        "candidate_thresholds": [
            "-inf"
            if np.isneginf(value)
            else "inf"
            if np.isposinf(value)
            else float(value)
            for value in thresholds
        ],
        "candidate_threshold_source": (
            "Quantiles of standalone Model V1 validation relative staleness plus +/- infinity."
        ),
        "validation_batches": [
            VALIDATION_BATCHES.start,
            VALIDATION_BATCHES.stop - 1,
        ],
        "test_batches_reserved": [85, 99],
        "kappa_reference": (
            "Median positive standalone Model V1 validation batch absolute proxy cost."
        ),
        "kappa_reference_value": reference_kappa,
        "kappa_multipliers": KAPPA_MULTIPLIERS,
        "default_kappa_multiplier": DEFAULT_KAPPA_MULTIPLIER,
        "selected_default_kappa": float(default_row["kappa"]),
        "selected_default_threshold": default_row["selected_threshold"],
        "selected_default_retrain_count": int(default_row["retrain_count"]),
        "reference_reset": (
            "After RETRAIN, estimate replacement-model reference cost on a seeded sample of its current training batch."
        ),
        "retraining": (
            f"Fresh weights, {RETRAIN_EPOCHS} epochs, batch size {RETRAIN_BATCH_SIZE}, "
            "current validation batch only."
        ),
        "decision_timing": (
            "Evaluate batch first; a trained replacement serves subsequent batches."
        ),
        "cost_definition": "KEEP pays current batch absolute proxy cost; RETRAIN pays kappa.",
        "limitations": [
            "Experimental CARA-T-inspired adaptation, not exact paper reproduction.",
            "Validation data is used for threshold selection and is not an unbiased test.",
            "The kappa values are proxy sensitivity settings, not measured operating cost.",
            "The source file order is not verified chronology.",
        ],
        "best_policy_per_kappa": best_rows,
    }
    with (CARA_DIR / "threshold_optimization_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)
    with (sensitivity_dir / "sensitivity_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    plot_dir = sensitivity_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cara_plot_dir = CARA_DIR / "plots"
    cara_plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 6))
    for multiplier in KAPPA_MULTIPLIERS:
        subset = search.loc[search["kappa_multiplier"] == multiplier].copy()
        subset["threshold_numeric"] = subset["threshold"].map(
            lambda value: (
                float("-inf")
                if str(value) == "-inf"
                else float("inf")
                if str(value) == "inf"
                else float(value)
            )
        )
        subset = subset.sort_values("threshold_numeric")
        finite = subset[np.isfinite(subset["threshold_numeric"])]
        if not finite.empty:
            axis.plot(
                finite["threshold_numeric"],
                finite["total_cost"],
                marker=".",
                label=f"kappa x {multiplier:g}",
            )
    axis.set(
        title="CoverType sequential validation threshold sensitivity",
        xlabel="Relative-staleness threshold",
        ylabel="Sequential validation proxy cost",
    )
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(cara_plot_dir / "threshold_sensitivity.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(best["kappa"], best["total_cost"], marker="o")
    axis.set(
        title="CoverType sequential validation proxy cost vs retraining cost",
        xlabel="Retraining cost (kappa)",
        ylabel="Selected sequential validation proxy cost",
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_cost_vs_kappa.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(best["kappa"], best["retrain_count"], marker="o", label="RETRAIN")
    axis.plot(best["kappa"], best["keep_count"], marker="s", label="KEEP")
    axis.set(
        title="CoverType selected validation decisions vs retraining cost",
        xlabel="Retraining cost (kappa)",
        ylabel="Number of validation decisions",
    )
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "decisions_vs_kappa.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(best["kappa"], best["mean_error"], marker="o")
    axis.set(
        title="CoverType selected validation error vs retraining cost",
        xlabel="Retraining cost (kappa)",
        ylabel="Mean batch error rate",
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "error_vs_kappa.png", dpi=180)
    plt.close(fig)

    print("\nSelected sequential validation policies")
    print(best.to_string(index=False))
    print(f"\nAll thresholds: {search_path}")
    print(f"Kappa sensitivity summary: {best_path}")
    print(f"Default kappa: {metadata['selected_default_kappa']:.8g}")
    print(f"Default threshold: {metadata['selected_default_threshold']}")


if __name__ == "__main__":
    main()
