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
    TEST_BATCHES,
    absolute_cost,
    batch_bounds,
    classification_metrics,
    dense_rows,
    load_processed_data,
    sample_batch_for_cost,
    sample_reference,
    train_replacement_model,
)


PERIODIC_INTERVAL = 3
RETRAIN_EPOCHS = 15
RETRAIN_BATCH_SIZE = 256
STRATEGIES = (
    ("Never retrain", "never_retrain"),
    ("Always retrain", "always_retrain"),
    ("Periodic (every 3 batches)", "periodic"),
    ("CARA-T-inspired", "cara_t"),
)


def main() -> None:
    features, labels, manifest, _ = load_processed_data()
    staleness_metadata_path = CARA_DIR / "staleness_metadata.json"
    reference_state_path = CARA_DIR / "reference_state.npz"
    thresholds_path = CARA_DIR / "best_thresholds.csv"
    for path in (MODEL_PATH, staleness_metadata_path, reference_state_path, thresholds_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required input is missing: {path}. Run the preceding pipeline scripts."
            )

    with staleness_metadata_path.open("r", encoding="utf-8") as file:
        staleness_metadata = json.load(file)
    thresholds = pd.read_csv(thresholds_path)
    default = thresholds.loc[np.isclose(thresholds["kappa_multiplier"], 1.0)]
    if len(default) != 1:
        raise ValueError("Expected one threshold row for the default kappa multiplier.")
    kappa = float(default.iloc[0]["kappa"])
    threshold = float(default.iloc[0]["selected_threshold"])
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
    initial_reference_cost = float(staleness_metadata["reference_absolute_cost"])

    CARA_DIR.mkdir(parents=True, exist_ok=True)
    logs = []
    summaries = []
    plot_data = {}

    for strategy, strategy_key in STRATEGIES:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")
        active_model = keras.models.load_model(MODEL_PATH)
        model_version = 1
        reference_data = initial_reference_data.copy()
        reference_labels = initial_reference_labels.copy()
        reference_queries = initial_reference_queries.copy()
        reference_cost = initial_reference_cost
        all_labels = []
        all_probabilities = []
        strategy_logs = []
        retrain_count = 0
        retraining_seconds = 0.0

        for position, batch_id in enumerate(TEST_BATCHES):
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
            batch_cost = absolute_cost(
                active_model, query_X, data_X, data_y, gamma
            )
            reference_cost_before_decision = reference_cost
            relative_staleness = batch_cost - reference_cost
            version_before = model_version

            if strategy == "Never retrain":
                action = "KEEP"
            elif strategy == "Always retrain":
                action = "RETRAIN"
            elif strategy == "Periodic (every 3 batches)":
                action = (
                    "RETRAIN"
                    if position % PERIODIC_INTERVAL == 0
                    else "KEEP"
                )
            elif strategy == "CARA-T-inspired":
                action = (
                    "RETRAIN"
                    if relative_staleness >= threshold
                    else "KEEP"
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            decision_cost = float(kappa if action == "RETRAIN" else batch_cost)
            trained_seconds = 0.0
            training_performed = False
            if action == "RETRAIN":
                active_model, trained_seconds = train_replacement_model(
                    X_batch,
                    y_batch.astype(np.float32),
                    seed=RANDOM_SEED + int(batch_id),
                    epochs=RETRAIN_EPOCHS,
                    batch_size=RETRAIN_BATCH_SIZE,
                )
                retraining_seconds += trained_seconds
                retrain_count += 1
                model_version += 1
                training_performed = True
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

            record = {
                "strategy": strategy,
                "strategy_key": strategy_key,
                "batch_id": int(batch_id),
                "n_records": int(end - start),
                "model_version_before_decision": int(version_before),
                **metrics,
                "absolute_cost": batch_cost,
                "reference_cost": float(reference_cost_before_decision),
                "relative_staleness": float(relative_staleness),
                "threshold": threshold if strategy == "CARA-T-inspired" else np.nan,
                "kappa": kappa,
                "decision": action,
                "proxy_cost": decision_cost,
                "training_performed": training_performed,
                "retraining_seconds": trained_seconds,
                "model_version_after_decision": int(model_version),
            }
            logs.append(record)
            strategy_logs.append(record)
            all_labels.append(y_batch)
            all_probabilities.append(probabilities)

        output_model_dir = CARA_DIR / "holdout_models" / strategy_key
        output_model_dir.mkdir(parents=True, exist_ok=True)
        active_model.save(output_model_dir / "final_model.keras")

        combined_labels = np.concatenate(all_labels)
        combined_probabilities = np.concatenate(all_probabilities)
        aggregate_metrics = classification_metrics(
            combined_labels, combined_probabilities
        )
        batch_frame = pd.DataFrame(strategy_logs)
        summaries.append(
            {
                "strategy": strategy,
                "strategy_key": strategy_key,
                "holdout_batches": int(len(batch_frame)),
                "holdout_records": int(len(combined_labels)),
                "retrain_count": int(retrain_count),
                "keep_count": int((batch_frame["decision"] == "KEEP").sum()),
                "total_proxy_cost": float(batch_frame["proxy_cost"].sum()),
                "total_retraining_seconds": float(retraining_seconds),
                "threshold": threshold if strategy == "CARA-T-inspired" else np.nan,
                "kappa": kappa,
                **aggregate_metrics,
                "mean_batch_accuracy": float(batch_frame["accuracy"].mean()),
                "mean_batch_f1": float(batch_frame["f1"].mean()),
                "mean_batch_error_rate": float(batch_frame["error_rate"].mean()),
            }
        )
        plot_data[strategy] = batch_frame
        print(
            f"{strategy}: retrains={retrain_count}, "
            f"accuracy={aggregate_metrics['accuracy']:.4f}, "
            f"proxy_cost={batch_frame['proxy_cost'].sum():.8g}"
        )

    result_dir = CARA_DIR / "test_holdout"
    result_dir.mkdir(parents=True, exist_ok=True)
    results_frame = pd.DataFrame(logs)
    summary_frame = pd.DataFrame(summaries)
    results_path = result_dir / "holdout_batch_results.csv"
    summary_path = result_dir / "holdout_strategy_summary.csv"
    results_frame.to_csv(results_path, index=False)
    summary_frame.to_csv(summary_path, index=False)

    experiment_metadata = {
        "evaluation": "sequential prequential holdout replay",
        "holdout_batches": [TEST_BATCHES.start, TEST_BATCHES.stop - 1],
        "prediction_order": (
            "Predict and score each batch before any update; updates affect only later batches."
        ),
        "final_batch": (
            "The policy decision is logged and a selected update is trained after "
            "scoring the final batch, matching the Electricity holdout replay; "
            "the final update has no effect on a later holdout prediction."
        ),
        "retraining_data": (
            "A replacement is trained from fresh weights using only the current "
            "batch, matching the Electricity experiment."
        ),
        "kappa": kappa,
        "threshold": threshold,
        "rbf_gamma": gamma,
        "proxy_cost_definition": (
            "RETRAIN pays kappa; KEEP pays current batch absolute proxy cost."
        ),
        "retraining_seconds_are_informational": True,
        "stream_order_is_verified_chronology": False,
        "limitations": [
            "LIBSVM input has no timestamps; preserved row order is not verified time.",
            "The binary dataset contains only original classes 1 and 2.",
            "The proxy cost and kappa are not real financial or compute costs.",
            "The test phase is a prequential replay: each batch is predicted before its labels are used for a later update.",
        ],
    }
    with (result_dir / "experiment_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(experiment_metadata, file, indent=2)

    plot_dir = result_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 6))
    for strategy, frame in plot_data.items():
        axis.plot(frame["batch_id"], frame["accuracy"], marker=".", label=strategy)
    axis.set(
        title="CoverType holdout accuracy by preserved-order batch",
        xlabel="Batch ID",
        ylabel="Accuracy",
        ylim=(0, 1),
    )
    axis.legend()
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "holdout_accuracy_by_batch.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 6))
    cara_frame = plot_data["CARA-T-inspired"]
    action_values = cara_frame["decision"].map(
        {"KEEP": 0, "RETRAIN": 1}
    )
    axis.step(cara_frame["batch_id"], action_values, where="mid")
    axis.set(
        title="CARA-T-inspired holdout decisions",
        xlabel="Batch ID",
        ylabel="Decision (0=KEEP, 1=RETRAIN)",
        yticks=[0, 1],
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "holdout_retraining_decisions.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(summary_frame["strategy"], summary_frame["total_proxy_cost"])
    axis.set(
        title="CoverType holdout total proxy cost",
        ylabel="Total proxy cost",
    )
    axis.tick_params(axis="x", rotation=20)
    axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "holdout_total_proxy_cost.png", dpi=180)
    plt.close(fig)

    print(f"\nBatch results: {results_path}")
    print(f"Strategy summary: {summary_path}")
    print(f"Final policy models: {CARA_DIR / 'holdout_models'}")


if __name__ == "__main__":
    main()
