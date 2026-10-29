import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow import keras

from covtype_pipeline import (
    CARA_DIR,
    MODEL_PATH,
    RANDOM_SEED,
    TRAIN_BATCHES,
    VALIDATION_BATCHES,
    absolute_cost,
    batch_bounds,
    classification_metrics,
    dense_rows,
    estimate_rbf_gamma,
    load_processed_data,
    sample_batch_for_cost,
    sample_reference,
)


def main() -> None:
    features, labels, manifest, _ = load_processed_data()
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Baseline model not found: {MODEL_PATH}")

    train_start, _ = batch_bounds(manifest, TRAIN_BATCHES.start)
    _, train_end = batch_bounds(manifest, TRAIN_BATCHES.stop - 1)
    X_train = features[train_start:train_end]
    y_train = labels[train_start:train_end]
    model = keras.models.load_model(MODEL_PATH)

    gamma = estimate_rbf_gamma(X_train, RANDOM_SEED)
    reference_data, reference_labels, reference_queries = sample_reference(
        X_train, y_train, RANDOM_SEED
    )
    reference_cost = absolute_cost(
        model,
        reference_queries,
        reference_data,
        reference_labels,
        gamma,
    )

    results = []
    for batch_id in VALIDATION_BATCHES:
        start, end = batch_bounds(manifest, batch_id)
        X_batch = dense_rows(features, start, end)
        y_batch = labels[start:end]
        probabilities = model.predict(X_batch, batch_size=2048, verbose=0).ravel()
        metrics = classification_metrics(y_batch, probabilities)

        query_X, data_X, data_y = sample_batch_for_cost(
            X_batch,
            y_batch,
            seed=RANDOM_SEED + batch_id,
        )
        batch_cost = absolute_cost(model, query_X, data_X, data_y, gamma)
        results.append(
            {
                "batch_id": int(batch_id),
                "n_records": int(end - start),
                **metrics,
                "reference_absolute_cost": reference_cost,
                "batch_absolute_cost": batch_cost,
                "relative_staleness": batch_cost - reference_cost,
            }
        )

    CARA_DIR.mkdir(parents=True, exist_ok=True)
    results_path = CARA_DIR / "validation_staleness.csv"
    pd.DataFrame(results).to_csv(results_path, index=False)
    np.savez_compressed(
        CARA_DIR / "reference_state.npz",
        reference_data=reference_data,
        reference_labels=reference_labels,
        reference_queries=reference_queries,
    )
    metadata = {
        "method": "RBF similarity-weighted 0-1 classification-error proxy",
        "gamma": gamma,
        "reference_absolute_cost": reference_cost,
        "validation_batches": [VALIDATION_BATCHES.start, VALIDATION_BATCHES.stop - 1],
        "reference_data_sample_size": len(reference_data),
        "reference_query_sample_size": len(reference_queries),
        "batch_data_sample_limit": 2000,
        "batch_query_sample_limit": 500,
        "relative_staleness_definition": (
            "batch_absolute_cost minus the original Model V1 training-reference cost"
        ),
        "rbf_kernel": "exp(-gamma * squared_euclidean_distance)",
        "loss": "0-1 loss at probability threshold 0.5",
        "cost_is_real_operational_cost": False,
        "stream_order_is_verified_chronology": False,
    }
    with (CARA_DIR / "staleness_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    plot_dir = CARA_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(frame["batch_id"], frame["relative_staleness"], marker="o")
    axis.axhline(0, color="black", linewidth=1)
    axis.set(
        title="CoverType validation relative staleness (Model V1)",
        xlabel="Preserved-order batch ID",
        ylabel="Batch proxy cost - training reference cost",
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "relative_staleness.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11, 5))
    axis.plot(frame["batch_id"], frame["error_rate"], marker="o")
    axis.set(
        title="CoverType validation error rate by batch (Model V1)",
        xlabel="Preserved-order batch ID",
        ylabel="Error rate",
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "validation_error_rate.png", dpi=180)
    plt.close(fig)

    print(f"RBF gamma: {gamma:.8g}")
    print(f"Reference proxy cost: {reference_cost:.8g}")
    print(frame[["batch_id", "accuracy", "error_rate", "relative_staleness"]].to_string(index=False))
    print(f"\nValidation staleness saved to: {results_path}")


if __name__ == "__main__":
    main()
