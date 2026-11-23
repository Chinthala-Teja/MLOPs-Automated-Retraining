import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras

from covtype_pipeline import (
    ARTIFACTS_DIR,
    BATCH_COUNT,
    MODEL_PATH,
    batch_bounds,
    classification_metrics,
    dense_rows,
    load_processed_data,
)


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Baseline model not found: {MODEL_PATH}. Run 03_train_baseline.py first."
        )
    features, labels, manifest, _ = load_processed_data()
    model = keras.models.load_model(MODEL_PATH)

    results = []
    for batch_id in range(BATCH_COUNT):
        start, end = batch_bounds(manifest, batch_id)
        X_batch = dense_rows(features, start, end)
        y_batch = labels[start:end]
        probabilities = model.predict(
            X_batch, batch_size=2048, verbose=0
        ).ravel()
        results.append(
            {
                "batch_id": batch_id,
                "split": str(manifest.loc[batch_id, "split"]),
                "n_records": end - start,
                **classification_metrics(y_batch, probabilities),
                "class_1_rate": float(np.mean(y_batch)),
                "predicted_class_1_rate": float(np.mean(probabilities >= 0.5)),
                "mean_probability": float(np.mean(probabilities)),
            }
        )

    result = pd.DataFrame(results)
    output_dir = ARTIFACTS_DIR / "monitoring"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "batch_metrics.csv"
    result.to_csv(output_path, index=False)

    split_summary = (
        result.groupby("split")[
            ["accuracy", "precision", "recall", "f1", "roc_auc", "error_rate"]
        ]
        .mean()
        .reset_index()
    )
    split_summary.to_csv(output_dir / "split_metrics_summary.csv", index=False)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(result["batch_id"], result["accuracy"], marker=".", linewidth=1)
    axis.axvline(69.5, color="grey", linestyle="--", linewidth=1)
    axis.axvline(84.5, color="grey", linestyle="--", linewidth=1)
    axis.set(
        title="CoverType Model V1 accuracy by preserved-order batch",
        xlabel="Batch ID",
        ylabel="Accuracy",
        ylim=(0, 1),
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "batch_accuracy.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 5))
    axis.plot(result["batch_id"], result["error_rate"], marker=".", linewidth=1)
    axis.axvline(69.5, color="grey", linestyle="--", linewidth=1)
    axis.axvline(84.5, color="grey", linestyle="--", linewidth=1)
    axis.set(
        title="CoverType Model V1 error rate by preserved-order batch",
        xlabel="Batch ID",
        ylabel="Error rate",
    )
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "batch_error_rate.png", dpi=180)
    plt.close(fig)

    print(f"Batch monitoring saved: {output_path}")
    print("\nMean metrics by split:")
    print(split_summary.round(4).to_string(index=False))
    print(
        f"\nBatch accuracy range: {result['accuracy'].min():.4f} "
        f"to {result['accuracy'].max():.4f}"
    )


if __name__ == "__main__":
    main()
