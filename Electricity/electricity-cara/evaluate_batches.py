
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

DATA_PATH = Path("data/elec/processed/electricity_batched.csv")
ARTIFACT_DIR = Path("artifacts/model_v1")
OUTPUT_DIR = Path("artifacts/monitoring")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]

df = pd.read_csv(DATA_PATH)
model = tf.keras.models.load_model(ARTIFACT_DIR / "model.keras")
scaler = joblib.load(ARTIFACT_DIR / "scaler.joblib")

results = []

for batch_id, batch in df.groupby("batch_id", sort=True):
    X = batch[FEATURES].to_numpy(dtype=np.float32)
    y = batch["class"].to_numpy(dtype=int)

    X_scaled = scaler.transform(X).astype(np.float32)
    probabilities = model.predict(X_scaled, verbose=0).ravel()
    predictions = (probabilities >= 0.5).astype(int)

    results.append({
        "batch_id": int(batch_id),
        "n_records": len(batch),
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(
            y, predictions, zero_division=0
        ),
        "recall": recall_score(y, predictions, zero_division=0),
        "f1": f1_score(y, predictions, zero_division=0),
        "roc_auc": (
            roc_auc_score(y, probabilities)
            if len(np.unique(y)) == 2
            else np.nan
        ),
        "class_1_rate": float(np.mean(y)),
        "predicted_class_1_rate": float(np.mean(predictions)),
        "mean_probability": float(np.mean(probabilities)),
    })

metrics_df = pd.DataFrame(results)
output_path = OUTPUT_DIR / "batch_metrics.csv"
metrics_df.to_csv(output_path, index=False)

print("Saved:", output_path.resolve())
print("\nFirst 10 batches:")
print(metrics_df.head(10).round(4).to_string(index=False))

print("\nSummary by existing split:")
metrics_df["split"] = metrics_df["batch_id"].map(
    lambda b: "train" if b <= 69
    else "validation" if b <= 84
    else "test"
)

print(
    metrics_df.groupby("split")[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].mean().round(4)
)

print("\nBatch accuracy range:")
print(f"Minimum: {metrics_df['accuracy'].min():.4f}")
print(f"Maximum: {metrics_df['accuracy'].max():.4f}")
