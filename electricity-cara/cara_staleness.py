
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.spatial.distance import cdist
from tensorflow import keras


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

DATA_PATH = Path("data/elec/processed/electricity_batched.csv")
MODEL_PATH = Path("artifacts/model_v1/model.keras")
SCALER_PATH = Path("artifacts/model_v1/scaler.joblib")

OUTPUT_DIR = Path("artifacts/cara_t")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]

TARGET = "class"

TRAIN_START, TRAIN_END = 0, 69
VAL_START, VAL_END = 70, 84

REFERENCE_DATA_SIZE = 2000
REFERENCE_QUERY_SIZE = 500
GAMMA_SAMPLE_SIZE = 1000
RANDOM_SEED = 42


# --------------------------------------------------
# 2. Load data, model, and scaler
# --------------------------------------------------

np.random.seed(RANDOM_SEED)

df = pd.read_csv(DATA_PATH)
model = keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

train_df = df[
    df["batch_id"].between(TRAIN_START, TRAIN_END)
].copy()

val_df = df[
    df["batch_id"].between(VAL_START, VAL_END)
].copy()

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation data is empty.")

X_train = scaler.transform(train_df[FEATURES])
y_train = train_df[TARGET].to_numpy().astype(int)

print("Training records:", len(train_df))
print("Validation records:", len(val_df))


# --------------------------------------------------
# 3. Model predictions and 0-1 classification loss
# --------------------------------------------------

def get_predictions(X):
    probabilities = model.predict(X, verbose=0).reshape(-1)
    return (probabilities >= 0.5).astype(int)


def get_losses(X, y):
    predictions = get_predictions(X)
    return (predictions != y).astype(float)


# --------------------------------------------------
# 4. Select reproducible reference samples
# --------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

data_size = min(REFERENCE_DATA_SIZE, len(X_train))
data_indices = rng.choice(
    len(X_train), size=data_size, replace=False
)

X_ref_data = X_train[data_indices]
y_ref_data = y_train[data_indices]

query_size = min(REFERENCE_QUERY_SIZE, len(X_train))
query_indices = rng.choice(
    len(X_train), size=query_size, replace=False
)

X_ref_queries = X_train[query_indices]

reference_losses = get_losses(X_ref_data, y_ref_data)


# --------------------------------------------------
# 5. Estimate RBF gamma from training data only
# --------------------------------------------------

gamma_size = min(GAMMA_SAMPLE_SIZE, len(X_train))
gamma_indices = rng.choice(
    len(X_train), size=gamma_size, replace=False
)

X_gamma = X_train[gamma_indices]

squared_distances = cdist(
    X_gamma, X_gamma, metric="sqeuclidean"
)

positive_distances = squared_distances[
    squared_distances > 0
]

if len(positive_distances) == 0:
    raise ValueError(
        "Could not estimate gamma: all sampled features are identical."
    )

median_squared_distance = float(np.median(positive_distances))
gamma = 1.0 / median_squared_distance

print(f"RBF gamma: {gamma:.8f}")


# --------------------------------------------------
# 6. Similarity-weighted loss
# --------------------------------------------------

def absolute_staleness(X_queries, X_data, losses):
    """
    Average similarity-weighted 0-1 loss.

    X_queries: feature vectors used as queries.
    X_data: labeled feature vectors where model loss is measured.
    losses: model's 0-1 loss on X_data.
    """

    if len(X_queries) == 0 or len(X_data) == 0:
        raise ValueError("Query and data sets must be non-empty.")

    if len(X_data) != len(losses):
        raise ValueError("Loss count must match data count.")

    distances = cdist(
        X_queries, X_data, metric="sqeuclidean"
    )

    similarities = np.exp(-gamma * distances)

    # Average over queries and labeled data points.
    cost = np.mean(similarities @ losses / len(X_data))

    return float(cost)


# Reference cost for Model V1 on its initial training period.
reference_cost = absolute_staleness(
    X_ref_queries,
    X_ref_data,
    reference_losses,
)

print(f"Reference absolute cost: {reference_cost:.8f}")


# --------------------------------------------------
# 7. Evaluate each validation batch
# --------------------------------------------------

results = []

for batch_id in range(VAL_START, VAL_END + 1):

    batch_df = val_df[val_df["batch_id"] == batch_id]

    if batch_df.empty:
        print(f"Warning: batch {batch_id} is empty; skipping.")
        continue

    X_batch = scaler.transform(batch_df[FEATURES])
    y_batch = batch_df[TARGET].to_numpy().astype(int)

    # Evaluate Model V1 on this batch's actual labels.
    batch_losses = get_losses(X_batch, y_batch)

    # Query set and labeled data are both from the current batch.
    batch_absolute_cost = absolute_staleness(
        X_batch,
        X_batch,
        batch_losses,
    )

    relative_cost = batch_absolute_cost - reference_cost

    results.append({
        "batch_id": int(batch_id),
        "n_records": int(len(batch_df)),
        "batch_error_rate": float(batch_losses.mean()),
        "reference_absolute_cost": reference_cost,
        "batch_absolute_cost": batch_absolute_cost,
        "relative_staleness": relative_cost,
    })

    print(
        f"Batch {batch_id:02d} | "
        f"Error={batch_losses.mean():.4f} | "
        f"Absolute={batch_absolute_cost:.6f} | "
        f"Relative={relative_cost:.6f}"
    )


# --------------------------------------------------
# 8. Save results and metadata
# --------------------------------------------------

results_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "validation_staleness.csv"
results_df.to_csv(csv_path, index=False)

metadata = {
    "model": "model_v1",
    "training_batches": [TRAIN_START, TRAIN_END],
    "validation_batches": [VAL_START, VAL_END],
    "reference_data_size": int(len(X_ref_data)),
    "reference_query_size": int(len(X_ref_queries)),
    "gamma_sample_size": int(gamma_size),
    "gamma": float(gamma),
    "median_squared_distance": median_squared_distance,
    "reference_absolute_cost": reference_cost,
    "kernel": "exp(-gamma * squared_euclidean_distance)",
    "loss": "0-1 classification loss at probability threshold 0.5",
    "relative_staleness_definition": (
        "current batch absolute cost minus initial training "
        "reference absolute cost"
    ),
    "normalization": "mean over queries and labeled data points",
    "seed": RANDOM_SEED,
    "note": (
        "Batch-specific adaptation of similarity-weighted loss. "
        "Reference and batch costs use their respective labeled "
        "data; the reference uses reproducible training samples."
    ),
}

metadata_path = OUTPUT_DIR / "staleness_metadata.json"

with open(metadata_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("\nStaleness calculation completed.")
print(f"Results: {csv_path}")
print(f"Metadata: {metadata_path}")
