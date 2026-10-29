
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



# --------------------------------------------------
# 9. Generate visualization artifacts
# --------------------------------------------------

import matplotlib
matplotlib.use("Agg")  # Save figures without opening windows
import matplotlib.pyplot as plt

PLOT_DIR = OUTPUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

if results_df.empty:
    raise ValueError("No validation results available to plot.")

# Shared plot settings
batch_ids = results_df["batch_id"].to_numpy()


# --------------------------------------------------
# Plot 1: Relative staleness by batch
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 5))

ax.bar(
    batch_ids.astype(str),
    results_df["relative_staleness"],
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=1.2,
    label="Reference cost",
)

ax.set_title("CARA-T: Relative Staleness by Validation Batch")
ax.set_xlabel("Batch ID")
ax.set_ylabel("Relative Staleness")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "relative_staleness.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# Plot 2: Classification error rate by batch
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    batch_ids,
    results_df["batch_error_rate"],
    marker="o",
    linewidth=1.8,
    label="Model V1 error rate",
)

ax.set_title("Model V1: Classification Error Across Batches")
ax.set_xlabel("Batch ID")
ax.set_ylabel("Error Rate")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "batch_error_rate.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# Plot 3: Absolute cost versus reference cost
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    batch_ids,
    results_df["batch_absolute_cost"],
    marker="o",
    linewidth=1.8,
    label="Batch absolute cost",
)

ax.axhline(
    reference_cost,
    linestyle="--",
    linewidth=1.5,
    label="Training reference cost",
)

ax.set_title("Similarity-Weighted Loss: Batch vs Reference")
ax.set_xlabel("Batch ID")
ax.set_ylabel("Absolute Cost")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "absolute_cost_vs_reference.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# Plot 4: Error rate versus relative staleness
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))

scatter = ax.scatter(
    results_df["batch_error_rate"],
    results_df["relative_staleness"],
    c=batch_ids,
    s=75,
)

for _, row in results_df.iterrows():
    ax.annotate(
        str(int(row["batch_id"])),
        (
            row["batch_error_rate"],
            row["relative_staleness"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

fig.colorbar(scatter, ax=ax, label="Batch ID")
ax.axhline(0, linestyle="--", linewidth=1)

ax.set_title("Classification Error vs Relative Staleness")
ax.set_xlabel("Classification Error Rate")
ax.set_ylabel("Relative Staleness")
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "error_vs_staleness.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# Plot 5: Compact monitoring dashboard
# --------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

axes[0].plot(
    batch_ids,
    results_df["batch_error_rate"],
    marker="o",
)
axes[0].set_ylabel("Error Rate")
axes[0].set_title("Validation Monitoring Dashboard")
axes[0].grid(True, alpha=0.3)

axes[1].plot(
    batch_ids,
    results_df["batch_absolute_cost"],
    marker="o",
)
axes[1].axhline(
    reference_cost,
    linestyle="--",
    label="Reference cost",
)
axes[1].set_ylabel("Absolute Cost")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].bar(
    batch_ids.astype(str),
    results_df["relative_staleness"],
)
axes[2].axhline(0, linestyle="--", linewidth=1)
axes[2].set_ylabel("Relative Staleness")
axes[2].set_xlabel("Batch ID")
axes[2].grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "monitoring_dashboard.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# 10. Save a concise summary
# --------------------------------------------------

summary = {
    "validation_batches": int(len(results_df)),
    "reference_absolute_cost": float(reference_cost),
    "mean_error_rate": float(results_df["batch_error_rate"].mean()),
    "mean_absolute_cost": float(
        results_df["batch_absolute_cost"].mean()
    ),
    "mean_relative_staleness": float(
        results_df["relative_staleness"].mean()
    ),
    "max_relative_staleness": float(
        results_df["relative_staleness"].max()
    ),
    "max_staleness_batch": int(
        results_df.loc[
            results_df["relative_staleness"].idxmax(),
            "batch_id",
        ]
    ),
    "min_relative_staleness": float(
        results_df["relative_staleness"].min()
    ),
    "min_staleness_batch": int(
        results_df.loc[
            results_df["relative_staleness"].idxmin(),
            "batch_id",
        ]
    ),
    "plots": [
        "relative_staleness.png",
        "batch_error_rate.png",
        "absolute_cost_vs_reference.png",
        "error_vs_staleness.png",
        "monitoring_dashboard.png",
    ],
}

with open(
    OUTPUT_DIR / "visualization_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)

print("\nVisualization artifacts generated:")

for filename in summary["plots"]:
    print(f"  {PLOT_DIR / filename}")

print(f"  {OUTPUT_DIR / 'visualization_summary.json'}")
