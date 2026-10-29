"""
Corrected sequential CARA-T-style replay with kappa sensitivity.
Run from the electricity-cara project root:
    python cara_sequential_replay.py

Experimental proxy objective:
  KEEP    -> current batch absolute cost
  RETRAIN -> kappa

This is not an exact reproduction of the paper's full CARA-T optimizer.
Validation batches 70-84 are used for policy selection; test batches 85-99
are not used by this script.
"""
import json
import random
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tensorflow import keras
import tensorflow as tf


# 1. Configuration
DATA_PATH = Path("data/elec/processed/electricity_batched.csv")
MODEL_PATH = Path("artifacts/model_v1/model.keras")
SCALER_PATH = Path("artifacts/model_v1/scaler.joblib")

OUTPUT_DIR = Path("artifacts/cara_t")
PLOT_DIR = OUTPUT_DIR / "plots"
SENSITIVITY_DIR = OUTPUT_DIR / "cost_sensitivity"
SENSITIVITY_PLOT_DIR = SENSITIVITY_DIR / "plots"
for directory in (OUTPUT_DIR, PLOT_DIR, SENSITIVITY_DIR, SENSITIVITY_PLOT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "period", "nswprice", "nswdemand",
    "vicprice", "vicdemand", "transfer",
]
TARGET = "class"
TRAIN_END = 69
VAL_START, VAL_END = 70, 84
RETRAIN_EPOCHS = 15
BATCH_SIZE = 32
SEED = 42

# Assumed normalized retraining costs for sensitivity analysis.
KAPPA_VALUES = [0.02, 0.04, 0.06, 0.08, 0.12]
N_THRESHOLDS = 7


# 2. Load data, scaler, and V1 model
for path in (DATA_PATH, MODEL_PATH, SCALER_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

df = pd.read_csv(DATA_PATH)
scaler = joblib.load(SCALER_PATH)
initial_model = keras.models.load_model(MODEL_PATH)

required = set(FEATURES + [TARGET, "batch_id"])
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Dataset missing columns: {sorted(missing)}")

train_df = df[df["batch_id"] <= TRAIN_END].copy()
val_df = df[df["batch_id"].between(VAL_START, VAL_END)].copy()
if train_df.empty or val_df.empty:
    raise ValueError("Training or validation split is empty.")

actual_val_batches = sorted(val_df["batch_id"].astype(int).unique().tolist())
expected_val_batches = list(range(VAL_START, VAL_END + 1))
if actual_val_batches != expected_val_batches:
    raise ValueError(
        f"Expected validation batches {expected_val_batches}; "
        f"found {actual_val_batches}."
    )

X_train = scaler.transform(train_df[FEATURES].to_numpy())
y_train = train_df[TARGET].to_numpy(dtype=int)

# Preserve architecture independently of the loaded V1 model's weights/state.
INITIAL_MODEL_JSON = initial_model.to_json()

print("Initial training records:", len(train_df))
print("Validation batches:", actual_val_batches)
print("Validation records:", len(val_df))


# 3. Estimate RBF gamma from training data only
rng = np.random.default_rng(SEED)
gamma_size = min(1000, len(X_train))
gamma_idx = rng.choice(len(X_train), gamma_size, replace=False)
gamma_dist = cdist(X_train[gamma_idx], X_train[gamma_idx], metric="sqeuclidean")
positive_dist = gamma_dist[gamma_dist > 0]
if len(positive_dist) == 0:
    raise ValueError("Cannot estimate RBF gamma: no positive distances.")
gamma = 1.0 / float(np.median(positive_dist))


def predict_labels(model, X):
    probabilities = model.predict(X, verbose=0).reshape(-1)
    return (probabilities >= 0.5).astype(int)


def classification_losses(model, X, y):
    return (predict_labels(model, X) != y).astype(float)


def absolute_cost(model, X_queries, X_data, y_data):
    """Mean RBF similarity-weighted 0-1 loss; normalized experimental units."""
    if len(X_queries) == 0 or len(X_data) == 0:
        raise ValueError("absolute_cost received an empty array.")
    losses = classification_losses(model, X_data, y_data)
    distances = cdist(X_queries, X_data, metric="sqeuclidean")
    similarities = np.exp(-gamma * distances)
    return float(np.mean(similarities @ losses / len(X_data)))


ref_rng = np.random.default_rng(SEED)
ref_data_idx = ref_rng.choice(len(X_train), min(2000, len(X_train)), replace=False)
ref_query_idx = ref_rng.choice(len(X_train), min(500, len(X_train)), replace=False)
X_initial_ref = X_train[ref_data_idx]
y_initial_ref = y_train[ref_data_idx]
X_initial_queries = X_train[ref_query_idx]

initial_reference_cost = absolute_cost(
    initial_model, X_initial_queries, X_initial_ref, y_initial_ref
)
print(f"RBF gamma: {gamma:.8f}")
print(f"Initial reference cost: {initial_reference_cost:.6f}")


# 4. Fresh replacement model
def train_replacement_model(X, y, seed):
    """Train from fresh random weights using V1's architecture and fixed scaler."""
    keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    replacement = keras.models.model_from_json(INITIAL_MODEL_JSON)
    replacement.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    replacement.fit(
        X, y,
        epochs=RETRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0,
    )
    return replacement


# 5. Candidate thresholds from the preliminary fixed-V1 validation staleness
preliminary_path = OUTPUT_DIR / "validation_staleness.csv"
if preliminary_path.exists():
    preliminary = pd.read_csv(preliminary_path)
    if "relative_staleness" not in preliminary.columns:
        raise ValueError(f"Missing relative_staleness in {preliminary_path}")
    prior_values = preliminary["relative_staleness"].dropna().to_numpy()
else:
    prior_values = np.array([])

if len(prior_values):
    finite_thresholds = np.unique(
        np.quantile(prior_values, np.linspace(0, 1, N_THRESHOLDS))
    )
else:
    finite_thresholds = np.linspace(-0.05, 0.15, N_THRESHOLDS)

thresholds = np.unique(np.concatenate((
    [float("-inf")],
    finite_thresholds,
    [float("inf")],
)))
print("Candidate thresholds:", thresholds)


# 6. Chronological replay for one threshold and one kappa
def replay_threshold(tau, kappa):
    """
    Evaluate batch t with the active model before deciding.
    If RETRAIN, train on batch t after evaluation; replacement serves t+1.
    """
    active_model = initial_model
    model_version = 1

    reference_X = X_initial_ref
    reference_y = y_initial_ref
    reference_queries = X_initial_queries
    ref_cost = initial_reference_cost
    logs = []

    for batch_id in range(VAL_START, VAL_END + 1):
        batch = val_df[val_df["batch_id"] == batch_id]
        if batch.empty:
            continue

        X_batch = scaler.transform(batch[FEATURES].to_numpy())
        y_batch = batch[TARGET].to_numpy(dtype=int)

        version_before = model_version
        error_rate = float(classification_losses(active_model, X_batch, y_batch).mean())
        batch_cost = absolute_cost(active_model, X_batch, X_batch, y_batch)

        reference_cost_before = ref_cost
        relative_staleness = batch_cost - reference_cost_before
        should_retrain = bool(relative_staleness >= tau)

        # Current batch was served by the old model. Retraining affects next batch.
        action = "RETRAIN" if should_retrain else "KEEP"
        decision_cost = float(kappa if should_retrain else batch_cost)

        if should_retrain:
            active_model = train_replacement_model(
                X_batch, y_batch, seed=SEED + int(batch_id)
            )
            model_version += 1

            # Establish the replacement model's reference on its training batch.
            reference_X = X_batch.copy()
            reference_y = y_batch.copy()
            reference_queries = X_batch.copy()
            ref_cost = absolute_cost(
                active_model, reference_queries, reference_X, reference_y
            )

        logs.append({
            "batch_id": int(batch_id),
            "model_version_before_decision": int(version_before),
            "model_version_after_decision": int(model_version),
            "error_rate": error_rate,
            "absolute_cost": float(batch_cost),
            "reference_cost_before_decision": float(reference_cost_before),
            "relative_staleness": float(relative_staleness),
            "threshold": float(tau),
            "kappa": float(kappa),
            "decision": action,
            "decision_cost": decision_cost,
        })

    log_df = pd.DataFrame(logs)
    if log_df.empty:
        raise ValueError("Replay produced no batch results.")

    return {
        "threshold": float(tau),
        "kappa": float(kappa),
        "log": log_df,
        "total_cost": float(log_df["decision_cost"].sum()),
        "mean_cost": float(log_df["decision_cost"].mean()),
        "retrain_count": int((log_df["decision"] == "RETRAIN").sum()),
        "keep_count": int((log_df["decision"] == "KEEP").sum()),
        "mean_error": float(log_df["error_rate"].mean()),
    }


# 7. Threshold optimization independently for each kappa
all_results = []
best_logs = []
best_summaries = []

for kappa in KAPPA_VALUES:
    print("\n" + "=" * 65)
    print(f"Testing retraining cost kappa = {kappa:.3f}")
    print("=" * 65)
    runs = []

    for i, tau in enumerate(thresholds, start=1):
        print(f"  Threshold {i}/{len(thresholds)}: {tau}")
        run = replay_threshold(tau, kappa)
        runs.append(run)
        print(
            f"    Cost={run['total_cost']:.6f}, "
            f"KEEP={run['keep_count']}, "
            f"RETRAIN={run['retrain_count']}, "
            f"Mean error={run['mean_error']:.4f}"
        )

    for run in runs:
        tau = run["threshold"]
        tau_label = (
            "-inf" if np.isneginf(tau)
            else "inf" if np.isposinf(tau)
            else float(tau)
        )
        all_results.append({
            "kappa": float(kappa),
            "threshold": tau_label,
            "total_cost": run["total_cost"],
            "mean_cost": run["mean_cost"],
            "keep_count": run["keep_count"],
            "retrain_count": run["retrain_count"],
            "mean_error": run["mean_error"],
        })

    best_idx = int(np.argmin([run["total_cost"] for run in runs]))
    best_run = runs[best_idx]
    best_run["log"].to_csv(
        SENSITIVITY_DIR / f"best_replay_kappa_{kappa:.3f}.csv",
        index=False,
    )
    best_logs.append(best_run["log"].assign(kappa=kappa))

    selected_tau = (
        "-inf" if np.isneginf(best_run["threshold"])
        else "inf" if np.isposinf(best_run["threshold"])
        else float(best_run["threshold"])
    )
    best_summaries.append({
        "kappa": float(kappa),
        "selected_threshold": selected_tau,
        "total_cost": best_run["total_cost"],
        "mean_cost": best_run["mean_cost"],
        "keep_count": best_run["keep_count"],
        "retrain_count": best_run["retrain_count"],
        "mean_error": best_run["mean_error"],
    })

    print("\nBest policy for this kappa:")
    print(f"  Threshold: {selected_tau}")
    print(f"  Total proxy cost: {best_run['total_cost']:.6f}")
    print(f"  KEEP: {best_run['keep_count']}")
    print(f"  RETRAIN: {best_run['retrain_count']}")
    print(f"  Mean batch error: {best_run['mean_error']:.4f}")


# 8. Save CSV and JSON outputs
comparison_df = pd.DataFrame(all_results)
best_df = pd.DataFrame(best_summaries)
combined_logs = pd.concat(best_logs, ignore_index=True)

comparison_df.to_csv(SENSITIVITY_DIR / "all_threshold_results.csv", index=False)
best_df.to_csv(SENSITIVITY_DIR / "best_policy_per_kappa.csv", index=False)
combined_logs.to_csv(SENSITIVITY_DIR / "best_policy_batch_logs.csv", index=False)

summary = {
    "method": "sequential validation replay with kappa sensitivity",
    "train_batches": [0, TRAIN_END],
    "validation_batches": [VAL_START, VAL_END],
    "test_batches_reserved": [85, 99],
    "kappa_values": KAPPA_VALUES,
    "candidate_thresholds": [
        "-inf" if np.isneginf(t) else
        "inf" if np.isposinf(t) else float(t)
        for t in thresholds
    ],
    "gamma": float(gamma),
    "initial_reference_cost": float(initial_reference_cost),
    "cost_definition": "KEEP pays current batch absolute cost; RETRAIN pays kappa.",
    "limitations": [
        "Experimental CARA-T-style adaptation, not exact paper reproduction.",
        "Threshold candidates come from standalone V1 validation staleness.",
        "Kappa values are assumed normalized costs, not measured monetary costs.",
        "Retraining uses only the current batch and reuses the fixed scaler.",
        "Validation is used for policy selection, not final unbiased testing.",
    ],
    "best_policy_per_kappa": best_summaries,
}
with open(SENSITIVITY_DIR / "sensitivity_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)


# 9. Plots
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(best_df["kappa"], best_df["total_cost"], marker="o")
ax.set_title("Best Validation Proxy Cost vs Retraining Cost")
ax.set_xlabel("Retraining cost (kappa)")
ax.set_ylabel("Minimum total proxy cost")
ax.set_xticks(KAPPA_VALUES)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SENSITIVITY_PLOT_DIR / "best_cost_vs_kappa.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(best_df["kappa"], best_df["retrain_count"], marker="o", label="RETRAIN")
ax.plot(best_df["kappa"], best_df["keep_count"], marker="s", label="KEEP")
ax.set_title("Selected Policy Decisions vs Retraining Cost")
ax.set_xlabel("Retraining cost (kappa)")
ax.set_ylabel("Number of decisions")
ax.set_xticks(KAPPA_VALUES)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SENSITIVITY_PLOT_DIR / "decisions_vs_kappa.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(best_df["kappa"], best_df["mean_error"], marker="o")
ax.set_title("Selected Policy Error vs Retraining Cost")
ax.set_xlabel("Retraining cost (kappa)")
ax.set_ylabel("Mean batch error rate")
ax.set_xticks(KAPPA_VALUES)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(SENSITIVITY_PLOT_DIR / "error_vs_kappa.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\n" + "=" * 65)
print("COST SENSITIVITY EXPERIMENT COMPLETE")
print("=" * 65)
print(best_df.to_string(index=False))
print(f"\nResults saved in: {SENSITIVITY_DIR}")
print(f"Plots saved in: {SENSITIVITY_PLOT_DIR}")
