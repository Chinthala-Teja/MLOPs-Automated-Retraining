"""
Compare retraining baselines on electricity validation batches 70-84.

Strategies:
  - Never retrain
  - Always retrain
  - Periodic retraining (every 3 batches)
  - CARA-T-inspired threshold policy selected in the prior sensitivity run

Run from the electricity-cara project root:
    python src/compare_baselines.py

This is an experimental proxy-cost comparison, not an exact reproduction of
the paper's full CARA-T optimizer. Validation metrics are for policy comparison;
do not treat them as an unbiased final test.
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow import keras


# Paths are relative to the project root; run this script from electricity-cara.
DATA_PATH = Path("data/elec/processed/electricity_batched.csv")
MODEL_PATH = Path("artifacts/model_v1/model.keras")
SCALER_PATH = Path("artifacts/model_v1/scaler.joblib")
POLICY_PATH = Path("artifacts/cara_t/cost_sensitivity/best_policy_per_kappa.csv")
OUTPUT_DIR = Path("artifacts/cara_t/baseline_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = ["period", "nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
TARGET = "class"
TRAIN_END = 69
VAL_START, VAL_END = 70, 84
RETRAIN_EPOCHS = 15
BATCH_SIZE = 32
SEED = 42
PERIODIC_INTERVAL = 3
CARA_KAPPA = 0.08

# Fallback is the threshold reported by the prior sensitivity experiment for kappa=0.08.
FALLBACK_CARA_THRESHOLD = -0.01685985


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def load_threshold():
    if POLICY_PATH.exists():
        policies = pd.read_csv(POLICY_PATH)
        if {"kappa", "selected_threshold"}.issubset(policies.columns):
            row = policies.iloc[(policies["kappa"] - CARA_KAPPA).abs().argsort()[:1]]
            if not row.empty and abs(float(row.iloc[0]["kappa"]) - CARA_KAPPA) < 1e-8:
                value = row.iloc[0]["selected_threshold"]
                if str(value).strip().lower() in ("-inf", "-infinity"):
                    return float("-inf")
                if str(value).strip().lower() in ("inf", "+inf", "infinity", "+infinity"):
                    return float("inf")
                return float(value)
    print("WARNING: prior policy summary not found; using recorded fallback threshold.")
    return FALLBACK_CARA_THRESHOLD


def predict_probabilities(model, X):
    return model.predict(X, verbose=0).reshape(-1)


def absolute_cost(model, X_queries, X_data, y_data, gamma):
    """Mean RBF-similarity-weighted 0-1 loss in the experiment's proxy units."""
    probs = predict_probabilities(model, X_data)
    predictions = (probs >= 0.5).astype(int)
    losses = (predictions != y_data).astype(float)
    distances = cdist(X_queries, X_data, metric="sqeuclidean")
    similarities = np.exp(-gamma * distances)
    return float(np.mean(similarities @ losses / len(X_data)))


def train_replacement_model(model_json, X, y, seed):
    """Train fresh weights using the initial model's architecture and fixed scaler."""
    set_seed(seed)
    model = keras.models.model_from_json(model_json)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(X, y, epochs=RETRAIN_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, verbose=0)
    return model


def batch_metrics(y_true, probs):
    preds = (probs >= 0.5).astype(int)
    try:
        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": auc,
        "error_rate": float(np.mean(preds != y_true)),
    }


def run_strategy(strategy, val_df, scaler, initial_model, model_json,
                 gamma, initial_ref_cost, initial_ref_X, initial_ref_y,
                 initial_ref_queries, kappa, cara_threshold):
    active_model = initial_model
    model_version = 1
    ref_cost = initial_ref_cost
    logs = []

    for batch_id in range(VAL_START, VAL_END + 1):
        batch = val_df[val_df["batch_id"] == batch_id]
        X = scaler.transform(batch[FEATURES].to_numpy())
        y = batch[TARGET].to_numpy(dtype=int)

        probs = predict_probabilities(active_model, X)
        metrics = batch_metrics(y, probs)
        cost = absolute_cost(active_model, X, X, y, gamma)
        relative_staleness = cost - ref_cost

        # Decide after evaluating this batch; any replacement serves the next batch.
        if strategy == "Never retrain":
            action = "KEEP"
        elif strategy == "Always retrain":
            action = "RETRAIN"
        elif strategy == "Periodic (every 3 batches)":
            action = "RETRAIN" if ((batch_id - VAL_START + 1) % PERIODIC_INTERVAL == 0) else "KEEP"
        elif strategy == "CARA-T-inspired":
            action = "RETRAIN" if relative_staleness >= cara_threshold else "KEEP"
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        decision_cost = float(kappa if action == "RETRAIN" else cost)
        version_before = model_version

        logs.append({
            "strategy": strategy,
            "batch_id": int(batch_id),
            "model_version_before_decision": int(version_before),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "error_rate": metrics["error_rate"],
            "absolute_cost": cost,
            "reference_cost": float(ref_cost),
            "relative_staleness": float(relative_staleness),
            "decision": action,
            "decision_cost": decision_cost,
            "model_version_after_decision": int(version_before + (action == "RETRAIN")),
        })

        if action == "RETRAIN":
            active_model = train_replacement_model(
                model_json, X, y, seed=SEED + int(batch_id)
            )
            model_version += 1
            # The replacement model's reference cost is measured on its training batch.
            ref_cost = absolute_cost(active_model, X, X, y, gamma)

    return pd.DataFrame(logs)


def main():
    for path in (DATA_PATH, MODEL_PATH, SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Run this script from the electricity-cara project root."
            )

    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET, "batch_id"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    train_df = data[data["batch_id"] <= TRAIN_END].copy()
    val_df = data[data["batch_id"].between(VAL_START, VAL_END)].copy()
    expected = list(range(VAL_START, VAL_END + 1))
    actual = sorted(val_df["batch_id"].astype(int).unique().tolist())
    if actual != expected:
        raise ValueError(f"Expected validation batches {expected}; found {actual}")

    scaler = joblib.load(SCALER_PATH)
    initial_model = keras.models.load_model(MODEL_PATH)
    model_json = initial_model.to_json()
    X_train = scaler.transform(train_df[FEATURES].to_numpy())
    y_train = train_df[TARGET].to_numpy(dtype=int)

    # Estimate gamma from training batches only.
    rng = np.random.default_rng(SEED)
    sample_size = min(1000, len(X_train))
    sample_idx = rng.choice(len(X_train), sample_size, replace=False)
    dists = cdist(X_train[sample_idx], X_train[sample_idx], metric="sqeuclidean")
    positive = dists[dists > 0]
    if len(positive) == 0:
        raise ValueError("Cannot estimate RBF gamma: no positive training distances.")
    gamma = 1.0 / float(np.median(positive))

    # Fixed initial reference sampled from training data only.
    ref_rng = np.random.default_rng(SEED)
    data_idx = ref_rng.choice(len(X_train), min(2000, len(X_train)), replace=False)
    query_idx = ref_rng.choice(len(X_train), min(500, len(X_train)), replace=False)
    ref_X, ref_y = X_train[data_idx], y_train[data_idx]
    ref_queries = X_train[query_idx]
    initial_ref_cost = absolute_cost(initial_model, ref_queries, ref_X, ref_y, gamma)
    cara_threshold = load_threshold()

    print(f"Validation batches: {VAL_START}-{VAL_END}")
    print(f"Validation records: {len(val_df)}")
    print(f"RBF gamma (train-only): {gamma:.8f}")
    print(f"Proxy retraining cost kappa: {CARA_KAPPA:.3f}")
    print(f"CARA-T-inspired threshold: {cara_threshold}")

    strategies = [
        "Never retrain",
        "Always retrain",
        f"Periodic (every {PERIODIC_INTERVAL} batches)",
        "CARA-T-inspired",
    ]
    all_logs = []
    for strategy in strategies:
        print(f"\nRunning: {strategy}")
        # Use the exact display name for periodic strategy in the runner.
        internal_name = "Periodic (every 3 batches)" if strategy.startswith("Periodic") else strategy
        log = run_strategy(
            internal_name, val_df, scaler, initial_model, model_json,
            gamma, initial_ref_cost, ref_X, ref_y, ref_queries,
            CARA_KAPPA, cara_threshold,
        )
        all_logs.append(log)

    logs = pd.concat(all_logs, ignore_index=True)
    logs.to_csv(OUTPUT_DIR / "baseline_batch_metrics.csv", index=False)

    summary_rows = []
    for strategy, group in logs.groupby("strategy", sort=False):
        summary_rows.append({
            "strategy": strategy,
            "kappa": CARA_KAPPA,
            "threshold": cara_threshold if strategy == "CARA-T-inspired" else np.nan,
            "periodic_interval": PERIODIC_INTERVAL if strategy.startswith("Periodic") else np.nan,
            "total_proxy_cost": float(group["decision_cost"].sum()),
            "mean_proxy_cost_per_batch": float(group["decision_cost"].mean()),
            "retrain_count": int((group["decision"] == "RETRAIN").sum()),
            "keep_count": int((group["decision"] == "KEEP").sum()),
            "mean_accuracy": float(group["accuracy"].mean()),
            "mean_precision": float(group["precision"].mean()),
            "mean_recall": float(group["recall"].mean()),
            "mean_f1": float(group["f1"].mean()),
            "mean_roc_auc": float(group["roc_auc"].mean()),
            "mean_error_rate": float(group["error_rate"].mean()),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "baseline_comparison.csv", index=False)
    metadata = {
        "validation_batches": [VAL_START, VAL_END],
        "training_batches": [0, TRAIN_END],
        "reserved_test_batches": [85, 99],
        "kappa": CARA_KAPPA,
        "cara_threshold": str(cara_threshold),
        "periodic_interval": PERIODIC_INTERVAL,
        "cost_definition": "KEEP pays current batch absolute proxy cost; RETRAIN pays kappa.",
        "limitations": [
            "CARA-T-inspired threshold policy, not exact reproduction of the paper optimizer.",
            "Threshold was selected using validation results, so this comparison is not an unbiased final test.",
            "Kappa is an assumed normalized proxy cost, not a measured monetary cost.",
            "Retraining uses only the current batch and reuses the fixed scaler.",
            "Metrics are calculated before the decision; a replacement model serves the next batch.",
            "The current implementation charges retraining cost for the final validation batch too, matching the prior replay convention, although that replacement is not used within this validation window."
        ],
        "gamma": gamma,
        "initial_reference_cost": initial_ref_cost,
    }
    with open(OUTPUT_DIR / "baseline_comparison_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Plot the main trade-offs.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].bar(summary["strategy"], summary["total_proxy_cost"])
    axes[0].set_title("Total proxy cost")
    axes[0].set_ylabel("Cost (normalized units)")
    axes[1].bar(summary["strategy"], summary["mean_error_rate"])
    axes[1].set_title("Mean batch error rate")
    axes[1].set_ylabel("Error rate")
    axes[2].bar(summary["strategy"], summary["retrain_count"])
    axes[2].set_title("Retraining count")
    axes[2].set_ylabel("Number of retrains")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    print("\n=== BASELINE COMPARISON ===")
    print(summary.to_string(index=False))
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
