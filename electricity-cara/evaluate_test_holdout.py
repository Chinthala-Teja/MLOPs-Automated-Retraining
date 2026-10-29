
"""
Electricity MLOps: Holdout evaluation and integrated audit.

Run from the electricity-cara project root:
    python -u evaluate_test_holdout.py

Includes:
- Sequential holdout replay, batches 85-99
- Never, always, periodic and CARA-T-inspired policies
- CSV summaries and experiment metadata
- PNG visualizations
- Complete terminal log
- Integrated threshold and proxy-cost audit

Important:
The CARA-T-inspired strategy is a project-specific adaptation,
not an exact reproduction of the original CARA-T algorithm.
"""

import atexit
import json
import random
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from tensorflow import keras


# ============================================================
# 1. CONFIGURATION AND OUTPUT DIRECTORIES
# ============================================================

DATA_PATH = Path("data/elec/processed/electricity_batched.csv")
MODEL_PATH = Path("artifacts/model_v1/model.keras")
SCALER_PATH = Path("artifacts/model_v1/scaler.joblib")

OUTPUT_DIR = Path("artifacts/cara_t/test_holdout")
PLOT_DIR = OUTPUT_DIR / "plots"

LOG_PATH = OUTPUT_DIR / "holdout_run_log.txt"
BATCH_RESULTS_PATH = OUTPUT_DIR / "test_batch_results.csv"
SUMMARY_PATH = OUTPUT_DIR / "test_strategy_summary.csv"
METADATA_PATH = OUTPUT_DIR / "experiment_metadata.json"
AUDIT_BATCH_PATH = OUTPUT_DIR / "holdout_audit.csv"
AUDIT_SUMMARY_PATH = OUTPUT_DIR / "holdout_audit_summary.csv"

VALIDATION_START = 70
VALIDATION_END = 84
N_THRESHOLD_QUANTILES = 11

VALIDATION_SENSITIVITY_PATH = (
    OUTPUT_DIR / "validation_threshold_sensitivity.csv"
)
VALIDATION_PLOT_PATH = (
    PLOT_DIR / "validation_threshold_sensitivity.png"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]
TARGET = "class"

TRAIN_END = 69
TEST_START = 85
TEST_END = 99

KAPPA = 0.08
VALIDATION_SELECTED_THRESHOLD = -0.01685984590411872

RETRAIN_EPOCHS = 15
BATCH_SIZE = 32
SEED = 42
PERIODIC_INTERVAL = 3

STRATEGIES = [
    "Never retrain",
    "Always retrain",
    "Periodic",
    "CARA-T-inspired",
]


# ============================================================
# 2. SAVE COMPLETE TERMINAL OUTPUT
# ============================================================

class Tee:
    """Write output to the terminal and a log file simultaneously."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()


_original_stdout = sys.stdout
_original_stderr = sys.stderr
_log_file = open(LOG_PATH, "w", encoding="utf-8", buffering=1)

sys.stdout = Tee(_original_stdout, _log_file)
sys.stderr = Tee(_original_stderr, _log_file)


def close_log():
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = _original_stdout
        sys.stderr = _original_stderr
        if not _log_file.closed:
            _log_file.close()


atexit.register(close_log)


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

def load_inputs():
    for path in (DATA_PATH, MODEL_PATH, SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    data = pd.read_csv(DATA_PATH)
    loaded_scaler = joblib.load(SCALER_PATH)
    loaded_model = keras.models.load_model(MODEL_PATH)

    required_columns = set(FEATURES + [TARGET, "batch_id"])
    missing = required_columns - set(data.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing)}"
        )

    if data[FEATURES + [TARGET, "batch_id"]].isnull().any().any():
        raise ValueError("Required dataset columns contain missing values.")

    training_data = data[data["batch_id"] <= TRAIN_END].copy()
    holdout_data = data[
        data["batch_id"].between(TEST_START, TEST_END)
    ].copy()

    expected = list(range(TEST_START, TEST_END + 1))
    actual = sorted(
        holdout_data["batch_id"].astype(int).unique().tolist()
    )

    if actual != expected:
        raise ValueError(
            f"Expected holdout batches {expected}; found {actual}"
        )

    if training_data.empty or holdout_data.empty:
        raise ValueError("Training or holdout data is empty.")

    train_features = loaded_scaler.transform(
        training_data[FEATURES].to_numpy()
    )
    train_labels = training_data[TARGET].to_numpy(dtype=int)

    for name, labels in (
        ("training", train_labels),
        ("holdout", holdout_data[TARGET].to_numpy(dtype=int)),
    ):
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError(
                f"{name.capitalize()} target must be binary 0/1."
            )

    return (
        data,
        training_data,
        holdout_data,
        loaded_scaler,
        loaded_model,
        train_features,
        train_labels,
        expected,
    )


(
    df,
    train_df,
    test_df,
    scaler,
    initial_model,
    X_train,
    y_train,
    expected_batches,
) = load_inputs()

MODEL_ARCHITECTURE = initial_model.to_json()


# ============================================================
# 4. RBF GAMMA AND TRAINING REFERENCE SAMPLE
# ============================================================

rng = np.random.default_rng(SEED)

gamma_size = min(1000, len(X_train))
gamma_indices = rng.choice(
    len(X_train), gamma_size, replace=False
)

distance_matrix = cdist(
    X_train[gamma_indices],
    X_train[gamma_indices],
    metric="sqeuclidean",
)
positive_distances = distance_matrix[distance_matrix > 0]

if len(positive_distances) == 0:
    raise ValueError("Unable to estimate RBF gamma.")

GAMMA = 1.0 / float(np.median(positive_distances))

reference_rng = np.random.default_rng(SEED)
reference_indices = reference_rng.choice(
    len(X_train), min(2000, len(X_train)), replace=False
)
query_indices = reference_rng.choice(
    len(X_train), min(500, len(X_train)), replace=False
)

INITIAL_REF_X = X_train[reference_indices].copy()
INITIAL_REF_Y = y_train[reference_indices].copy()
INITIAL_QUERY_X = X_train[query_indices].copy()

print("=" * 75)
print("ELECTRICITY MLOPS: HOLDOUT EVALUATION AND AUDIT")
print("=" * 75)
print("Training records:", len(train_df))
print("Holdout batches:", expected_batches)
print("Holdout records:", len(test_df))
print(f"RBF gamma: {GAMMA:.8f}")
print(
    "Initial threshold before validation selection: "
    f"{VALIDATION_SELECTED_THRESHOLD:.8f}"
)
print(f"Kappa: {KAPPA:.3f}")
print("Log file:", LOG_PATH)


# ============================================================
# 5. PREDICTION, COST AND METRICS
# ============================================================

def predict_probabilities(model, X):
    return model.predict(X, verbose=0).reshape(-1)


def predict_labels(model, X):
    return (predict_probabilities(model, X) >= 0.5).astype(int)


def absolute_cost(model, X_queries, X_data, y_data):
    """
    Mean RBF-similarity-weighted 0-1 loss.
    This is a project-specific normalized proxy, not monetary cost.
    """
    if len(X_queries) == 0 or len(X_data) == 0:
        raise ValueError("Cannot compute cost on empty data.")

    errors = (
        predict_labels(model, X_data) != y_data
    ).astype(float)

    distances = cdist(
        X_queries, X_data, metric="sqeuclidean"
    )
    similarities = np.exp(-GAMMA * distances)

    return float(
        np.mean(similarities @ errors / len(X_data))
    )


def calculate_metrics(model, X, y):
    probabilities = predict_probabilities(model, X)
    predictions = (probabilities >= 0.5).astype(int)

    try:
        auc = float(roc_auc_score(y, probabilities))
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": auc,
        "error_rate": float(np.mean(predictions != y)),
    }


def train_replacement_model(X, y, seed):
    """Train on the current batch after evaluating it."""
    keras.utils.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    model = keras.models.model_from_json(MODEL_ARCHITECTURE)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        X,
        y,
        epochs=RETRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0,
    )
    return model


# ============================================================
# 6. SEQUENTIAL REPLAY
# ============================================================

def replay_strategy(strategy):
    active_model = keras.models.load_model(MODEL_PATH)
    model_version = 1

    reference_X = INITIAL_REF_X.copy()
    reference_y = INITIAL_REF_Y.copy()
    reference_queries = INITIAL_QUERY_X.copy()

    reference_cost = absolute_cost(
        active_model,
        reference_queries,
        reference_X,
        reference_y,
    )

    logs = []

    for batch_id in expected_batches:
        batch = test_df[test_df["batch_id"] == batch_id]

        X_batch = scaler.transform(batch[FEATURES].to_numpy())
        y_batch = batch[TARGET].to_numpy(dtype=int)

        # Evaluate before making the retraining decision.
        metrics = calculate_metrics(active_model, X_batch, y_batch)
        batch_cost = absolute_cost(
            active_model, X_batch, X_batch, y_batch
        )
        relative_staleness = batch_cost - reference_cost

        if strategy == "Never retrain":
            should_retrain = False
        elif strategy == "Always retrain":
            should_retrain = True
        elif strategy == "Periodic":
            should_retrain = (
                (batch_id - TEST_START) % PERIODIC_INTERVAL == 0
            )
        elif strategy == "CARA-T-inspired":
            should_retrain = (
                relative_staleness
                >= VALIDATION_SELECTED_THRESHOLD
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        action = "RETRAIN" if should_retrain else "KEEP"
        proxy_cost = KAPPA if should_retrain else batch_cost
        version_before = model_version

        # Retrained model serves the next batch, not this one.
        if should_retrain:
            active_model = train_replacement_model(
                X_batch, y_batch, seed=SEED + int(batch_id)
            )
            model_version += 1

            reference_X = X_batch.copy()
            reference_y = y_batch.copy()
            reference_queries = X_batch.copy()

            reference_cost = absolute_cost(
                active_model,
                reference_queries,
                reference_X,
                reference_y,
            )

        logs.append({
            "strategy": strategy,
            "batch_id": int(batch_id),
            "model_version_before": version_before,
            "model_version_after": model_version,
            **metrics,
            "absolute_cost": float(batch_cost),
            "relative_staleness": float(relative_staleness),
            "threshold": (
                VALIDATION_SELECTED_THRESHOLD
                if strategy == "CARA-T-inspired"
                else np.nan
            ),
            "kappa": KAPPA,
            "decision": action,
            "proxy_decision_cost": float(proxy_cost),
        })

        print(
            f"{strategy:20s} | batch={batch_id:02d} | "
            f"error={metrics['error_rate']:.3f} | "
            f"staleness={relative_staleness:.4f} | {action}"
        )

    return pd.DataFrame(logs)


# ============================================================
# VALIDATION-ONLY THRESHOLD SENSITIVITY AND SELECTION
# ============================================================

def tune_threshold_on_validation():
    """
    Select a threshold using validation batches 70-84 only.

    Candidate thresholds come from quantiles of the standalone
    V1 relative-staleness values on validation data.

    Each candidate is evaluated with sequential test-then-train
    replay on validation batches. The existing project-specific
    proxy-cost objective is kept unchanged.
    """
    global VALIDATION_SELECTED_THRESHOLD

    validation_df = df[
        df["batch_id"].between(VALIDATION_START, VALIDATION_END)
    ].copy()

    validation_batches = list(
        range(VALIDATION_START, VALIDATION_END + 1)
    )
    actual_validation_batches = sorted(
        validation_df["batch_id"].astype(int).unique().tolist()
    )

    if actual_validation_batches != validation_batches:
        raise ValueError(
            "Validation data must contain batches "
            f"{VALIDATION_START}-{VALIDATION_END}. "
            f"Found: {actual_validation_batches}"
        )

    # Build candidate thresholds from the original V1 model.
    # This is validation-only and does not use holdout batches.
    v1_model = keras.models.load_model(MODEL_PATH)

    initial_reference_cost = absolute_cost(
        v1_model,
        INITIAL_QUERY_X,
        INITIAL_REF_X,
        INITIAL_REF_Y,
    )

    standalone_staleness = []

    for batch_id in validation_batches:
        batch = validation_df[
            validation_df["batch_id"] == batch_id
        ]

        X_batch = scaler.transform(
            batch[FEATURES].to_numpy()
        )
        y_batch = batch[TARGET].to_numpy(dtype=int)

        batch_cost = absolute_cost(
            v1_model,
            X_batch,
            X_batch,
            y_batch,
        )

        standalone_staleness.append(
            batch_cost - initial_reference_cost
        )

    quantiles = np.linspace(
        0.0, 1.0, N_THRESHOLD_QUANTILES
    )

    quantile_thresholds = np.quantile(
        standalone_staleness,
        quantiles,
    )

    # Include always-retrain and never-retrain boundary policies.
    thresholds = sorted(set(
        [-np.inf]
        + [float(value) for value in quantile_thresholds]
        + [np.inf]
    ))

    candidate_rows = []

    print("\n" + "=" * 75)
    print("VALIDATION-ONLY THRESHOLD SENSITIVITY")
    print("=" * 75)
    print(
        f"Validation batches: "
        f"{VALIDATION_START}-{VALIDATION_END}"
    )
    print(f"Candidate thresholds: {len(thresholds)}")
    print("Objective: existing normalized proxy cost")

    for candidate_index, threshold in enumerate(thresholds):
        active_model = keras.models.load_model(MODEL_PATH)
        model_version = 1

        reference_X = INITIAL_REF_X.copy()
        reference_y = INITIAL_REF_Y.copy()
        reference_queries = INITIAL_QUERY_X.copy()

        reference_cost = absolute_cost(
            active_model,
            reference_queries,
            reference_X,
            reference_y,
        )

        total_proxy_cost = 0.0
        retrain_count = 0
        keep_count = 0
        batch_errors = []

        for batch_id in validation_batches:
            batch = validation_df[
                validation_df["batch_id"] == batch_id
            ]

            X_batch = scaler.transform(
                batch[FEATURES].to_numpy()
            )
            y_batch = batch[TARGET].to_numpy(dtype=int)

            # Evaluate the current model before making the decision.
            metrics = calculate_metrics(
                active_model, X_batch, y_batch
            )

            batch_cost = absolute_cost(
                active_model,
                X_batch,
                X_batch,
                y_batch,
            )

            relative_staleness = batch_cost - reference_cost

            should_retrain = (
                relative_staleness >= threshold
            )

            if should_retrain:
                action_cost = KAPPA
                retrain_count += 1

                # Train only after the current batch has been evaluated.
                active_model = train_replacement_model(
                    X_batch,
                    y_batch,
                    seed=SEED + int(batch_id),
                )
                model_version += 1

                # Reset the reference condition after retraining.
                reference_X = X_batch.copy()
                reference_y = y_batch.copy()
                reference_queries = X_batch.copy()

                reference_cost = absolute_cost(
                    active_model,
                    reference_queries,
                    reference_X,
                    reference_y,
                )
            else:
                action_cost = batch_cost
                keep_count += 1

            total_proxy_cost += float(action_cost)
            batch_errors.append(metrics["error_rate"])

        candidate_rows.append({
            "candidate_index": candidate_index,
            "threshold": threshold,
            "threshold_label": (
                "-inf" if np.isneginf(threshold)
                else "+inf" if np.isposinf(threshold)
                else f"{threshold:.8f}"
            ),
            "total_proxy_cost": total_proxy_cost,
            "mean_proxy_cost": (
                total_proxy_cost / len(validation_batches)
            ),
            "retrain_count": retrain_count,
            "keep_count": keep_count,
            "mean_error_rate": float(np.mean(batch_errors)),
            "selection_split": "validation",
        })

    sensitivity = pd.DataFrame(candidate_rows)

    # Select lowest proxy cost. For ties, prefer fewer retrains.
    sensitivity = sensitivity.sort_values(
        ["total_proxy_cost", "retrain_count", "threshold"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    best_row = sensitivity.iloc[0]
    VALIDATION_SELECTED_THRESHOLD = float(best_row["threshold"])

    # Save every candidate, including the selected threshold.
    sensitivity["selected"] = np.isclose(
        sensitivity["threshold"].to_numpy(dtype=float),
        VALIDATION_SELECTED_THRESHOLD,
        equal_nan=False,
    )

    # Handle infinities explicitly because np.isclose handles them,
    # but retain a clear textual representation in the CSV.
    sensitivity.to_csv(
        VALIDATION_SENSITIVITY_PATH,
        index=False,
    )

    print("\nCandidate results ranked by proxy cost:")
    print(
        sensitivity[
            [
                "threshold_label",
                "total_proxy_cost",
                "retrain_count",
                "keep_count",
                "mean_error_rate",
                "selected",
            ]
        ].round(5).to_string(index=False)
    )

    print("\nSelected validation threshold:")
    print(f"  {VALIDATION_SELECTED_THRESHOLD}")
    print(f"  Validation proxy cost: {best_row['total_proxy_cost']:.6f}")
    print(f"  Retraining decisions: {int(best_row['retrain_count'])}")
    print(
        "  Mean validation error: "
        f"{best_row['mean_error_rate']:.6f}"
    )

    # Plot threshold against validation proxy cost.
    # Use candidate index on x-axis so +/- infinity are displayable.
    plot_data = sensitivity.sort_values("candidate_index")

    plt.figure(figsize=(12, 6))
    plt.plot(
        plot_data["candidate_index"],
        plot_data["total_proxy_cost"],
        marker="o",
    )

    selected_index = int(best_row["candidate_index"])
    selected_cost = float(best_row["total_proxy_cost"])

    plt.scatter(
        [selected_index],
        [selected_cost],
        marker="*",
        s=220,
        label="Selected on validation",
    )

    plt.xlabel("Threshold candidate index")
    plt.ylabel("Total validation proxy cost")
    plt.title("Validation-Only Threshold Sensitivity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(VALIDATION_PLOT_PATH, dpi=180)
    plt.close()

    print("\nSaved validation sensitivity CSV:")
    print(" ", VALIDATION_SENSITIVITY_PATH)
    print("Saved validation sensitivity PNG:")
    print(" ", VALIDATION_PLOT_PATH)

    return sensitivity


# Run threshold selection BEFORE evaluating holdout batches.
validation_sensitivity = tune_threshold_on_validation()


# ============================================================
# 7. RUN STRATEGIES AND SAVE RESULTS
# ============================================================

all_logs = []

for strategy in STRATEGIES:
    print("\n" + "=" * 75)
    print("Running:", strategy)
    print("=" * 75)
    all_logs.append(replay_strategy(strategy))

batch_results = pd.concat(all_logs, ignore_index=True)
batch_results.to_csv(BATCH_RESULTS_PATH, index=False)


# ============================================================
# 8. AGGREGATE SUMMARY AND METADATA
# ============================================================

summary = (
    batch_results.groupby("strategy", sort=False)
    .agg(
        total_proxy_cost=("proxy_decision_cost", "sum"),
        mean_proxy_cost=("proxy_decision_cost", "mean"),
        retrain_count=(
            "decision",
            lambda values: int((values == "RETRAIN").sum()),
        ),
        keep_count=(
            "decision",
            lambda values: int((values == "KEEP").sum()),
        ),
        mean_accuracy=("accuracy", "mean"),
        mean_precision=("precision", "mean"),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
        mean_error_rate=("error_rate", "mean"),
    )
    .reset_index()
)
summary.to_csv(SUMMARY_PATH, index=False)

metadata = {
    "experiment": "Electricity chronological holdout replay",
    "training_batches": [0, TRAIN_END],
    "validation_batches": [70, 84],
    "holdout_batches": expected_batches,
    "training_records": int(len(train_df)),
    "holdout_records": int(len(test_df)),
    "threshold_source": (
        "Selected by sequential proxy-cost minimization on "
        "validation batches 70-84"
    ),
    "validation_sensitivity_csv": str(
        VALIDATION_SENSITIVITY_PATH
    ),
    "validation_sensitivity_plot": str(
        VALIDATION_PLOT_PATH
    ),
    "threshold": VALIDATION_SELECTED_THRESHOLD,
    "kappa": KAPPA,
    "periodic_interval": PERIODIC_INTERVAL,
    "retrain_epochs": RETRAIN_EPOCHS,
    "batch_size": BATCH_SIZE,
    "random_seed": SEED,
    "rbf_gamma": GAMMA,
    "note": (
        "CARA-T-inspired adaptation, not exact paper reproduction. "
        "Proxy cost is kappa for RETRAIN and absolute batch cost for KEEP. "
        "Batch labels are assumed available after evaluation for the "
        "test-then-train simulation. Holdout batches were previously "
        "inspected during EDA; threshold is not tuned on them here."
    ),
}
with open(METADATA_PATH, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=2)


# ============================================================
# 9. GENERATE FOUR MAIN PNG PLOTS
# ============================================================

plt.figure(figsize=(10, 6))
plt.bar(summary["strategy"], summary["total_proxy_cost"])
plt.ylabel("Total proxy decision cost (normalized units)")
plt.title("Holdout: Total Proxy Cost by Retraining Strategy")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "holdout_total_proxy_cost.png", dpi=180)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(summary["strategy"], summary["mean_error_rate"])
plt.ylabel("Mean batch error rate")
plt.title("Holdout: Predictive Error by Retraining Strategy")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "holdout_error_rate.png", dpi=180)
plt.close()

plt.figure(figsize=(12, 6))
for strategy in STRATEGIES:
    subset = batch_results[batch_results["strategy"] == strategy]
    plt.plot(
        subset["batch_id"],
        subset["accuracy"],
        marker="o",
        label=strategy,
    )
plt.xlabel("Chronological batch ID")
plt.ylabel("Accuracy")
plt.title("Holdout: Accuracy Across Batches 85-99")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "holdout_accuracy_by_batch.png", dpi=180)
plt.close()

plt.figure(figsize=(12, 5))
decision_colors = {"KEEP": "tab:blue", "RETRAIN": "tab:orange"}

for y_position, strategy in enumerate(STRATEGIES):
    subset = batch_results[batch_results["strategy"] == strategy]
    for _, row in subset.iterrows():
        plt.scatter(
            row["batch_id"],
            y_position,
            color=decision_colors[row["decision"]],
            s=65,
            marker="s",
        )

plt.yticks(range(len(STRATEGIES)), STRATEGIES)
plt.xticks(expected_batches)
plt.xlabel("Chronological batch ID")
plt.title("Holdout: KEEP vs RETRAIN Decisions")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(PLOT_DIR / "holdout_retraining_decisions.png", dpi=180)
plt.close()


# ============================================================
# 10. INTEGRATED AUDIT: NO SECOND FILE REQUIRED
# ============================================================

print("\n" + "=" * 75)
print("INTEGRATED HOLDOUT AUDIT")
print("=" * 75)

audit_rows = []
expected_batch_set = set(expected_batches)

print("\nA. Batch coverage")

for strategy in STRATEGIES:
    subset = batch_results[batch_results["strategy"] == strategy]
    observed_batches = set(subset["batch_id"].astype(int))

    complete = (
        len(subset) == len(expected_batches)
        and observed_batches == expected_batch_set
    )

    print(
        f"{strategy:20s}: rows={len(subset)}, "
        f"distinct batches={len(observed_batches)}, "
        f"complete={complete}"
    )

    if not complete:
        print("  WARNING: Missing or duplicate batch records.")

print("\nB. Threshold decision verification")

cara = batch_results[
    batch_results["strategy"] == "CARA-T-inspired"
].copy()

cara["expected_decision"] = np.where(
    cara["relative_staleness"] >= VALIDATION_SELECTED_THRESHOLD,
    "RETRAIN",
    "KEEP",
)
cara["decision_matches_threshold"] = (
    cara["decision"] == cara["expected_decision"]
)

threshold_matches = int(
    cara["decision_matches_threshold"].sum()
)
threshold_total = len(cara)

print(
    f"Threshold = {VALIDATION_SELECTED_THRESHOLD:.8f}; "
    f"matching decisions = {threshold_matches}/{threshold_total}"
)

if threshold_matches != threshold_total:
    print("WARNING: Threshold and recorded decisions disagree.")

cara["threshold"] = VALIDATION_SELECTED_THRESHOLD
cara.to_csv(AUDIT_BATCH_PATH, index=False)

print("\nC. Proxy-cost reconciliation")

for strategy in STRATEGIES:
    subset = batch_results[batch_results["strategy"] == strategy]

    retrain_mask = subset["decision"] == "RETRAIN"
    keep_mask = subset["decision"] == "KEEP"

    retrain_count = int(retrain_mask.sum())
    keep_count = int(keep_mask.sum())

    retrain_cost = float(
        subset.loc[retrain_mask, "proxy_decision_cost"].sum()
    )
    keep_cost = float(
        subset.loc[keep_mask, "proxy_decision_cost"].sum()
    )
    total_cost = float(subset["proxy_decision_cost"].sum())

    expected_retrain_cost = retrain_count * KAPPA
    retrain_cost_matches = np.isclose(
        retrain_cost, expected_retrain_cost, atol=1e-6
    )

    saved = summary[summary["strategy"] == strategy]
    saved_total = (
        float(saved.iloc[0]["total_proxy_cost"])
        if not saved.empty else float("nan")
    )
    summary_matches = np.isclose(
        total_cost, saved_total, atol=1e-6
    )

    audit_rows.append({
        "strategy": strategy,
        "batch_count": len(subset),
        "retrain_count": retrain_count,
        "keep_count": keep_count,
        "retrain_cost": retrain_cost,
        "keep_cost": keep_cost,
        "recalculated_total_proxy_cost": total_cost,
        "saved_summary_total_proxy_cost": saved_total,
        "mean_proxy_cost": total_cost / len(subset),
        "mean_error_rate": float(subset["error_rate"].mean()),
        "retrain_cost_matches_kappa": bool(retrain_cost_matches),
        "summary_matches_batch_results": bool(summary_matches),
    })

    print(
        f"{strategy:20s} | KEEP={keep_count:2d} | "
        f"RETRAIN={retrain_count:2d} | "
        f"retrain cost={retrain_cost:.5f} | "
        f"keep cost={keep_cost:.5f} | "
        f"total={total_cost:.5f} | "
        f"summary consistent={summary_matches}"
    )

audit_summary = pd.DataFrame(audit_rows)
audit_summary.to_csv(AUDIT_SUMMARY_PATH, index=False)

print("\nD. Audit plot")

plt.figure(figsize=(12, 6))
plt.plot(
    cara["batch_id"],
    cara["relative_staleness"],
    marker="o",
    label="Relative staleness",
)
plt.axhline(
    VALIDATION_SELECTED_THRESHOLD,
    linestyle="--",
    label="Fixed validation threshold",
)
plt.axhline(
    0,
    linestyle=":",
    label="Zero relative staleness",
)

for _, row in cara.iterrows():
    label = "R" if row["decision"] == "RETRAIN" else "K"
    plt.annotate(
        label,
        (row["batch_id"], row["relative_staleness"]),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
    )

plt.xlabel("Holdout batch ID")
plt.ylabel("Relative staleness")
plt.title("Holdout Audit: Staleness and Fixed Threshold")
plt.xticks(expected_batches)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "holdout_threshold_audit.png", dpi=180)
plt.close()

print(" ", PLOT_DIR / "holdout_threshold_audit.png")

print("\nE. Final audit artifacts")
print(" ", AUDIT_BATCH_PATH)
print(" ", AUDIT_SUMMARY_PATH)

print("\n" + "=" * 75)
print("FINAL HOLDOUT SUMMARY")
print("=" * 75)
print(summary.round(4).to_string(index=False))

print("\nSaved artifacts:")
for path in (
    LOG_PATH,
    BATCH_RESULTS_PATH,
    SUMMARY_PATH,
    METADATA_PATH,
    AUDIT_BATCH_PATH,
    AUDIT_SUMMARY_PATH,
):
    print(" ", path)

print("\nSaved PNG plots:")
for path in sorted(PLOT_DIR.glob("*.png")):
    print(" ", path)

print("\nEvaluation and integrated audit completed.")
print("=" * 75)
