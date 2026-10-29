
from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

# --------------------------------------------------
# 1. Reproducibility
# --------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# --------------------------------------------------
# 2. Paths and configuration
# Run this script from the electricity-cara folder.
# --------------------------------------------------
ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "elec" / "raw" / "elec.csv"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Run this script from the electricity-cara folder."
    )

# --------------------------------------------------
# 3. Load and validate dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

FEATURES = [
    "period",
    "nswprice",
    "nswdemand",
    "vicprice",
    "vicdemand",
    "transfer",
]
TARGET = "class"

required_columns = FEATURES + [TARGET]
missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

if df[required_columns].isnull().any().any():
    raise ValueError("Dataset contains missing values.")

if not set(df[TARGET].unique()).issubset({0, 1}):
    raise ValueError("Expected binary target labels 0 and 1.")

X = df[FEATURES].astype("float32").to_numpy()
y = df[TARGET].astype("int32").to_numpy()

# --------------------------------------------------
# 4. Chronological split: 70% / 15% / 15%
# Do not randomly shuffle the records.
# --------------------------------------------------
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print("\n--- CHRONOLOGICAL SPLIT ---")
print(f"Training records:   {len(X_train)}")
print(f"Validation records: {len(X_val)}")
print(f"Test records:       {len(X_test)}")
print(f"Total records:      {n}")

# --------------------------------------------------
# 5. Fit preprocessing ONLY on training records
# --------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train).astype("float32")
X_val_scaled = scaler.transform(X_val).astype("float32")
X_test_scaled = scaler.transform(X_test).astype("float32")

# --------------------------------------------------
# 6. Build Keras neural network
# --------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(len(FEATURES),)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

# --------------------------------------------------
# 7. Train Model V1
# --------------------------------------------------
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=256,
    shuffle=False,
    callbacks=[early_stopping],
    verbose=1,
)

# --------------------------------------------------
# 8. Evaluate on the later, unseen test period
# --------------------------------------------------
probabilities = model.predict(X_test_scaled, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype("int32")

metrics = {
    "model_version": "V1",
    "model_type": "Keras MLP",
    "seed": SEED,
    "train_records": int(len(X_train)),
    "validation_records": int(len(X_val)),
    "test_records": int(len(X_test)),
    "epochs_completed": int(len(history.history["loss"])),
    "accuracy": float(accuracy_score(y_test, predictions)),
    "precision": float(
        precision_score(y_test, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, predictions, zero_division=0)
    ),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
}

print("\n--- MODEL V1 TEST RESULTS ---")
for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, predictions, zero_division=0))

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# 9. Save model, scaler, metrics, and test predictions
# --------------------------------------------------
model.save(MODEL_DIR / "electricity_v1.keras")
joblib.dump(scaler, MODEL_DIR / "electricity_v1_scaler.joblib")

with open(RESULTS_DIR / "v1_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame({
    "row_index": np.arange(val_end, n),
    "actual_class": y_test,
    "predicted_class": predictions,
    "probability_class_1": probabilities,
}).to_csv(RESULTS_DIR / "v1_test_predictions.csv", index=False)

print("\nSaved artifacts:")
print(f"  Model:       {MODEL_DIR / 'electricity_v1.keras'}")
print(f"  Scaler:      {MODEL_DIR / 'electricity_v1_scaler.joblib'}")
print(f"  Metrics:     {RESULTS_DIR / 'v1_metrics.json'}")
print(f"  Predictions: {RESULTS_DIR / 'v1_test_predictions.csv'}")
print("\nBaseline training completed successfully!")
