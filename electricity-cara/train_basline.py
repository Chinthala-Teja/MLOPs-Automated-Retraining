
from pathlib import Path
import json
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
)
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------
# 1. Configuration
# -------------------------------
SEED = 42
tf.keras.utils.set_random_seed(SEED)

DATA_PATH = Path("data/elec/processed/electricity_batched.csv")
OUTPUT_DIR = Path("artifacts/model_v1")
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

# -------------------------------
# 2. Load and validate data
# -------------------------------
df = pd.read_csv(DATA_PATH)

required_columns = FEATURES + [TARGET, "batch_id"]
missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in required columns.")

# Temporal split using the existing batch manifest
train_df = df[df["batch_id"].between(0, 69)].copy()
val_df = df[df["batch_id"].between(70, 84)].copy()

X_train = train_df[FEATURES].to_numpy(dtype=np.float32)
y_train = train_df[TARGET].to_numpy(dtype=np.float32)

X_val = val_df[FEATURES].to_numpy(dtype=np.float32)
y_val = val_df[TARGET].to_numpy(dtype=np.int32)

print("Training records:", len(train_df))
print("Validation records:", len(val_df))

# -------------------------------
# 3. Fit preprocessing on training data only
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
X_val_scaled = scaler.transform(X_val).astype(np.float32)

# -------------------------------
# 4. Build Keras classifier
# -------------------------------
model = keras.Sequential([
    keras.Input(shape=(len(FEATURES),)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.AUC(name="auc"),
    ],
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

# -------------------------------
# 5. Train Model V1
# -------------------------------
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=128,
    callbacks=[early_stopping],
    verbose=1,
)

# -------------------------------
# 6. Evaluate on validation data
# -------------------------------
probabilities = model.predict(X_val_scaled, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "accuracy": float(accuracy_score(y_val, predictions)),
    "precision": float(
        precision_score(y_val, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_val, predictions, zero_division=0)
    ),
    "f1": float(f1_score(y_val, predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_val, probabilities)),
}

print("\nModel V1 validation metrics")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# -------------------------------
# 7. Save artifacts
# -------------------------------
model.save(OUTPUT_DIR / "model.keras")
joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")

metadata = {
    "model_version": "v1",
    "algorithm": "Keras binary classifier",
    "features": FEATURES,
    "target": TARGET,
    "training_batches": [0, 69],
    "validation_batches": [70, 84],
    "training_records": int(len(train_df)),
    "validation_records": int(len(val_df)),
    "random_seed": SEED,
    "metrics": metrics,
}

with open(OUTPUT_DIR / "metadata.json", "w") as file:
    json.dump(metadata, file, indent=2)

print(f"\nArtifacts saved to: {OUTPUT_DIR.resolve()}")
