from __future__ import annotations

import bz2
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import load_npz


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RAW_DATA_PATH = WORKSPACE_ROOT / "covtype.libsvm.binary.scale.bz2"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PROCESSED_FEATURES_PATH = DATA_DIR / "X.npz"
PROCESSED_TARGET_PATH = DATA_DIR / "y.npy"
MANIFEST_PATH = DATA_DIR / "batch_manifest.csv"
STREAM_METADATA_PATH = DATA_DIR / "stream_metadata.json"
MODEL_DIR = ARTIFACTS_DIR / "model_v1"
MODEL_PATH = MODEL_DIR / "model.keras"
MODEL_METADATA_PATH = MODEL_DIR / "metadata.json"
CARA_DIR = ARTIFACTS_DIR / "cara_t"
FEATURE_COUNT = 54
BATCH_COUNT = 100
TRAIN_BATCHES = range(0, 70)
VALIDATION_BATCHES = range(70, 85)
TEST_BATCHES = range(85, 100)
RANDOM_SEED = 42
FEATURE_NAMES = [
    "elevation",
    "aspect",
    "slope",
    "horizontal_distance_to_hydrology",
    "vertical_distance_to_hydrology",
    "horizontal_distance_to_roadways",
    "hillshade_9am",
    "hillshade_noon",
    "hillshade_3pm",
    "horizontal_distance_to_fire_points",
    *[f"wilderness_area_{index}" for index in range(1, 5)],
    *[f"soil_type_{index}" for index in range(1, 41)],
]


def load_raw_dataset():
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"CoverType LIBSVM file not found: {RAW_DATA_PATH}")

    from sklearn.datasets import load_svmlight_file

    with bz2.open(RAW_DATA_PATH, "rb") as compressed_file:
        features, original_labels = load_svmlight_file(
            compressed_file,
            n_features=FEATURE_COUNT,
            dtype=np.float32,
        )

    features = features.tocsr()
    labels = np.asarray(original_labels)
    if features.shape[1] != FEATURE_COUNT:
        raise ValueError(
            f"Expected {FEATURE_COUNT} features; found {features.shape[1]}."
        )
    if not np.isfinite(features.data).all():
        raise ValueError("The CoverType dataset contains non-finite feature values.")
    if not np.isin(labels, [1.0, 2.0]).all():
        raise ValueError(
            "Expected the binary LIBSVM labels 1 and 2; "
            f"found {np.unique(labels).tolist()}."
        )
    if set(np.unique(labels).tolist()) != {1.0, 2.0}:
        raise ValueError("Both original target labels 1 and 2 must be present.")

    binary_labels = (labels == 2.0).astype(np.int8)
    return features, binary_labels, labels


def load_processed_data():
    required_paths = (
        PROCESSED_FEATURES_PATH,
        PROCESSED_TARGET_PATH,
        MANIFEST_PATH,
        STREAM_METADATA_PATH,
    )
    missing = [path for path in required_paths if not path.is_file()]
    if missing:
        paths = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Prepared CoverType data is missing. Run "
            "Covertype\\src\\02_prepare_batches.py first:\n" + paths
        )

    features = load_npz(PROCESSED_FEATURES_PATH).tocsr()
    labels = np.load(PROCESSED_TARGET_PATH, allow_pickle=False)
    manifest = pd.read_csv(MANIFEST_PATH)
    with STREAM_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if features.shape != (len(labels), FEATURE_COUNT):
        raise ValueError(
            f"Processed data shape mismatch: features={features.shape}, "
            f"labels={labels.shape}."
        )
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Processed target labels must be encoded as 0 and 1.")
    if manifest["batch_id"].tolist() != list(range(BATCH_COUNT)):
        raise ValueError("Batch manifest must contain batch IDs 0 through 99.")
    manifest_columns = {
        "start_row",
        "end_row_exclusive",
        "n_records",
        "split",
    }
    missing_columns = manifest_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Batch manifest is missing columns: {sorted(missing_columns)}"
        )
    starts = manifest["start_row"].to_numpy(dtype=np.int64)
    ends = manifest["end_row_exclusive"].to_numpy(dtype=np.int64)
    counts = manifest["n_records"].to_numpy(dtype=np.int64)
    if (
        starts[0] != 0
        or ends[-1] != len(labels)
        or np.any(starts[1:] != ends[:-1])
        or np.any(ends <= starts)
        or np.any(counts != ends - starts)
    ):
        raise ValueError("Batch manifest has gaps, overlaps, or invalid row counts.")
    if manifest["split"].value_counts().to_dict() != {
        "train": 70,
        "validation": 15,
        "test": 15,
    }:
        raise ValueError("Expected 70 train, 15 validation, and 15 test batches.")
    if int(metadata.get("records", -1)) != len(labels):
        raise ValueError("Stream metadata record count does not match the data.")
    return features, labels.astype(np.int8, copy=False), manifest, metadata


def batch_bounds(manifest: pd.DataFrame, batch_id: int) -> tuple[int, int]:
    matches = manifest.loc[manifest["batch_id"] == batch_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one manifest entry for batch {batch_id}.")
    row = matches.iloc[0]
    return int(row["start_row"]), int(row["end_row_exclusive"])


def dense_rows(features, start: int, end: int) -> np.ndarray:
    if end <= start:
        raise ValueError(f"Empty or invalid row range: [{start}, {end}).")
    return features[start:end].toarray().astype(np.float32, copy=False)


def estimate_rbf_gamma(training_features, seed: int = RANDOM_SEED) -> float:
    sample_size = min(1000, training_features.shape[0])
    if sample_size < 2:
        raise ValueError("At least two training rows are required to estimate gamma.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(training_features.shape[0], sample_size, replace=False)
    sample = training_features[indices].toarray().astype(np.float32, copy=False)
    squared_distances = cdist(sample, sample, metric="sqeuclidean")
    positive_distances = squared_distances[squared_distances > 0]
    if positive_distances.size == 0:
        raise ValueError("Cannot estimate RBF gamma: sampled feature rows are identical.")
    return float(1.0 / np.median(positive_distances))


def sample_reference(
    features,
    labels: np.ndarray,
    seed: int,
    data_size: int = 2000,
    query_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if features.shape[0] == 0:
        raise ValueError("Cannot sample a reference from an empty feature matrix.")
    rng = np.random.default_rng(seed)
    data_indices = rng.choice(
        features.shape[0], min(data_size, features.shape[0]), replace=False
    )
    query_indices = rng.choice(
        features.shape[0], min(query_size, features.shape[0]), replace=False
    )
    reference_data = features[data_indices].toarray().astype(np.float32, copy=False)
    reference_labels = labels[data_indices].astype(np.int8, copy=False)
    reference_queries = features[query_indices].toarray().astype(np.float32, copy=False)
    return reference_data, reference_labels, reference_queries


def sample_batch_for_cost(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    data_size: int = 2000,
    query_size: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(features) == 0:
        raise ValueError("Cannot estimate batch cost on an empty batch.")
    rng = np.random.default_rng(seed)
    data_indices = rng.choice(len(features), min(data_size, len(features)), replace=False)
    query_indices = rng.choice(
        len(features), min(query_size, len(features)), replace=False
    )
    return (
        features[query_indices],
        features[data_indices],
        labels[data_indices],
    )


def absolute_cost(
    model,
    query_features: np.ndarray,
    data_features: np.ndarray,
    labels: np.ndarray,
    gamma: float,
) -> float:
    if len(query_features) == 0 or len(data_features) == 0:
        raise ValueError("RBF cost requires non-empty query and labeled data.")
    if len(data_features) != len(labels):
        raise ValueError("RBF cost data and target lengths do not match.")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError(f"RBF gamma must be finite and positive; received {gamma}.")

    probabilities = model.predict(data_features, verbose=0).reshape(-1)
    predictions = (probabilities >= 0.5).astype(np.int8)
    losses = (predictions != labels).astype(np.float32)
    squared_distances = cdist(query_features, data_features, metric="sqeuclidean")
    similarities = np.exp(-gamma * squared_distances)
    return float(np.mean(similarities @ losses / len(data_features)))


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    predictions = (probabilities >= 0.5).astype(np.int8)
    auc = (
        float(roc_auc_score(labels, probabilities))
        if np.unique(labels).size == 2
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": auc,
        "error_rate": float(np.mean(predictions != labels)),
    }


def build_model(seed: int = RANDOM_SEED):
    from tensorflow import keras
    from tensorflow.keras import layers

    keras.utils.set_random_seed(seed)
    model = keras.Sequential(
        [
            keras.Input(shape=(FEATURE_COUNT,)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train_replacement_model(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epochs: int = 15,
    batch_size: int = 256,
) -> tuple[Any, float]:
    model = build_model(seed)
    started_at = time.perf_counter()
    model.fit(
        features,
        labels,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    return model, float(time.perf_counter() - started_at)
