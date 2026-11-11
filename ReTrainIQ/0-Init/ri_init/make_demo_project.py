"""make_demo_project.py
Creates synthetic flight-like dataset and trains a baseline model for ReTrainIQ.
- 50,000 reference rows
- 6,000 holdout rows
- 50 stream batches of 2,000 rows
- 2 numeric columns: dep_delay, distance
- 3 categorical columns: carrier, origin, dest
- 1 binary target: delayed (0 or 1)
- Model: Pipeline(OneHotEncoder(handle_unknown='ignore') + RandomForestClassifier(seed=42))
- Concept drift introduced after stream batch 25.
- Writes workspace/demo/env.json with python and scikit-learn versions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from ri_init.config import WORKSPACE

def generate_flight_data(n_rows: int, seed: int, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    carriers = ["AA", "DL", "UA", "WN", "B6"]
    origins = ["JFK", "ORD", "ATL", "DFW", "LAX", "DEN"]
    dests = ["SFO", "SEA", "MIA", "BOS", "PHX", "LAS"]

    carrier_col = rng.choice(carriers, size=n_rows)
    origin_col = rng.choice(origins, size=n_rows)
    dest_col = rng.choice(dests, size=n_rows)

    dep_delay = rng.normal(loc=10.0, scale=15.0, size=n_rows)
    distance = rng.uniform(low=200.0, high=2500.0, size=n_rows)

    # Base probability function
    # Delay probability correlates with dep_delay and carrier
    logits = 0.08 * (dep_delay - 10.0) + 0.0005 * (distance - 1000.0)
    carrier_effect = {"AA": 0.2, "DL": -0.2, "UA": 0.1, "WN": -0.1, "B6": 0.3}
    logits += np.array([carrier_effect[c] for c in carrier_col])

    if drift:
        # Concept drift: relationship changes drastically
        # e.g. severe weather / winter operations reverse carrier delay rates
        # and distance has a strong negative correlation or threshold shift
        logits = -0.05 * (dep_delay - 10.0) + (distance > 1500).astype(float) * 2.5 - 1.5

    prob = 1.0 / (1.0 + np.exp(-logits))
    delayed = (rng.uniform(size=n_rows) < prob).astype(int)

    df = pd.DataFrame({
        "dep_delay": dep_delay,
        "distance": distance,
        "carrier": carrier_col,
        "origin": origin_col,
        "dest": dest_col,
        "delayed": delayed,
    })
    return df

def main() -> None:
    demo_dir = WORKSPACE / "demo"
    data_dir = demo_dir / "data"
    stream_dir = data_dir / "stream"
    models_dir = demo_dir / "models" / "v1"

    data_dir.mkdir(parents=True, exist_ok=True)
    stream_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Write env.json
    env_info = {
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
    }
    env_path = demo_dir / "env.json"
    with open(env_path, "w", encoding="utf-8") as f:
        json.dump(env_info, f, indent=2)
    print(f"Saved env metadata to {env_path}")

    print("Generating reference dataset (50,000 rows)...")
    ref_df = generate_flight_data(50000, seed=42, drift=False)
    ref_path = data_dir / "reference.csv"
    ref_df.to_csv(ref_path, index=False)

    print("Generating holdout dataset (6,000 rows)...")
    holdout_df = generate_flight_data(6000, seed=1042, drift=False)
    holdout_path = data_dir / "holdout.csv"
    holdout_df.to_csv(holdout_path, index=False)

    print("Generating 50 stream batches (2,000 rows each)...")
    for i in range(50):
        # Drift after batch 25 (batches 26-49)
        has_drift = (i >= 26)
        batch_df = generate_flight_data(2000, seed=2000 + i, drift=has_drift)
        batch_file = stream_dir / f"batch_{i:03d}.csv"
        batch_df.to_csv(batch_file, index=False)

    # Train model
    print("Training RandomForest Pipeline on reference data...")
    numeric_features = ["dep_delay", "distance"]
    categorical_features = ["carrier", "origin", "dest"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ])

    X_ref = ref_df.drop(columns=["delayed"])
    y_ref = ref_df["delayed"]
    pipeline.fit(X_ref, y_ref)

    model_path = models_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")

    # Write profile.json
    profile = {
        "project_id": "demo",
        "problem_statement": "Flight departure delay prediction with online monitoring and retraining",
        "task": {
            "type": "classification",
            "subtype": "binary",
            "target": "delayed",
            "positive_class": 1,
        },
        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
        },
        "model": {
            "framework": "sklearn",
            "artifact": str(model_path.relative_to(demo_dir.parents[1]).as_posix()),
            "algorithm": "RandomForestClassifier",
        },
        "data": {
            "reference_path": str(ref_path.relative_to(demo_dir.parents[1]).as_posix()),
            "holdout_path": str(holdout_path.relative_to(demo_dir.parents[1]).as_posix()),
            "id_column": None,
            "ordered": True,
        },
        "primary_metric": "recall",
        "batch_size_B": 2000,
        "offline_batches_N": 25,
        "W": 1,
        "seed": 42,
    }

    profile_path = demo_dir / "profile.json"
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print(f"Saved profile to {profile_path}")

if __name__ == "__main__":
    main()
