from __future__ import annotations

import json
import joblib
import pandas as pd
import sklearn
from sklearn.metrics import recall_score
from ri_init.config import WORKSPACE

def test_demo_files_exist():
    demo_dir = WORKSPACE / "demo"
    assert (demo_dir / "profile.json").exists()
    assert (demo_dir / "env.json").exists()
    assert (demo_dir / "data" / "reference.csv").exists()
    assert (demo_dir / "data" / "holdout.csv").exists()
    assert (demo_dir / "models" / "v1" / "model.joblib").exists()

    stream_files = list((demo_dir / "data" / "stream").glob("batch_*.csv"))
    assert len(stream_files) == 50

def test_sklearn_version_match():
    demo_dir = WORKSPACE / "demo"
    env_file = demo_dir / "env.json"
    assert env_file.exists(), "workspace/demo/env.json does not exist. Run python -m ri_init.make_demo_project."

    with open(env_file, "r", encoding="utf-8") as f:
        env_info = json.load(f)

    expected_sklearn = env_info.get("sklearn_version")
    actual_sklearn = sklearn.__version__
    assert actual_sklearn == expected_sklearn, (
        f"Installed scikit-learn version '{actual_sklearn}' does not match the version used to train the "
        f"demo model '{expected_sklearn}' recorded in workspace/demo/env.json. "
        f"Please install scikit-learn=={expected_sklearn}."
    )

def test_demo_model_predict_and_drift():
    demo_dir = WORKSPACE / "demo"
    model_path = demo_dir / "models" / "v1" / "model.joblib"
    model = joblib.load(model_path)

    # Read batch 000 (pre-drift)
    b0_path = demo_dir / "data" / "stream" / "batch_000.csv"
    b0_df = pd.read_csv(b0_path)
    X0 = b0_df.drop(columns=["delayed"])
    y0 = b0_df["delayed"]
    pred0 = model.predict(X0)
    assert len(pred0) == 2000
    rec0 = recall_score(y0, pred0)

    # Read batch 030 (post-drift after batch 25)
    b30_path = demo_dir / "data" / "stream" / "batch_030.csv"
    b30_df = pd.read_csv(b30_path)
    X30 = b30_df.drop(columns=["delayed"])
    y30 = b30_df["delayed"]
    pred30 = model.predict(X30)
    assert len(pred30) == 2000
    rec30 = recall_score(y30, pred30)

    # Check recall differs between pre-drift and post-drift batches
    assert abs(rec0 - rec30) > 0.05, f"Expected noticeable recall change, got rec0={rec0:.3f}, rec30={rec30:.3f}"
