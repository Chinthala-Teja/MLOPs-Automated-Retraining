
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT
    / "artifacts"
    / "cara_t"
    / "sequential_threshold_comparison.csv"
)
OUTPUT = ROOT / "artifacts" / "cara_t" / "cost_sensitivity"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

print("Available columns:")
print(df.columns.tolist())
print("\nExisting replay results:")
print(df.to_string(index=False))

# The existing replay was evaluated using kappa = 0.04.
# This summarizes that experiment; it does NOT simulate
# alternative retraining costs.
if "threshold" not in df.columns:
    raise ValueError("Expected a 'threshold' column.")

if "total_cost" not in df.columns:
    raise ValueError(
        "Expected 'total_cost'. Check the actual column names above "
        "and update this script to match the CSV."
    )

best = df.loc[df["total_cost"].idxmin()]

print("\nBest policy in the existing replay:")
print(best.to_string())

df.to_csv(OUTPUT / "existing_replay_summary.csv", index=False)

print(f"\nSaved summary to: {OUTPUT}")
