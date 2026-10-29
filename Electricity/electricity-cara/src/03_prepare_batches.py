from pathlib import Path
import json
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths and configuration
# Run this script from the electricity-cara folder.
# --------------------------------------------------
ROOT = Path.cwd()
DATA_PATH = ROOT / "data" / "elec" / "raw" / "elec.csv"
OUTPUT_DIR = ROOT / "data" / "elec" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}\n"
        "Run this script from the electricity-cara folder."
    )

# --------------------------------------------------
# 2. Load dataset
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
n = len(df)

# Chronological split boundaries (70% / 15% / 15%)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

# Total number of batches across the stream
N_BATCHES = 100

# --------------------------------------------------
# 3. Construct batches securely within each split
# --------------------------------------------------
split_ranges = [
    ("train", 0, train_end, 70),
    ("validation", train_end, val_end, 15),
    ("test", val_end, n, 15),
]

manifest = []
batch_arrays = []
batch_id = 0

for split, start, end, count in split_ranges:
    local_batches = np.array_split(np.arange(start, end), count)

    for indices in local_batches:
        batch_arrays.append(indices)
        manifest.append({
            "batch_id": batch_id,
            "start_row": int(indices[0]),
            "end_row_exclusive": int(indices[-1]) + 1,
            "n_records": len(indices),
            "split": split,
            "class_0": int((df.iloc[indices][TARGET] == 0).sum()),
            "class_1": int((df.iloc[indices][TARGET] == 1).sum()),
        })
        batch_id += 1

batch_indices = batch_arrays

# Save batch metadata for reproducible experiments
manifest_df = pd.DataFrame(manifest)
manifest_path = OUTPUT_DIR / "batch_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)

# --------------------------------------------------
# 4. Save the original ordered data with batch IDs
# --------------------------------------------------
df["row_index"] = np.arange(n)
df["batch_id"] = -1
for b_id, indices in enumerate(batch_indices):
    df.loc[indices, "batch_id"] = b_id

stream_path = OUTPUT_DIR / "electricity_batched.csv"
df.to_csv(stream_path, index=False)

# --------------------------------------------------
# 5. Record experiment metadata
# --------------------------------------------------
metadata = {
    "dataset": "Electricity",
    "records": int(n),
    "number_of_batches": N_BATCHES,
    "train_end_exclusive": train_end,
    "validation_end_exclusive": val_end,
    "train_rows": [0, train_end],
    "validation_rows": [train_end, val_end],
    "test_rows": [val_end, n],
    "features": FEATURES,
    "target": TARGET,
    "row_order_preserved": True,
}

with open(OUTPUT_DIR / "stream_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# --------------------------------------------------
# 6. Terminal Summary
# --------------------------------------------------
print("\n--- BATCH PREPARATION COMPLETED ---")
print(f"Records: {n}")
print(f"Batches: {N_BATCHES}")

print("\n--- SPLIT SUMMARY ---")
print(
    manifest_df.groupby("split")
    .agg(batches=("batch_id", "count"), records=("n_records", "sum"))
    .to_string()
)

print("\n--- FIRST 5 BATCHES ---")
print(manifest_df.head().to_string(index=False))

print("\n--- LAST 5 BATCHES ---")
print(manifest_df.tail().to_string(index=False))

print("\nSaved artifacts:")
print(f"  Manifest: {manifest_path}")
print(f"  Stream:   {stream_path}")
print(f"  Metadata: {OUTPUT_DIR / 'stream_metadata.json'}")