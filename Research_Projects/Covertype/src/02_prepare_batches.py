import json

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from covtype_pipeline import (
    BATCH_COUNT,
    DATA_DIR,
    FEATURE_COUNT,
    FEATURE_NAMES,
    STREAM_METADATA_PATH,
    load_raw_dataset,
)


def main() -> None:
    features, labels, original_labels = load_raw_dataset()
    record_count = len(labels)
    train_end = int(record_count * 0.70)
    validation_end = int(record_count * 0.85)
    split_ranges = [
        ("train", 0, train_end, 70),
        ("validation", train_end, validation_end, 15),
        ("test", validation_end, record_count, 15),
    ]

    manifest_rows = []
    batch_id = 0
    for split_name, split_start, split_end, batch_count in split_ranges:
        boundaries = np.linspace(
            split_start, split_end, batch_count + 1, dtype=np.int64
        )
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            batch_labels = labels[start:end]
            manifest_rows.append(
                {
                    "batch_id": batch_id,
                    "start_row": int(start),
                    "end_row_exclusive": int(end),
                    "n_records": int(end - start),
                    "split": split_name,
                    "class_0": int(np.count_nonzero(batch_labels == 0)),
                    "class_1": int(np.count_nonzero(batch_labels == 1)),
                    "original_class_1": int(np.count_nonzero(original_labels[start:end] == 1)),
                    "original_class_2": int(np.count_nonzero(original_labels[start:end] == 2)),
                }
            )
            batch_id += 1

    if batch_id != BATCH_COUNT:
        raise AssertionError(f"Expected {BATCH_COUNT} batches; created {batch_id}.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_npz(DATA_DIR / "X.npz", features.astype(np.float32))
    np.save(DATA_DIR / "y.npy", labels.astype(np.int8), allow_pickle=False)
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(DATA_DIR / "batch_manifest.csv", index=False)

    metadata = {
        "dataset": "Covertype binary LIBSVM scaled",
        "source_file": "covtype.libsvm.binary.scale.bz2",
        "records": int(record_count),
        "features": FEATURE_NAMES,
        "feature_count": FEATURE_COUNT,
        "target": "binary label: original class 1 -> 0; original class 2 -> 1",
        "feature_groups": {
            "1-10": "terrain and distance measurements",
            "11-14": "wilderness-area indicators",
            "15-54": "soil-type indicators",
        },
        "original_class_counts": {
            "1": int(np.count_nonzero(original_labels == 1)),
            "2": int(np.count_nonzero(original_labels == 2)),
        },
        "number_of_batches": BATCH_COUNT,
        "train_end_exclusive": train_end,
        "validation_end_exclusive": validation_end,
        "train_rows": [0, train_end],
        "validation_rows": [train_end, validation_end],
        "test_rows": [validation_end, record_count],
        "split_batch_ids": {
            "train": [0, 69],
            "validation": [70, 84],
            "test": [85, 99],
        },
        "row_order_preserved": True,
        "row_order_is_verified_chronology": False,
        "scaling": "Input features were already scaled in the supplied LIBSVM file.",
        "storage": "CSR sparse matrix in X.npz and integer target array in y.npy.",
    }
    with STREAM_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print(f"Prepared {record_count:,} records and {batch_id} batches.")
    print(
        manifest.groupby("split")
        .agg(batches=("batch_id", "count"), records=("n_records", "sum"))
        .to_string()
    )
    print(f"\nSparse features: {DATA_DIR / 'X.npz'}")
    print(f"Targets:         {DATA_DIR / 'y.npy'}")
    print(f"Batch manifest:  {DATA_DIR / 'batch_manifest.csv'}")
    print(f"Metadata:        {STREAM_METADATA_PATH}")


if __name__ == "__main__":
    main()
