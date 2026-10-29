import json

import numpy as np

from covtype_pipeline import (
    ARTIFACTS_DIR,
    FEATURE_COUNT,
    RAW_DATA_PATH,
    load_raw_dataset,
)


def main() -> None:
    features, labels, original_labels = load_raw_dataset()
    unique_labels, counts = np.unique(original_labels, return_counts=True)
    report = {
        "source_file": str(RAW_DATA_PATH),
        "records": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "expected_feature_count": FEATURE_COUNT,
        "labels_original": {
            str(label): int(count)
            for label, count in zip(unique_labels, counts)
        },
        "labels_binary": {
            "0 (original class 1)": int(np.count_nonzero(labels == 0)),
            "1 (original class 2)": int(np.count_nonzero(labels == 1)),
        },
        "sparse_nonzero_values": int(features.nnz),
        "sparse_density": float(features.nnz / (features.shape[0] * features.shape[1])),
        "feature_value_min": float(features.data.min()),
        "feature_value_max": float(features.data.max()),
        "row_order_is_verified_chronology": False,
        "notes": [
            "This is the binary LIBSVM CoverType variant, with original labels 1 and 2.",
            "The 54 supplied feature columns are already scaled; no additional scaler is fitted.",
            "The file has no timestamp field; row order is preserved only as a reproducible pseudo-stream.",
        ],
    }
    output_dir = ARTIFACTS_DIR / "data_inspection"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "inspection.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nInspection report saved to: {report_path}")


if __name__ == "__main__":
    main()
