
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

INPUT_PATH = Path(
    "artifacts/cara_t/validation_staleness.csv"
)

OUTPUT_DIR = Path("artifacts/cara_t")
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load validation staleness
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

required_columns = {
    "batch_id",
    "relative_staleness",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

staleness = df["relative_staleness"].to_numpy(dtype=float)

if len(staleness) == 0 or not np.isfinite(staleness).all():
    raise ValueError("Staleness data is empty or contains invalid values.")


# --------------------------------------------------
# 3. Define candidate thresholds and retraining costs
# --------------------------------------------------

# Candidate thresholds span the observed validation values.
thresholds = np.unique(
    np.concatenate([
        [staleness.min() - 1e-9],
        staleness,
        [staleness.max() + 1e-9],
    ])
)

# Experimental costs, in the same normalized units
# as the current staleness approximation.
# These are sensitivity-analysis values, NOT measured
# real-world retraining costs.
kappa_values = [0.005, 0.01, 0.02, 0.04, 0.08]


# --------------------------------------------------
# 4. Evaluate each candidate
# --------------------------------------------------

records = []

for kappa in kappa_values:
    for tau in thresholds:

        # CARA-T decision convention:
        # KEEP when relative staleness < tau.
        # RETRAIN when relative staleness >= tau.
        retrain_mask = staleness >= tau
        keep_mask = ~retrain_mask

        # Experimental proxy objective:
        # keep cost = relative staleness
        # retrain cost = kappa
        #
        # This objective uses fixed Model V1 staleness.
        # It does not simulate a replacement model.
        per_batch_cost = np.where(
            keep_mask,
            staleness,
            kappa,
        )

        records.append({
            "kappa": float(kappa),
            "threshold": float(tau),
            "total_proxy_cost": float(per_batch_cost.sum()),
            "mean_proxy_cost": float(per_batch_cost.mean()),
            "keep_count": int(keep_mask.sum()),
            "retrain_count": int(retrain_mask.sum()),
        })

results = pd.DataFrame(records)

results.to_csv(
    OUTPUT_DIR / "threshold_search.csv",
    index=False,
)


# --------------------------------------------------
# 5. Find best threshold for each kappa
# --------------------------------------------------

best_rows = []

for kappa in kappa_values:
    candidates = results[results["kappa"] == kappa]

    best = candidates.loc[
        candidates["total_proxy_cost"].idxmin()
    ]

    best_rows.append(best.to_dict())

best_df = pd.DataFrame(best_rows)

best_df.to_csv(
    OUTPUT_DIR / "best_thresholds.csv",
    index=False,
)

print("\nPreliminary threshold calibration")
print("=" * 65)

for _, row in best_df.iterrows():
    print(
        f"kappa={row['kappa']:.4f} | "
        f"tau={row['threshold']:.6f} | "
        f"proxy_cost={row['total_proxy_cost']:.6f} | "
        f"KEEP={int(row['keep_count'])} | "
        f"RETRAIN={int(row['retrain_count'])}"
    )


# --------------------------------------------------
# 6. Plot threshold sensitivity
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for kappa in kappa_values:
    subset = results[results["kappa"] == kappa].sort_values(
        "threshold"
    )

    ax.plot(
        subset["threshold"],
        subset["total_proxy_cost"],
        label=f"kappa={kappa:.3f}",
    )

ax.set_title("Preliminary CARA-T Threshold Sensitivity")
ax.set_xlabel("Candidate threshold")
ax.set_ylabel("Total proxy cost across validation batches")
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(
    PLOT_DIR / "threshold_sensitivity.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# 7. Save selected thresholds and limitations
# --------------------------------------------------

summary = {
    "method": "fixed-model preliminary threshold calibration",
    "validation_batches": [
        int(df["batch_id"].min()),
        int(df["batch_id"].max()),
    ],
    "candidate_kappa_values": kappa_values,
    "best_thresholds": best_df.to_dict(orient="records"),
    "limitations": [
        "Uses Model V1 staleness values only.",
        "Does not train replacement models.",
        "Does not account for future costs after retraining.",
        "Kappa values are experimental sensitivity settings, "
        "not measured operational costs.",
        "Not the complete CARA-T strategy-cost optimization."
    ],
    "next_step": (
        "Implement sequential validation replay with model "
        "updates and a consistent strategy-cost calculation."
    ),
}

with open(
    OUTPUT_DIR / "threshold_optimization_summary.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=4)

print("\nSaved:")
print("  artifacts/cara_t/threshold_search.csv")
print("  artifacts/cara_t/best_thresholds.csv")
print("  artifacts/cara_t/threshold_optimization_summary.json")
print("  artifacts/cara_t/plots/threshold_sensitivity.png")
print("\nThis is a preliminary calibration, not final CARA-T.")
