
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("artifacts/cara_t/test_holdout")
PLOTS = ROOT / "plots"

results_path = ROOT / "test_batch_results.csv"
summary_path = ROOT / "test_strategy_summary.csv"

if not results_path.exists() or not summary_path.exists():
    raise FileNotFoundError(
        "Run evaluate_test_holdout.py successfully first."
    )

results = pd.read_csv(results_path)
summary = pd.read_csv(summary_path)

cara = results[results["strategy"] == "CARA-T-inspired"].copy()

print("\n1. STRATEGY SUMMARY")
print(summary.round(4).to_string(index=False))

print("\n2. CARA-T-INSPIRED DECISIONS")
print(
    cara[
        [
            "batch_id",
            "relative_staleness",
            "threshold",
            "decision",
            "absolute_cost",
            "proxy_decision_cost",
            "error_rate",
            "model_version_before",
            "model_version_after",
        ]
    ].round(5).to_string(index=False)
)

print("\n3. DECISION RULE CHECK")
cara["expected_decision"] = cara["relative_staleness"].apply(
    lambda value: (
        "RETRAIN"
        if value >= cara["threshold"].iloc[0]
        else "KEEP"
    )
)
cara["decision_matches_rule"] = (
    cara["decision"] == cara["expected_decision"]
)

print(
    "Decisions matching fixed threshold:",
    int(cara["decision_matches_rule"].sum()),
    "out of",
    len(cara),
)
if not cara["decision_matches_rule"].all():
    print("WARNING: Some recorded decisions do not match the threshold.")

print("\n4. COST COMPONENTS")
for strategy, group in results.groupby("strategy", sort=False):
    retrain_cost = group.loc[
        group["decision"] == "RETRAIN", "proxy_decision_cost"
    ].sum()
    keep_cost = group.loc[
        group["decision"] == "KEEP", "proxy_decision_cost"
    ].sum()

    print(
        f"{strategy:20s} | "
        f"retraining cost={retrain_cost:.5f} | "
        f"keep cost={keep_cost:.5f} | "
        f"total={retrain_cost + keep_cost:.5f}"
    )

print("\n5. RELATIVE STALENESS VS FIXED THRESHOLD")
plt.figure(figsize=(12, 6))
plt.plot(
    cara["batch_id"],
    cara["relative_staleness"],
    marker="o",
    label="Observed relative staleness",
)
plt.axhline(
    cara["threshold"].iloc[0],
    linestyle="--",
    label="Validation-selected threshold",
)
plt.axhline(0, linestyle=":", label="Zero relative staleness")

for _, row in cara.iterrows():
    if row["decision"] == "RETRAIN":
        plt.annotate(
            "R",
            (row["batch_id"], row["relative_staleness"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )

plt.xlabel("Chronological batch ID")
plt.ylabel("Relative staleness")
plt.title("Holdout Audit: Staleness, Threshold and Retraining Decisions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
PLOTS.mkdir(parents=True, exist_ok=True)
plt.savefig(PLOTS / "holdout_threshold_audit.png", dpi=180)
plt.close()

print("\nSaved plot:")
print(PLOTS / "holdout_threshold_audit.png")
print("\nAudit complete.")
