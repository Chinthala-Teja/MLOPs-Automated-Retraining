import numpy as np
import pandas as pd

from covtype_pipeline import CARA_DIR, TEST_BATCHES


STRATEGY_NAMES = (
    "Never retrain",
    "Always retrain",
    "Periodic (every 3 batches)",
    "CARA-T-inspired",
)
PERIODIC_INTERVAL = 3


def expected_decision(strategy: str, position: int, relative_staleness: float, threshold: float) -> str:
    if strategy == "Never retrain":
        return "KEEP"
    if strategy == "Always retrain":
        return "RETRAIN"
    if strategy == "Periodic (every 3 batches)":
        return "RETRAIN" if position % PERIODIC_INTERVAL == 0 else "KEEP"
    if strategy == "CARA-T-inspired":
        return "RETRAIN" if relative_staleness >= threshold else "KEEP"
    raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    result_path = CARA_DIR / "test_holdout" / "holdout_batch_results.csv"
    summary_path = CARA_DIR / "test_holdout" / "holdout_strategy_summary.csv"
    if not result_path.is_file():
        raise FileNotFoundError(
            f"Holdout results not found: {result_path}. "
            "Run 06_evaluate_holdout.py first."
        )
    if not summary_path.is_file():
        raise FileNotFoundError(f"Holdout strategy summary not found: {summary_path}.")
    results = pd.read_csv(result_path)
    summaries = pd.read_csv(summary_path)
    expected_batches = list(TEST_BATCHES)
    audit_rows = []

    for strategy in STRATEGY_NAMES:
        subset = results.loc[results["strategy"] == strategy].sort_values("batch_id")
        actual_batches = subset["batch_id"].astype(int).tolist()
        coverage_ok = actual_batches == expected_batches
        decision_checks = []
        cost_checks = []
        version_checks = []

        if coverage_ok:
            for position, (_, row) in enumerate(subset.iterrows()):
                expected = expected_decision(
                    strategy,
                    position,
                    float(row["relative_staleness"]),
                    float(row["threshold"])
                    if strategy == "CARA-T-inspired"
                    else float("nan"),
                )
                decision_checks.append(row["decision"] == expected)
                expected_training = expected == "RETRAIN"
                version_checks.append(
                    bool(row["training_performed"]) == expected_training
                    and int(row["model_version_after_decision"])
                    == int(row["model_version_before_decision"])
                    + int(expected_training)
                    and (
                        position == 0
                        or int(row["model_version_before_decision"])
                        == int(subset.iloc[position - 1]["model_version_after_decision"])
                    )
                )
                expected_cost = (
                    float(row["kappa"])
                    if expected == "RETRAIN"
                    else float(row["absolute_cost"])
                )
                cost_checks.append(
                    np.isclose(
                        float(row["proxy_cost"]),
                        expected_cost,
                        rtol=1e-8,
                        atol=1e-12,
                    )
                )

        strategy_summary = summaries.loc[summaries["strategy"] == strategy]
        summary_ok = len(strategy_summary) == 1
        if summary_ok:
            summary_row = strategy_summary.iloc[0]
            summary_ok = (
                int(summary_row["retrain_count"])
                == int((subset["decision"] == "RETRAIN").sum())
                and np.isclose(
                    float(summary_row["total_proxy_cost"]),
                    float(subset["proxy_cost"].sum()),
                    rtol=1e-8,
                    atol=1e-12,
                )
            )
        audit_rows.append(
            {
                "strategy": strategy,
                "row_count": int(len(subset)),
                "distinct_batch_count": int(subset["batch_id"].nunique()),
                "batch_coverage_ok": bool(coverage_ok),
                "decision_checks_passed": int(sum(decision_checks)),
                "decision_checks_total": int(len(decision_checks)),
                "decision_rule_ok": bool(decision_checks) and all(decision_checks),
                "model_version_checks_passed": int(sum(version_checks)),
                "model_version_checks_total": int(len(version_checks)),
                "model_version_progression_ok": bool(version_checks)
                and all(version_checks),
                "recorded_proxy_cost": float(subset["proxy_cost"].sum()),
                "recalculated_proxy_cost": float(
                    sum(
                        float(row["kappa"])
                        if row["decision"] == "RETRAIN"
                        else float(row["absolute_cost"])
                        for _, row in subset.iterrows()
                    )
                ),
                "proxy_cost_reconciles": bool(cost_checks) and all(cost_checks),
                "summary_reconciles": bool(summary_ok),
            }
        )

    audit = pd.DataFrame(audit_rows)
    audit_path = CARA_DIR / "test_holdout" / "holdout_audit_summary.csv"
    audit.to_csv(audit_path, index=False)
    passed = bool(
        audit["batch_coverage_ok"].all()
        and audit["decision_rule_ok"].all()
        and audit["model_version_progression_ok"].all()
        and audit["proxy_cost_reconciles"].all()
        and audit["summary_reconciles"].all()
    )
    print(audit.to_string(index=False))
    print(f"\nAudit passed: {passed}")
    print(f"Audit summary: {audit_path}")
    if not passed:
        raise ValueError("Holdout audit failed; inspect the saved audit summary.")


if __name__ == "__main__":
    main()
