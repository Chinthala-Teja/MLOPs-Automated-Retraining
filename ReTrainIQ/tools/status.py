"""status.py
Inspects modules across all 7 packages and frontend, checks IS_STUB,
and verifies required public functions against CONTRACT.md signatures.
Exits with code 1 if any CONTRACT.md function is missing or has parameter mismatches.
"""
from __future__ import annotations

import importlib
import inspect
import sys
from typing import Any, Dict, List, Tuple

# Expected signatures from CONTRACT.md:
# module_path -> dict of {function_name: [param_names]}
EXPECTED_SIGNATURES: Dict[str, Dict[str, List[str]]] = {
    "ri_init.store": {
        "init_db": [],
        "add_project": ["project_id", "source", "profile_json"],
        "get_project": ["project_id"],
        "add_batch": ["project_id", "file", "n_rows", "n_labelled"],
        "get_batch": ["batch_id"],
        "get_batches": ["project_id"],
        "update_batch": ["batch_id", "n_rows", "n_labelled"],
        "add_report": ["project_id", "batch_id", "drift_json", "cara_json", "quality_json", "score_json", "zone", "summary"],
        "get_reports": ["project_id"],
        "add_decision": ["report_id", "action", "by", "note"],
        "add_version": ["project_id", "version", "path", "trained_on"],
        "get_versions": ["project_id"],
        "set_active_version": ["project_id", "version"],
        "get_active_version": ["project_id"],
    },
    "ri_init.worker": {
        "run_in_worker": ["callable_path", "kwargs"],
    },
    "ri_profile.understand": {
        "fetch_repo": ["url", "dest"],
        "extract_zip_safe": ["zip_path", "dest"],
        "scan_project": ["path"],
        "build_context": ["scan"],
        "extract_profile": ["context", "llm"],
        "heuristic_profile": [],
        "verify_profile": ["profile"],
        "finalize_profile": ["profile"],
    },
    "ri_drift.drift": {
        "run_drift": ["reference_df", "new_df", "profile"],
        "validate_schema": ["df", "profile"],
        "fit_reference": ["reference_df", "profile"],
        "prediction_shift": ["ref_pred", "new_pred"],
    },
    "ri_drift.plots": {
        "drift_bar": ["result"],
        "distribution_overlay": ["reference_df", "new_df", "feature", "ftype"],
        "top_drifted": ["result"],
    },
    "ri_drift.scenarios": {
        "inject_covariate_shift": ["df", "profile", "feature", "magnitude", "start_row"],
        "inject_concept_shift": ["df", "profile", "region_rule", "flip_fraction", "start_row"],
        "make_invalid_upload": ["df", "kind"],
        "hold_back_labels": ["df", "fraction"],
        "build_scenario_stream": ["batches", "scenario_name"],
    },
    "ri_cara.cara_t": {
        "calibrate": ["model_path", "history_df", "profile", "cfg"],
        "online_staleness": ["model_path", "new_df", "train_sample_df", "calib", "profile"],
        "estimate_staleness": ["model_path", "new_df", "train_sample_df", "calib", "profile"],
    },
    "ri_cara.replay_baselines": {
        "run_baselines": ["batches", "calib", "profile", "cfg"],
    },
    "ri_cara.proxy_eval": {
        "run_proxy_eval": ["profile", "cfg"],
    },
    "ri_metrics.metrics": {
        "default_plan": ["profile"],
        "validate_plan": ["plan"],
        "run_metrics": ["model_path", "df_labelled", "profile", "plan"],
        "compute_baseline": ["model_path", "holdout_df", "profile", "plan"],
        "compare": ["baseline", "current", "primary_metric"],
    },
    "ri_analysis.summary": {
        "summarize": ["report"],
        "validate_summary": ["text", "report"],
        "template_summary": ["report"],
    },
    "ri_analysis.chat": {
        "chat": ["question", "project_id"],
    },
    "ri_decision.decision": {
        "decide": ["drift", "cara", "quality"],
        "what_if": ["drift", "cara", "quality"],
    },
    "ri_decision.retrain": {
        "train_candidate": ["profile", "train_df", "out_path", "seed", "champion_path"],
        "build_training_data": ["project_id", "profile"],
        "verify": ["candidate_path", "holdout_df", "profile", "champion_path", "v_df", "margin", "tol", "recall_guard", "n_boot", "seed"],
        "promote": ["candidate_path", "active_path"],
        "rollback": ["previous_path", "active_path"],
        "save_bundle": ["project_id", "version"],
        "load_bundle": ["project_id", "version"],
        "run_retrain_cycle": ["project_id", "trigger", "report_id", "force"],
        "shadow_check": ["project_id", "batch_df"],
        "rollback_project": ["project_id"],
    },
    "ri_decision.orchestrator": {
        "setup_project": ["profile"],
        "on_new_data": ["project_id", "file_path"],
        "attach_labels": ["project_id", "batch_id", "labels_path"],
        "resolve_human": ["project_id", "report_id", "choice"],
        "rollback": ["project_id"],
    },
}

MODULES_TO_INSPECT = [
    ("0-Init", "ri_init.config"),
    ("0-Init", "ri_init.setup_env"),
    ("0-Init", "ri_init.store"),
    ("0-Init", "ri_init.worker"),
    ("0-Init", "ri_init.make_demo_project"),
    ("1-Profile", "ri_profile.understand"),
    ("2-Drift", "ri_drift.drift"),
    ("2-Drift", "ri_drift.plots"),
    ("2-Drift", "ri_drift.scenarios"),
    ("3-CARA", "ri_cara.cara_t"),
    ("3-CARA", "ri_cara.replay_baselines"),
    ("3-CARA", "ri_cara.proxy_eval"),
    ("4-Metrics", "ri_metrics.metrics"),
    ("5-Analysis", "ri_analysis.llm"),
    ("5-Analysis", "ri_analysis.summary"),
    ("5-Analysis", "ri_analysis.chat"),
    ("6-Decision", "ri_decision.decision"),
    ("6-Decision", "ri_decision.retrain"),
    ("6-Decision", "ri_decision.orchestrator"),
    ("6-Decision", "ri_decision.cli"),
    ("6-Decision", "ri_decision.calibrate_score"),
    ("6-Decision", "ri_decision.sensitivity"),
    ("frontend", "app"),
]

def check_status() -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for pkg, mod_name in MODULES_TO_INSPECT:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            errors.append(f"Failed to import {mod_name}: {e}")
            continue

        is_stub = getattr(mod, "IS_STUB", None)
        status_label = "STUB" if is_stub is True else ("IMPLEMENTED" if is_stub is False else "N/A")

        # Check public functions if expected
        expected = EXPECTED_SIGNATURES.get(mod_name, {})
        found_fns = []
        for fn_name, req_params in expected.items():
            if not hasattr(mod, fn_name):
                errors.append(f"{mod_name}: Missing expected function '{fn_name}'")
                found_fns.append(f"MISSING({fn_name})")
            else:
                fn_obj = getattr(mod, fn_name)
                if not callable(fn_obj):
                    errors.append(f"{mod_name}: '{fn_name}' is not callable")
                else:
                    sig = inspect.signature(fn_obj)
                    params = list(sig.parameters.keys())
                    missing_p = [p for p in req_params if p not in params]
                    if missing_p:
                        errors.append(f"{mod_name}.{fn_name}: Missing expected parameters {missing_p}. Got {params}")
                    found_fns.append(fn_name)

        rows.append({
            "package": pkg,
            "module": mod_name,
            "status": status_label,
            "contract_functions": ", ".join(found_fns) if found_fns else "—",
        })

    return rows, errors

def main() -> None:
    rows, errors = check_status()

    # Print table
    header = f"{'Package':<12} {'Module':<28} {'Status':<14} {'Contract Functions':<40}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['package']:<12} {r['module']:<28} {r['status']:<14} {r['contract_functions']:<40}")

    if errors:
        print("\nErrors found:")
        for err in errors:
            print(f"  [ERROR] {err}")
        sys.exit(1)
    else:
        print("\nAll CONTRACT.md signatures and modules verified successfully.")

if __name__ == "__main__":
    main()
