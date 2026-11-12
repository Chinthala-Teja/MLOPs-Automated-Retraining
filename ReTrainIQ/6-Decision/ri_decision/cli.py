"""cli.py
Command-line interface for ReTrainIQ orchestrator and replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from ri_init.config import WORKSPACE
import ri_decision.orchestrator as orch
import ri_init.store as store

IS_STUB = False


def run_setup(project_id: str) -> None:
    demo_profile = WORKSPACE / project_id / "profile.json"
    if not demo_profile.exists():
        print(f"Error: {demo_profile} does not exist.")
        return

    with open(demo_profile, "r", encoding="utf-8") as f:
        profile = json.load(f)

    res = orch.setup_project(profile)
    print(f"Setup completed for project '{project_id}': status={res.get('status')}")
    if res.get("issues"):
        print("Issues:")
        for iss in res["issues"]:
            print(f"  - {iss}")


def run_rollback(project_id: str) -> None:
    result = orch.rollback(project_id)
    print(
        f"Rollback completed for project '{project_id}': "
        f"status={result['status']}, active_version={result['active_version']}"
    )


def run_versions(project_id: str) -> None:
    versions = store.get_versions(project_id)
    print(f"{'Version':<12} {'Status':<12} {'Trained On':<24} {'Created'}")
    print("-" * 72)
    for version in versions:
        print(
            f"{version['version']:<12} {version['status']:<12} "
            f"{version['trained_on']:<24} {version['created_at']}"
        )


def run_replay(
    project_id: str,
    batches_dir: str | Path,
    limit: Optional[int] = None,
    label_lag: int = 0,
) -> None:
    b_dir = Path(batches_dir).resolve()
    if not b_dir.exists():
        print(f"Error: batches directory {b_dir} not found.")
        return

    batch_files = sorted(list(b_dir.glob("batch_*.csv")))
    if limit is not None:
        batch_files = batch_files[:limit]

    profile_path = WORKSPACE / project_id / "profile.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)
    target_col = profile.get("task", {}).get("target", "target")

    print(f"\nReplaying {len(batch_files)} batches with label_lag={label_lag}...\n")
    header = f"{'Batch':<14} {'Labelled':<10} {'DriftShare':<12} {'Staleness':<12} {'Basis':<10} {'P':<8} {'Zone':<10} {'Action':<18}"
    print(header)
    print("-" * len(header))

    # Maintain queue of detached labels for lag: list of (batch_name, batch_id, label_csv_path)
    label_queue = []
    total_sent = 0
    total_attached = 0
    total_human = 0
    total_errors = 0

    for idx, f_path in enumerate(batch_files):
        total_sent += 1
        if label_lag > 0:
            # Separate features and labels
            df_full = pd.read_csv(f_path)
            unlabelled_df = df_full.drop(columns=[target_col], errors="ignore")
            tmp_unlabelled = WORKSPACE / project_id / "data" / f"tmp_unlab_{f_path.name}"
            unlabelled_df.to_csv(tmp_unlabelled, index=False)

            tmp_labels = WORKSPACE / project_id / "data" / f"tmp_labels_{f_path.name}"
            if target_col in df_full.columns:
                df_full[[target_col]].to_csv(tmp_labels, index=False)
            else:
                df_full.iloc[:, [-1]].to_csv(tmp_labels, index=False)

            res = orch.on_new_data(project_id, tmp_unlabelled)
            tmp_unlabelled.unlink(missing_ok=True)

            if res.get("status") in ("BLOCKED", "ERROR"):
                total_errors += 1

            if res.get("reports"):
                rep = res["reports"][-1]
                batch_id = rep["batch_id"]
                label_queue.append((f_path.name, batch_id, tmp_labels))

                drift_share = rep["drift"].get("drift_share", 0.0)
                staleness = rep["cara"].get("staleness", rep["cara"].get("staleness_est", 0.0))
                basis = rep["cara"].get("basis", "none")
                P = rep["decision"].get("P", 0.0)
                zone = rep["zone"]
                action = rep["action"]
                labelled_str = str(rep["labelled"])
                if zone == "HUMAN":
                    total_human += 1

                print(f"{f_path.name:<14} {labelled_str:<10} {drift_share:<12.3f} {staleness:<12.3f} {basis:<10} {P:<8.3f} {zone:<10} {action:<18}")

            # If lag reached, attach labels to older batch
            if len(label_queue) > label_lag:
                old_name, old_batch_id, old_labels_path = label_queue.pop(0)
                orch.attach_labels(project_id, old_batch_id, old_labels_path)
                old_labels_path.unlink(missing_ok=True)
                total_attached += 1

                # Read updated report from store to display attachment line
                all_reps = store.get_reports(project_id)
                up_rep = next((r for r in all_reps if r["batch_id"] == old_batch_id), None)
                if up_rep:
                    cara_up = json.loads(up_rep["cara_json"])
                    drift_up = json.loads(up_rep["drift_json"])
                    dec_up = json.loads(up_rep["score_json"])
                    drift_share = drift_up.get("drift_share", 0.0)
                    staleness = cara_up.get("staleness", 0.0)
                    basis = cara_up.get("basis", "measured")
                    P = dec_up.get("P", 0.0)
                    zone = up_rep["zone"]
                    p_err = dec_up.get("proxy_error")
                    err_str = f"err={p_err:+.3f}" if p_err is not None else ""
                    action_str = f"labels attached {err_str}".strip()

                    print(f"  -> {old_name:<9} True       {drift_share:<12.3f} {staleness:<12.3f} {basis:<10} {P:<8.3f} {zone:<10} {action_str:<18}")
        else:
            # Labelled replay
            res = orch.on_new_data(project_id, f_path)
            if res.get("status") in ("BLOCKED", "ERROR"):
                total_errors += 1
            for rep in res.get("reports", []):
                drift_share = rep["drift"].get("drift_share", 0.0)
                staleness = rep["cara"].get("staleness", 0.0)
                basis = rep["cara"].get("basis", "measured")
                P = rep["decision"].get("P", 0.0)
                zone = rep["zone"]
                action = rep["action"]
                labelled_str = str(rep["labelled"])
                if zone == "HUMAN":
                    total_human += 1

                print(f"{f_path.name:<14} {labelled_str:<10} {drift_share:<12.3f} {staleness:<12.3f} {basis:<10} {P:<8.3f} {zone:<10} {action:<18}")

    # Summary line
    print("-" * len(header))
    print(f"Summary: batches sent: {total_sent}, labels attached: {total_attached}, human-pending count: {total_human}, errors: {total_errors}\n")


def run_reset(project_id: str) -> None:
    res = orch.reset_project(project_id)
    print(f"Project '{project_id}' reset and set up again: status={res.get('status')}")
    for issue in res.get("issues", []):
        print(f"  - {issue}")


def run_resolve(project_id: str, report_id: int, choice: str, force: bool = False) -> None:
    res = orch.resolve_human(project_id, report_id, choice.upper(), force=force)
    print(f"Report {report_id}: choice={res['choice']}, action={res['action']}")


def run_evaluate(project_id: str, batches_dir: Optional[str] = None, limit: Optional[int] = None) -> None:
    """Baselines, sensitivity, score calibration and the label-free evaluation. All SIMULATIONS."""
    from ri_cara import proxy_eval, replay_baselines
    from ri_decision.calibrate_score import calibrate_score, load_project
    from ri_decision.sensitivity import run_sensitivity

    out_dir = WORKSPACE / project_id / "reports" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_project(project_id, batches_dir, limit)
    print(f"Building the model bank for {len(data['batches'])} batches (one model per batch)...")
    bank = replay_baselines.build_bank(data["batches"], data["calib"], data["profile"], {})
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    base = replay_baselines.run_baselines(data["batches"], data["calib"], data["profile"], {"bank": bank})
    base.to_csv(out_dir / "baselines.csv", index=False)
    print(f"\n== Strategies (simulated replay; kappa={base.attrs['kappa']:.3f}, tau*={base.attrs['tau_star']:.3f}) ==")
    print(base.drop(columns=["retrain_at"]).round(3).to_string(index=False))

    sens = run_sensitivity(project_id, batches_dir=batches_dir, limit=limit, bank=bank)
    sens.to_csv(out_dir / "sensitivity.csv", index=False)
    print("\n== Sensitivity (retrain limit and kappa multiplier) ==")
    print(sens.drop(columns=["retrain_at"]).round(3).to_string(index=False))

    cal = calibrate_score(project_id, batches_dir=batches_dir, limit=limit, bank=bank)
    (out_dir / "calibration.json").write_text(json.dumps(cal, indent=2), encoding="utf-8")
    print("\n== Decision-score calibration (fit on earlier points, test on later ones) ==")
    print(json.dumps({k: cal.get(k) for k in ("status", "n_train", "n_test", "fitted_coefficients", "brier",
                                               "calibrated", "note")}, indent=2))

    proxy = proxy_eval.run_proxy_eval(data["profile"], {})
    proxy.to_csv(out_dir / "proxy_eval.csv", index=False)
    print("\n== CARA-T-L label-free staleness under label delay ==")
    cols = [c for c in ("scenario", "label_lag", "onset_batch", "measured_actionable_at", "estimate_alert_at",
                        "estimate_lower_bound_alert_at", "drift_alert_at", "false_alarm_rate_lower_bound",
                        "spearman_estimate_vs_measured") if c in proxy.columns]
    print(proxy[cols].round(3).to_string(index=False))
    print(f"\nFiles written to {out_dir}")


_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_interval(text: str) -> int:
    """'5h', '1d', '1w', '30m' or plain seconds -> seconds."""
    text = text.strip().lower()
    if text and text[-1] in _UNITS:
        return int(float(text[:-1]) * _UNITS[text[-1]])
    return int(float(text))


def check_incoming(project_id: str, incoming: str | Path) -> dict:
    """One scheduled check: process every new file in the drop folder, oldest first.

    Data files (*.csv, *.parquet) go through on_new_data. A file named
    labels_<batch_id>.csv is attached to that stored batch. Processed files are
    moved to <incoming>/processed so a file is never handled twice.
    """
    folder = Path(incoming)
    done = folder / "processed"
    done.mkdir(parents=True, exist_ok=True)
    files = sorted((f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in (".csv", ".parquet")),
                   key=lambda f: f.stat().st_mtime)
    summary = {"files": 0, "batches": 0, "labels_attached": 0, "blocked": 0, "zones": []}
    for f in files:
        summary["files"] += 1
        if f.stem.lower().startswith("labels_") and f.stem.split("_", 1)[1].isdigit():
            res = orch.attach_labels(project_id, int(f.stem.split("_", 1)[1]), f)
            summary["labels_attached"] += 1
            summary["zones"].append(res.get("zone"))
        else:
            res = orch.on_new_data(project_id, f)
            if res.get("status") == "BLOCKED":
                summary["blocked"] += 1
            summary["batches"] += int(res.get("batches_processed", 0))
            summary["zones"].extend(r["zone"] for r in res.get("reports", []))
        f.replace(done / f.name)
    return summary


def run_watch(project_id: str, incoming: str, every: str, once: bool = False) -> None:
    import time

    seconds = parse_interval(every)
    Path(incoming).mkdir(parents=True, exist_ok=True)
    print(f"Watching {incoming} for project '{project_id}', checking every {seconds} s. Ctrl+C to stop.")
    while True:
        summary = check_incoming(project_id, incoming)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] files={summary['files']} batches={summary['batches']} "
              f"labels={summary['labels_attached']} blocked={summary['blocked']} zones={summary['zones']}")
        if once:
            return
        time.sleep(seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="ReTrainIQ Orchestrator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Setup
    setup_p = subparsers.add_parser("setup", help="Setup project environment and state")
    setup_p.add_argument("--project", type=str, default="demo", help="Project ID")

    # Replay
    replay_p = subparsers.add_parser("replay", help="Replay data batches through pipeline")
    replay_p.add_argument("--project", type=str, default="demo", help="Project ID")
    replay_p.add_argument("--batches", type=str, required=True, help="Directory containing batch files")
    replay_p.add_argument("--limit", type=int, default=None, help="Limit number of batches to process")
    replay_p.add_argument("--label-lag", type=int, default=0, help="Number of batches to lag labels behind")

    rollback_p = subparsers.add_parser("rollback", help="Restore the previous model version")
    rollback_p.add_argument("--project", type=str, default="demo", help="Project ID")

    versions_p = subparsers.add_parser("versions", help="List registered model versions")
    versions_p.add_argument("--project", type=str, default="demo", help="Project ID")

    reset_p = subparsers.add_parser("reset", help="Wipe run state and set the project up again")
    reset_p.add_argument("--project", type=str, default="demo")

    resolve_p = subparsers.add_parser("resolve", help="Record the human decision for a pending report")
    resolve_p.add_argument("--project", type=str, default="demo")
    resolve_p.add_argument("--report", type=int, required=True)
    resolve_p.add_argument("--choice", type=str, required=True, choices=["RETRAIN", "KEEP", "retrain", "keep"])
    resolve_p.add_argument("--force", action="store_true", help="Promote even if verification fails")

    eval_p = subparsers.add_parser("evaluate", help="Baselines, sensitivity, calibration, label-free evaluation")
    eval_p.add_argument("--project", type=str, default="demo")
    eval_p.add_argument("--batches", type=str, default=None)
    eval_p.add_argument("--limit", type=int, default=None)

    watch_p = subparsers.add_parser("watch", help="Check a drop folder on a schedule (5h, 1d, 1w ...)")
    watch_p.add_argument("--project", type=str, default="demo")
    watch_p.add_argument("--incoming", type=str, required=True, help="Folder where new data files are dropped")
    watch_p.add_argument("--every", type=str, default="1d", help="Check interval, e.g. 5h, 1d, 1w")
    watch_p.add_argument("--once", action="store_true", help="Run one check and exit")

    args = parser.parse_args()
    if args.command == "setup":
        run_setup(args.project)
    elif args.command == "replay":
        run_replay(args.project, args.batches, limit=args.limit, label_lag=args.label_lag)
    elif args.command == "rollback":
        run_rollback(args.project)
    elif args.command == "versions":
        run_versions(args.project)
    elif args.command == "reset":
        run_reset(args.project)
    elif args.command == "resolve":
        run_resolve(args.project, args.report, args.choice, force=args.force)
    elif args.command == "evaluate":
        run_evaluate(args.project, args.batches, args.limit)
    elif args.command == "watch":
        run_watch(args.project, args.incoming, args.every, once=args.once)


if __name__ == "__main__":
    main()
