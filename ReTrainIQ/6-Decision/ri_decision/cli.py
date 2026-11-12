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

    args = parser.parse_args()
    if args.command == "setup":
        run_setup(args.project)
    elif args.command == "replay":
        run_replay(args.project, args.batches, limit=args.limit, label_lag=args.label_lag)


if __name__ == "__main__":
    main()
