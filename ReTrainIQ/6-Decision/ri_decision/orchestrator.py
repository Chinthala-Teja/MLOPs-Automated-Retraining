"""orchestrator.py
Stage 2 Orchestrator: manages project setup, data streams, model invocations via worker,
cooldown, decision handling, label attachment, and human resolution.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ri_init.config import ROOT, WORKSPACE
import ri_init.store as store
from ri_init.worker import run_in_worker

import ri_drift.drift as drift_mod
import ri_metrics.metrics as metrics_mod
import ri_decision.decision as decision_mod
import ri_decision.retrain as retrain_mod

IS_STUB = False


def _resolve_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    return (ROOT / path_obj).resolve()


def _to_rel_path(p: str | Path | None) -> str | None:
    if p is None:
        return None
    path_obj = Path(p).resolve()
    try:
        return str(path_obj.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path_obj).replace("\\", "/")


def _load_state(project_id: str) -> Dict[str, Any]:
    state_file = WORKSPACE / project_id / "state.json"
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "active_version": "v1",
        "cooldown_left": 0,
        "buffer_rows": 0,
        "batch_counter": 0,
    }


def _save_state(project_id: str, state: Dict[str, Any]) -> None:
    proj_dir = WORKSPACE / project_id
    proj_dir.mkdir(parents=True, exist_ok=True)
    state_file = proj_dir / "state.json"
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def setup_project(profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Initialize DB, folders, state, baseline, calib, and reference stats.
    """
    project_id = profile.get("project_id", "demo")
    proj_dir = WORKSPACE / project_id
    data_dir = proj_dir / "data"
    batches_dir = data_dir / "batches"
    rejected_dir = data_dir / "rejected"
    models_dir = proj_dir / "models"
    reports_dir = proj_dir / "reports"

    for d in [data_dir, batches_dir, rejected_dir, models_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    store.init_db()

    # Register project in store
    store.add_project(
        project_id=project_id,
        source=profile.get("source", "setup"),
        profile_json=json.dumps(profile),
        status="active",
    )

    # Register version v1 as active
    model_art_raw = profile.get("model", {}).get("artifact")
    model_art = _to_rel_path(_resolve_path(model_art_raw))
    store.add_version(
        project_id=project_id,
        version="v1",
        path=model_art or "",
        trained_on="reference",
        status="active",
    )
    store.set_active_version(project_id, "v1")

    issues: List[str] = []

    # Read reference data
    ref_path = _resolve_path(profile.get("data", {}).get("reference_path"))
    if ref_path and ref_path.exists():
        if ref_path.suffix.lower() == ".parquet":
            ref_df = pd.read_parquet(ref_path)
        else:
            ref_df = pd.read_csv(ref_path)
    else:
        ref_df = pd.DataFrame()
        issues.append("Reference data path not found.")

    # Save 2000-row seeded sample of reference data
    sample_size = min(2000, len(ref_df))
    seed = profile.get("seed", 42)
    if sample_size > 0:
        train_sample_df = ref_df.sample(n=sample_size, random_state=seed)
    else:
        train_sample_df = ref_df

    train_sample_path = data_dir / "train_sample.csv"
    train_sample_df.to_csv(train_sample_path, index=False)

    # Metric plan & baseline
    plan = metrics_mod.default_plan(profile)
    holdout_path_raw = profile.get("data", {}).get("holdout_path")
    holdout_path = _resolve_path(holdout_path_raw)

    if holdout_path and holdout_path.exists():
        if holdout_path.suffix.lower() == ".parquet":
            holdout_df = pd.read_parquet(holdout_path)
        else:
            holdout_df = pd.read_csv(holdout_path)

        baseline_res = run_in_worker(
            "ri_metrics.metrics:compute_baseline",
            {
                "model_path": str(_resolve_path(model_art_raw)),
                "holdout_df": holdout_df,
                "profile": profile,
                "plan": plan,
            },
        )
    else:
        baseline_res = {"status": "PENDING", "module": "quality"}
        issues.append("Holdout path missing; baseline quality comparison will be off.")

    baseline_file = proj_dir / "baseline.json"
    with open(baseline_file, "w", encoding="utf-8") as f:
        json.dump(baseline_res, f, indent=2)

    # Calibrate on ordered reference data
    calib_res = run_in_worker(
        "ri_cara.cara_t:calibrate",
        {
            "model_path": str(_resolve_path(model_art_raw)),
            "history_df": train_sample_df,
            "profile": profile,
            "cfg": {},
        },
    )
    calib_file = proj_dir / "calib.json"
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_res, f, indent=2)

    # Drift reference stats
    ref_stats = drift_mod.fit_reference(train_sample_df, profile)
    ref_stats_file = proj_dir / "reference_stats.json"
    with open(ref_stats_file, "w", encoding="utf-8") as f:
        json.dump(ref_stats, f, indent=2)

    # Initialize state
    state = {
        "active_version": "v1",
        "cooldown_left": 0,
        "buffer_rows": 0,
        "batch_counter": 0,
    }
    _save_state(project_id, state)

    return {
        "status": "OK",
        "project_id": project_id,
        "issues": issues,
    }


def on_new_data(
    project_id: str,
    file_path: Path | str,
    labels_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """
    Process incoming data stream file.
    Validates schema, buffers rows, runs batches when buffer >= B, evaluates drift,
    cara, quality, decision, applies cooldown, retrain or human fallback.
    """
    proj_dir = WORKSPACE / project_id
    state = _load_state(project_id)

    # Load profile
    profile_path = proj_dir / "profile.json"
    if not profile_path.exists():
        # Fallback to store
        proj_row = store.get_project(project_id)
        if proj_row and proj_row.get("profile_json"):
            profile = json.loads(proj_row["profile_json"])
        else:
            raise FileNotFoundError(f"Project profile not found for {project_id}")
    else:
        with open(profile_path, "r", encoding="utf-8") as f:
            profile = json.load(f)

    # Read incoming data file
    in_p = Path(file_path).resolve()
    if not in_p.exists():
        raise FileNotFoundError(f"Incoming data file not found: {in_p}")

    df_in = pd.read_parquet(in_p) if in_p.suffix.lower() == ".parquet" else pd.read_csv(in_p)

    # Validate schema
    schema_res = drift_mod.validate_schema(df_in, profile)
    if schema_res.get("status") == "BLOCKED" or not schema_res.get("valid", True):
        # Move file to rejected
        rejected_dir = proj_dir / "data" / "rejected"
        rejected_dir.mkdir(parents=True, exist_ok=True)
        dest_rejected = rejected_dir / in_p.name
        try:
            df_in.to_csv(dest_rejected, index=False)
        except Exception:
            pass

        # Store report with zone BLOCKED
        batch_id = store.add_batch(project_id, str(_to_rel_path(dest_rejected)), len(df_in), 0)
        report_id = store.add_report(
            project_id=project_id,
            batch_id=batch_id,
            drift_json=json.dumps(schema_res),
            cara_json="{}",
            quality_json="{}",
            score_json="{}",
            zone="BLOCKED",
            summary="Schema validation failed; file rejected.",
        )
        store.add_decision(report_id, "BLOCKED", "system", "Schema error")
        return {
            "status": "BLOCKED",
            "issues": schema_res.get("missing_columns", ["Schema mismatch"]),
        }

    # If labels_path is provided, join labels
    target_col = profile.get("task", {}).get("target", "target")
    if labels_path:
        lp = Path(labels_path).resolve()
        df_labels = pd.read_parquet(lp) if lp.suffix.lower() == ".parquet" else pd.read_csv(lp)

        id_col = profile.get("data", {}).get("id_column")
        if id_col and id_col in df_in.columns and id_col in df_labels.columns:
            df_in = df_in.merge(df_labels[[id_col, target_col]], on=id_col, how="left")
        else:
            if len(df_in) != len(df_labels):
                raise ValueError(
                    f"Labels length ({len(df_labels)}) does not match data length ({len(df_in)}) and no id_column specified."
                )
            if target_col in df_labels.columns:
                df_in[target_col] = df_labels[target_col].values
            else:
                df_in[target_col] = df_labels.iloc[:, -1].values

    # Buffer management
    buffer_file = proj_dir / "data" / "buffer.csv"
    if buffer_file.exists():
        df_buffer = pd.read_csv(buffer_file)
        df_combined = pd.concat([df_buffer, df_in], ignore_index=True)
    else:
        df_combined = df_in

    B = int(profile.get("batch_size_B", 2000))
    batches_dir = proj_dir / "data" / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline, calib, train_sample
    train_sample_path = proj_dir / "data" / "train_sample.csv"
    if train_sample_path.exists():
        train_sample_df = pd.read_csv(train_sample_path)
    else:
        train_sample_df = pd.DataFrame()

    calib_file = proj_dir / "calib.json"
    if calib_file.exists():
        with open(calib_file, "r", encoding="utf-8") as f:
            calib = json.load(f)
    else:
        calib = {"module": "cara_t", "status": "STUB"}

    baseline_file = proj_dir / "baseline.json"
    if baseline_file.exists():
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline = json.load(f)
    else:
        baseline = {"module": "quality", "status": "STUB"}

    active_ver = store.get_active_version(project_id)
    model_path = _resolve_path(active_ver["path"]) if active_ver else _resolve_path(profile["model"]["artifact"])

    reports_out: List[Dict[str, Any]] = []
    n_batches_processed = 0

    while len(df_combined) >= B:
        batch_df = df_combined.iloc[:B].copy()
        df_combined = df_combined.iloc[B:].reset_index(drop=True)

        batch_idx = state["batch_counter"]
        batch_filename = f"batch_{batch_idx:03d}.csv"
        batch_path = batches_dir / batch_filename
        batch_df.to_csv(batch_path, index=False)
        state["batch_counter"] += 1
        n_batches_processed += 1

        # Check if labelled (at least 90% non-null target)
        n_rows = len(batch_df)
        if target_col in batch_df.columns:
            n_labelled = int(batch_df[target_col].notna().sum())
        else:
            n_labelled = 0

        is_labelled = (n_labelled / n_rows) >= 0.90 if n_rows > 0 else False

        # 1. Drift
        drift_res = drift_mod.run_drift(train_sample_df, batch_df, profile)

        # 2. CARA & Quality
        if is_labelled:
            cara_res = run_in_worker(
                "ri_cara.cara_t:online_staleness",
                {
                    "model_path": str(model_path),
                    "new_df": batch_df,
                    "train_sample_df": train_sample_df,
                    "calib": calib,
                    "profile": profile,
                    "seed": profile.get("seed", 42),
                },
            )
            plan = metrics_mod.default_plan(profile)
            quality_raw = run_in_worker(
                "ri_metrics.metrics:run_metrics",
                {
                    "model_path": str(model_path),
                    "df_labelled": batch_df,
                    "profile": profile,
                    "plan": plan,
                },
            )
            quality_res = metrics_mod.compare(baseline, quality_raw, profile.get("primary_metric", "recall"))
        else:
            cara_res = run_in_worker(
                "ri_cara.cara_t:estimate_staleness",
                {
                    "model_path": str(model_path),
                    "new_df": batch_df,
                    "train_sample_df": train_sample_df,
                    "calib": calib,
                    "profile": profile,
                    "seed": profile.get("seed", 42),
                },
            )
            quality_res = {"module": "quality", "status": "PENDING", "z_m": 0.0}

        # 3. Decision
        dec_res = decision_mod.decide(drift_res, cara_res, quality_res)

        # For unlabelled batch, contract specifies:
        # "an unlabelled batch runs drift + estimate_staleness and ends in HUMAN with the alert 'request labels'"
        if not is_labelled:
            dec_res["zone"] = "HUMAN"
            dec_res["alert"] = "request labels"

        zone = dec_res["zone"]

        # Handle cooldown: if cooldown_left > 0 and zone is RETRAIN -> downgrade to HUMAN
        if state["cooldown_left"] > 0 and zone == "RETRAIN":
            zone = "HUMAN"
            dec_res["zone"] = "HUMAN"
            dec_res["reason"] = "cooldown"

        # Action handling
        action_recorded = "KEEP"
        decision_by = "system"
        decision_note = dec_res.get("reason", "")

        if zone == "RETRAIN":
            # Check if retrain module is stub
            if getattr(retrain_mod, "IS_STUB", True):
                action_recorded = "retrain skipped (stub)"
            else:
                # Run train_candidate, verify, promote
                next_v_idx = int(state.get("active_version", "v1").replace("v", "")) + 1
                cand_ver = f"v{next_v_idx}"
                cand_path = proj_dir / "models" / cand_ver / "model.joblib"
                cand_path.parent.mkdir(parents=True, exist_ok=True)

                train_res = retrain_mod.train_candidate(profile, batch_df, cand_path)
                holdout_df_cand = pd.read_csv(profile["data"]["holdout_path"]) if profile["data"].get("holdout_path") else None
                ver_res = retrain_mod.verify(cand_path, holdout_df_cand, profile)

                if ver_res.get("verified", False):
                    active_path = proj_dir / "models" / cand_ver / "model.joblib"
                    retrain_mod.promote(cand_path, active_path)
                    store.add_version(project_id, cand_ver, str(_to_rel_path(active_path)), batch_filename, "active")
                    store.set_active_version(project_id, cand_ver)
                    state["active_version"] = cand_ver
                    model_path = active_path
                    action_recorded = "PROMOTED"

                    # Rerun calibrate, refresh train_sample and baseline, set cooldown
                    calib = run_in_worker("ri_cara.cara_t:calibrate", {"model_path": str(model_path), "history_df": batch_df, "profile": profile, "cfg": {}})
                    with open(calib_file, "w", encoding="utf-8") as f:
                        json.dump(calib, f, indent=2)
                    state["cooldown_left"] = int(profile.get("W", 1))
                else:
                    action_recorded = "VERIFY_FAILED"

        elif zone == "HUMAN":
            action_recorded = "pending"
            if not is_labelled:
                dec_res["alert"] = "request labels"
                decision_note += " (request labels)"

        elif zone == "KEEP":
            action_recorded = "KEEP"

        # Decrement cooldown after each labelled batch
        if is_labelled and state["cooldown_left"] > 0:
            state["cooldown_left"] -= 1

        # Store batch, report, and decision
        rel_batch_path = _to_rel_path(batch_path)
        batch_id = store.add_batch(project_id, rel_batch_path or str(batch_path), n_rows, n_labelled)

        summary_text = f"Batch {batch_filename}: Zone {zone}. {decision_note}"
        report_id = store.add_report(
            project_id=project_id,
            batch_id=batch_id,
            drift_json=json.dumps(drift_res),
            cara_json=json.dumps(cara_res),
            quality_json=json.dumps(quality_res),
            score_json=json.dumps(dec_res),
            zone=zone,
            summary=summary_text,
        )

        store.add_decision(report_id, action_recorded, decision_by, decision_note)

        rep_dict = {
            "report_id": report_id,
            "batch_id": batch_id,
            "batch_file": batch_filename,
            "labelled": is_labelled,
            "drift": drift_res,
            "cara": cara_res,
            "quality": quality_res,
            "decision": dec_res,
            "zone": zone,
            "action": action_recorded,
        }
        reports_out.append(rep_dict)

    # Save remaining buffer
    state["buffer_rows"] = len(df_combined)
    _save_state(project_id, state)
    if len(df_combined) > 0:
        df_combined.to_csv(buffer_file, index=False)
    else:
        if buffer_file.exists():
            buffer_file.unlink(missing_ok=True)

    return {
        "status": "OK",
        "batches_processed": n_batches_processed,
        "reports": reports_out,
        "buffer_rows": len(df_combined),
    }


def attach_labels(project_id: str, batch_id: int, labels_path: Path | str) -> Dict[str, Any]:
    """
    Attach late labels to stored batch, compute measured cara and quality,
    record proxy_error on report, re-evaluate decision, and update store.
    """
    proj_dir = WORKSPACE / project_id
    report_row = store.get_report(batch_id)  # report by id or lookup by batch_id
    # Look up reports for project to find exact report
    all_reports = store.get_reports(project_id)
    matching_report = next((r for r in all_reports if r["batch_id"] == batch_id), None)
    if not matching_report:
        raise ValueError(f"Report for batch_id {batch_id} not found in project {project_id}")

    report_id = matching_report["id"]
    profile_path = proj_dir / "profile.json"
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    # Load stored batch CSV
    target_col = profile.get("task", {}).get("target", "target")
    batch_record = store.get_batch(batch_id)
    if batch_record and batch_record.get("file"):
        batch_file = _resolve_path(batch_record["file"])
    else:
        batch_file = None

    if not batch_file or not batch_file.exists():
        # Fallback to search in batches directory
        matching_files = sorted(list((proj_dir / "data" / "batches").glob("batch_*.csv")))
        if matching_files:
            batch_file = matching_files[-1]
        else:
            raise FileNotFoundError(f"Batch file for batch_id {batch_id} not found.")

    batch_df = pd.read_csv(batch_file)
    lp = Path(labels_path).resolve()
    df_labels = pd.read_parquet(lp) if lp.suffix.lower() == ".parquet" else pd.read_csv(lp)

    id_col = profile.get("data", {}).get("id_column")
    if id_col and id_col in batch_df.columns and id_col in df_labels.columns:
        batch_df = batch_df.merge(df_labels[[id_col, target_col]], on=id_col, how="left")
    else:
        if len(batch_df) != len(df_labels):
            raise ValueError(f"Labels length ({len(df_labels)}) != batch length ({len(batch_df)})")
        batch_df[target_col] = df_labels[target_col].values if target_col in df_labels.columns else df_labels.iloc[:, -1].values

    # Overwrite batch file with labelled data
    batch_df.to_csv(batch_file, index=False)

    train_sample_df = pd.read_csv(proj_dir / "data" / "train_sample.csv")
    with open(proj_dir / "calib.json", "r", encoding="utf-8") as f:
        calib = json.load(f)
    with open(proj_dir / "baseline.json", "r", encoding="utf-8") as f:
        baseline = json.load(f)

    active_ver = store.get_active_version(project_id)
    model_path = _resolve_path(active_ver["path"]) if active_ver else _resolve_path(profile["model"]["artifact"])

    # Compute measured cara and quality
    cara_measured = run_in_worker(
        "ri_cara.cara_t:online_staleness",
        {
            "model_path": str(model_path),
            "new_df": batch_df,
            "train_sample_df": train_sample_df,
            "calib": calib,
            "profile": profile,
            "seed": profile.get("seed", 42),
        },
    )

    plan = metrics_mod.default_plan(profile)
    quality_raw = run_in_worker(
        "ri_metrics.metrics:run_metrics",
        {
            "model_path": str(model_path),
            "df_labelled": batch_df,
            "profile": profile,
            "plan": plan,
        },
    )
    quality_measured = metrics_mod.compare(baseline, quality_raw, profile.get("primary_metric", "recall"))

    # Load drift from original report
    drift_orig = json.loads(matching_report["drift_json"])
    cara_est = json.loads(matching_report["cara_json"])

    # Proxy error = cara_estimated.staleness_est - measured staleness
    est_staleness = cara_est.get("staleness_est")
    meas_staleness = cara_measured.get("staleness", 0.0)
    proxy_error = (est_staleness - meas_staleness) if est_staleness is not None else None

    # Re-evaluate decision
    dec_res = decision_mod.decide(drift_orig, cara_measured, quality_measured)
    dec_res["proxy_error"] = proxy_error
    zone = dec_res["zone"]

    # Update report
    store.update_report(
        report_id=report_id,
        cara_json=json.dumps(cara_measured),
        quality_json=json.dumps(quality_measured),
        score_json=json.dumps(dec_res),
        zone=zone,
        summary=f"Late labels attached. Re-evaluated zone: {zone}.",
    )

    action = "KEEP" if zone == "KEEP" else ("retrain skipped (stub)" if zone == "RETRAIN" else "pending")
    store.add_decision(report_id, action, "late_labels", f"proxy_error={proxy_error}")

    return {
        "status": "OK",
        "report_id": report_id,
        "proxy_error": proxy_error,
        "zone": zone,
        "decision": dec_res,
    }


def resolve_human(project_id: str, report_id: int, choice: str, force: bool = False) -> Dict[str, Any]:
    """
    Resolve human decision for report: 'RETRAIN' or 'KEEP'.
    """
    if choice not in ("RETRAIN", "KEEP"):
        raise ValueError(f"Invalid human choice: {choice}. Must be 'RETRAIN' or 'KEEP'.")

    proj_dir = WORKSPACE / project_id
    report = store.get_report(report_id)
    if not report:
        raise ValueError(f"Report id {report_id} not found")

    action = choice
    if choice == "RETRAIN":
        if getattr(retrain_mod, "IS_STUB", True):
            action = "retrain skipped (stub)"
        else:
            profile_path = proj_dir / "profile.json"
            with open(profile_path, "r", encoding="utf-8") as f:
                profile = json.load(f)

            batch_id = report["batch_id"]
            batch_file = proj_dir / "data" / "batches" / f"batch_{batch_id - 1:03d}.csv"
            batch_df = pd.read_csv(batch_file)

            cand_path = proj_dir / "models" / "v_human" / "model.joblib"
            cand_path.parent.mkdir(parents=True, exist_ok=True)
            _ = retrain_mod.train_candidate(profile, batch_df, cand_path)

            holdout_df = pd.read_csv(profile["data"]["holdout_path"]) if profile["data"].get("holdout_path") else None
            ver = retrain_mod.verify(cand_path, holdout_df, profile)

            if not ver.get("verified", False) and not force:
                return {"status": "VERIFY_FAILED"}

            retrain_mod.promote(cand_path, proj_dir / "models" / "active" / "model.joblib")
            action = "PROMOTED"

    store.add_decision(report_id, action, "human", f"Human resolved as {choice}")
    store.update_report(report_id, zone=choice)

    return {
        "status": "OK",
        "choice": choice,
        "action": action,
    }
