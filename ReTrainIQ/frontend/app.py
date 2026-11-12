"""ReTrainIQ dashboard (Streamlit). Run:  streamlit run frontend/app.py

Pages: Setup, Monitor, Report, Chat. The app only calls orchestrator / module
functions; every number shown comes from code, never from the LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
for _folder in ("0-Init", "1-Profile", "2-Drift", "3-CARA", "4-Metrics", "5-Analysis", "6-Decision"):
    if str(ROOT_DIR / _folder) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR / _folder))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import ri_decision.orchestrator as orch  # noqa: E402
import ri_init.store as store  # noqa: E402
import ri_profile.understand as understand  # noqa: E402
from ri_analysis.chat import chat as chat_fn  # noqa: E402
from ri_analysis.chat import default_tools  # noqa: E402
from ri_analysis.llm import get_client  # noqa: E402
from ri_analysis.summary import report_from_row, summarize_with_meta, template_summary  # noqa: E402
from ri_drift import plots as drift_plots  # noqa: E402
from ri_drift.scenarios import inject_covariate_shift  # noqa: E402
from ri_init.config import WORKSPACE, load_config  # noqa: E402

ZONE_COLOR = {"RETRAIN": "#c0392b", "HUMAN": "#d68910", "KEEP": "#1e8449", "BLOCKED": "#566573"}
ZONE_TEXT = {"RETRAIN": "RETRAIN (automatic)", "HUMAN": "HUMAN REVIEW", "KEEP": "KEEP", "BLOCKED": "BLOCKED"}


# ----------------------------------------------------------------------------- data access
def list_projects() -> List[str]:
    if not WORKSPACE.is_dir():
        return []
    return sorted(p.name for p in WORKSPACE.iterdir() if (p / "profile.json").is_file())


def load_profile(project_id: str) -> Dict[str, Any]:
    return json.loads((WORKSPACE / project_id / "profile.json").read_text(encoding="utf-8"))


def is_set_up(project_id: str) -> bool:
    return (WORKSPACE / project_id / "state.json").is_file() and store.get_active_version(project_id) is not None


def load_reports(project_id: str) -> List[Dict[str, Any]]:
    profile = load_profile(project_id)
    return [report_from_row(r, profile) for r in store.get_reports(project_id)]


def decisions_by_report(project_id: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for d in store.get_decisions(project_id):
        out.setdefault(d["report_id"], []).append(d)
    return out


def save_upload(uploaded: Any, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(uploaded.name).name
    path.write_bytes(uploaded.getbuffer())
    return path


def history_frame(project_id: str) -> pd.DataFrame:
    actions = decisions_by_report(project_id)
    rows = []
    for rep in load_reports(project_id):
        if rep["zone"] == "BLOCKED":
            continue
        primary = rep["quality"].get("primary_metric")
        metric = rep["quality"].get("metrics", {}).get(primary or "", {}).get("value")
        acts = [d["action"] for d in actions.get(rep["report_id"], [])]
        rows.append({
            "check": rep["report_id"], "zone": rep["zone"], "P": rep["decision"].get("P"),
            "staleness": rep["cara"].get("staleness"), "tau_star": rep["cara"].get("tau_star"),
            "basis": rep["decision"].get("basis"), "drift_share": rep["drift"].get("drift_share"),
            "primary_metric": metric,
            "event": next((a for a in acts if a in ("PROMOTED", "VERIFY_FAILED", "ROLLED_BACK")), ""),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- charts
def gauge(p: float, cfg: Dict[str, Any]) -> go.Figure:
    keep, retrain = 100 * cfg["keep_limit"], 100 * cfg["retrain_limit"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * p, number={"suffix": "%"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#2c3e50"},
               "steps": [{"range": [0, keep], "color": "#d5f5e3"},
                         {"range": [keep, retrain], "color": "#fdebd0"},
                         {"range": [retrain, 100], "color": "#f5b7b1"}]},
        title={"text": "Decision score"}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def timeline(hist: pd.DataFrame, cfg: Dict[str, Any]) -> go.Figure:
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Decision score", "Staleness vs tau*", "Primary metric"))
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["P"], mode="lines+markers", name="score",
                             marker=dict(color=[ZONE_COLOR.get(z, "#555") for z in hist["zone"]], size=9),
                             line=dict(color="#95a5a6")), row=1, col=1)
    fig.add_hline(y=cfg["retrain_limit"], line_dash="dash", line_color=ZONE_COLOR["RETRAIN"], row=1, col=1)
    fig.add_hline(y=cfg["keep_limit"], line_dash="dash", line_color=ZONE_COLOR["KEEP"], row=1, col=1)
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["staleness"], mode="lines+markers", name="staleness",
                             line=dict(color="#2c3e50")), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["tau_star"], mode="lines", name="tau*",
                             line=dict(color=ZONE_COLOR["RETRAIN"], dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist["check"], y=hist["primary_metric"], mode="lines+markers", name="metric",
                             line=dict(color="#1f618d")), row=3, col=1)
    for _, row in hist[hist["event"] != ""].iterrows():
        color = {"PROMOTED": "#1e8449", "VERIFY_FAILED": "#d68910", "ROLLED_BACK": "#c0392b"}[row["event"]]
        fig.add_vline(x=row["check"], line_color=color, line_width=2, opacity=0.6)
        fig.add_annotation(x=row["check"], y=1.0, yref="paper", text=row["event"].replace("_", " ").title(),
                           showarrow=False, font=dict(color=color, size=11), yshift=12)
    fig.update_layout(height=620, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# ----------------------------------------------------------------------------- pages
def page_setup(project_id: Optional[str]) -> None:
    st.header("Setup")
    st.caption("Once per ML project: read the project, confirm the profile, register the model.")
    left, right = st.columns(2)
    with left:
        st.subheader("Demo project")
        st.write("A synthetic flight-delay project: 50,000 reference rows, a Random Forest pipeline and "
                 "50 stream batches with a simulated concept shift after batch 25.")
        if st.button("Create / reset the demo project"):
            with st.spinner("Generating data, training v1 and calibrating CARA-T (about a minute)..."):
                if not (WORKSPACE / "demo" / "profile.json").is_file():
                    from ri_init import make_demo_project
                    make_demo_project.main()
                res = orch.reset_project("demo")
            st.success(f"Demo project ready: {res['status']}")
            for issue in res.get("issues", []):
                st.warning(issue)
    with right:
        st.subheader("Status")
        if project_id and is_set_up(project_id):
            prof, active = load_profile(project_id), store.get_active_version(project_id)
            st.write(f"**{project_id}**: {prof.get('problem_statement', '')}")
            st.write(f"Model: {prof['model'].get('algorithm')} | active version: **{active['version']}** | "
                     f"primary metric: {prof.get('primary_metric')} | batch size: {prof.get('batch_size_B')}")
            st.write("LLM: " + ("configured" if get_client() else "not configured (template summaries, manual profile)"))
        else:
            st.info("No project is set up yet.")

    st.divider()
    st.subheader("Register your own project")
    st.caption("Version 1 supports a scikit-learn Pipeline that accepts raw rows and tabular data. "
               "Repository code is scanned as text only and is never executed.")
    c1, c2 = st.columns(2)
    new_id = c1.text_input("Project id (letters, digits, dashes)", value="myproject")
    url = c1.text_input("GitHub URL (https://github.com/owner/repo)")
    zip_file = c1.file_uploader("...or project ZIP", type=["zip"])
    model_file = c2.file_uploader("Saved model (.joblib / .pkl)", type=["joblib", "pkl"])
    ref_file = c2.file_uploader("Reference data = the model's training data (CSV / Parquet)", type=["csv", "parquet"])
    hold_file = c2.file_uploader("Holdout the model never saw (optional but recommended)", type=["csv", "parquet"])
    target = c2.text_input("Target column (blank = last column)")
    if st.button("Scan and draft profile"):
        draft_profile(new_id.strip(), url.strip(), zip_file, model_file, ref_file, hold_file, target.strip())
    if "draft" in st.session_state:
        confirm_profile()


def draft_profile(new_id: str, url: str, zip_file: Any, model_file: Any, ref_file: Any, hold_file: Any, target: str) -> None:
    if not new_id.replace("-", "").replace("_", "").isalnum():
        st.error("Project id may contain only letters, digits, dashes and underscores.")
        return
    if ref_file is None or model_file is None:
        st.error("The reference data and the saved model are required.")
        return
    proj = WORKSPACE / new_id
    scan: Optional[Dict[str, Any]] = None
    try:
        if url:
            scan = understand.scan_project(understand.fetch_repo(url, proj / "repo"))
        elif zip_file is not None:
            zpath = save_upload(zip_file, proj / "upload")
            understand.extract_zip_safe(zpath, proj / "repo")
            scan = understand.scan_project(proj / "repo")
    except Exception as exc:  # shown to the user; nothing was executed
        st.error(f"Could not read the project: {exc}")
        return
    ref_path = save_upload(ref_file, proj / "data")
    model_path = save_upload(model_file, proj / "models" / "v1")
    if model_path.name != "model.joblib":
        model_path = model_path.replace(model_path.with_name("model.joblib"))
    hold_path = save_upload(hold_file, proj / "data") if hold_file is not None else None
    llm, source = get_client(), "heuristic"
    profile = understand.heuristic_profile(scan=scan, dataset_path=str(ref_path), target=target or None)
    if llm is not None and scan is not None:
        try:
            extracted = understand.extract_profile(understand.build_context(scan), llm)
            for key in ("problem_statement", "task", "features", "primary_metric"):
                if extracted.get(key) not in (None, "unknown", {}, []):
                    profile[key] = extracted[key]
            source = "LLM draft (verify it)"
        except Exception as exc:
            st.warning(f"LLM profile extraction failed ({exc}); using the heuristic profile.")
    profile["project_id"] = new_id
    profile.setdefault("model", {})["artifact"] = str(model_path)
    profile.setdefault("data", {})["reference_path"] = str(ref_path)
    profile["data"]["holdout_path"] = str(hold_path) if hold_path else None
    st.session_state["draft"] = {"profile": profile, "source": source}


def confirm_profile() -> None:
    draft = st.session_state["draft"]
    st.write(f"Profile draft ({draft['source']}). Edit the JSON if anything is wrong, then confirm.")
    text = st.text_area("Profile", json.dumps(draft["profile"], indent=2), height=380)
    if st.button("Confirm profile and register the model"):
        try:
            profile = json.loads(text)
        except ValueError as exc:
            st.error(f"Profile is not valid JSON: {exc}")
            return
        with st.spinner("Verifying the profile, computing the baseline and calibrating CARA-T..."):
            final = understand.finalize_profile(profile)
            blocking = [i for i in final.get("issues", []) if "ERROR" in i.upper() or "cannot" in i.lower()]
            for issue in final.get("issues", []):
                st.warning(issue)
            if blocking:
                st.error("Fix the issues above before registering.")
                return
            res = orch.setup_project({k: v for k, v in final.items() if k != "issues"})
        st.success(f"Project '{final['project_id']}' registered: {res['status']}")
        for issue in res.get("issues", []):
            st.warning(issue)
        del st.session_state["draft"]


def show_results(result: Dict[str, Any]) -> None:
    if result.get("status") == "BLOCKED":
        st.error("Upload blocked: " + "; ".join(str(i) for i in result.get("issues", [])))
    elif result.get("status") == "DUPLICATE":
        st.warning("This exact file was already processed.")
    else:
        st.success(f"{result.get('batches_processed', 0)} batch(es) checked; "
                   f"{result.get('buffer_rows', 0)} rows waiting for a full batch.")
        for rep in result.get("reports", []):
            st.markdown(f"Check **{rep['report_id']}** -> :{'red' if rep['zone'] == 'RETRAIN' else 'orange' if rep['zone'] == 'HUMAN' else 'green'}"
                        f"[{ZONE_TEXT.get(rep['zone'], rep['zone'])}] (score {rep['decision'].get('P', 0):.2f}, action {rep['action']})")


def page_monitor(project_id: str) -> None:
    st.header("Monitor")
    cfg = load_config(project_id)
    profile = load_profile(project_id)
    up, late = st.columns(2)
    with up:
        st.subheader("New data")
        data_file = st.file_uploader("Data file (features, plus the target if known)", type=["csv", "parquet"], key="new")
        labels_file = st.file_uploader("Labels for this file (optional)", type=["csv", "parquet"], key="newlab")
        if st.button("Run check") and data_file is not None:
            inbox = WORKSPACE / project_id / "data" / "uploads"
            path = save_upload(data_file, inbox)
            lab = save_upload(labels_file, inbox) if labels_file is not None else None
            with st.spinner("Drift, staleness, quality, decision..."):
                show_results(orch.on_new_data(project_id, path, lab))
    with late:
        st.subheader("Labels that arrived later")
        pending = [r for r in load_reports(project_id)
                   if r["decision"].get("basis") in ("estimated", "none") and "proxy_error" not in r["decision"]
                   and r["zone"] != "BLOCKED"]
        if pending:
            pick = st.selectbox("Check waiting for labels", [r["report_id"] for r in pending])
            late_file = st.file_uploader("Labels file (target column, same row order or id column)",
                                         type=["csv", "parquet"], key="late")
            if st.button("Attach labels") and late_file is not None:
                rep = next(r for r in pending if r["report_id"] == pick)
                path = save_upload(late_file, WORKSPACE / project_id / "data" / "uploads")
                with st.spinner("Computing measured staleness and quality..."):
                    res = orch.attach_labels(project_id, rep["batch_id"], path)
                err = res.get("proxy_error")
                st.success(f"Labels attached. Zone now {res['zone']}."
                           + (f" Estimate minus measured staleness: {err:+.3f}." if err is not None else ""))
        else:
            st.caption("No check is waiting for labels.")

    st.divider()
    chart_slot = st.empty()
    hist = history_frame(project_id)
    if len(hist):
        chart_slot.plotly_chart(timeline(hist, cfg), use_container_width=True, key="timeline")
        st.dataframe(hist, use_container_width=True, hide_index=True)
    else:
        chart_slot.info("No checks yet. Upload data or run the demo below.")

    st.divider()
    st.subheader("Demo mode (simulated replay)")
    stream = sorted((WORKSPACE / project_id / "data" / "stream").glob("batch_*.csv"))
    if not stream:
        st.caption("This project has no data/stream folder to replay.")
        return
    st.caption("Replays stored stream batches through the full loop. Drift here is SIMULATED, not real.")
    c1, c2, c3 = st.columns(3)
    first, last = c1.slider("Batches to replay", 0, len(stream) - 1, (min(20, len(stream) - 1), min(31, len(stream) - 1)))
    scenario = c2.selectbox("Scenario", ["stream as stored", "labels arrive 2 batches late",
                                         "inject covariate shift from the middle"])
    fresh = c3.checkbox("Reset the project first", value=True)
    if st.button("Run demo"):
        run_demo(project_id, profile, stream[first:last + 1], scenario, fresh, chart_slot, cfg)


def run_demo(project_id: str, profile: Dict[str, Any], files: List[Path], scenario: str, fresh: bool,
             chart_slot: Any, cfg: Dict[str, Any]) -> None:
    target = profile["task"]["target"]
    tmp = WORKSPACE / project_id / "data" / "uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    if fresh:
        with st.spinner("Resetting the project..."):
            orch.reset_project(project_id)
    progress, log = st.progress(0.0), st.empty()
    queue: List[tuple] = []
    for i, f in enumerate(files):
        frame = pd.read_csv(f)
        if scenario.startswith("inject") and i >= len(files) // 2:
            numeric = profile["features"]["numeric"]
            frame = inject_covariate_shift(frame, {**profile, "seed": 42 + i}, numeric[0], 1.5, 0)
        send = tmp / f"demo_{i:03d}_{f.name}"
        if scenario.startswith("labels"):
            labels = tmp / f"demo_labels_{i:03d}_{f.name}"
            frame[[target]].to_csv(labels, index=False)
            frame.drop(columns=[target]).to_csv(send, index=False)
        else:
            frame.to_csv(send, index=False)
        res = orch.on_new_data(project_id, send)
        for rep in res.get("reports", []):
            log.write(f"{f.name}: zone {rep['zone']}, score {rep['decision'].get('P', 0):.2f}, action {rep['action']}")
            if scenario.startswith("labels"):
                queue.append((rep["batch_id"], labels))
        if scenario.startswith("labels") and len(queue) > 2:
            batch_id, labels_path = queue.pop(0)
            late = orch.attach_labels(project_id, batch_id, labels_path)
            log.write(f"labels arrived for an earlier batch: zone {late['zone']}")
        hist = history_frame(project_id)
        if len(hist):
            chart_slot.plotly_chart(timeline(hist, cfg), use_container_width=True, key=f"tl_{i}")
        progress.progress((i + 1) / len(files))
    st.success("Demo finished. Open the Report page for the details of each check.")


def evidence_cards(project_id: str, rep: Dict[str, Any]) -> None:
    drift, cara, quality = rep["drift"], rep["cara"], rep["quality"]
    t1, t2, t3 = st.tabs(["Drift (Module 2)", "CARA-T staleness (Module 3)", "Quality (Module 4)"])
    with t1:
        st.write(f"Drift share: **{drift.get('drift_share', 0):.2f}** ({drift.get('status')}). "
                 "Drift is a warning only; it cannot trigger retraining by itself.")
        if drift.get("per_feature"):
            st.plotly_chart(drift_plots.drift_bar(drift), use_container_width=True)
            st.dataframe(pd.DataFrame(drift["per_feature"]), use_container_width=True, hide_index=True)
            batch = store.get_batch(rep["batch_id"])
            ref_path = WORKSPACE / project_id / "data" / "train_sample.csv"
            batch_path = orch._resolve_path(batch["file"]) if batch else None
            if batch_path and batch_path.is_file() and ref_path.is_file():
                ref_df, new_df = pd.read_csv(ref_path), pd.read_csv(batch_path)
                cols = st.columns(3)
                for col, feat in zip(cols, drift_plots.top_drifted(drift, 3)):
                    col.plotly_chart(drift_plots.distribution_overlay(ref_df, new_df, feat["feature"], feat["type"]),
                                     use_container_width=True)
    with t2:
        basis = rep["decision"].get("basis", cara.get("basis"))
        m1, m2, m3, m4 = st.columns(4)
        if basis == "estimated":
            m1.metric("Estimated staleness", f"{cara.get('staleness_est', 0):.2f}")
            m2.metric("Lower bound", f"{cara.get('staleness_est_lo', 0):.2f}")
            st.info("Labels are pending: staleness is ESTIMATED from model confidence (CARA-T-L). "
                    "An estimate can never trigger automatic retraining; upload labels to measure it.")
        elif basis == "measured":
            m1.metric("Measured staleness", f"{cara.get('staleness', 0):.2f}")
            m2.metric("Lower bound (5%)", f"{cara.get('staleness_lo', cara.get('staleness', 0)):.2f}")
        else:
            st.info("Staleness is not available for this check (awaiting labels).")
        m3.metric("Threshold tau*", f"{cara.get('tau_star', 0):.2f}")
        m4.metric("Retraining cost kappa", f"{cara.get('kappa', 0):.2f}")
        if rep["decision"].get("proxy_error") is not None:
            before = rep["decision"].get("estimated_before_labels", {})
            st.write(f"Before labels arrived the estimate was {before.get('staleness_est', float('nan')):.2f}; "
                     f"estimate minus measured = {rep['decision']['proxy_error']:+.2f}.")
    with t3:
        if quality.get("table"):
            st.write(f"z versus baseline: **{quality.get('z_m', 0):.2f}** "
                     f"({'degraded' if quality.get('degraded') else 'not degraded'}; n = {quality.get('n')})")
            st.dataframe(pd.DataFrame(quality["table"]).round(4), use_container_width=True, hide_index=True)
            if quality.get("confusion_matrix"):
                st.write("Confusion matrix (rows = actual, columns = predicted)")
                st.dataframe(pd.DataFrame(quality["confusion_matrix"], index=quality.get("labels"),
                                          columns=quality.get("labels")))
        else:
            st.info(quality.get("note") or "Quality is pending: no labels for this batch yet.")


def page_report(project_id: str) -> None:
    st.header("Report")
    reports = load_reports(project_id)
    if not reports:
        st.info("No checks yet.")
        return
    cfg = load_config(project_id)
    ids = [r["report_id"] for r in reports]
    rid = st.selectbox("Check", ids, index=len(ids) - 1)
    rep = next(r for r in reports if r["report_id"] == rid)
    acts = decisions_by_report(project_id).get(rid, [])
    rep["action"] = acts[-1]["action"] if acts else None
    zone, dec = rep["zone"], rep["decision"]
    left, right = st.columns([1, 2])
    with left:
        st.markdown(f"<div style='padding:14px;border-radius:8px;background:{ZONE_COLOR.get(zone, '#555')};"
                    f"color:white;font-size:22px;text-align:center'><b>{ZONE_TEXT.get(zone, zone)}</b></div>",
                    unsafe_allow_html=True)
        if "P" in dec:
            st.plotly_chart(gauge(dec["P"], cfg), use_container_width=True)
        if dec.get("alert"):
            st.warning(f"Alert: {dec['alert']}")
    with right:
        st.subheader("Summary")
        key = f"ai_{rid}"
        st.text(st.session_state.get(key, {}).get("text") or template_summary(rep))
        if st.button("Write AI summary (numbers are checked against the report)"):
            st.session_state[key] = summarize_with_meta(rep)
            st.rerun()
        if key in st.session_state:
            st.caption(f"Source: {st.session_state[key]['source']}")
        if acts:
            st.caption("Recorded: " + " | ".join(f"{a['action']} by {a['by']}" for a in acts))
        if dec.get("retrain", {}).get("verify"):
            v = dec["retrain"]["verify"]
            st.write(f"Candidate {dec['retrain'].get('version')}: accuracy gain on the newest slice "
                     f"{v.get('gain', 0):+.3f} (lower bound {v.get('gain_lb', 0):+.3f}); holdout gate "
                     f"{v.get('holdout_gate', 'applied')}; result **{dec['retrain']['status']}**.")
            for reason in v.get("reasons", []):
                st.caption(f"- {reason}")
    if zone == "BLOCKED":
        st.error("; ".join(rep["drift"].get("issues", [])))
        return
    evidence_cards(project_id, rep)

    st.divider()
    w1, w2 = st.columns(2)
    with w1:
        st.subheader("What would change this decision?")
        mult = st.slider("Retraining-cost (kappa) multiplier", 0.25, 4.0, 1.0, 0.25)
        limit = st.slider("Automatic-retrain limit", 0.55, 0.90, float(cfg["retrain_limit"]), 0.05)
        res = default_tools(project_id)["what_if"](
            {"kappa_multiplier": mult, "retrain_limit": limit, "keep_limit": round(1 - limit, 2)}, rid)
        if res:
            st.write(f"Score **{res['P']:.2f}** -> zone **{res['zone']}** (stored: {zone}).")
    with w2:
        st.subheader("Your decision")
        b1, b2, b3 = st.columns(3)
        force = st.checkbox("Promote even if verification fails (needs a reason you can defend)")
        if b1.button("Retrain"):
            with st.spinner("Training a candidate and verifying it..."):
                out = orch.resolve_human(project_id, rid, "RETRAIN", force=force)
            st.write(f"Result: **{out['action']}**")
        if b2.button("Keep"):
            orch.resolve_human(project_id, rid, "KEEP")
            st.write("Recorded: KEEP for this batch.")
        if b3.button("Rollback"):
            try:
                out = orch.rollback(project_id)
                st.write(f"Rolled back. Active version: {out.get('active_version')}")
            except Exception as exc:
                st.error(str(exc))
    st.subheader("Model versions")
    st.dataframe(pd.DataFrame(store.get_versions(project_id))[["version", "status", "trained_on", "created_at"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Download this report (Markdown)", report_markdown(project_id, rep),
                       file_name=f"{project_id}_check_{rid}.md")


def report_markdown(project_id: str, rep: Dict[str, Any]) -> str:
    lines = [f"# ReTrainIQ check {rep['report_id']} ({project_id})", "", f"Zone: **{rep['zone']}**", "",
             "## Summary", "", template_summary(rep), ""]
    if rep["drift"].get("per_feature"):
        lines += ["## Drift", "", pd.DataFrame(rep["drift"]["per_feature"]).round(4).to_csv(index=False, sep="|"), ""]
    if rep["quality"].get("table"):
        lines += ["## Quality versus baseline", "", pd.DataFrame(rep["quality"]["table"]).round(4).to_csv(index=False, sep="|"), ""]
    keys = ("staleness", "staleness_lo", "staleness_est", "staleness_est_lo", "tau_star", "kappa", "s", "basis", "decision")
    lines += ["## CARA-T", "", json.dumps({k: rep["cara"].get(k) for k in keys}, indent=2), "",
              "## Decision score", "", json.dumps({k: v for k, v in rep["decision"].items() if k != "retrain"}, indent=2, default=str)]
    return "\n".join(lines)


def page_chat(project_id: str) -> None:
    st.header("Chat")
    st.caption("Answers come only from the stored reports. Try: 'why this decision?', 'which features drifted?', "
               "'show the history', 'what if kappa is 2?', 'which versions exist?'")
    log = st.session_state.setdefault(f"chat_{project_id}", [])
    for role, text in log:
        st.chat_message(role).write(text)
    question = st.chat_input("Ask about this project")
    if question:
        st.chat_message("user").write(question)
        res = chat_fn(question, project_id)
        st.chat_message("assistant").write(res["answer"])
        st.caption(f"mode: {res['mode']}; sources: {', '.join(res.get('sources', []))}")
        log.extend([("user", question), ("assistant", res["answer"])])


def main() -> None:
    st.set_page_config(page_title="ReTrainIQ", layout="wide")
    store.init_db()
    st.sidebar.title("ReTrainIQ")
    st.sidebar.caption("AI-based retraining advisor")
    projects = list_projects()
    project_id = st.sidebar.selectbox("Project", projects) if projects else None
    page = st.sidebar.radio("Page", ["Setup", "Monitor", "Report", "Chat"])
    st.sidebar.caption("Python computes every number and the decision score. The LLM only drafts the "
                       "profile, proposes metrics and explains. Replays are simulations.")
    if page == "Setup" or project_id is None:
        page_setup(project_id)
    elif not is_set_up(project_id):
        st.warning("This project is not set up yet. Open the Setup page.")
    elif page == "Monitor":
        page_monitor(project_id)
    elif page == "Report":
        page_report(project_id)
    else:
        page_chat(project_id)


if __name__ == "__main__":
    main()
