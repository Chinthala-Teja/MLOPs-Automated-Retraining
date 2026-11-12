"""ReTrainIQ dashboard (Streamlit). Run:  streamlit run frontend/app.py

One product, seven sections: Overview, Project Setup, Monitoring, Analysis & Reports,
Retraining History, AI Assistant, Documentation. The app calls the orchestrator and the
module functions directly (there is no HTTP layer); every number shown is computed by
the backend, never by the LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
for _folder in ("0-Init", "1-Profile", "2-Drift", "3-CARA", "4-Metrics", "5-Analysis", "6-Decision", "frontend"):
    if str(ROOT_DIR / _folder) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR / _folder))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import demo as demo_mod  # noqa: E402
import ri_decision.orchestrator as orch  # noqa: E402
import ri_init.store as store  # noqa: E402
import ri_profile.understand as understand  # noqa: E402
import ui  # noqa: E402
from ri_analysis.chat import chat as chat_fn  # noqa: E402
from ri_analysis.chat import default_tools  # noqa: E402
from ri_analysis.llm import get_client  # noqa: E402
from ri_analysis.summary import report_from_row, summarize_with_meta, template_summary  # noqa: E402
from ri_drift import plots as drift_plots  # noqa: E402
from ri_init.config import WORKSPACE, load_config  # noqa: E402
from ri_metrics.metrics import REGISTRY, default_plan  # noqa: E402

PAGES = ["Overview", "Project Setup", "Monitoring", "Analysis & Reports", "Retraining History",
         "AI Assistant", "Documentation"]
DOCS = {"User manual": "USER_MANUAL.md", "Project explanation": "PROJECT_EXPLANATION.md",
        "System architecture": "SYSTEM_ARCHITECTURE.md", "Data flow": "DATA_FLOW.md",
        "API reference": "API_REFERENCE.md", "Tech stack": "TECH_STACK.md"}
MANUALS = ["ReTrainIQ_Quick_Start.png", "ReTrainIQ_Workflow.png", "ReTrainIQ_Module_Guide.png",
           "ReTrainIQ_Retraining_Lifecycle.png", "ReTrainIQ_System_Architecture.png",
           "ReTrainIQ_API_Data_Flow.png"]


# ----------------------------------------------------------------------------- data access
def list_projects() -> List[str]:
    if not WORKSPACE.is_dir():
        return []
    return sorted(p.name for p in WORKSPACE.iterdir() if (p / "profile.json").is_file())


def load_profile(project_id: str) -> Dict[str, Any]:
    return json.loads((WORKSPACE / project_id / "profile.json").read_text(encoding="utf-8"))


def is_set_up(project_id: Optional[str]) -> bool:
    return bool(project_id) and (WORKSPACE / project_id / "state.json").is_file() \
        and store.get_active_version(project_id) is not None


def decisions_by_report(project_id: str) -> Dict[int, List[Dict[str, Any]]]:
    out: Dict[int, List[Dict[str, Any]]] = {}
    for d in store.get_decisions(project_id):
        out.setdefault(d["report_id"], []).append(d)
    return out


def load_reports(project_id: str) -> List[Dict[str, Any]]:
    profile, actions = load_profile(project_id), decisions_by_report(project_id)
    reports = []
    for row in store.get_reports(project_id):
        rep = report_from_row(row, profile)
        rep["decisions"] = actions.get(rep["report_id"], [])
        rep["action"] = rep["decisions"][-1]["action"] if rep["decisions"] else None
        reports.append(rep)
    return reports


def history_frame(reports: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rep in reports:
        if rep["zone"] == "BLOCKED":
            continue
        primary = rep["quality"].get("primary_metric")
        acts = [d["action"] for d in rep["decisions"]]
        basis = rep["decision"].get("basis")
        rows.append({
            "check": rep["report_id"], "zone": rep["zone"], "P": rep["decision"].get("P"),
            "staleness": rep["cara"].get("staleness") if basis in ("measured", "estimated") else None,
            "tau_star": rep["cara"].get("tau_star"), "basis": basis,
            "drift_share": rep["drift"].get("drift_share"),
            "primary_metric": rep["quality"].get("metrics", {}).get(primary or "", {}).get("value"),
            "event": next((a for a in acts if a in ("PROMOTED", "VERIFY_FAILED", "ROLLED_BACK")), ""),
        })
    return pd.DataFrame(rows, columns=["check", "zone", "P", "staleness", "tau_star", "basis", "drift_share",
                                       "primary_metric", "event"])


def save_upload(uploaded: Any, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(uploaded.name).name
    path.write_bytes(uploaded.getbuffer())
    return path


def fmt(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None or (isinstance(value, float) and pd.isna(value)) else f"{float(value):.{digits}f}"


# ----------------------------------------------------------------------------- Overview
def page_overview(project_id: Optional[str]) -> None:
    ui.hero("Welcome to ReTrainIQ — Intelligent ML Retraining Advisor",
            "Register a deployed scikit-learn model, send it new data, and get an evidence-backed answer to one "
            "question: keep the model, ask a human, or retrain it.")
    ui.flow_strip()
    st.write("")
    if not is_set_up(project_id):
        st.info("No project is registered yet. Open **Project Setup** to register your model, or load the demo project there.")
        return
    profile, cfg = load_profile(project_id), load_config(project_id)
    reports = load_reports(project_id)
    hist = history_frame(reports)
    active = store.get_active_version(project_id)
    last = reports[-1] if reports else None
    pending = sum(1 for r in reports if r["zone"] == "HUMAN" and r["action"] in (None, "pending"))
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        ui.card("Active model", active["version"], profile["model"].get("algorithm", ""), "info")
    with c2:
        ui.card("Checks run", str(len(reports)), f"batch size {profile.get('batch_size_B')}")
    with c3:
        zone = last["zone"] if last else None
        ui.card("Latest decision", ui.ZONE_LABEL.get(zone, "none yet"), f"check {last['report_id']}" if last else "",
                ui.ZONE_TONE.get(zone or "", "idle"))
    with c4:
        ui.card("Latest score", fmt(last["decision"].get("P") * 100, 1) if last and "P" in last["decision"] else "n/a",
                f"retrain at {cfg['retrain_limit'] * 100:.0f}, keep at {cfg['keep_limit'] * 100:.0f}")
    with c5:
        ui.card("Waiting for you", str(pending), "checks in human review", "warn" if pending else "good")
    st.write("")
    st.markdown(f"**{project_id}** — {profile.get('problem_statement', '')}")
    if len(hist):
        st.plotly_chart(ui.timeline(hist, cfg), use_container_width=True)
        st.caption("Dot colour = zone (green keep, amber human review, red retrain). Vertical lines mark "
                   "promotions, rejected candidates and rollbacks.")
    else:
        st.info("No checks yet. Go to **Monitoring** to upload a batch or run the guided demo.")


# ----------------------------------------------------------------------------- Project Setup (Module 1)
def page_setup(project_id: Optional[str]) -> None:
    ui.page_title("Project Setup", "Module 1 — understand the ML project. Done once per project.")
    tab_new, tab_demo, tab_status = st.tabs(["Register a project", "Demo project", "Current registration"])
    with tab_demo:
        st.write("A synthetic flight-delay project: 50,000 reference rows, a Random Forest pipeline, a 6,000-row "
                 "holdout and 50 stream batches with a simulated concept shift after batch 25.")
        ui.simulated_banner("The demo project is synthetic. Results on it say nothing about real data.")
        if st.button("Load / reset the demo project", type="primary"):
            with st.status("Preparing the demo project...", expanded=True) as status:
                if not (WORKSPACE / "demo" / "profile.json").is_file():
                    st.write("Generating data and training model v1...")
                    from ri_init import make_demo_project
                    make_demo_project.main()
                st.write("Registering v1, computing the holdout baseline, calibrating CARA-T...")
                res = orch.reset_project("demo")
                status.update(label=f"Demo project ready ({res['status']})", state="complete")
            for issue in res.get("issues", []):
                st.warning(issue)
    with tab_status:
        if is_set_up(project_id):
            profile, active = load_profile(project_id), store.get_active_version(project_id)
            st.markdown(f"**{project_id}** {ui.pill('REGISTERED', 'good')}", unsafe_allow_html=True)
            st.json({k: profile.get(k) for k in ("problem_statement", "task", "features", "model", "data",
                                                 "primary_metric", "batch_size_B", "offline_batches_N", "W")})
            st.caption(f"Active version: {active['version']}. LLM: "
                       + ("configured" if get_client() else "not configured — heuristic profile and template summaries are used."))
        else:
            st.info("Nothing registered for the selected project.")
    with tab_new:
        register_form()
        if "scan_log" in st.session_state:
            module1_progress(st.session_state["scan_log"])
        if "draft" in st.session_state:
            profile_editor()


def register_form() -> None:
    st.markdown("#### 1. Point ReTrainIQ at your project")
    st.caption("Repository code is read as text only and is never executed. Version 1 supports a scikit-learn "
               "Pipeline that accepts raw rows, and tabular data.")
    c1, c2 = st.columns(2)
    new_id = c1.text_input("Project id", value="myproject", help="Letters, digits, dashes, underscores.")
    url = c1.text_input("GitHub repository URL", placeholder="https://github.com/owner/repo")
    zip_file = c1.file_uploader("...or upload the project ZIP", type=["zip"])
    ref_file = c2.file_uploader("Training / reference dataset (CSV or Parquet)", type=["csv", "parquet"])
    model_file = c2.file_uploader("Trained model artifact (.joblib / .pkl) — needed if not in the repository",
                                  type=["joblib", "pkl"])
    hold_file = c2.file_uploader("Holdout the model never saw (recommended)", type=["csv", "parquet"])
    target = c2.text_input("Target column", placeholder="leave blank to use the last column")
    if st.button("Analyse Project", type="primary"):
        analyse_project(new_id.strip(), url.strip(), zip_file, model_file, ref_file, hold_file, target.strip())


def analyse_project(new_id: str, url: str, zip_file: Any, model_file: Any, ref_file: Any, hold_file: Any,
                    target: str) -> None:
    if not new_id.replace("-", "").replace("_", "").isalnum():
        st.error("Project id may contain only letters, digits, dashes and underscores.")
        return
    if ref_file is None:
        st.error("The training / reference dataset is required.")
        return
    proj = WORKSPACE / new_id
    log: List[Dict[str, str]] = []
    scan: Optional[Dict[str, Any]] = None
    with st.status("Module 1 is reading the project...", expanded=True) as status:
        try:
            if url:
                st.write("Cloning the repository (depth 1)...")
                scan = understand.scan_project(understand.fetch_repo(url, proj / "repo"))
            elif zip_file is not None:
                st.write("Extracting the ZIP safely...")
                understand.extract_zip_safe(save_upload(zip_file, proj / "upload"), proj / "repo")
                scan = understand.scan_project(proj / "repo")
        except Exception as exc:  # nothing was executed; show the reason
            status.update(label="Could not read the project", state="error")
            st.error(str(exc))
            return
        source = "GitHub repository" if url else "ZIP archive" if zip_file is not None else "no repository (dataset only)"
        log.append({"title": "Repository / ZIP scanning", "detail": source, "state": "done" if scan else "skip"})
        if scan:
            log.append({"title": "File structure analysis", "detail": f"{len(scan['file_tree'])} entries listed (depth 3)", "state": "done"})
            log.append({"title": "README and dependency inspection",
                        "detail": f"README {'found' if scan['readme'] else 'not found'}; requirements "
                                  f"{'found' if scan['requirements'] else 'not found'}", "state": "done"})
            log.append({"title": "Model artifact discovery",
                        "detail": ", ".join(Path(m).name for m in scan["model_files"][:4]) or "none in the repository",
                        "state": "done" if scan["model_files"] else "skip"})
            log.append({"title": "Dataset identification",
                        "detail": f"{len(scan['data_files'])} data file(s) in the repository", "state": "done"})
        ref_path = save_upload(ref_file, proj / "data")
        hold_path = save_upload(hold_file, proj / "data") if hold_file is not None else None
        model_path: Optional[Path] = None
        if model_file is not None:
            model_path = save_upload(model_file, proj / "models" / "v1")
            if model_path.name != "model.joblib":
                model_path = model_path.replace(model_path.with_name("model.joblib"))
        elif scan and scan["model_files"]:
            model_path = Path(scan["model_files"][0])
            if not model_path.is_absolute():
                model_path = Path(scan["root"]) / model_path
        if model_path is None:
            status.update(label="No model artifact", state="error")
            st.error("No model file was uploaded and none was found in the repository.")
            return
        st.write("Drafting the profile...")
        profile = understand.heuristic_profile(scan=scan, dataset_path=str(ref_path), target=target or None)
        drafted_by = "heuristics (column types, last column as target)"
        llm = get_client()
        if llm is not None and scan is not None:
            try:
                extracted = understand.extract_profile(understand.build_context(scan), llm)
                for key in ("problem_statement", "task", "features", "primary_metric"):
                    if extracted.get(key) not in (None, "unknown", {}, []):
                        profile[key] = extracted[key]
                drafted_by = "LLM draft, checked by code"
            except Exception as exc:
                st.warning(f"LLM profile extraction failed ({exc}); the heuristic profile is used.")
        profile["project_id"] = new_id
        profile.setdefault("model", {})["artifact"] = str(model_path)
        profile.setdefault("data", {})["reference_path"] = str(ref_path)
        profile["data"]["holdout_path"] = str(hold_path) if hold_path else None
        feats = profile.get("features", {})
        log.append({"title": "Feature and target identification",
                    "detail": f"target '{profile.get('task', {}).get('target')}', {len(feats.get('numeric', []))} numeric, "
                              f"{len(feats.get('categorical', []))} categorical", "state": "done"})
        log.append({"title": "ML task identification",
                    "detail": f"{profile.get('task', {}).get('type')} / {profile.get('task', {}).get('subtype')}", "state": "done"})
        log.append({"title": "Model framework and algorithm",
                    "detail": f"{profile['model'].get('framework')} / {profile['model'].get('algorithm')}", "state": "done"})
        log.append({"title": "Profile generation", "detail": drafted_by, "state": "done"})
        st.write("Verifying the profile (columns, class balance, model smoke test in the sandbox)...")
        issues = understand.verify_profile(profile)
        log.append({"title": "Profile validation",
                    "detail": "; ".join(issues) if issues else "no issues found",
                    "state": "fail" if any("error" in i.lower() for i in issues) else "done"})
        status.update(label="Module 1 finished — review the profile below", state="complete")
    st.session_state["scan_log"] = log
    st.session_state["draft"] = profile


def module1_progress(log: List[Dict[str, str]]) -> None:
    st.markdown("#### 2. What Module 1 found")
    ui.steps(log)


def profile_editor() -> None:
    draft = st.session_state["draft"]
    st.markdown("#### 3. Review and correct the profile")
    st.caption("Nothing is registered until you confirm. Correct anything Module 1 got wrong.")
    ref = understand._resolve_path(draft["data"]["reference_path"])
    columns = list((pd.read_parquet(ref) if str(ref).endswith(".parquet") else pd.read_csv(ref, nrows=5)).columns)
    task, feats = draft.get("task", {}), draft.get("features", {})
    with st.form("profile_form"):
        problem = st.text_area("Problem statement", draft.get("problem_statement", ""), height=70)
        c1, c2, c3 = st.columns(3)
        ttype = c1.selectbox("Task", ["classification", "regression"],
                             index=0 if task.get("type", "classification") == "classification" else 1)
        subtype = c2.selectbox("Classification kind", ["binary", "multiclass"],
                               index=0 if task.get("subtype", "binary") != "multiclass" else 1)
        target = c3.selectbox("Target column", columns,
                              index=columns.index(task["target"]) if task.get("target") in columns else len(columns) - 1)
        others = [c for c in columns if c != target]
        numeric = st.multiselect("Numeric features", others, [c for c in feats.get("numeric", []) if c in others])
        categorical = st.multiselect("Categorical features", others, [c for c in feats.get("categorical", []) if c in others])
        c4, c5, c6 = st.columns(3)
        algorithm = c4.text_input("Model algorithm", draft.get("model", {}).get("algorithm", ""))
        trial = {"task": {"type": ttype, "subtype": subtype if ttype == "classification" else None}}
        options = default_plan({**trial, "primary_metric": draft.get("primary_metric")})
        primary = c5.selectbox("Primary evaluation metric", options,
                               index=options.index(draft.get("primary_metric")) if draft.get("primary_metric") in options else 0)
        batch = c6.number_input("Batch size B (rows per check)", 200, 100000, int(draft.get("batch_size_B", 2000)), 100)
        st.caption(f"Model artifact: `{draft['model']['artifact']}`  |  Reference dataset: `{draft['data']['reference_path']}`  |  "
                   f"Holdout: `{draft['data'].get('holdout_path') or 'none — quality comparison will be switched off'}`")
        confirmed = st.form_submit_button("Confirm Profile", type="primary")
    if not confirmed:
        return
    draft.update({"problem_statement": problem, "primary_metric": primary, "batch_size_B": int(batch)})
    draft["task"] = {**task, "type": ttype, "subtype": subtype if ttype == "classification" else None, "target": target}
    draft["features"] = {"numeric": numeric, "categorical": categorical}
    draft["model"]["algorithm"] = algorithm
    with st.status("Registering the project...", expanded=True) as status:
        st.write("Verifying the confirmed profile...")
        final = understand.finalize_profile(draft)
        for issue in final.get("issues", []):
            st.warning(issue)
        st.write("Loading the model in the sandbox, computing the holdout baseline, calibrating CARA-T...")
        try:
            res = orch.setup_project({k: v for k, v in final.items() if k != "issues"})
        except Exception as exc:
            status.update(label="Registration failed", state="error")
            st.error(str(exc))
            return
        status.update(label=f"Project '{final['project_id']}' registered", state="complete")
    for issue in res.get("issues", []):
        st.warning(issue)
    st.success("Registered. Select the project in the sidebar and open Monitoring.")
    st.session_state.pop("draft", None)
    st.session_state.pop("scan_log", None)


# ----------------------------------------------------------------------------- Monitoring
def show_results(result: Dict[str, Any]) -> None:
    if result.get("status") == "BLOCKED":
        st.error("INVALID DATA — upload blocked: " + "; ".join(str(i) for i in result.get("issues", [])))
    elif result.get("status") == "DUPLICATE":
        st.warning("This exact file was already processed; nothing was done.")
    else:
        st.success(f"{result.get('batches_processed', 0)} batch(es) checked. {result.get('buffer_rows', 0)} row(s) "
                   "are waiting until a full batch has accumulated.")
        for rep in result.get("reports", []):
            st.markdown(f"Check **{rep['report_id']}** &nbsp; {ui.zone_pill(rep['zone'])} &nbsp; score "
                        f"{fmt(rep['decision'].get('P', 0) * 100, 1)} &nbsp; action: {rep['action']}", unsafe_allow_html=True)
        if result.get("reports"):
            st.caption("Open Analysis & Reports for the module-by-module evidence.")


def page_monitoring(project_id: str) -> None:
    ui.page_title("Monitoring", "Send new data. Each full batch runs Module 2, then Modules 3 and 4, then the decision.")
    cfg, profile = load_config(project_id), load_profile(project_id)
    tab_upload, tab_demo, tab_timeline = st.tabs(["Upload data", "Guided demo", "Timeline"])
    with tab_upload:
        up, late = st.columns(2)
        with up:
            st.markdown("#### New incoming data")
            data_file = st.file_uploader("Data file: feature columns, plus the target column if known",
                                         type=["csv", "parquet"], key="new")
            labels_file = st.file_uploader("Labels for this file (optional, same row order or id column)",
                                           type=["csv", "parquet"], key="newlab")
            if st.button("Run check", type="primary", disabled=data_file is None):
                inbox = WORKSPACE / project_id / "data" / "uploads"
                path = save_upload(data_file, inbox)
                lab = save_upload(labels_file, inbox) if labels_file is not None else None
                with st.status("Checking the batch...", expanded=True) as status:
                    st.write("Schema validation, drift (Module 2), staleness (Module 3), quality (Module 4), decision...")
                    result = orch.on_new_data(project_id, path, lab)
                    status.update(label=f"Finished: {result.get('status')}", state="complete")
                show_results(result)
        with late:
            st.markdown("#### Labels that arrived later")
            waiting = [r for r in load_reports(project_id)
                       if r["decision"].get("basis") in ("estimated", "none")
                       and "proxy_error" not in r["decision"] and r["zone"] != "BLOCKED"]
            if not waiting:
                st.caption("No check is waiting for labels.")
            else:
                pick = st.selectbox("Check waiting for labels", [r["report_id"] for r in waiting])
                late_file = st.file_uploader("Labels file", type=["csv", "parquet"], key="late")
                if st.button("Attach labels", disabled=late_file is None):
                    rep = next(r for r in waiting if r["report_id"] == pick)
                    path = save_upload(late_file, WORKSPACE / project_id / "data" / "uploads")
                    with st.spinner("Measuring staleness and quality with the labels..."):
                        res = orch.attach_labels(project_id, rep["batch_id"], path)
                    err = res.get("proxy_error")
                    st.success(f"Labels attached. Zone is now {ui.ZONE_LABEL.get(res['zone'], res['zone'])}."
                               + (f" Estimate minus measured staleness: {err:+.3f}." if err is not None else ""))
    with tab_demo:
        guided_demo(project_id, profile)
    with tab_timeline:
        hist = history_frame(load_reports(project_id))
        if len(hist):
            st.plotly_chart(ui.timeline(hist, cfg), use_container_width=True)
            st.dataframe(hist, use_container_width=True, hide_index=True)
        else:
            st.info("No checks yet.")


def guided_demo(project_id: str, profile: Dict[str, Any]) -> None:
    if not demo_mod.stream_files(project_id):
        st.info("The guided demo needs a `data/stream` folder. Select the demo project.")
        return
    ui.simulated_banner()
    st.write("")
    key = f"demo_{project_id}"
    state = st.session_state.setdefault(key, {"next": 0, "results": {}})
    c1, c2, c3 = st.columns([2, 2, 5])
    done = state["next"] >= len(demo_mod.DEMO_STEPS)
    if c1.button("Run next step", type="primary", disabled=done):
        index = state["next"]
        with st.status(demo_mod.DEMO_STEPS[index]["title"], expanded=True) as status:
            st.write(demo_mod.DEMO_STEPS[index]["what"])
            state["results"][index] = demo_mod.run_demo_step(project_id, index, profile)
            status.update(state="complete")
        state["next"] = index + 1
    if c2.button("Restart demo"):
        st.session_state[key] = {"next": 0, "results": {}}
        st.rerun()
    c3.progress(min(state["next"], len(demo_mod.DEMO_STEPS)) / len(demo_mod.DEMO_STEPS),
                text=f"{state['next']} of {len(demo_mod.DEMO_STEPS)} steps done")
    for index, step in enumerate(demo_mod.DEMO_STEPS):
        ran = index in state["results"]
        with st.expander(("✓ " if ran else "") + step["title"], expanded=ran or index == state["next"]):
            st.write(step["what"])
            if ran:
                frame = pd.DataFrame(state["results"][index])
                st.dataframe(frame, use_container_width=True, hide_index=True)
                promoted = [r for r in state["results"][index] if r.get("action") == "PROMOTED"]
                if promoted:
                    active = store.get_active_version(project_id)
                    st.success(f"Candidate verified and promoted. Active model version is now {active['version']}.")
    reports = load_reports(project_id)
    hist = history_frame(reports)
    if state["results"] and len(hist):
        st.plotly_chart(ui.timeline(hist, load_config(project_id)), use_container_width=True)
    if done:
        st.success("Demo finished. Open Analysis & Reports to walk through any check, or Retraining History for versions.")


# ----------------------------------------------------------------------------- Analysis & Reports
def module2_panel(project_id: str, rep: Dict[str, Any], hist: pd.DataFrame) -> None:
    drift = rep["drift"]
    status = "INVALID DATA" if rep["zone"] == "BLOCKED" else "DRIFT DETECTED" if drift.get("status") == "DRIFT" else "STABLE"
    tone = "idle" if status == "INVALID DATA" else "warn" if status == "DRIFT DETECTED" else "good"
    st.markdown(f"#### Module 2 — Data drift &nbsp; {ui.pill(status, tone)}", unsafe_allow_html=True)
    st.caption("Question: does the incoming data look different from the reference data? Drift is an early "
               "warning. It can never trigger retraining on its own.")
    if rep["zone"] == "BLOCKED":
        for issue in drift.get("issues", []):
            st.error(issue)
        st.write("Fix the file and upload it again. Nothing else was computed for this upload.")
        return
    feats = drift.get("per_feature", [])
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.card("Drift share d", fmt(drift.get("drift_share"), 2), "flagged features / all features", tone)
    with c2:
        ui.card("Features flagged", f"{sum(1 for f in feats if f.get('flag'))} / {len(feats)}",
                "PSI ≥ 0.2 (categorical); KS p < 0.01 and statistic ≥ 0.1 (numeric)")
    with c3:
        ui.card("Rows compared", f"{drift.get('n_new', 0):,}", f"against {drift.get('n_ref', 0):,} reference rows")
    st.write("")
    schema = drift.get("schema") or {}
    checks = [{"title": "Schema validation", "detail": "all feature columns present; numeric columns can be cast", "state": "done"},
              {"title": "Unseen categories",
               "detail": "; ".join(f"{k}: {', '.join(map(str, v[:5]))}" for k, v in schema.get("unseen_categories", {}).items()) or "none",
               "state": "fail" if schema.get("unseen_categories") else "done"},
              {"title": "Values outside the reference 1st-99th percentile",
               "detail": "; ".join(f"{k}: {v} rows" for k, v in schema.get("out_of_range", {}).items()) or "none reported",
               "state": "done"}]
    with st.expander("Validation checks on this upload"):
        ui.steps(checks)
    if not feats:
        return
    left, right = st.columns([3, 2])
    left.plotly_chart(drift_plots.drift_bar(drift), use_container_width=True)
    right.dataframe(pd.DataFrame(feats).round(4), use_container_width=True, hide_index=True)
    batch = store.get_batch(rep["batch_id"])
    ref_path = WORKSPACE / project_id / "data" / "train_sample.csv"
    batch_path = orch._resolve_path(batch["file"]) if batch else None
    if batch_path and batch_path.is_file() and ref_path.is_file():
        st.markdown("**Top three features: reference versus incoming**")
        ref_df, new_df = pd.read_csv(ref_path), pd.read_csv(batch_path)
        for col, feat in zip(st.columns(3), drift_plots.top_drifted(drift, 3)):
            col.plotly_chart(drift_plots.distribution_overlay(ref_df, new_df, feat["feature"], feat["type"]),
                             use_container_width=True)
        st.caption("The reference shown is the stored 2,000-row sample of the active model's training data.")
    if len(hist):
        st.bar_chart(hist.set_index("check")["drift_share"], height=180)
        st.caption("Drift share over all checks.")


def module3_panel(rep: Dict[str, Any], hist: pd.DataFrame) -> None:
    cara, basis = rep["cara"], rep["decision"].get("basis", rep["cara"].get("basis", "none"))
    status = {"measured": cara.get("decision", "KEEP"), "estimated": "ESTIMATED", "none": "PENDING"}.get(basis, "PENDING")
    tone = {"RETRAIN": "bad", "KEEP": "good", "ESTIMATED": "warn", "PENDING": "idle"}[status]
    st.markdown(f"#### Module 3 — CARA-T staleness &nbsp; {ui.pill('CARA-T: ' + status, tone)}", unsafe_allow_html=True)
    st.caption("Question: has the model gone stale for the kind of inputs it is being asked about now?")
    with st.expander("How CARA-T works, in four lines", expanded=False):
        st.markdown(
            "1. Take the rows of the new batch as **queries** and its labelled rows as **data points**.\n"
            "2. For every query, average the model's 0-1 loss over the data points, weighting each point by an "
            "**RBF similarity** `exp(-gamma * distance²)`: errors near the query count more.\n"
            "3. Do the same with the **same queries** against a sample of the model's own training data, and subtract. "
            "The difference is the **relative staleness**.\n"
            "4. Compare it with **tau\\***, the threshold that gave the lowest total cost (staleness + retraining "
            "cost kappa) when replayed on historical batches at setup.")
        st.latex(r"\overline{\Psi} = \Psi(Q_{new}, D_{new}, M) - \Psi(Q_{new}, D_{train}, M)\qquad "
                 r"\text{RETRAIN if } \overline{\Psi} \ge \tau^*")
    if basis == "none":
        st.info("PENDING — this batch has no (or too few) labels and no label-free estimate is available. "
                "No staleness value is shown because none was computed.")
    c = st.columns(5)
    if basis == "measured":
        with c[0]:
            ui.card("Staleness", fmt(cara.get("staleness"), 2), "measured with labels",
                    "bad" if cara.get("decision") == "RETRAIN" else "good")
        with c[1]:
            ui.card("Lower bound (5%)", fmt(cara.get("staleness_lo", cara.get("staleness")), 2), "used by the decision score")
    elif basis == "estimated":
        with c[0]:
            ui.card("Estimated staleness", fmt(cara.get("staleness_est"), 2), "from model confidence (CARA-T-L)", "warn")
        with c[1]:
            ui.card("Lower bound", fmt(cara.get("staleness_est_lo"), 2), "estimate minus offline error margin")
    else:
        with c[0]:
            ui.card("Staleness", "PENDING", "awaiting labels", "idle")
        with c[1]:
            ui.card("Lower bound", "n/a", "")
    with c[2]:
        ui.card("Threshold tau*", fmt(cara.get("tau_star"), 2), "calibrated offline")
    with c[3]:
        ui.card("Retraining cost kappa", fmt(cara.get("kappa"), 2), "same units as staleness")
    with c[4]:
        ui.card("Samples used", f"{cara.get('n_points', 0):,}", f"data points; {cara.get('n_queries', 0)} queries")
    if basis == "measured" and cara.get("psi_current") is not None:
        st.caption(f"Kernel-weighted loss on the new batch: {cara['psi_current']:.2f}; on the training sample with "
                   f"the same queries: {cara['psi_reference']:.2f}; difference = relative staleness.")
    if basis == "estimated":
        st.warning("Labels are pending. This value is an ESTIMATE and can only send the check to human review; "
                   "it never triggers automatic retraining. A pure concept shift is invisible to it. Upload labels to measure.")
    if rep["decision"].get("proxy_error") is not None:
        before = rep["decision"].get("estimated_before_labels", {})
        st.caption(f"Before labels arrived the estimate was {fmt(before.get('staleness_est'), 2)}; "
                   f"estimate minus measured = {rep['decision']['proxy_error']:+.2f}.")
    if len(hist):
        st.plotly_chart(ui.staleness_chart(hist, rep["report_id"]), use_container_width=True)


def module4_panel(rep: Dict[str, Any], hist: pd.DataFrame) -> None:
    quality = rep["quality"]
    state = quality.get("status", "PENDING")
    label = {"OK": "DEGRADED" if quality.get("degraded") else "NOT DEGRADED", "NO_BASELINE": "NO BASELINE"}.get(state, "PENDING")
    tone = {"DEGRADED": "bad", "NOT DEGRADED": "good", "NO BASELINE": "warn", "PENDING": "idle"}[label]
    st.markdown(f"#### Module 4 — Model quality &nbsp; {ui.pill(label, tone)}", unsafe_allow_html=True)
    st.caption("Question: are the model's predictions measurably worse than on the holdout it was accepted with?")
    if state not in ("OK", "NO_BASELINE"):
        st.info("PENDING — quality needs true labels. No metric was computed for this batch.")
        return
    primary = quality.get("primary_metric")
    row = next((r for r in quality.get("table", []) if r["metric"] == primary), None)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.card(f"Current {primary}", fmt(quality["metrics"].get(primary, {}).get("value")),
                f"± {fmt(quality['metrics'].get(primary, {}).get('se'))} bootstrap SE")
    with c2:
        ui.card("Baseline (holdout)", fmt(row["reference"]) if row else "n/a", "never-seen data")
    with c3:
        ui.card("Degradation z", fmt(quality.get("z_m"), 2), "≥ 1.64 counts as degraded", tone)
    with c4:
        ui.card("Labelled rows", f"{quality.get('n', 0):,}", f"coverage {fmt(quality.get('label_coverage'), 2)}")
    st.write("")
    if state == "NO_BASELINE":
        st.warning(quality.get("note", "No holdout baseline; the quality comparison is switched off."))
    st.caption("Metrics come from a fixed registry: " + ", ".join(
        f"{name} ({REGISTRY[name]['description']})" for name in list(quality.get("metrics", {}))[:4] if REGISTRY[name]["description"]) + ".")
    left, right = st.columns(2)
    if quality.get("table"):
        left.plotly_chart(ui.metric_bars(quality["table"]), use_container_width=True)
    if quality.get("confusion_matrix"):
        right.plotly_chart(ui.confusion_heatmap(quality["confusion_matrix"], quality.get("labels") or []), use_container_width=True)
    curves = quality.get("curves") or {}
    if curves.get("roc"):
        c5, c6 = st.columns(2)
        c5.plotly_chart(ui.curve_chart(curves["roc"], "ROC curve", "false positive rate", "true positive rate", True), use_container_width=True)
        c6.plotly_chart(ui.curve_chart(curves["pr"], "Precision-recall curve", "recall", "precision", False), use_container_width=True)
    if quality.get("per_class_recall"):
        st.markdown("**Per-class recall**")
        st.dataframe(pd.DataFrame([quality["per_class_recall"]]).round(4), use_container_width=True, hide_index=True)
    metric_rows = [{"metric": k, "value": v["value"], "bootstrap_se": v["se"]} for k, v in quality.get("metrics", {}).items()]
    st.dataframe(pd.DataFrame(metric_rows).round(4), use_container_width=True, hide_index=True)
    series = hist.dropna(subset=["primary_metric"]) if len(hist) else hist
    if len(series):
        st.line_chart(series.set_index("check")["primary_metric"], height=180)
        st.caption(f"{primary} over all labelled checks.")


def decision_panel(project_id: str, rep: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    dec, zone = rep["decision"], rep["zone"]
    st.markdown(f"#### Decision engine &nbsp; {ui.zone_pill(zone)}", unsafe_allow_html=True)
    meaning = {"RETRAIN": "Automatic retraining is permitted, subject to cooldown and verification.",
               "HUMAN": "The evidence is insufficient or incomplete. Your decision is required.",
               "KEEP": "Continue using the existing model."}.get(zone, "")
    st.caption(meaning)
    if "P" not in dec:
        st.info("No decision score for this upload.")
        return
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(ui.gauge(dec["P"], cfg), use_container_width=True)
        st.caption("A decision score, not a calibrated probability.")
    with right:
        st.latex(r"z_c=\frac{\text{staleness}-\tau^*}{s}\qquad z_m=\text{quality degradation}")
        st.latex(r"P=\sigma(b+a_1 z_c+a_2 z_m+a_3 d)")
        st.markdown(f"`b = {cfg['b']}`, `a1 = {cfg['a1']}`, `a2 = {cfg['a2']}`, `a3 = {cfg['a3']}`; "
                    f"retrain at `P ≥ {cfg['retrain_limit']}`, keep at `P ≤ {cfg['keep_limit']}`; "
                    f"z values are clipped to ±{cfg['z_clip']}.")
        c1, c2, c3 = st.columns(3)
        c1.metric("z_c (staleness)", fmt(dec.get("zc"), 2))
        c2.metric("z_m (quality)", fmt(dec.get("zm"), 2))
        c3.metric("d (drift share)", fmt(dec.get("d"), 2))
        if dec.get("terms"):
            st.plotly_chart(ui.contribution_bars(dec["terms"]), use_container_width=True)
    notes = [dec.get("reason", "")] + [f"flag: {f}" for f in dec.get("flags", [])]
    if dec.get("alert"):
        notes.append(f"alert: {dec['alert']}")
    st.markdown("**Evidence:** " + " · ".join(n for n in notes if n))
    st.caption("Rules that always hold: missing labels force HUMAN REVIEW; an estimate never triggers RETRAIN; "
               "drift alone cannot reach the retrain line; after a promotion there is a cooldown.")
    st.divider()
    st.markdown("#### What if?")
    w1, w2, w3 = st.columns(3)
    mult = w1.slider("Retraining-cost (kappa) multiplier", 0.25, 4.0, 1.0, 0.25,
                     help="Higher cost raises tau*, so more staleness is tolerated.")
    limit = w2.slider("Automatic-retrain limit", 0.55, 0.90, float(cfg["retrain_limit"]), 0.05)
    keep = w3.slider("Keep limit", 0.10, 0.45, float(cfg["keep_limit"]), 0.05)
    res = default_tools(project_id)["what_if"]({"kappa_multiplier": mult, "retrain_limit": limit, "keep_limit": keep},
                                               rep["report_id"])
    if res:
        st.markdown(f"With these settings: score **{res['P'] * 100:.1f}** → {ui.zone_pill(res['zone'])} "
                    f"(stored decision: {ui.zone_pill(zone)}). Nothing is saved; this only recomputes the formula.",
                    unsafe_allow_html=True)


def retrain_pipeline(rep: Dict[str, Any], profile: Dict[str, Any]) -> None:
    info = rep["decision"].get("retrain")
    if not info:
        st.info("No retraining was attempted for this check.")
        return
    verify = info.get("verify") or {}
    status = info.get("status")
    if status == "INSUFFICIENT_DATA":
        st.warning(f"Retraining was refused: {info.get('reason')}")
        return
    passed, waived = bool(verify.get("verified")), verify.get("holdout_gate") == "waived"
    promoted = status == "PROMOTED"
    window = profile.get("W", 1)
    items = [
        {"title": "Select recent labelled batches", "detail": f"the last W = {window} labelled batch(es)", "state": "done"},
        {"title": "Split verification data", "detail": "the newest 20% of the newest batch is held out", "state": "done"},
        {"title": "Clone the existing sklearn pipeline", "detail": "same steps and hyper-parameters, fresh weights", "state": "done"},
        {"title": "Train the candidate in the sandbox worker", "detail": f"{fmt(info.get('seconds'), 2)} s", "state": "done"},
        {"title": "Evaluate the candidate on the verification slice",
         "detail": f"accuracy {fmt(verify.get('candidate_v_accuracy'))}; {profile.get('primary_metric')} {fmt(verify.get('candidate_v'))}", "state": "done"},
        {"title": "Compare against the champion",
         "detail": f"champion accuracy {fmt(verify.get('champion_v_accuracy'))}; gain {fmt(verify.get('gain'))}, "
                   f"5% lower bound {fmt(verify.get('gain_lb'))}",
         "state": "done" if (verify.get("gain_lb") or 0) >= 0 else "fail"},
        {"title": "Evaluate against the reference holdout",
         "detail": "waived: the champion itself has degraded on current data, so the old holdout is out of date"
                   if waived else f"drop {fmt(verify.get('holdout_drop'))} (tolerance 0.02)",
         "state": "skip" if waived else "done" if (verify.get("holdout_drop") or 0) <= 0.02 else "fail"},
        {"title": "Approve or reject", "detail": "; ".join(verify.get("reasons", [])) or "all gates passed",
         "state": "done" if passed else "fail"},
        {"title": "Promote the candidate", "detail": f"version {info.get('version')}" if promoted else "not promoted; the champion stays",
         "state": "done" if promoted else "skip"},
        {"title": "Update the active model version", "detail": "atomic file replace; previous version kept for rollback" if promoted else "",
         "state": "done" if promoted else "skip"},
        {"title": "Recalibrate CARA-T", "detail": "; ".join(info.get("warnings") or []) or ("tau* and kappa recomputed for the new model" if promoted else ""),
         "state": "done" if promoted else "skip"},
    ]
    st.markdown(f"Result: {ui.pill(status.replace('_', ' '), 'good' if promoted else 'warn')}", unsafe_allow_html=True)
    ui.steps(items)


def page_analysis(project_id: str) -> None:
    ui.page_title("Analysis & Reports", "One check at a time, module by module, ending in the decision.")
    reports = load_reports(project_id)
    if not reports:
        st.info("No checks yet. Upload data or run the guided demo in Monitoring.")
        return
    cfg, profile, hist = load_config(project_id), load_profile(project_id), history_frame(reports)
    ids = [r["report_id"] for r in reports]
    top = st.columns([2, 5])
    rid = top[0].selectbox("Check", ids, index=len(ids) - 1,
                           format_func=lambda i: f"Check {i} — {next(r['zone'] for r in reports if r['report_id'] == i)}")
    rep = next(r for r in reports if r["report_id"] == rid)
    top[1].markdown(f"<div style='padding-top:30px'>{ui.zone_pill(rep['zone'])} &nbsp; "
                    + " ".join(ui.pill(f"{d['action']} · {d['by']}", "idle") for d in rep["decisions"]) + "</div>",
                    unsafe_allow_html=True)
    tabs = st.tabs(["Summary", "① Drift", "② Staleness", "③ Quality", "④ Decision", "⑤ Retraining"])
    with tabs[0]:
        key = f"ai_{rid}"
        st.text(st.session_state.get(key, {}).get("text") or template_summary(rep))
        c1, c2 = st.columns([2, 5])
        if c1.button("Write AI summary"):
            with st.spinner("Asking the LLM and checking its numbers against the report..."):
                st.session_state[key] = summarize_with_meta(rep)
            st.rerun()
        c2.caption(f"Source: {st.session_state[key]['source']}" if key in st.session_state else
                   "Template summary. The AI summary is used only if every number in it matches the report.")
        if rep["zone"] != "BLOCKED":
            st.markdown("#### Your decision")
            b1, b2, b3 = st.columns([1, 1, 4])
            force = b3.checkbox("Promote even if verification fails")
            if b1.button("Retrain"):
                with st.spinner("Training a candidate and verifying it..."):
                    out = orch.resolve_human(project_id, rid, "RETRAIN", force=force)
                st.write(f"Result: **{out['action']}**")
            if b2.button("Keep"):
                orch.resolve_human(project_id, rid, "KEEP")
                st.write("Recorded: KEEP for this batch.")
        st.download_button("Download this report (Markdown)", report_markdown(project_id, rep),
                           file_name=f"{project_id}_check_{rid}.md")
    with tabs[1]:
        module2_panel(project_id, rep, hist)
    if rep["zone"] == "BLOCKED":
        for tab in tabs[2:]:
            tab.info("Not computed: the upload was blocked at schema validation.")
        return
    with tabs[2]:
        module3_panel(rep, hist)
    with tabs[3]:
        module4_panel(rep, hist)
    with tabs[4]:
        decision_panel(project_id, rep, cfg)
    with tabs[5]:
        st.markdown("#### Retraining and verification")
        st.caption("A candidate replaces the current model only if it passes verification. A failed candidate is never promoted automatically.")
        retrain_pipeline(rep, profile)


def report_markdown(project_id: str, rep: Dict[str, Any]) -> str:
    lines = [f"# ReTrainIQ check {rep['report_id']} ({project_id})", "", f"Zone: **{rep['zone']}**", "",
             "## Summary", "", template_summary(rep), ""]
    if rep["drift"].get("per_feature"):
        lines += ["## Drift", "", pd.DataFrame(rep["drift"]["per_feature"]).round(4).to_csv(index=False, sep="|"), ""]
    if rep["quality"].get("table"):
        lines += ["## Quality versus baseline", "", pd.DataFrame(rep["quality"]["table"]).round(4).to_csv(index=False, sep="|"), ""]
    keys = ("staleness", "staleness_lo", "staleness_est", "staleness_est_lo", "tau_star", "kappa", "s", "basis", "decision")
    lines += ["## CARA-T", "", json.dumps({k: rep["cara"].get(k) for k in keys}, indent=2), "",
              "## Decision score", "", json.dumps(rep["decision"], indent=2, default=str)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- Retraining History
def page_history(project_id: str) -> None:
    ui.page_title("Retraining History", "Every candidate that was trained, what verification said, and the version in use.")
    versions = store.get_versions(project_id)
    active = store.get_active_version(project_id)
    attempts = []
    for rep in load_reports(project_id):
        info = rep["decision"].get("retrain")
        if info and info.get("version"):
            v = info.get("verify") or {}
            attempts.append({"check": rep["report_id"], "version": info["version"], "result": info["status"],
                             "trigger": info.get("trigger", "automatic"),
                             "training_seconds": info.get("seconds"), "candidate_accuracy": v.get("candidate_v_accuracy"),
                             "champion_accuracy": v.get("champion_v_accuracy"), "gain": v.get("gain"),
                             "gain_lower_bound": v.get("gain_lb"), "holdout_gate": v.get("holdout_gate"),
                             "reasons": "; ".join(v.get("reasons", []))})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ui.card("Active version", active["version"] if active else "n/a", "serving now", "info")
    with c2:
        ui.card("Versions registered", str(len(versions)), "including rejected candidates")
    with c3:
        ui.card("Promotions", str(sum(1 for a in attempts if a["result"] == "PROMOTED")), "passed verification", "good")
    with c4:
        ui.card("Rejected candidates", str(sum(1 for a in attempts if a["result"] == "VERIFY_FAILED")), "champion kept", "warn")
    st.write("")
    st.markdown("#### Model versions")
    frame = pd.DataFrame(versions)
    if len(frame):
        st.dataframe(frame[["version", "status", "trained_on", "created_at"]], use_container_width=True, hide_index=True)
    st.markdown("#### Retraining attempts")
    if attempts:
        st.dataframe(pd.DataFrame(attempts).round(4), use_container_width=True, hide_index=True)
        st.caption("Automatic, late-label and human-requested attempts. Open the check in Analysis & Reports → ⑤ Retraining for the full pipeline.")
    else:
        st.caption("No candidate has been trained yet.")
    st.markdown("#### Rollback")
    state_path = WORKSPACE / project_id / "state.json"
    previous = json.loads(state_path.read_text(encoding="utf-8")).get("previous_version") if state_path.is_file() else None
    if previous:
        st.write(f"Restores **{previous}** together with its calibration, baseline and drift reference.")
        if st.button(f"Roll back to {previous}"):
            try:
                out = orch.rollback(project_id)
                st.success(f"Rolled back. Active version: {out.get('active_version')}")
            except Exception as exc:
                st.error(str(exc))
    else:
        st.caption("No previous version to roll back to.")


# ----------------------------------------------------------------------------- AI Assistant
def page_assistant(project_id: str) -> None:
    ui.page_title("AI Assistant", "Ask about this project. Answers come only from the stored reports.")
    llm_on = get_client() is not None
    st.markdown(ui.pill("LLM CONNECTED", "good") if llm_on else ui.pill("NO LLM — RULE-BASED ANSWERS", "idle"),
                unsafe_allow_html=True)
    st.caption("Try: “why this decision?”, “which features drifted?”, “show the history”, “what if kappa is 2?”, "
               "“which versions exist?”. With an LLM, every number in its answer is checked against the reports; "
               "an answer that fails the check is replaced by the rule-based one.")
    log = st.session_state.setdefault(f"chat_{project_id}", [])
    for role, text in log:
        st.chat_message(role).write(text)
    question = st.chat_input("Ask about this project")
    if question:
        st.chat_message("user").write(question)
        res = chat_fn(question, project_id)
        st.chat_message("assistant").write(res["answer"])
        st.caption(f"mode: {res['mode']} · sources: {', '.join(res.get('sources', []))}")
        log.extend([("user", question), ("assistant", res["answer"])])


# ----------------------------------------------------------------------------- Documentation
def page_docs() -> None:
    ui.page_title("Documentation", "How the system works and how to use it.")
    docs_dir = ROOT_DIR / "docs"
    tab_guides, tab_docs = st.tabs(["Visual guides", "Written documentation"])
    with tab_guides:
        found = [m for m in MANUALS if (docs_dir / "manual" / m).is_file()]
        if not found:
            st.info("The PNG guides are expected in docs/manual/.")
        else:
            name = st.selectbox("Guide", found, format_func=lambda m: m.replace("ReTrainIQ_", "").replace(".png", "").replace("_", " "))
            st.image(str(docs_dir / "manual" / name), use_container_width=True)
    with tab_docs:
        available = {title: f for title, f in DOCS.items() if (docs_dir / f).is_file()}
        if not available:
            st.info("The Markdown documents are expected in docs/.")
        else:
            title = st.selectbox("Document", list(available))
            st.markdown((docs_dir / available[title]).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- shell
def main() -> None:
    st.set_page_config(page_title="ReTrainIQ", page_icon="🔁", layout="wide")
    ui.inject_theme()
    store.init_db()
    st.sidebar.markdown("<p class='rq-brand'>ReTrain<span>IQ</span></p><p class='rq-sub'>ML retraining advisor</p>",
                        unsafe_allow_html=True)
    projects = list_projects()
    project_id = st.sidebar.selectbox("Project", projects) if projects else None
    page = st.sidebar.radio("Navigate", PAGES)
    if is_set_up(project_id):
        active = store.get_active_version(project_id)
        st.sidebar.markdown(f"Active model **{active['version']}**")
    st.sidebar.caption("Python computes every number and the decision score. The LLM only drafts the profile, "
                       "proposes metric names and writes explanations.")
    if page == "Overview":
        page_overview(project_id)
    elif page == "Project Setup":
        page_setup(project_id)
    elif page == "Documentation":
        page_docs()
    elif not is_set_up(project_id):
        st.warning("This section needs a registered project. Open Project Setup first.")
    elif page == "Monitoring":
        page_monitoring(project_id)
    elif page == "Analysis & Reports":
        page_analysis(project_id)
    elif page == "Retraining History":
        page_history(project_id)
    else:
        page_assistant(project_id)


if __name__ == "__main__":
    main()
