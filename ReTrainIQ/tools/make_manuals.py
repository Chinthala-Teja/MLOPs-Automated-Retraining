"""make_manuals.py - draws the six PNG guides in docs/manual/ (needs matplotlib; documentation tool only).

Run:  python tools/make_manuals.py
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "manual"
INK, MUTED = "#111827", "#4B5563"
C = {"setup": ("#EEF2FF", "#4F46E5"), "module": ("#FFF7ED", "#C2410C"), "decision": ("#FEF3C7", "#B45309"),
     "good": ("#ECFDF5", "#047857"), "bad": ("#FEF2F2", "#B91C1C"), "store": ("#F1F5F9", "#334155"),
     "llm": ("#F5F3FF", "#6D28D9"), "plain": ("#FFFFFF", "#9CA3AF")}


def canvas(title: str, subtitle: str, height: float = 62.0):
    fig, ax = plt.subplots(figsize=(16, 16 * height / 100))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, height)
    ax.axis("off")
    ax.text(50, height - 2.6, title, ha="center", va="center", fontsize=21, fontweight="bold", color=INK)
    ax.text(50, height - 5.6, subtitle, ha="center", va="center", fontsize=11.5, color=MUTED)
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), kind="plain", chars=None, size=9.2, title_size=10.5):
    """Rounded box. Text is wrapped to the box width and the font shrinks until it fits the height."""
    face, edge = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8", fc=face, ec=edge, lw=1.6))
    title_fit = min(title_size, (w - 1.5) / (0.066 * max(len(title), 1)))
    ax.text(x + w / 2, y + h - 1.7, title, ha="center", va="center", fontsize=title_fit, fontweight="bold", color=edge)
    while True:
        width = max(8, int((w - 2.0) / (size * 0.061)))
        wrapped = [piece for line in lines for piece in (textwrap.wrap(line, width) or [""])]
        if len(wrapped) * size * 0.14 <= h - 4.6 or size <= 6.4:
            break
        size -= 0.3
    ax.text(x + 1.0, y + h - 3.6, "\n".join(wrapped), ha="left", va="top", fontsize=size, color=INK, linespacing=1.35)


def arrow(ax, start, end, label="", dashed=False, rad=0.0, color=INK, label_dy=0.9):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=color,
                                 linestyle="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + label_dy, label, ha="center", va="center",
                fontsize=8.6, color=MUTED, bbox=dict(fc="white", ec="none", pad=1.2))


def band(ax, y, h, label, kind):
    face, edge = C[kind]
    ax.add_patch(FancyBboxPatch((1, y), 98, h, boxstyle="round,pad=0,rounding_size=0.8", fc=face, ec=edge, lw=1.0, alpha=0.45))
    ax.text(2.2, y + h - 1.5, label, ha="left", va="center", fontsize=10.5, fontweight="bold", color=edge)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=170, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def quick_start():
    fig, ax = canvas("ReTrainIQ Quick Start", "From a fresh checkout to a registered project in five steps", 50)
    box(ax, 3, 36, 94, 6.5, "Open the website", [
        "pip install -r requirements.txt      python 0-Init/ri_init/setup_env.py      streamlit run frontend/app.py"
        "      then open http://localhost:8501 in a browser"], "store", chars=170, size=9.6)
    steps = [
        ("1  Project Setup", ["Sidebar: Project Setup, tab 'Register a project'.", "Type a project id.",
                              "To try it first: tab 'Demo project', button 'Load / reset the demo project'."]),
        ("2  Give the project", ["Paste a GitHub URL (https://github.com/owner/repo)", "or upload the project ZIP.",
                                 "The code is read as text. It is never run."]),
        ("3  Upload the data", ["Training / reference dataset (CSV or Parquet): required.",
                                "Model file (.joblib / .pkl) if it is not in the repository.",
                                "Holdout the model never saw: recommended.", "Target column, if not the last one."]),
        ("4  Analyse Project", ["Press 'Analyse Project'.", "Module 1 scans the files, drafts a profile and checks it.",
                                "Read the list 'What Module 1 found'."]),
        ("5  Confirm Profile", ["Correct task, target, features, primary metric and batch size.", "Press 'Confirm Profile'.",
                                "The model is registered: holdout baseline, drift reference and CARA-T threshold are computed."]),
    ]
    for i, (title, lines) in enumerate(steps):
        x = 3 + i * 19
        box(ax, x, 9, 17, 23, title, lines, "setup" if i < 4 else "good", chars=27)
        if i < 4:
            arrow(ax, (x + 17, 20.5), (x + 19, 20.5))
    ax.text(50, 4.5, "Next: sidebar → Monitoring → upload a new batch (or run the guided demo), then Analysis & Reports.",
            ha="center", fontsize=11, color=MUTED)
    save(fig, "ReTrainIQ_Quick_Start.png")


def workflow():
    fig, ax = canvas("ReTrainIQ Workflow", "Project upload → monitoring → analysis → decision → retraining. "
                     "Python computes every number; the LLM only drafts, proposes metric names and explains.", 66)
    band(ax, 44, 14.5, "SETUP (once per project)", "setup")
    xs = [3, 22.5, 42, 61.5, 81]
    setup = [("Inputs", ["GitHub URL or ZIP", "reference dataset", "model file, holdout"], "plain"),
             ("Fetch and scan", ["shallow clone or safe unzip", "static scan only", "code is never executed"], "plain"),
             ("MODULE 1  Understand", ["profile draft (LLM or heuristics)", "code verifies columns, model smoke test, holdout"], "module"),
             ("User confirms", ["edits task, target, features, metric", "presses Confirm Profile"], "plain"),
             ("Registration", ["baseline on holdout (Module 4)", "drift reference (Module 2)", "CARA-T tau*, kappa (Module 3)"], "setup")]
    for (t, body, k), x in zip(setup, xs):
        box(ax, x, 45.5, 16, 10.5, t, body, k, chars=27, size=8.8)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 16, 50.7), (b, 50.7))
    band(ax, 25, 16.5, "MONITORING LOOP (every new upload restarts at Module 2)", "module")
    loop = [("New data (+ labels)", ["upload, drop folder or schedule", "schema check; INVALID DATA is blocked",
                                     "rows accumulate to batch size B"], "plain"),
            ("MODULE 2  Drift", ["PSI (categorical), KS + PSI (numeric)", "drift share d", "warning only"], "module"),
            ("MODULE 3  CARA-T", ["labels: measured staleness vs tau*", "no labels: estimate from confidence"], "module"),
            ("MODULE 4  Quality", ["registry metrics, bootstrap SE", "z versus holdout baseline", "no labels: PENDING"], "module"),
            ("Decision score P", ["sigmoid(b + a1*zc + a2*zm + a3*d)", "drift alone stays below the retrain line"], "decision")]
    for (t, body, k), x in zip(loop, xs):
        box(ax, x, 26, 16, 12.5, t, body, k, chars=27, size=8.8)
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + 16, 32), (b, 32))
    arrow(ax, (89, 45.5), (89, 38.5), "reference, baseline, tau*", dashed=True, label_dy=0)
    band(ax, 3, 18.5, "ACTION (chosen by the score zone)", "decision")
    box(ax, 3, 4.5, 26, 14, "P ≤ 0.35   KEEP", ["Continue with the current model.", "Decision and evidence are logged.",
                                               "The next upload re-enters at Module 2."], "good", chars=44)
    box(ax, 32, 4.5, 30, 14, "0.35 < P < 0.65, or labels missing   HUMAN REVIEW",
        ["Report, evidence and summary are shown.", "Buttons: Retrain / Keep.",
         "Retrain runs the same verification; a forced promotion needs an explicit tick."], "decision", chars=52)
    box(ax, 65, 4.5, 32, 14, "P ≥ 0.65   RETRAIN (automatic)",
        ["1 train a candidate on the last W labelled batches", "2 verify against the champion",
         "3 pass: promote; fail: keep the champion", "4 rollback restores the previous version"], "bad", chars=56)
    ax.plot([89, 89], [26, 23.2], color=INK, lw=1.6)
    ax.plot([27, 89], [23.2, 23.2], color=INK, lw=1.6)
    for x in (27, 47, 81):
        arrow(ax, (x, 23.2), (x, 18.5))
    save(fig, "ReTrainIQ_Workflow.png")


def module_guide():
    fig, ax = canvas("ReTrainIQ Module Guide", "What each module is for, what goes in, what it does and what comes out", 66)
    cols = [
        ("MODULE 1  Understand the project", "Know what the model predicts and how to feed it.",
         ["GitHub URL or ZIP", "reference dataset", "model artifact, optional holdout"],
         ["static file scan (tree, README, requirements, model and data files)", "profile draft by the LLM or by heuristics",
          "code checks: columns exist, class balance, model loads and predicts on raw rows in the sandbox"],
         ["profile.json: task, target, features, model, primary metric, batch size", "list of issues for the user"]),
        ("MODULE 2  Data drift", "Has the incoming data changed compared with the reference?",
         ["reference sample of the training data", "new batch (labels not needed)"],
         ["schema validation (missing columns, bad types block the upload)", "categorical: PSI, flag at 0.2 or more",
          "numeric: KS test, flag if p < 0.01 and statistic ≥ 0.1"],
         ["per-feature table and charts", "drift share d = flagged / all features", "status STABLE, DRIFT or INVALID DATA"]),
        ("MODULE 3  CARA-T staleness", "Has the model gone stale for the inputs it now receives?",
         ["new batch with labels", "stored sample of the model's training data", "offline calibration: tau*, kappa, s"],
         ["RBF-similarity-weighted 0-1 loss on the new batch minus the same on the training sample (same queries)",
          "bootstrap lower bound", "labels late: estimate from calibrated confidence (CARA-T-L)"],
         ["staleness, lower bound, tau*, kappa", "basis: measured, estimated or PENDING", "raw CARA-T result KEEP / RETRAIN"]),
        ("MODULE 4  Model quality", "Are predictions measurably worse than on the holdout?",
         ["new batch with labels", "baseline metrics on the never-seen holdout", "metric plan (registry names)"],
         ["compute each metric, 200 bootstrap resamples for the standard error",
          "z = (baseline − current) / combined SE on the primary metric"],
         ["metric table, confusion matrix, ROC and PR curves", "z_m and the degraded flag (z ≥ 1.64)",
          "no labels: PENDING, nothing is computed"]),
    ]
    rows = [("PURPOSE", 46, 8.5), ("INPUTS", 34.5, 10.5), ("PROCESSING", 17.5, 16), ("OUTPUTS", 3, 13.5)]
    for i, (title, purpose, inputs, processing, outputs) in enumerate(cols):
        x = 8 + i * 23
        ax.text(x + 10.75, 56.6, title, ha="center", fontsize=11.5, fontweight="bold", color=C["module"][1])
        for (label, y, h), content, kind in zip(rows, ([purpose], inputs, processing, outputs),
                                                ("setup", "plain", "module", "good")):
            box(ax, x, y, 21.5, h, label, ["• " + c if label != "PURPOSE" else c for c in content], kind, chars=40,
                size=8.6, title_size=9)
    for label, y, h in rows:
        ax.text(4, y + h / 2, label, ha="center", va="center", rotation=90, fontsize=10, fontweight="bold", color=MUTED)
    save(fig, "ReTrainIQ_Module_Guide.png")


def lifecycle():
    fig, ax = canvas("ReTrainIQ Retraining Lifecycle", "Decision score → retraining → verification → promotion → rollback", 62)
    box(ax, 3, 39, 28, 14, "1  Decision score", ["zc = (staleness lower bound − tau*) / s", "zm = quality degradation z",
                                                  "d = drift share", "P = sigmoid(b + 1.5 zc + 1.0 zm + 0.5 d)",
                                                  "P is a score, not a probability"], "decision", chars=46)
    box(ax, 36, 46.5, 27, 6.5, "P ≤ 0.35  KEEP", ["log and continue monitoring"], "good", chars=44)
    box(ax, 36, 39, 27, 6.5, "between, or no labels  HUMAN REVIEW", ["user presses Retrain or Keep"], "decision", chars=44)
    box(ax, 68, 39, 29, 14, "P ≥ 0.65  RETRAIN", ["automatic, unless a cooldown is active (W labelled batches after a promotion)",
                                                  "an estimate (labels pending) can never start it"], "bad", chars=48)
    arrow(ax, (31, 49.5), (36, 49.5))
    arrow(ax, (31, 43), (36, 42.5))
    arrow(ax, (63, 42.5), (68, 42.5))
    ax.text(65.5, 44.2, "Retrain", ha="center", fontsize=8.2, color=MUTED)
    box(ax, 3, 20, 22, 14, "2  Train the candidate", ["data: last W labelled batches", "newest 20% of the newest batch is held out",
                                                      "clone(pipeline).fit in the sandbox worker", "training seconds recorded"], "module", chars=36)
    box(ax, 29, 20, 34, 14, "3  Verify against the champion",
        ["accuracy gain on the held-out slice: 5% bootstrap lower bound ≥ 0",
         "primary metric drop ≤ 0.02 and recall drop ≤ 0.02",
         "reference-holdout drop ≤ 0.02 (waived, and reported, when the champion itself has degraded on current data)"],
        "module", chars=57)
    box(ax, 67, 27.5, 30, 6.5, "REJECT", ["champion stays; reason logged; cost still counted"], "bad", chars=50)
    box(ax, 67, 20, 30, 6.5, "4  PROMOTE", ["write models/vN, then switch the active model atomically"], "good", chars=50)
    arrow(ax, (82, 39), (14, 34), rad=0.0)
    arrow(ax, (25, 27), (29, 27))
    arrow(ax, (63, 30), (67, 30.5))
    ax.text(65, 32.6, "fail", ha="center", fontsize=8.6, color=MUTED)
    arrow(ax, (63, 24), (67, 23.5))
    ax.text(65, 21.4, "pass", ha="center", fontsize=8.6, color=MUTED)
    box(ax, 3, 3, 30, 12, "5  After promotion", ["version bundle saved: model, calibration, baseline, training sample, drift reference",
                                                "CARA-T recalibrated; cooldown starts"], "setup", chars=50)
    box(ax, 37, 3, 28, 12, "6  Shadow check", ["next 3 labelled batches: previous model is scored on the same data",
                                              "new model worse by more than 0.02 on average: rollback suggested"], "decision", chars=46)
    box(ax, 69, 3, 28, 12, "7  Rollback", ["button or CLI command", "restores the previous model and its whole bundle",
                                          "automatic only if auto_rollback is switched on"], "bad", chars=46)
    arrow(ax, (82, 20), (18, 15), rad=0.0)
    arrow(ax, (33, 9), (37, 9))
    arrow(ax, (65, 9), (69, 9))
    ax.text(67, 13.2, "if worse", ha="center", fontsize=8.6, color=MUTED)
    save(fig, "ReTrainIQ_Retraining_Lifecycle.png")


def architecture():
    fig, ax = canvas("ReTrainIQ System Architecture", "One Python process for the UI and the pipeline; model code runs in a separate sandbox process", 64)
    box(ax, 3, 48, 94, 8, "FRONTEND   Streamlit app (frontend/app.py, ui.py, demo.py)",
        ["Overview · Project Setup · Monitoring · Analysis & Reports · Retraining History · AI Assistant · Documentation"
         "        Calls Python functions directly; there is no HTTP API between UI and backend."], "setup", chars=165)
    box(ax, 3, 36, 60, 8.5, "ORCHESTRATOR   ri_decision.orchestrator (+ cli.py for replay, evaluate, watch)",
        ["setup_project · on_new_data · attach_labels · resolve_human · rollback · reset_project"], "decision", chars=104)
    box(ax, 67, 36, 30, 8.5, "ANALYSIS   ri_analysis", ["summary.py, chat.py: text only, numbers checked", "llm.py: the only network caller"], "llm", chars=50)
    names = [("MODULE 1", "ri_profile.understand", "scan, profile, verify"), ("MODULE 2", "ri_drift", "drift, schema, plots"),
             ("MODULE 3", "ri_cara", "CARA-T, CARA-T-L, baselines"), ("MODULE 4", "ri_metrics", "registry, bootstrap, compare"),
             ("DECISION", "ri_decision.decision", "score P and zone"), ("RETRAIN", "ri_decision.retrain", "train, verify, promote")]
    for i, (a, b, c) in enumerate(names):
        box(ax, 3 + i * 15.2, 22, 13.6, 9.5, a, [b, c], "module" if i < 4 else "decision", size=8.6)
        top = (9.8 + i * 15.2, 31.5)
        arrow(ax, (top[0], 36) if i < 4 else (58 + (i - 4) * 3, 36), top)
    box(ax, 3, 4, 21, 13, "SANDBOX WORKER", ["ri_init.worker: subprocess", "loads models, predicts, trains", "timeout; memory and CPU limits on Linux / macOS",
                                            "minimal environment"], "bad", chars=34)
    box(ax, 27, 4, 21, 13, "SQLITE", ["workspace/retrainiq.db", "projects, model_versions, batches, reports, decisions, llm_log"], "store", chars=34)
    box(ax, 51, 4, 22, 13, "MODEL AND FILE STORAGE", ["workspace/<project>/", "models/vN bundles + active", "data: batches, samples, holdout",
                                                    "calib.json, baseline.json, state.json"], "store", chars=36)
    box(ax, 76, 4, 21, 13, "LLM PROVIDER (external, optional)", ["Anthropic Messages API or an OpenAI-compatible endpoint",
                                                                "key from .env; without it templates are used"], "llm", chars=34)
    arrow(ax, (50, 48), (40, 44.5))
    arrow(ax, (75, 48), (80, 44.5))
    arrow(ax, (20, 22), (13.5, 17), "model code")
    arrow(ax, (40, 22), (37.5, 17), "rows")
    arrow(ax, (84, 22), (62, 17), "artifacts")
    arrow(ax, (95, 36), (95, 17), "HTTPS", dashed=True, label_dy=0)
    save(fig, "ReTrainIQ_System_Architecture.png")


def data_flow():
    fig, ax = canvas("ReTrainIQ API and Data Flow", "What happens, in order, when a new batch is uploaded (function calls, not HTTP requests, except step 9)", 66)
    lanes = [("Browser", "setup"), ("Streamlit app", "setup"), ("Orchestrator", "decision"), ("Modules + decision", "module"),
             ("Sandbox worker", "bad"), ("SQLite + files", "store"), ("LLM (optional)", "llm")]
    xs = [8 + i * 14 for i in range(7)]
    for (name, kind), x in zip(lanes, xs):
        face, edge = C[kind]
        ax.add_patch(FancyBboxPatch((x - 6, 53), 12, 4, boxstyle="round,pad=0,rounding_size=0.6", fc=face, ec=edge, lw=1.5))
        ax.text(x, 55, name, ha="center", va="center", fontsize=9.2, fontweight="bold", color=edge)
        ax.plot([x, x], [3, 53], color="#D1D5DB", lw=1.1, linestyle=":")
    flows = [(0, 1, "1  upload CSV / Parquet (+ labels)", False), (1, 2, "2  on_new_data(project_id, file, labels)", False),
             (2, 3, "3  validate_schema → VALID / BLOCKED", False), (2, 5, "4  buffer rows until batch size B; store the batch file", False),
             (2, 3, "5  run_drift(reference sample, batch)", False), (2, 4, "6  online_staleness / estimate_staleness, run_metrics (pickled args)", False),
             (4, 2, "7  result dicts (staleness, metrics)", True), (2, 3, "8  compare → z_m;  decide → P, zone", False),
             (2, 4, "8b zone RETRAIN: train candidate, verify, promote", False), (2, 5, "8c add_report, add_decision, state.json, model bundle", False),
             (2, 1, "8d reports returned to the page", True), (1, 6, "9  only on 'Write AI summary' or chat: HTTPS POST with the compact report", False),
             (6, 1, "10 text back; every number is checked against the report, else the template is used", True),
             (1, 0, "11 charts, zone, evidence, summary", True)]
    y = 49.5
    for a, b, label, back in flows:
        arrow(ax, (xs[a], y), (xs[b], y), dashed=back)
        left = min(xs[a], xs[b])
        ax.text(left + 0.8, y + 1.0, label, ha="left", va="center", fontsize=8.8, color=INK,
                bbox=dict(fc="white", ec="none", pad=0.8))
        y -= 3.3
    save(fig, "ReTrainIQ_API_Data_Flow.png")


def main() -> None:
    for draw in (quick_start, workflow, module_guide, lifecycle, architecture, data_flow):
        draw()
    print(f"Wrote 6 PNG files to {OUT}")


if __name__ == "__main__":
    main()
