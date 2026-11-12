# ReTrainIQ

ReTrainIQ is a retraining advisor for a deployed scikit-learn model. It reads an existing ML project,
watches new data, measures drift, CARA-T staleness and model quality, and turns them into one decision
score with three zones: KEEP, HUMAN review, or automatic RETRAIN. A retrained candidate replaces the
current model only after it passes verification, and every promotion can be rolled back.

Python computes every number and the score. The LLM only drafts the project profile, proposes metric
names from a fixed registry and writes explanations whose numbers are checked against the stored report.

## Modules

1. **Module 1, understand (`1-Profile`)**: static scan of a GitHub repo or ZIP (never executed), profile draft, code-side verification, model smoke test in a sandbox.
2. **Module 2, drift (`2-Drift`)**: PSI and KS per feature, drift share `d`, schema validation. A warning only, never a trigger.
3. **Module 3, CARA-T (`3-CARA`)**: relative staleness from Mahadevan and Mathioudakis (2024) against an offline threshold `tau*`. While labels are pending, **CARA-T-L** estimates staleness from calibrated model confidence; an estimate can never trigger automatic retraining.
4. **Module 4, quality (`4-Metrics`)**: registry metrics with bootstrap standard errors, compared with a never-seen holdout (`z_m`).
5. **Decision (`6-Decision`)**: `P = sigmoid(b + a1*zc + a2*zm + a3*d)`; `P >= 0.65` RETRAIN, `P <= 0.35` KEEP, otherwise HUMAN. `P` is a decision score, not a probability (the calibration test on the demo did not support calling it one).
6. **Analysis (`5-Analysis`)** and **frontend**: summaries, chat, Streamlit dashboard with a demo mode.

All packages are implemented (`python tools/status.py`). One developer owns everything.

## Setup Instructions

### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
python 0-Init/ri_init/setup_env.py
python -m ri_init.make_demo_project
pytest -q
python tools/status.py
```

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python 0-Init/ri_init/setup_env.py
python -m ri_init.make_demo_project
pytest -q
python tools/status.py
```

## Commands

```bash
python -m ri_decision.cli setup    --project demo          # baseline + CARA-T calibration (about 30 s)
python -m ri_decision.cli replay   --project demo --batches workspace/demo/data/stream --limit 40
python -m ri_decision.cli replay   --project demo --batches workspace/demo/data/stream --limit 30 --label-lag 2
python -m ri_decision.cli evaluate --project demo          # baselines, sensitivity, calibration, CARA-T-L
python -m ri_decision.cli resolve  --project demo --report 12 --choice KEEP
python -m ri_decision.cli versions --project demo
python -m ri_decision.cli rollback --project demo
python -m ri_decision.cli reset    --project demo          # wipe run state, set up again
python -m ri_decision.cli watch    --project demo --incoming workspace/demo/incoming --every 5h
streamlit run frontend/app.py                              # dashboard
```

`watch` checks a drop folder on a schedule (`5h`, `1d`, `1w`, or seconds). Data files go through the full
check; a file named `labels_<batch_id>.csv` is attached to that stored batch. Uploads are accumulated
until a full batch of `batch_size_B` rows exists, so irregular uploads are fine.

To use an LLM, copy `.env.example` to `.env` and fill in provider, key and model. Without it the system
uses a manual / heuristic profile, the default metric plan and template summaries.

## Verify on your machine

```powershell
pip install -r requirements.txt -r requirements-dev.txt
python 0-Init/ri_init/setup_env.py
python -m ri_init.make_demo_project
pytest -q                      # about 10 minutes: the suite runs several real calibrations
ruff check .
python tools/status.py
python -m ri_decision.cli reset --project demo
python -m ri_decision.cli replay --project demo --batches workspace/demo/data/stream --limit 30
```

## Security Rules

- **Zero Untrusted Code Execution:** Uploaded repositories, archives, and files are treated strictly as passive DATA. Code from uploaded projects is never executed or imported.
- **Isolated Sandbox Execution:** Model loading and prediction smoke-testing run exclusively in a restricted subprocess via `ri_init.worker.run_in_worker` with memory and CPU boundaries.
- **Deterministic Number Origin:** All evidence scores ($d$, $z_c$, $z_m$, $P$) are computed purely in deterministic Python and scikit-learn code, never synthesized by LLMs.
- **Path Escape Prevention:** `extract_zip_safe` strictly forbids path traversal (`..` escapes), absolute target paths, and archive symlinks.

## Known limitations

- Replays and injected shifts are simulations; nothing here is evidence of real temporal drift.
- CARA-T is an adaptation of the paper, not an exact reproduction (see the notes at the top of `3-CARA/ri_cara/cara_t.py`).
- Version 1 supports a scikit-learn Pipeline that accepts raw rows, tabular data, classification first. Retraining verification supports classification metrics only.
- CARA-T-L cannot see a pure concept shift (the model stays confident while wrong); `cli evaluate` shows this.
- The sandbox worker has memory and CPU limits on Linux and macOS; on Windows only the timeout applies.
- Demo results come from one synthetic stream and one seed.

More: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/STATUS.md](docs/STATUS.md), [CONTRACT.md](CONTRACT.md).
