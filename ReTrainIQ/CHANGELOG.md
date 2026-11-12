# Changelog

All notable changes to ReTrainIQ are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased] — Audit & Hardening

### Fixed
- **worker.py**: Replaced `subprocess.run(timeout=)` with `Popen + communicate(timeout=) + proc.kill()` so the sandbox worker is reliably killed on timeout rather than raising an unhandled exception.
- **worker.py**: Built `PYTHONPATH` from `sys.path` so the subprocess finds all packages without relying on the `.pth` file from `setup_env.py`.
- **orchestrator.py** (deduplication): Added SHA-256 content-hash deduplication per uploaded file; duplicate uploads return `{"status": "DUPLICATE"}`.
- **orchestrator.py** (unlabelled batches): Unlabelled batches now force `zone="HUMAN"` and `alert="request labels"` instead of incorrectly delegating to `decide()` which returned KEEP from the stub.
- **orchestrator.py** (idempotency): `attach_labels` returns `{"status": "ALREADY_ATTACHED"}` when `proxy_error` is already present in the stored report; prevents double-labelling.
- **orchestrator.py** (batch path lookup): `attach_labels` now calls `store.get_batch(batch_id)` to retrieve the actual stored file path instead of guessing from a sequential index.
- **store.py**: Added `update_report`, `get_report`, and `get_batch` functions with roundtrip tests.
- **cli.py** (`run_replay`): Rewritten to display attachment rows (basis=measured, proxy_error), print a summary line (`batches sent: N, labels attached: N, human-pending: N, errors: N`), and correctly call `attach_labels` when `len(label_queue) > label_lag`.
- **tests/test_contract.py**: Removed unused import; rewrote `test_imports` to use an `importlib` loop.
- **tests/test_decision.py**: Removed unused base variables; fixed `what_if` and monotonicity assertions.

### Added
- **tools/status.py**: CONTRACT.md signature checker; exits 0 with a table of all modules; `tests/test_status.py` added.
- **.gitignore**: Added `.ruff_cache/` and `retrainiq.db`.
- **.gitattributes**: `* text=auto eol=lf` for consistent line endings.
- **.env.example**: Template for `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`.
- **requirements-dev.txt**: Pins `ruff==0.9.10` for static analysis.
- **Stage 3 retraining**: Added worker-isolated candidate training, paired V-slice and reference-holdout verification, atomic promotion and rollback, version bundles, labelled-history refresh, and three-batch shadow checks.
- **Orchestrator and CLI**: Routed automatic, late-label, and human retrains through one cycle; added rollback and version-list commands.
- **Store**: Added ordered batch/version retrieval, label-count updates, and previous/active version status handling.
- **docs/ARCHITECTURE.md**: Data-flow diagram, decision-score formula, Contract v2 explanation.
- **docs/STATUS.md**: Per-stage tracker and evidence table (real vs stub).
- **CHANGELOG.md**: This file.

### Changed
- **README.md**: Full rewrite — problem statement, module map, setup commands (PowerShell + bash), replay command, component status table, security rules, known limitations.
- **README.md**: Replaced machine-specific absolute links with repository-relative documentation links.
- **CONTRACT.md**: Updated to match final signatures including `update_report`, `get_report`, `get_batch`, `estimate_staleness`, Contract v2 fields.

### Validation
- **2026-11-12** V0 audit: local `pytest -q` — 37 passed; `ruff check .` — clean; `python tools/status.py` — all contract signatures verified; fresh Python 3.10 clone — setup/demo generation succeeded and 37 tests passed.
- **2026-11-12** Stage 3: `pytest -q` — 50 passed in 131.44s; `ruff check .` — clean; `python tools/status.py` — all contract signatures verified.

---

## [Stage 2] — 2026-11-11

### Added
- **ri_decision/decision.py** (`IS_STUB=False`): Sigmoid decision score, what-if analysis, safety guard, estimated-basis path (uses `staleness_est_lo`, caps zone at HUMAN).
- **ri_decision/orchestrator.py** (`IS_STUB=False`): Batch buffering, cooldown enforcement, late-label workflow, project lifecycle.
- **ri_decision/cli.py** (`IS_STUB=False`): `setup` and `replay` subcommands with label-lag support.
- **config.py**: Extended `DEFAULT_CONFIG` with `z_clip=3.0` and `s_floor=1e-6`.
- **tests/test_decision.py**: 14 tests covering monotonicity, what-if, estimated path, safety guard, edge cases.
- **tests/test_orchestrator.py**: 12 tests covering full batch lifecycle, deduplication, cooldown, attach idempotency.

---

## [Stage 1] — 2026-11-11

### Added
- **ri_profile/understand.py** (`IS_STUB=False`): `fetch_repo` (GitHub URL validation, `git clone --depth 1`), `extract_zip_safe` (path-traversal and symlink rejection), `scan_project` (file tree, model/data/script discovery), `build_context`, `extract_profile` (LLM-based with schema validation and retry), `heuristic_profile` (no-LLM, up to 200k rows), `verify_profile`.
- **5-Analysis/ri_analysis/llm.py**: `LLMClient` Protocol and `MockLLM` stub (`IS_STUB=True`).
- **tests/test_profile.py**: 8 tests covering zip safety, heuristic profiling, schema validation, verify_profile.

---

## [Contract v2] — 2026-11-11

### Added
- `cara_t` result dict gains `basis`, `staleness_est`, `staleness_est_lo`, `decision_est` keys; `status` may be `'ESTIMATED'`.
- `ri_cara.cara_t` gains `estimate_staleness(model_path, new_df, train_sample_df, calib, profile, seed=42) -> dict`.
- `calibrate()` result gains `proxy` sub-dict (`conf_cal`, `map`, `spearman`).
- Decision path: if `cara['basis'] == 'estimated'`, use `staleness_est_lo` for `zc`, cap zone at `'HUMAN'`, annotate reason.
- Updated stubs in `ri_cara` and `ri_decision`; `tests/test_contract.py` updated.

---

## [Stage 0C] — 2026-11-11

### Added
- **ri_init/make_demo_project.py**: Generates `workspace/demo/` — 50 stream batches of synthetic flight-like data, RandomForest model, concept drift after batch 25, `env.json` with Python and scikit-learn versions.
- **tests/test_demo.py**: Checks `env.json` scikit-learn version matches installed version.

---

## [Stage 0B] — 2026-11-11

### Added
- **ri_init/store.py**: SQLite-backed store with tables for projects, batches, reports, labels; full roundtrip tests.
- **ri_init/worker.py** + **worker_main.py**: Sandbox worker that runs user code in a subprocess with timeout.

---

## [Stage 0A] — 2026-11-11

### Added
- Project skeleton: 7 packages (`ri_init`, `ri_profile`, `ri_drift`, `ri_cara`, `ri_metrics`, `ri_analysis`, `ri_decision`), `frontend/`, `tests/`, `workspace/`.
- **CONTRACT.md**: Interface contract for all modules.
- **requirements.txt**: Pinned direct dependencies (`pandas==2.3.3`, `numpy==2.2.6`, `scipy==1.15.3`, `scikit-learn==1.7.2`, `joblib==1.5.1`, `plotly==6.1.2`, `streamlit==1.45.1`, `requests==2.32.3`, `pytest==8.3.5`).
- **ri_init/setup_env.py**: Adds all package folders to `sys.path` via a `.pth` file; cross-platform (pathlib, no shell-specific code).
- **pytest.ini**: `testpaths = tests`.
- **README.md**: Initial setup commands.
