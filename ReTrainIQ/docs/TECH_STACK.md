# Tech stack

Only what the repository uses. Versions are pinned in `requirements.txt`.

| Technology | What it does here |
|---|---|
| Python 3.10+ | Everything: pipeline, UI, tests. Developed on 3.10 (Windows) and checked on 3.13 (Linux). |
| Streamlit | The dashboard (`frontend/`). It runs the pipeline functions in-process; there is no separate web server or REST API. |
| pandas | Reading CSV / Parquet uploads, batching, label joins, all tabular handling. |
| NumPy | The CARA-T kernel maths (500 x 2,000 similarity matrices), bootstrap resampling, PSI. |
| SciPy | Two-sample Kolmogorov-Smirnov test (drift), Spearman correlation (label-free evaluation). |
| scikit-learn | The supported model type (a `Pipeline`), `clone().fit()` for retraining, all metric functions, isotonic regression for confidence calibration, logistic regression for the score-calibration test. |
| joblib | Loading and saving model artifacts. Loading happens only inside the sandbox worker. |
| Plotly | Every chart: timeline, gauge, drift bars, distribution overlays, staleness history, ROC and PR curves, confusion matrix. |
| SQLite (`sqlite3`, standard library) | One local file `workspace/retrainiq.db` with projects, model versions, batches, reports, decisions and the LLM log. |
| requests | The HTTPS call to the LLM provider. Nothing else uses the network except `git clone`. |
| pytest | The test suite (`tests/`). |
| ruff | Linting (development only). |
| matplotlib | Development only: `tools/make_manuals.py` draws the PNG guides. Not needed to run the product. |
| git (command) | Shallow clone of a public GitHub repository in Module 1. |

Not used: no LLM SDK, no FastAPI or Flask, no Docker, no MLflow, no message broker, no cloud service. The FastAPI prediction service described in the earlier semester report is not part of this repository.
