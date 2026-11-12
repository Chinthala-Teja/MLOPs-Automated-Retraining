# ReTrainIQ implementation status

**Date:** 2026-11-12. **Owner of every item:** Teja (single developer; the earlier three-person split no longer applies).

## Verified in the cloud session of 2026-11-12 (Linux, Python 3.13, scikit-learn 1.7.2)

- `pytest -q`: see the last line of CHANGELOG "Validation" for the count and time.
- `ruff check --select E4,E7,E9,F .` clean (run with ruff 0.16; the pinned ruff 0.9.10 was not run).
- `python tools/status.py`: all contract signatures present, no module is a stub.
- Demo replay end to end (`cli replay`), delayed-label replay (`--label-lag 2`), `cli evaluate`.

**Not verified:** Windows / Python 3.10 (your machine), a real LLM call (no API key in the session),
the Streamlit UI in a real browser (only a headless render test), the real Airlines project,
GitHub cloning (`fetch_repo`) against the network.

## Stage tracker

| Stage | Content | Status |
|---|---|---|
| 0 | Skeleton, store, sandbox worker, demo project | Done |
| 1 | Module 1: understand the project (scan, profile, verify) | Done |
| 2 | Decision score, orchestrator, CLI | Done |
| 3 | Retrain, verify, promote, rollback, shadow check | Done (gates corrected, see CHANGELOG) |
| M2 | Drift, schema check, plots, scenario injector | Done |
| M3 | CARA-T calibration + measured staleness + CARA-T-L estimate | Done |
| M4 | Metric registry, bootstrap SE, baseline comparison | Done |
| 4 | LLM client, metric plan, validated summary, chat | Done (real LLM call untested) |
| 5 | Streamlit UI: seven sections, guided four-module workflow, guided demo | Done (redesigned 2026-11-12; rendered in a headless browser, screenshots in docs/screenshots) |
| 5b | Documentation: API reference, architecture, data flow, tech stack, project explanation, user manual, six PNG guides | Done |
| 6 | Baselines, sensitivity, score calibration, label-free evaluation | Done on the demo project |
| 7 | Airlines / Electricity / Covertype runs, 5 seeds, report rewrite | **Open** |

## Where each number comes from

Every value shown to the user is computed by code: drift share (ri_drift), staleness, tau*, kappa, s
(ri_cara), metrics and z_m (ri_metrics), P and zone (ri_decision). The LLM writes text only, and every
number in its text is checked against the stored report.

## Demo results (synthetic data, one seed, SIMULATED concept shift after batch 25)

Full loop, labels on time (`cli replay --limit 40`): 26 stable batches gave 22 KEEP and 4 HUMAN,
no automatic retrain; batch 26 scored 0.999, a candidate was trained, verified (holdout gate waived
because the champion had degraded) and promoted; the following 13 batches were KEEP.

Strategy comparison on the 50-batch stream (`cli evaluate`; kappa 67.62, tau* 69.06):

| Strategy | Retrains | Total cost J | % above Oracle |
|---|---|---|---|
| Never | 0 | 4482.9 | 37.7 |
| Always | 50 | 6598.3 | 102.6 |
| Periodic(5) | 10 | 4046.5 | 24.3 |
| Markov | 21 | 4629.7 | 42.2 |
| CARA-T | 10 | 3859.4 | 18.5 |
| ReTrainIQ score (auto zone, no verification) | 6 | 3607.4 | 10.8 |
| Oracle (retrospective) | 1 | 3256.5 | 0 |

Score calibration: fitted coefficients were worse out of time than the starting ones (Brier 0.050
against 0.019), so the coefficients stay as they are and P stays a *decision score*, not a probability.

CARA-T-L (label-free estimate), onset at batch 8: under a covariate shift towards inputs the model is
unsure about, the estimate raised the alert at batch 8 while measured staleness became actionable at
8 + label lag (10 and 13 for lags 2 and 5); no false alarms on stable batches. Under a pure concept
shift the estimate saw nothing (expected failure: the model stays confident while wrong).

These are results on one synthetic stream with one seed. They are not evidence about real data.

## Remaining work (in order)

1. Local check on Windows / Python 3.10: commands in README "Verify on your machine".
2. Open the UI in your own browser and click through the seven sections and the guided demo.
3. Put an LLM key in `.env`, test profile extraction and the AI summary.
4. Airlines project: export the notebook model as one scikit-learn Pipeline, register it through Setup,
   run `cli replay` and `cli evaluate`; then Electricity and Covertype with scikit-learn models.
5. Five seeds for the final numbers; paired bootstrap intervals; a final holdout used once.
6. Literature search before any novelty claim for CARA-T-L.
7. Rewrite the report (main.tex) for the four-module design; record the demo video.
