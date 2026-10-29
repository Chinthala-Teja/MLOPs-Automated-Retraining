# Electricity MLOps — Cost-Aware Retraining

The notebook companion [Electricity_CARA_T.ipynb](electricity-cara/Electricity_CARA_T.ipynb)
follows the Airline notebook's research/evaluation flow, cites the supplied
CARA paper, and reads saved results by default. Its final cell can optionally
rerun the canonical holdout evaluator.

## Overview

This project implements an MLOps workflow for electricity price-movement classification. It combines chronological dataset batching, neural-network training, batch monitoring, RBF-based staleness estimation, and a threshold-based retraining policy inspired by CARA-T.

The research question is: **When should an ML model be retrained instead of continuing to use the current model?**

## Dataset

The processed dataset contains 45,312 records divided into 100 chronological batches.

| Split | Batch IDs | Records |
|---|---|---:|
| Training | 0–69 | 31,718 |
| Validation | 70–84 | 6,797 |
| Holdout | 85–99 | 6,797 |

The holdout data was inspected during earlier exploratory analysis, so it should not be described as completely untouched throughout the project.

## Architecture

Raw data → preprocessing and chronological batching → Model V1 → batch monitoring → staleness estimation → KEEP/RETRAIN decision → sequential replay → baseline comparison and audit.

## Model

The baseline is a neural-network binary classifier using six standardized features. Its artifacts are stored under `artifacts/model_v1/`.

- `model.keras`: trained model
- `scaler.joblib`: fitted feature scaler
- `metadata.json`: training configuration and validation metrics

The scaler is fitted on the training split.

## Retraining policies

The experiments compare:
1. Never retrain
2. Always retrain
3. Periodic retraining
4. CARA-T-inspired threshold retraining

The project-specific proxy cost assigns κ to a RETRAIN decision and the estimated absolute batch cost to a KEEP decision. This is an adaptation inspired by CARA-T, not an exact reproduction of the paper's objective.

## Main scripts

Run commands from the project root.

| Script | Purpose |
|---|---|
| `src/03_prepare_batches.py` | Prepare chronological batches |
| `train_baseline.py` | Train the canonical Model V1 and save its artifacts |
| `evaluate_batches.py` | Generate batch-level monitoring metrics |
| `cara_staleness.py` | Estimate and visualize staleness |
| `optimize_cara_threshold.py` | Run the existing threshold optimization experiment |
| `cara_sequential_replay.py` | Run sequential replay and cost-sensitivity experiments |
| `compare_baselines.py` | Compare retraining strategies |
| `evaluate_test_holdout.py` | Run validation threshold selection, holdout comparison, and integrated audit |
| `audit_test_holdout.py` | Re-audit saved holdout results |

Run only the scripts needed for the experiment you want to reproduce. Do not rerun training after final evaluation unless you intend to regenerate the model and rerun dependent experiments.

## Holdout outputs

The principal results are saved in `artifacts/cara_t/test_holdout/`:

- `test_batch_results.csv`
- `test_strategy_summary.csv`
- `experiment_metadata.json`
- `holdout_audit.csv`
- `holdout_audit_summary.csv`
- `holdout_run_log.txt`
- `plots/`

## Recorded holdout results

| Policy | Retrains | Proxy cost | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Never retrain | 0 | 1.2928 | 78.96% | 0.7860 |
| Always retrain | 15 | 1.2000 | 81.07% | 0.7900 |
| Periodic | 5 | 1.1300 | 80.42% | 0.7644 |
| CARA-T-inspired | 14 | 1.1657 | 81.17% | 0.7914 |

These values correspond to the recorded experiment and may change if the model, environment, or random seed changes. The proxy cost is not a measured monetary cost.

## Limitations

- Retraining costs are assumed normalized values.
- The CARA-T-inspired policy is not an exact reproduction of the paper.
- Retraining uses the current batch and retains the original scaler.
- The workflow simulates a streaming process using chronological batches; live automated deployment has not been established by these experiments.
- Earlier exploratory inspection of holdout data limits claims of a fully untouched holdout.

## Environment

Install dependencies with:

`python -m pip install -r requirements.txt`

Run the holdout evaluation with:

`python -u evaluate_test_holdout.py`

Run the saved-results audit with:

`python audit_test_holdout.py`
