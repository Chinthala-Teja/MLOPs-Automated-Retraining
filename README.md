# Electricity MLOps: Cost-Aware Model Retraining

## 1. Project Overview

This project implements an end-to-end MLOps workflow for electricity price-movement classification. It trains a baseline model, monitors performance across chronological data batches, estimates model staleness, and evaluates retraining policies using a cost-aware decision mechanism inspired by CARA-T.

The main research question is:

**When should an ML model be retrained, and when should the existing model continue serving predictions?**

The system compares threshold-based retraining against never-retrain, always-retrain, and periodic-retraining baselines.

## 2. Architecture

Raw Electricity Dataset  
→ Data Preparation and Chronological Batching  
→ Model Training and Preprocessing  
→ Model V1 Artifacts  
→ Batch-Level Monitoring  
→ RBF-Based Staleness Estimation  
→ Retraining Decision  
→ KEEP or RETRAIN  
→ Updated Model Evaluation  
→ Cost and Performance Comparison

## 3. Dataset and Splits

The processed dataset contains 45,312 records divided into 100 chronological batches.

| Split | Batches | Records |
|---|---|---:|
| Training | 0–69 | 31,718 |
| Validation | 70–84 | 6,797 |
| Holdout | 85–99 | 6,797 |

The holdout batches are used for final policy comparison. Earlier exploratory analysis included inspection of holdout data, so the holdout should not be described as completely untouched throughout the entire project history.

## 4. Baseline Model

The baseline is a neural-network binary classifier using six standardized input features.

Architecture:
- Dense layer with 32 ReLU units
- Dropout layer with rate 0.2
- Dense layer with 16 ReLU units
- Sigmoid output layer

The scaler is fitted on training data. Training uses Adam, binary cross-entropy, and early stopping.

## 5. Monitoring and Staleness

The monitoring pipeline evaluates model performance across chronological batches and records classification metrics.

The staleness component uses an RBF similarity kernel and prediction errors to estimate the cost associated with using a model on a batch. Relative staleness compares the current batch's estimated cost against a reference cost.

A threshold-based decision determines whether the policy keeps the active model or retrains it.

## 6. Retraining Policies

The evaluation compares:
1. Never retrain
2. Always retrain
3. Periodic retraining, every three batches
4. CARA-T-inspired threshold-based retraining

The project uses a normalized proxy-cost objective: KEEP incurs the estimated batch cost, while RETRAIN incurs the configured retraining cost, κ.

This is a project-specific adaptation inspired by CARA-T, not an exact reproduction of the paper's optimization procedure.

## 7. Holdout Results

| Policy | Retrains | Proxy Cost | Accuracy | F1 |
|---|---:|---:|---:|---:|
| Never retrain | 0 | 1.2928 | 78.96% | 0.7860 |
| Always retrain | 15 | 1.2000 | 81.07% | 0.7900 |
| Periodic | 5 | 1.1300 | 80.42% | 0.7644 |
| CARA-T-inspired | 14 | 1.1657 | 81.17% | 0.7914 |

These results are from the recorded holdout run. The proxy cost is an experimental objective, not a monetary-cost measurement.

## 8. Key Findings

- Periodic retraining achieved the lowest proxy cost in the recorded holdout experiment.
- The CARA-T-inspired policy achieved the highest mean accuracy and F1 in that run.
- The threshold policy retrained on 14 of 15 holdout batches, so it reduced retraining only slightly compared with always retraining.
- Lower proxy cost does not necessarily imply higher predictive performance.
- The current results motivate further work on cost calibration, robust threshold selection, and retraining on accumulated data.

## 9. Artifacts

Important output locations:

- `artifacts/model_v1/` — baseline model, scaler, and metadata
- `artifacts/monitoring/` — batch-level monitoring metrics
- `artifacts/cara_t/` — staleness, threshold, and sequential replay experiments
- `artifacts/cara_t/test_holdout/` — holdout results, audit CSVs, run log, and plots

## 10. Reproducibility

Run scripts from the project root using the Python environment configured for the project. Verify each script's input and output paths before running it.

The principal experiment outputs should be preserved alongside the configuration and metadata used to generate them.

## 11. Limitations

- Retraining costs are normalized assumptions rather than measured operational costs.
- The threshold-based method is CARA-T-inspired, not an exact paper reproduction.
- Retraining currently uses the current batch and retains the original scaler.
- Batch-based evaluation is a simulation of a streaming MLOps workflow.
- The reported results apply to this dataset, model, split, and experiment configuration; broader generalization has not been established.

## 12. Future Work

Potential improvements include measuring actual training and serving costs, evaluating retraining on accumulated data, tracking data and model versions more comprehensively, and testing the workflow on a live stream.
