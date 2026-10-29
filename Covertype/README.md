# CoverType cost-aware retraining experiment

The notebook companion [Covertype_CARA_T.ipynb](Covertype_CARA_T.ipynb)
follows the Airline notebook's research/evaluation flow and cites the supplied
CARA paper. It analyzes saved artifacts by default; its final cell can
optionally rerun the full canonical pipeline.

This project ports the batch-monitoring and CARA-T-inspired retraining workflow used in `Electricity\electricity-cara` to the supplied binary CoverType LIBSVM data. It is an experiment/replay, not a live production deployment and not an exact reproduction of the CARA-T research paper.

## What the supplied data actually is

The input file is `..\covtype.libsvm.binary.scale.bz2`, relative to this project folder. Inspection of the full file found:

- 581,012 records and 54 sparse feature columns.
- Original target labels 1 and 2 only: 297,711 records of class 1 and 283,301 of class 2. The model maps original class 1 to 0 and original class 2 to 1.
- Nonzero feature values range from 0.00050025 to 1.0. The input is already scaled, so this pipeline does not fit another scaler.
- The file contains no timestamp. The pipeline preserves its row order for reproducible splits and batch replay, but does **not** claim the batches are chronological or prove data drift.

The original CovType dataset is multiclass; this uploaded `binary.scale` version is a binary classification task.

The 54 columns follow the standard CovType feature layout: features 1–10 are terrain and distance measurements; 11–14 are wilderness-area indicators; and 15–54 are soil-type indicators.

## What is carried over from Electricity and the Airlines analysis

The Electricity implementation established the main lifecycle: inspect and prepare data, train a saved Model V1, monitor batches, estimate RBF similarity-weighted error staleness, select a validation threshold, sequentially compare KEEP/RETRAIN strategies, and audit decisions and proxy costs. Its recorded holdout comparison found periodic retraining had the lowest proxy cost, while CARA-T-inspired retraining had the best mean accuracy/F1 but retrained on 14 of 15 batches. The Electricity cost values and threshold do not transfer to CoverType because the feature dimension and similarity scale differ.

The supplied Airlines conclusions reinforce three design requirements adopted here: choose thresholds only on a separate validation segment; compare multiple baselines across a sensitivity range for retraining cost; and state plainly that thresholds/costs are dataset-specific assumptions. Both earlier projects also emphasize the limitations of single-seed, batch-only retraining, delayed labels, and unverified temporal ordering.

At the Airline notebook's illustrative `kappa=26.1941`, CARA-T retrained 40 times over 75 online batches, reported query accuracy of 64.10%, and total strategy cost 1,438.85; the retrospective Oracle cost 1,367.58 but is not deployable. The kappa sweep changed retraining counts from 67 at the low-cost setting to 1 at the highest settings. Those results are specific to that notebook and are not CoverType results.

## Dataset preparation and split

The scripts create 100 contiguous batches without crossing split boundaries:

| Split | Batches | Rows (end exclusive) | Purpose |
|---|---:|---:|---|
| Train | 0–69 | 0–406,708 | Fit Model V1 and the RBF reference |
| Validation | 70–84 | 406,708–493,860 | Estimate staleness and select the threshold |
| Holdout replay | 85–99 | 493,860–581,012 | Sequential policy comparison |

Features remain sparse on disk (`data\processed\X.npz`); targets, the batch manifest, and stream metadata are saved alongside it. Only an individual training split or batch is densified when passed to Keras.

## Model and policy

Model V1 is a Keras binary classifier with 54 inputs, Dense(32, ReLU), Dropout(0.2), Dense(16, ReLU), and a sigmoid output. It is trained on batches 0–69 with early stopping against batches 70–84. Model artifacts and validation metrics are saved under `artifacts\model_v1`.

The staleness signal is a sampled RBF similarity-weighted 0–1 classification-loss proxy:

`absolute_cost = mean(RBF_similarity(query, labeled_row) * classification_loss)`, averaged over the sampled query and labeled rows.

`relative_staleness = current_batch_absolute_cost - reference_absolute_cost`.

The policy retrains when `relative_staleness >= threshold`. Candidate thresholds come from quantiles of standalone Model V1 validation staleness and include always-KEEP/always-RETRAIN endpoints. Threshold selection then runs a **sequential validation replay**: for each candidate threshold and `kappa`, the active model predicts each validation batch; if triggered, a fresh model trains on that batch and resets the reference cost for the next batch. KEEP pays the current batch's absolute proxy cost, RETRAIN pays `kappa`. The reference `kappa` is the median positive Model V1 validation batch proxy cost; multipliers 0.25, 0.5, 1, 2, and 4 provide sensitivity analysis. This scaling is necessary because an Electricity `kappa` value is not portable. Neither `kappa` nor the RBF loss proxy is a measured dollar, compute, or energy cost.

The holdout replay compares Never retrain, Always retrain, Periodic (every 3 batches), and CARA-T-inspired threshold retraining. The model predicts and is scored on each batch before labels from that batch can be used to train a replacement for later batches. Replacement models start with fresh weights and train on the current batch only, matching the Electricity experiment. The final batch still receives its policy decision and a triggered update is trained after scoring it, matching Electricity; that last update cannot affect holdout predictions.

The corrected sequential validation search selected `kappa=0.164186`, `threshold=0.026606`, with seven validation retrains. The completed holdout replay produced:

| Holdout strategy | Retrains | Accuracy | F1 | Total proxy cost |
|---|---:|---:|---:|---:|
| Never retrain | 0 | 63.85% | 0.4988 | 2.9384 |
| Always retrain | 15 | **73.54%** | **0.6728** | 2.4628 |
| Periodic (every 3 batches) | 5 | 64.87% | 0.5628 | 2.8059 |
| CARA-T-inspired | 11 | 71.48% | 0.6461 | **2.1887** |

For this run, Always retrain achieved the strongest predictive metrics, while CARA-T-inspired achieved the lowest defined proxy cost, retraining four fewer times. These are one-run results for the supplied ordered data, model, split, and proxy-cost definition; they do not prove chronological drift or real operational savings. Full aggregate and per-batch results are in `artifacts\cara_t\test_holdout\holdout_strategy_summary.csv` and `holdout_batch_results.csv`. The completed audit is in `holdout_audit_summary.csv`.

## Run from the workspace root

Install dependencies into the Python environment you intend to use:

```powershell
python -m pip install -r Covertype\requirements.txt
python Covertype\src\01_inspect_data.py
python Covertype\src\02_prepare_batches.py
python Covertype\src\03_train_baseline.py
python Covertype\src\08_evaluate_batches.py
python Covertype\src\04_compute_staleness.py
python Covertype\src\05_optimize_policy.py
python Covertype\src\06_evaluate_holdout.py
python Covertype\src\07_audit_results.py
```

The inspection, preparation, baseline training, calibration, replay, and audit are separate steps so that intermediate outputs can be reviewed and reproduced. Do not select a threshold using holdout results.

## Important files

- `src\covtype_pipeline.py`: shared paths, validation, sparse loading, RBF-cost, metrics, and replacement-training helpers.
- `src\01_inspect_data.py`: validates and summarizes the uploaded file.
- `src\02_prepare_batches.py`: writes the sparse processed data and 70/15/15 chronological-style splits.
- `src\03_train_baseline.py`: trains and saves Model V1.
- `src\04_compute_staleness.py`: estimates validation batch staleness and saves reusable reference state.
- `src\05_optimize_policy.py`: sequentially replays validation batches for candidate thresholds and a kappa sensitivity sweep.
- `src\06_evaluate_holdout.py`: replays four policies on batches 85–99, logs metrics/decisions/costs, and saves final models.
- `src\07_audit_results.py`: checks batch coverage, decision rules, and proxy-cost reconciliation.
- `src\08_evaluate_batches.py`: writes Model V1 metrics across all 100 batches.
- `artifacts\cara_t\test_holdout`: batch results, strategy summaries, audit, metadata, plots, and final policy models.

## Interpretation and limitations

- This is **CARA-T-inspired**, not the paper's complete optimizer.
- The RBF cost is an error-related proxy, not a pure feature-drift detector.
- Staleness uses true batch labels after prediction; a live system with delayed labels cannot make the same immediate decision.
- Retraining uses only the current batch, which can discard older patterns; compare rolling/accumulated training as future work.
- The supplied ordering is not verified as time, so conclusions are about this reproducible ordered replay rather than real-world temporal drift.
- A single dataset, model, split, and random seed do not establish statistical robustness or generality.
- Retraining time is recorded as an operational measurement, but the policy's `kappa` remains a proxy setting and is not inferred from that time.
- The final holdout is prequential: each batch is evaluated before its labels inform later model updates. It is not an untouched fixed test set after the replay begins.

Report actual generated results from `artifacts\cara_t\test_holdout\holdout_strategy_summary.csv`; do not pre-fill conclusions from the Electricity or Airlines runs.
