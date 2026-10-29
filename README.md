# Cost-Aware Model Retraining Experiments

This repository contains three experiments about **when to retrain an existing
machine-learning model as new batches of data arrive**:

1. **Electricity** — a binary price-movement classifier and batch replay.
2. **Airlines** — a flight-delay CARA-T research notebook with offline threshold
   selection and online policy comparisons.
3. **CoverType** — a binary forest-cover classifier porting the batch workflow
   to the supplied LIBSVM data.

The experiments compare keeping a model, retraining on every batch, periodic
retraining, and a cost-aware threshold policy. They are research/replay
workflows, not production deployments. The threshold policy is CARA-T-inspired;
the implementations are not exact reproductions of the full published method.

## Repository contents

| Path | Purpose |
|---|---|
| `Electricity\electricity-cara\` | Electricity training, monitoring, sequential replay, policy comparison, and audit scripts |
| `AirLine.ipynb` | Executed Airlines research notebook; experiment results and plots are embedded in the notebook |
| `Covertype\` | CoverType pipeline, scripts, project documentation, and saved results |
| `Covertype\artifacts\` | CoverType baseline, validation, holdout, monitoring, and audit outputs |
| `Electricity\electricity-cara\artifacts\` | Electricity models, monitoring outputs, policy results, and holdout audit |

More detailed project-specific instructions are in the [Electricity README](Electricity/README.md)
and [CoverType README](Covertype/README.md).

## Shared experimental workflow

Each workflow follows the same broad lifecycle:

1. Divide the dataset into ordered batches and define training, validation,
   holdout, and/or online phases.
2. Train and save a baseline model (Model V1).
3. Measure batch-level predictive performance and estimate error-based
   staleness using an RBF similarity measure.
4. Choose a retraining threshold on offline/validation data, using an assumed
   retraining penalty, `kappa` (κ).
5. Replay policies in test-then-train order: evaluate the current batch before
   using its labels to retrain a model for later batches.
6. Compare prediction metrics, retraining counts, and the experiment's proxy
   cost; audit saved decisions and calculations where the project provides an
   audit.

The reported proxy costs are **not monetary, energy, or measured compute costs**.
They depend on project-specific assumptions and are not directly comparable
between datasets. Since the staleness calculation uses labeled examples, these
experiments also do not demonstrate immediate retraining decisions where labels
are delayed.

## 1. Electricity

The Electricity experiment uses 45,312 records in 100 ordered batches. It
trains a neural-network binary classifier on six standardized features and
compares four policies on batches 85–99. Its data split is batches 0–69 for
training, 70–84 for validation, and 85–99 for holdout. Earlier exploratory
analysis included inspection of holdout data, so the holdout should not be
described as completely untouched throughout the project.

### Recorded holdout results

| Policy | Retrains | Accuracy | F1 | Total proxy cost |
|---|---:|---:|---:|---:|
| Never retrain | 0 | 78.96% | 0.7860 | 1.29281 |
| Always retrain | 15 | 81.07% | 0.7900 | 1.20000 |
| Periodic (every 3 batches) | 5 | 80.42% | 0.7644 | **1.13004** |
| CARA-T-inspired | 14 | **81.17%** | **0.7914** | 1.16573 |

In this recorded run, periodic retraining had the lowest proxy cost, while the
CARA-T-inspired policy had the strongest accuracy and F1, by a small margin.
The CARA-T-inspired policy retrained on 14 of 15 batches, so this run did not
show substantial retraining reduction relative to always retraining.

### Run the integrated evaluation

From the repository root in PowerShell:

```powershell
python -m pip install -r Electricity\electricity-cara\requirements.txt
Set-Location Electricity\electricity-cara
python -u evaluate_test_holdout.py
python audit_test_holdout.py
```

The evaluation writes policy results and audit files under
`Electricity\electricity-cara\artifacts\cara_t\test_holdout\`. See the
[Electricity README](Electricity/README.md) for individual scripts and
interpretation notes.

## 2. Airlines

The Airline experiment is in [`AirLine.ipynb`](AirLine.ipynb). The checked-in
notebook contains saved outputs from an execution: it validates the input
batches, checks the scikit-learn version against the saved Model V1 metadata,
recomputes the baseline holdout metrics, runs the cost sensitivity experiment,
and reports an audit/leakage check. Results below are the recorded notebook
outputs, not a fresh run performed by this README update.

The notebook uses 539,383 records arranged as 100 batch CSV files. Model V1 and
offline threshold selection use batches 0–21; batches 22–24 form a baseline
holdout; batches 25–99 are used for online replay. Model V1 is a scikit-learn
pipeline with a `ColumnTransformer`, one-hot encoding, and a
`RandomForestClassifier`. The notebook compares CARA-T with Never Retrain,
Markov, Always Retrain, Periodic Retraining, and a retrospective Oracle.

### Recorded findings

At the illustrative/project-selected `kappa = 26.1941`, the notebook records
`tau* = 17.9245`. Over the 75 online batches:

| Strategy | Retrains | Total strategy cost | Average query accuracy |
|---|---:|---:|---:|
| CARA-T | 40 | 1,438.85 | 64.10% |
| Never Retrain | 0 | 2,376.29 | 60.30% |
| Markov | 32 | 1,544.74 | 63.33% |
| Always Retrain | 75 | 1,964.56 | 63.75% |
| Periodic | 9 | 2,145.75 | 61.45% |
| Oracle (retrospective) | 28 | 1,367.58 | 64.48% |

The sensitivity sweep is the notebook's primary result: at lower assumed
retraining cost the selected policy retrained more often (67 online retrains at
`kappa = 6.549`); at higher assumed costs it retrained less often (one at
`kappa = 104.777` and above in the recorded sweep). The Oracle uses future
online labels and is a retrospective reference, not a deployable policy.

### Running the notebook

The notebook is configured for **Kaggle**, not direct execution from the
Windows repository root. It expects Kaggle's `/kaggle/input` and
`/kaggle/working` directories, the prepared 100 batch CSV files, and the
previously generated Model V1 pipeline and metadata/reference artifacts. It
also checks that the installed scikit-learn version matches the version used
to serialize Model V1 (the recorded run used scikit-learn 1.6.1). Attach or
restore those inputs in Kaggle before running the notebook top to bottom. Its
generated result files are written under `/kaggle/working/results`; they are
not a separate checked-in artifact directory in this repository.

The batch CSVs contain no timestamps. The notebook validates their file index
and replays that order, but the order alone does not establish real arrival
chronology or prove concept drift. Its `kappa` values are illustrative, not
measured operating costs.

## 3. CoverType

CoverType ports the batch/retraining workflow to the supplied
`covtype.libsvm.binary.scale.bz2` data. This is a binary dataset variant with
581,012 rows and 54 already-scaled features; labels 1 and 2 are mapped to
binary targets 0 and 1. It is not the original seven-class task. The source
contains no timestamps, so the pipeline preserves file order for reproducible
splits and does not claim verified chronology or temporal drift.

The split is batches 0–69 for training (406,708 rows), 70–84 for validation
(87,152 rows), and 85–99 for holdout replay (87,152 rows). Model V1 is a Keras
binary classifier. Threshold selection uses sequential validation replay and
a `kappa` sensitivity sweep.

### Recorded holdout results

| Policy | Retrains | Accuracy | F1 | Total proxy cost |
|---|---:|---:|---:|---:|
| Never retrain | 0 | 63.85% | 0.4988 | 2.9384 |
| Always retrain | 15 | **73.54%** | **0.6728** | 2.4628 |
| Periodic (every 3 batches) | 5 | 64.87% | 0.5628 | 2.8059 |
| CARA-T-inspired | 11 | 71.48% | 0.6461 | **2.1887** |

For this run, always retraining achieved the strongest predictive metrics;
CARA-T-inspired retraining had the lowest defined proxy cost, retraining four
fewer times. The recorded audit passed the checks for batch coverage, policy
decisions, model-version progression, and proxy-cost reconciliation. These
results are specific to this dataset order, split, model, and proxy-cost
definition.

### Reproduce the CoverType pipeline

The source dataset is expected at the repository root as
`covtype.libsvm.binary.scale.bz2`. Install the project dependencies, then run
the pipeline from the repository root:

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

Threshold selection must use validation data, not holdout results. Processed
data, models, per-batch results, plots, and audit summaries are saved under
`Covertype\data\processed\` and `Covertype\artifacts\`. More detail is in the
[CoverType README](Covertype/README.md).

## Limitations and interpretation

- The ordered-batch replays are simulations. Where source data lacks timestamps,
  file order is not proof of actual time or drift.
- Error-based staleness requires labels after prediction; production use would
  need to account for label delay.
- `kappa` and proxy costs are assumptions, not measured financial, compute, or
  energy costs. Costs across Electricity, Airlines, and CoverType are on
  different scales and should not be compared numerically.
- Several workflows retrain from the current batch only. This can discard
  useful history and should be compared with rolling-window or cumulative
  training.
- Results come from one recorded run/seed per project and do not establish
  statistical robustness or generality.
- The repository demonstrates experimental pipelines and sequential
  evaluation, not continuous ingestion, production serving, candidate-model
  approval, alerting, or rollback.

## Data and generated files

Some source datasets and archives are local inputs and are excluded by
`.gitignore`. Restore the appropriate dataset before rerunning a workflow.
The Electricity and CoverType repositories include saved models and experiment
outputs; the Airline notebook has its recorded results embedded in the
notebook and writes generated files to its Kaggle working directory when run.
