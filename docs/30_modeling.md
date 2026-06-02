# Modeling

This document tracks all modeling decisions, architectures, and training configurations used across sprints.

---

## 1. Classical ML Baseline (Sprint 1D)

### 1.1 Training Pipeline

The training pipeline (`src/neuro_eeg_cdss/training/trainer.py`) is designed to be model-agnostic. It separates concerns into:

* **Feature/label separation** — Automatically identifies feature columns (everything not in the metadata or label set).
* **Optional scaling** — StandardScaler fitted on training data, applied consistently at inference.
* **Model fitting** — Factory pattern supporting multiple sklearn estimators.
* **Artifact serialization** — Full round-trip save/load of model, scaler, config, and metadata.

### 1.2 Models

#### Logistic Regression

A linear model that learns a weighted combination of features and applies a sigmoid threshold. Serves as the simplest possible baseline — any model that cannot beat LR has a fundamental problem.

| Parameter | Value |
|-----------|-------|
| Solver | L-BFGS |
| Max iterations | 1000 |
| Class weight | balanced |
| Scaling | StandardScaler |

#### Random Forest

An ensemble of 200 decision trees that vote on the final prediction. Captures non-linear feature interactions and provides feature importance rankings.

| Parameter | Value |
|-----------|-------|
| n_estimators | 200 |
| Class weight | balanced |
| n_jobs | -1 (all cores) |
| Scaling | StandardScaler (consistency) |

### 1.3 Class Imbalance Handling

With ~0.3% positive samples, `class_weight="balanced"` adjusts the loss function so that each positive sample contributes ~300x more than each negative sample. Without this, a model would learn to always predict "no seizure" and achieve 99.7% accuracy while detecting zero seizures.

### 1.4 Reproducibility

All randomness is controlled through a single seed (default: 42):

* `np.random.seed()` for global numpy state
* `random_state` parameter injected into every sklearn constructor

### 1.5 Prediction Decoupling

Raw predictions (y_true, y_pred, y_proba) are saved as parquet files per split. This decouples training from evaluation: subsequent sprints can compute any metric without re-running training.

### 1.6 Baseline Results — Preliminary (Sprint 1D)

| Model | Split | Accuracy | Recall | Precision |
|-------|-------|----------|--------|-----------|
| Logistic Regression | Train | 0.9016 | 0.8454 | 0.0219 |
| Logistic Regression | Val | 0.8628 | 0.1589 | 0.0061 |
| Logistic Regression | Test | 0.9251 | 0.2314 | 0.0178 |
| Random Forest | Train | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | Val | 0.9951 | 0.0826 | 0.7358 |
| Random Forest | Test | 0.9943 | 0.0000 | 0.0000 |

### 1.7 Clinical Evaluation — Full Metrics (Sprint 1E)

**Test set performance:**

| Metric | Logistic Regression | Random Forest |
|--------|:-------------------:|:-------------:|
| Sensitivity | 0.2335 | 0.0000 |
| Specificity | 0.9281 | 0.9998 |
| Precision | 0.0178 | 0.0000 |
| NPV | 0.9954 | 0.9945 |
| F1 | 0.0331 | 0.0000 |
| F2 | 0.0682 | 0.0000 |
| Balanced Accuracy | 0.5808 | 0.4999 |
| **AUROC** | **0.6622** | **0.7104** |
| **AUPRC** | **0.0123** | **0.0572** |

**Key findings:**

* **LR generalizes poorly but learns something.** Train recall 84.5% drops to test 23.1%, indicating the linear boundary doesn't transfer across patients.
* **RF overfits severely.** Perfect train performance collapses to 0% test recall. The trees memorize patient-specific seizure signatures rather than generalizable patterns.
* **RF discriminates better than LR.** Despite 0% recall at threshold 0.5, RF has higher AUROC (0.71 vs 0.66) and AUPRC (0.057 vs 0.012). The model ranks seizures above non-seizures but its probabilities are uncalibrated.
* **Both models beat random.** Random AUROC = 0.5, random AUPRC ≈ 0.006. Both models significantly exceed these baselines.
* **AUPRC is the honest metric.** Both values are very low, reflecting the fundamental challenge of achieving useful precision with 0.55% prevalence.
* **Performance floor established.** Future models must exceed AUROC > 0.71 and AUPRC > 0.057 to demonstrate progress.

### 1.7 Artifact Structure

```text
experiments/baseline/
├── all_results.json
├── logistic_regression/
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── train_config.json
│   ├── feature_names.json
│   ├── train_metadata.json
│   ├── quick_results.json
│   ├── predictions_train.parquet
│   ├── predictions_val.parquet
│   └── predictions_test.parquet
└── random_forest/
    └── (same structure)
```

---

## 2. Future Models

Planned in upcoming sprints:

* **Sprint 3A–3B:** 1D CNN on raw signal (PyTorch)
* **Sprint 3C:** Sequence models (LSTM/GRU)
* **Sprint 3D:** Transformer-based temporal encoder
* **Sprint 3E:** Systematic comparison across all model families
