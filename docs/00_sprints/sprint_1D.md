# Sprint 1D — Baseline Models

## Status

Completed

---

## 1. Objective

Train the first seizure detection models using classical machine learning on the handcrafted feature dataset. These models serve as the performance baseline against which all future models (deep learning, transformers) will be compared.

---

## 2. Context

At this point the pipeline provides:

* A labeled feature dataset with 144 features per segment (`features.parquet`)
* A patient-independent split ensuring zero data leakage (`data/splits/`)
* 531,300 training samples (1,378 positive, 529,922 negative)

---

## 3. Models Trained

### 3.1 Logistic Regression

A linear model that learns a weighted combination of features and applies a sigmoid threshold to produce probabilities.

**Configuration:**
* Solver: L-BFGS
* Max iterations: 1000
* Class weight: balanced
* Feature scaling: StandardScaler

**Why include it:** Logistic Regression is the simplest possible baseline. It is fully interpretable (each feature gets a coefficient) and provides a lower bound on what any model should achieve. If a complex model cannot beat LR, something is wrong with the pipeline.

### 3.2 Random Forest

An ensemble of 200 decision trees that vote on the final prediction. Each tree is trained on a random subset of samples and features.

**Configuration:**
* Number of estimators: 200
* Class weight: balanced
* Parallelization: all available cores
* Feature scaling: StandardScaler (applied for consistency, not strictly required)

**Why include it:** Random Forest captures non-linear feature interactions that LR cannot. It also provides feature importance rankings, which will be useful for explainability (Sprint 4C).

---

## 4. Key Design Decisions

### 4.1 Class weight: balanced

With ~0.3% positive samples, an unweighted model would learn to always predict "no seizure" and achieve 99.7% accuracy while detecting zero seizures. `class_weight="balanced"` adjusts the loss function so that each positive sample contributes ~300× more than each negative sample.

### 4.2 StandardScaler before training

Features are normalized to zero mean and unit variance. This is critical for Logistic Regression (which is sensitive to feature scales) and harmless for Random Forest (which is scale-invariant).

### 4.3 Seed control

All randomness is controlled through a single seed (default: 42). This includes:
* `np.random.seed()` for global numpy state
* `random_state` parameter injected into every sklearn estimator

### 4.4 Prediction saving

Raw predictions (y_true, y_pred, y_proba) are saved as parquet files for each split. This decouples training from evaluation: Sprint 1E can compute any metric without re-running training.

---

## 5. Results

### 5.1 Logistic Regression

| Split | Accuracy | Recall | Precision | TP | FN | FP | TN |
|-------|----------|--------|-----------|-----|-----|-------|---------|
| Train | 0.9016 | 0.8454 | 0.0219 | 1165 | 213 | 52069 | 477853 |
| Val | 0.8628 | 0.1589 | 0.0061 | 75 | 397 | 12133 | 78722 |
| Test | 0.9251 | 0.2314 | 0.0178 | 109 | 362 | 6000 | 78426 |

### 5.2 Random Forest

| Split | Accuracy | Recall | Precision | TP | FN | FP | TN |
|-------|----------|--------|-----------|-----|-----|-------|---------|
| Train | 1.0000 | 1.0000 | 1.0000 | 1378 | 0 | 0 | 529922 |
| Val | 0.9951 | 0.0826 | 0.7358 | 39 | 433 | 14 | 90841 |
| Test | 0.9943 | 0.0000 | 0.0000 | 0 | 471 | 15 | 84411 |

---

## 6. Interpretation of Results

### 6.1 Logistic Regression

**Train recall = 84.5%** means LR learned a reasonable decision boundary on training data. However, **val recall drops to 15.9%** and **test recall = 23.1%**. This large gap indicates that LR's linear boundary does not generalize well across patients.

**Precision is extremely low** (~2% on train, <1% on val). This means the model generates thousands of false alarms for every true seizure detected. This is expected at the baseline stage and will improve with better models and post-processing.

### 6.2 Random Forest

**Perfect train performance** (100% recall, 100% precision) is classic overfitting: the trees memorize training samples. On unseen patients, **val recall = 8.3%** and **test recall = 0%** — the model essentially fails to detect any seizures on new patients.

This is a critical finding: Random Forest overfits severely to patient-specific patterns despite `class_weight="balanced"`. The model learns to recognize individual patients' seizure signatures rather than generalizable seizure features.

### 6.3 What these results mean

1. **Patient-independent evaluation is working correctly.** The train→test gap proves that the split is doing its job: preventing inflated metrics from patient leakage.

2. **Baseline performance is poor but expected.** With only 144 handcrafted features and extreme class imbalance, these results are consistent with the literature for patient-independent seizure detection.

3. **Random Forest overfitting motivates regularization.** Future sprints should explore: fewer trees, max depth limits, or feature selection.

4. **These results establish the floor.** Any model that cannot exceed ~23% test recall on this split is not learning generalizable seizure patterns.

---

## 7. Implementation

### 7.1 Core module

`src/neuro_eeg_cdss/training/trainer.py`

Key components:

* `TrainConfig` — model type, seed, scaling, hyperparameters
* `train_model()` — full training pipeline
* `predict()` — generate predictions from a trained result
* `save_train_result()` / `load_train_result()` — artifact serialization

### 7.2 Script

`scripts/training/train_baseline.py`

Usage:

```bash
python scripts/training/train_baseline.py
python scripts/training/train_baseline.py --seed 123
```

### 7.3 Output structure

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

### 7.4 Tests

`tests/test_trainer.py` — 14 tests covering:

* Feature/label separation
* LR and RF training
* Unsupported model rejection
* Empty dataset and no-positives handling
* Determinism
* Scaling toggle
* Prediction and missing-features detection
* Save/load round-trip

---

## 8. Limitations

* Only two model types evaluated
* No hyperparameter tuning performed
* No threshold optimization (default 0.5 decision boundary)
* No feature selection or dimensionality reduction
* Evaluation metrics are preliminary (full clinical evaluation in Sprint 1E)

---

## 9. Contribution to the Overall System

This sprint provides:

* The first working seizure detectors in the pipeline
* Saved prediction files that Sprint 1E will use for clinical evaluation
* Evidence that patient-independent evaluation produces honest metrics
* A performance floor for all future modeling work

---

## 10. Next Steps

Sprint 1E — Clinical Evaluation (window-level):

* Compute full clinical metrics: sensitivity, specificity, F1, AUROC
* Generate confusion matrices and ROC curves
* Analyze the trade-off between recall and false alarm rate
* Compare LR vs RF systematically
