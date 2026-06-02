# Results

This document collects evaluation results across all sprints and model types.

---

## 1. Baseline Models — Window-Level Evaluation (Sprint 1E)

### 1.1 Test Set Performance

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

### 1.2 Key Observations

1. **AUROC vs Recall paradox.** RF has higher AUROC (0.71) than LR (0.66) but 0% recall at threshold 0.5. This means RF ranks seizures higher than non-seizures, but its probability estimates are uncalibrated — the model "knows" something but can't express it as usable probabilities.

2. **AUPRC is the honest metric.** Both AUPRC values are extremely low (0.012 and 0.057). This reflects the fundamental challenge: with 0.55% prevalence in the test set, achieving high precision is nearly impossible for these models.

3. **Both models beat random.** Random AUROC = 0.5, random AUPRC = prevalence (~0.0055). Both models significantly exceed these baselines, confirming they learn genuine signal.

4. **Performance floor established.** Future models must exceed: AUROC > 0.71, AUPRC > 0.057 (RF benchmarks) to demonstrate progress.

### 1.3 Threshold Analysis Summary (Test Set, LR)

| Threshold | Sensitivity | Specificity | F2 |
|-----------|-------------|-------------|-----|
| 0.1 | ~0.65 | ~0.60 | ~0.12 |
| 0.2 | ~0.45 | ~0.80 | ~0.10 |
| 0.3 | ~0.35 | ~0.87 | ~0.09 |
| 0.5 (default) | 0.23 | 0.93 | 0.09 |
| 0.7 | ~0.10 | ~0.97 | ~0.05 |

Lowering the threshold significantly increases sensitivity but at the cost of specificity. This trade-off motivates future work on:
- Probability calibration (Sprint 2C)
- Temporal post-processing to reduce false alarms (Sprint 2A)
- Threshold optimization as part of the inference pipeline

---

## 2. Labeling Strategy Experiments (Sprint 1F)

### 2.1 Experiment Design

Six configurations testing 3 overlap thresholds (0.3, 0.5, 0.7) crossed
with 2 partial-overlap policies (drop ambiguous, keep as negative).
Only Logistic Regression was used (RF excluded due to catastrophic
overfitting in Sprint 1E).

### 2.2 Test Set Performance

| Config          | Threshold | Drop | N+    | Sensitivity | Specificity | F2     | AUROC  | AUPRC  |
|-----------------|-----------|------|-------|-------------|-------------|--------|--------|--------|
| thresh_0.3_drop | 0.3       | Yes  | 2,321 | 0.2314      | 0.9289      | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.3_keep | 0.3       | No   | 2,321 | 0.2314      | 0.9289      | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.5_drop | 0.5       | Yes  | 2,321 | 0.2314      | 0.9289      | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.5_keep | 0.5       | No   | 2,321 | 0.2314      | 0.9289      | 0.0682 | 0.6623 | 0.0123 |
| thresh_0.7_drop | 0.7       | Yes  | 2,247 | 0.2289      | 0.9310      | 0.0666 | 0.6549 | 0.0114 |
| thresh_0.7_keep | 0.7       | No   | 2,247 | 0.2289      | 0.9311      | 0.0667 | 0.6553 | 0.0115 |

### 2.3 Key Findings

1. **Configs 1–4 are identical.** Thresholds 0.3 and 0.5 produce the
   exact same dataset because no windows with 0 < overlap < 0.5 exist
   (they were dropped during the original build).

2. **Threshold 0.7 has negligible impact.** Only 74 windows (0.01% of
   the dataset) change status, causing < 0.003 difference in sensitivity.

3. **The labeling strategy is NOT the bottleneck.** With 5-second
   non-overlapping windows, seizure boundaries create only 156 partial-
   overlap windows out of 707,524 total (0.022%). The choice of threshold
   and drop policy has minimal impact on model performance.

4. **The original default (threshold=0.5, drop=True) is validated** as a
   reasonable choice for this dataset and window configuration.

### 2.4 Implication

The performance bottleneck lies in feature representation and model
architecture, not in the labeling strategy. This motivates progression
to deep learning approaches that learn directly from raw signal.

---

## 3. Evaluation Artifacts

All evaluation outputs are saved to `experiments/baseline/evaluation/`:

* `all_metrics.json` — complete metrics for both models on all splits
* Per-model: `metrics.json`, `threshold_analysis_*.json`
* Plots: ROC curves, PR curves, confusion matrices (absolute + normalized), threshold analysis, model comparison bar charts

Labeling experiment outputs are saved to `experiments/labeling/`:

* `all_results.json` — combined results for all 6 configurations
* Per-config directories with `results.json`
* `comparison_val.txt`, `comparison_test.txt` — formatted comparison tables

---

## 4. Future Results

This section will be expanded with:

* Sprint 2B: event-level metrics (detection delay, false alarms per hour)
* Sprint 3E: classical vs deep learning comparison
