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
| F2 | 0.0882 | 0.0000 |
| Balanced Accuracy | 0.5808 | 0.4999 |
| **AUROC** | **0.7060** | **0.7709** |
| **AUPRC** | **0.0194** | **0.0510** |

### 1.2 Key Observations

1. **AUROC vs Recall paradox.** RF has higher AUROC (0.77) than LR (0.71) but 0% recall at threshold 0.5. This means RF ranks seizures higher than non-seizures, but its probability estimates are uncalibrated — the model "knows" something but can't express it as usable probabilities.

2. **AUPRC is the honest metric.** Both AUPRC values are extremely low (0.019 and 0.051). This reflects the fundamental challenge: with 0.55% prevalence in the test set, achieving high precision is nearly impossible for these models.

3. **Both models beat random.** Random AUROC = 0.5, random AUPRC = prevalence (~0.0055). Both models significantly exceed these baselines, confirming they learn genuine signal.

4. **Performance floor established.** Future models must exceed: AUROC > 0.77, AUPRC > 0.051 (RF benchmarks) to demonstrate progress.

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

## 2. Evaluation Artifacts

All evaluation outputs are saved to `experiments/baseline/evaluation/`:

* `all_metrics.json` — complete metrics for both models on all splits
* Per-model: `metrics.json`, `threshold_analysis_*.json`
* Plots: ROC curves, PR curves, confusion matrices (absolute + normalized), threshold analysis, model comparison bar charts

---

## 3. Future Results

This section will be expanded with:

* Sprint 1F: labeling strategy comparison
* Sprint 2B: event-level metrics (detection delay, false alarms per hour)
* Sprint 3E: classical vs deep learning comparison
