# Sprint 1E — Clinical Evaluation (Window-Level)

## Status

Completed

---

## 1. Objective

Evaluate the baseline seizure detection models using clinically relevant metrics. This sprint transforms raw predictions (saved in Sprint 1D) into a comprehensive analysis that answers: *How well do these models actually detect seizures, and what are the trade-offs?*

---

## 2. Context

Sprint 1D produced:

* Saved prediction files (y_true, y_pred, y_proba) for both LR and RF on train/val/test splits
* Preliminary metrics (accuracy, recall, precision) that are insufficient for clinical evaluation

This sprint adds:

* Threshold-independent metrics (AUROC, AUPRC)
* Clinically prioritized metrics (sensitivity, specificity, F2, NPV)
* Visual analysis (ROC curves, PR curves, confusion matrices)
* Threshold analysis (sensitivity-specificity trade-off at different operating points)

---

## 3. Metrics Computed

### 3.1 Core Clinical Metrics

| Metric | Formula | Clinical Meaning |
|--------|---------|------------------|
| **Sensitivity** | TP / (TP + FN) | Proportion of real seizures detected. The most critical metric — a missed seizure can be life-threatening. |
| **Specificity** | TN / (TN + FP) | Proportion of non-seizure windows correctly identified. Controls alarm fatigue. |
| **Precision (PPV)** | TP / (TP + FP) | Proportion of alarms that are real seizures. Low precision means many false alarms. |
| **NPV** | TN / (TN + FN) | Proportion of non-alarms that are truly non-seizure. High NPV means the system is safe when it stays quiet. |
| **F1** | Harmonic mean of precision and sensitivity | Balanced trade-off between precision and recall. |
| **F2** | Weighted harmonic mean (beta=2) | Favors sensitivity over precision. More appropriate for seizure detection where missing a seizure is worse than a false alarm. |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 | Not inflated by the majority class, unlike raw accuracy. |

### 3.2 Threshold-Independent Metrics

| Metric | Meaning |
|--------|---------|
| **AUROC** | Area under the ROC curve. Measures discrimination ability across all thresholds. A random model scores 0.5. |
| **AUPRC** | Area under the Precision-Recall curve. More informative than AUROC for imbalanced datasets because it is sensitive to the minority class performance. |

### 3.3 Why These Metrics Matter

With ~0.3% prevalence:

* **Accuracy is misleading.** A model predicting "no seizure" 100% of the time achieves 99.7% accuracy. Balanced accuracy fixes this.
* **AUROC can be optimistic.** With massive class imbalance, even small FPR improvements look good on the ROC curve. AUPRC is the more honest metric.
* **F2 > F1.** In a clinical setting, a false negative (missed seizure) is far more dangerous than a false positive (false alarm). F2 with beta=2 weights sensitivity 4× more than precision.

---

## 4. Results

### 4.1 Logistic Regression — Full Clinical Metrics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Sensitivity | 0.8447 | 0.1589 | 0.2335 |
| Specificity | 0.9017 | 0.8663 | 0.9281 |
| Precision | 0.0219 | 0.0061 | 0.0178 |
| NPV | 0.9996 | 0.9956 | 0.9954 |
| F1 | 0.0427 | 0.0118 | 0.0331 |
| F2 | 0.1435 | 0.0331 | 0.0882 |
| Balanced Accuracy | 0.8732 | 0.5126 | 0.5808 |
| AUROC | 0.9479 | 0.5923 | 0.7060 |
| AUPRC | 0.0759 | 0.0065 | 0.0194 |

### 4.2 Random Forest — Full Clinical Metrics

| Metric | Train | Val | Test |
|--------|-------|-----|------|
| Sensitivity | 1.0000 | 0.0826 | 0.0000 |
| Specificity | 1.0000 | 0.9998 | 0.9998 |
| Precision | 1.0000 | 0.7358 | 0.0000 |
| NPV | 1.0000 | 0.9953 | 0.9945 |
| F1 | 1.0000 | 0.1486 | 0.0000 |
| F2 | 1.0000 | 0.0992 | 0.0000 |
| Balanced Accuracy | 1.0000 | 0.5412 | 0.4999 |
| AUROC | 1.0000 | 0.8330 | 0.7709 |
| AUPRC | 1.0000 | 0.1361 | 0.0510 |

---

## 5. Interpretation

### 5.1 AUROC Tells a Different Story Than Recall

On the test set, **RF has higher AUROC (0.77) than LR (0.71)**. This means that *if we optimize the threshold*, RF can actually discriminate seizures better than LR. The problem is that the default 0.5 threshold is completely wrong for RF — its probability distribution is so skewed by overfitting that almost no test sample gets a probability above 0.5.

**Key insight:** AUROC measures the model's ranking ability, not its calibration. RF ranks seizure windows higher than non-seizure windows (on average), but its probability estimates are unreliable.

### 5.2 AUPRC Reveals the Real Challenge

AUPRC on test is 0.019 (LR) and 0.051 (RF). Both are extremely low. For context, a random model would score ~0.003 (the prevalence). So the models are doing better than random, but not by much.

**Why AUPRC is so low:** With 471 seizures in ~85,000 test windows, even a small FPR translates to thousands of false alarms. To achieve useful precision, the model would need to be nearly perfect at distinguishing seizure from non-seizure.

### 5.3 Specificity Is Deceptively High

Both models show >92% specificity on test. But with 84,426 negative windows, even 93% specificity means ~5,900 false alarms per LR evaluation — that is roughly one false alarm every 5 seconds in a real monitoring scenario.

### 5.4 NPV Is Near-Perfect (But Meaningless)

NPV > 99.5% for both models. This is purely an artifact of prevalence: if you predict "no seizure" for everything, NPV is already 99.7%. It's not a meaningful metric in this context.

### 5.5 F2 vs F1

LR test F2 (0.088) is higher than F1 (0.033) because F2 gives more weight to sensitivity. For a seizure detection system, F2 is the more relevant metric — but both are very low at baseline.

### 5.6 Threshold Optimization Potential

The threshold analysis reveals that lowering the decision threshold from 0.5 to ~0.2-0.3 could significantly increase sensitivity at the cost of specificity. This is a design choice: in ICU monitoring, high sensitivity with more false alarms may be preferable to missing seizures.

---

## 6. Visualizations Generated

```text
experiments/baseline/evaluation/
├── all_metrics.json                              # Combined metrics for both models
├── logistic_regression/
│   ├── metrics.json                              # Full metrics per split
│   ├── threshold_analysis_val.json               # Metrics at each threshold (val)
│   ├── threshold_analysis_test.json              # Metrics at each threshold (test)
│   └── plots/
│       ├── confusion_matrix_train.png
│       ├── confusion_matrix_val.png
│       ├── confusion_matrix_test.png
│       ├── confusion_matrix_normalized_train.png
│       ├── confusion_matrix_normalized_val.png
│       ├── confusion_matrix_normalized_test.png
│       ├── threshold_analysis_val.png
│       └── threshold_analysis_test.png
├── random_forest/
│   └── (same structure)
└── comparison/
    ├── roc_comparison_val.png
    ├── roc_comparison_test.png
    ├── pr_comparison_val.png
    ├── pr_comparison_test.png
    ├── model_comparison_val.png
    └── model_comparison_test.png
```

---

## 7. Implementation

### 7.1 Core module

`src/neuro_eeg_cdss/evaluation/metrics.py`

Key components:

* `WindowMetrics` — frozen dataclass holding all 19 metric fields
* `compute_window_metrics()` — computes all metrics from (y_true, y_pred, y_proba)
* `compute_roc_curve()` / `compute_pr_curve()` — curve data for plotting
* `compute_threshold_analysis()` — metrics at configurable thresholds
* `format_metrics_report()` — human-readable multi-split report

### 7.2 Visualization module

`src/neuro_eeg_cdss/evaluation/plots.py`

Key components:

* `plot_roc_curves()` — overlayed ROC curves with AUROC annotations
* `plot_pr_curves()` — overlayed PR curves with prevalence baseline
* `plot_confusion_matrix()` — absolute and normalized versions
* `plot_threshold_analysis()` — sensitivity/specificity/F2 vs threshold
* `plot_model_comparison()` — grouped bar chart of key metrics

### 7.3 Script

`scripts/evaluation/evaluate_baseline.py`

Usage:

```bash
python scripts/evaluation/evaluate_baseline.py
python scripts/evaluation/evaluate_baseline.py --no-plots  # metrics only
```

### 7.4 Tests

* `tests/test_metrics.py` — 23 tests covering metric computation, edge cases, curves, threshold analysis, and report formatting
* `tests/test_plots.py` — 8 tests verifying all plot types generate and save correctly

---

## 8. Limitations

* Evaluation is window-level only. Event-level evaluation (Sprint 2B) will provide clinically more relevant metrics like detection delay and false alarms per hour.
* Threshold optimization is exploratory. No systematic optimal threshold selection was performed.
* All metrics use the default 0.5 threshold for binary predictions. The threshold analysis shows better operating points exist.
* No confidence intervals or statistical significance tests. Would require cross-validation or bootstrap.

---

## 9. Contribution to the Overall System

This sprint provides:

* The first rigorous clinical assessment of the seizure detection pipeline
* Evidence that both baseline models perform poorly on patient-independent evaluation (as expected)
* AUROC evidence that models learn *something* useful (above random), even if the default threshold is suboptimal
* A reusable evaluation framework that all future models will use
* Visual artifacts ready for documentation and potential publication
* Threshold analysis that motivates Sprint 2A (temporal post-processing) and Sprint 2C (probability calibration)

---

## 10. Next Steps

Sprint 1F — Labeling/Threshold Experiments:

* Compare systematic labeling strategies (30%, 50%, 70% overlap thresholds)
* Compare ambiguous window policies (drop vs keep as negative)
* Analyze impact on all clinical metrics
* This is the most paper-worthy experimental sprint in Block 1
