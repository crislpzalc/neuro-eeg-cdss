# Sprint 2C — Probability Calibration

## Objective

Make model output probabilities *interpretable*. When an uncalibrated
model says "0.3 probability of seizure," that number is often
meaningless — it doesn't mean 30% of similar cases are actually
seizures. Calibration corrects the probability scale so the model's
confidence matches observed reality.

## Motivation

Two specific problems from previous sprints motivate this:

1. **Random Forest's probability paradox**: RF has AUROC 0.71 (good
   discrimination) but 0% test recall at threshold 0.5. Its
   probabilities are compressed near zero — the model "knows" which
   windows are more suspicious, but can't express it as useful
   probabilities. Calibration can fix this.

2. **Threshold sensitivity**: Sprint 1E showed that lowering the
   threshold from 0.5 to 0.1 dramatically changes behavior. With
   calibrated probabilities, threshold selection becomes principled
   rather than arbitrary — a 0.3 threshold means "flag anything with
   ≥30% chance of being a seizure."

## Key Concepts

### Platt Scaling

Fits a logistic regression on the uncalibrated probabilities:

```
P_calibrated = sigmoid(a * P_uncalibrated + b)
```

Only 2 parameters (a, b), making it very resistant to overfitting.
Best when the calibration curve is roughly sigmoid-shaped.

### Isotonic Regression

Non-parametric monotone fit: learns a step function that maps
uncalibrated → calibrated probabilities. More flexible than Platt
but can overfit with small calibration sets. Our validation set
(91K+ samples) is large enough to avoid this.

### Calibration Metrics

- **ECE** (Expected Calibration Error): Divide [0,1] into bins. For
  each bin, measure |mean predicted - observed frequency|. ECE is the
  weighted average across bins. Lower = better calibrated.
- **MCE** (Maximum Calibration Error): Worst-case bin error.
- **Brier Score**: Mean squared error of probabilities = mean((p - y)²).
  Decomposes into calibration error + discrimination + uncertainty.
- **Log Loss**: Cross-entropy. Heavily penalizes confident wrong
  predictions (saying 0.99 when the truth is 0).

### Reliability Diagram

A plot with:
- **X-axis**: Mean predicted probability per bin
- **Y-axis**: Observed fraction of positives per bin
- **Diagonal**: Perfect calibration line (predicted = observed)

Points above the diagonal = model under-confident (says 0.3 but
truth is 0.5). Points below = model over-confident.

## Design Decisions

### D1: Fit on validation, evaluate on test

Calibrators are fitted exclusively on validation set predictions and
then evaluated on the test set. This prevents data leakage — the
calibrator never sees test data during fitting. This is the standard
protocol in the literature.

### D2: Two methods (Platt + Isotonic)

Both are implemented because they represent different trade-offs:
- **Platt** = 2 parameters, simple, robust, good default
- **Isotonic** = non-parametric, flexible, needs more data

With 91K validation samples, both should work well. Comparing them
reveals whether the calibration curve is simple (Platt enough) or
complex (isotonic needed).

### D3: Equal-width bins for ECE

We use equal-width bins (dividing [0,1] into 10 equal intervals)
rather than equal-count bins. This is the standard approach in the
literature and matches the reliability diagram visualization.

### D4: Save calibrated prediction parquets

Calibrated predictions are saved as separate parquet files, preserving
the original uncalibrated predictions. This enables downstream
analysis (e.g., re-running event-level evaluation with calibrated
probabilities) without retraining.

## Deliverables

- `src/neuro_eeg_cdss/calibration/calibrator.py` — core module
- `src/neuro_eeg_cdss/calibration/plots.py` — reliability diagrams
- `scripts/calibration/run_calibration.py` — experiment script
- `tests/test_calibration.py` — tests
- `experiments/calibration/` — result artifacts

## How to Run

### Tests
```bash
python -m pytest tests/test_calibration.py -v
```

### Experiment
```bash
python scripts/calibration/run_calibration.py
```

## Commit Message

```
Sprint 2C: Probability calibration — Platt scaling + isotonic regression with ECE/Brier metrics and reliability diagrams
```
