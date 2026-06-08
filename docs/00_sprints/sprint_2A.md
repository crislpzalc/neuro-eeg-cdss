# Sprint 2A — Temporal Post-Processing

## Objective

Improve the temporal coherence of seizure predictions by applying
post-processing filters that exploit the fact that real seizures
persist across multiple consecutive EEG windows while false positives
tend to be isolated.

## Motivation

In Sprint 1E, the Logistic Regression model achieved 23.4% sensitivity
and 92.8% specificity at threshold 0.5. Many false positives are
isolated single-window detections that a neurologist would immediately
dismiss — real seizures last seconds to minutes (2–12+ consecutive
5-second windows). Temporal post-processing can filter these out.

## Strategies Implemented

### 1. Median Filter (`median_filter`)

Replaces each window's predicted probability with the median of its
local neighborhood (kernel_size windows). Isolated probability spikes
are suppressed because the median of surrounding low-probability
windows dominates.

### 2. Moving Average (`moving_average`)

Replaces each probability with the mean of its neighborhood. Provides
smoother transitions but is less effective at completely eliminating
isolated spikes compared to the median filter.

### 3. Minimum Duration Filter (`min_duration`)

Operates on binary predictions rather than probabilities. Removes
positive detection "runs" (consecutive positive windows) shorter than
min_windows. This directly encodes the clinical prior that seizures
have a minimum duration.

## Design Decisions

- **Per-recording processing**: Filters are applied within each
  recording independently. Temporal context never crosses recording
  boundaries, which would be physiologically meaningless.

- **Prediction enrichment**: Current prediction files store only
  (y_true, y_pred, y_proba). The `enrich_predictions` utility joins
  predictions with feature metadata (subject, path, start_sec) by
  positional alignment, validated by checking y_true matches labels.

- **Non-destructive**: Original predictions are preserved in the
  output DataFrame alongside post-processed columns (y_pred_post,
  y_proba_post), enabling direct before/after comparison.

## Configurations Tested

| Config | Strategy | Parameters |
|--------|----------|------------|
| median_k3_t0.5 | Median filter | kernel=3, threshold=0.5 |
| median_k5_t0.5 | Median filter | kernel=5, threshold=0.5 |
| median_k7_t0.5 | Median filter | kernel=7, threshold=0.5 |
| mavg_k3_t0.5 | Moving average | kernel=3, threshold=0.5 |
| mavg_k5_t0.5 | Moving average | kernel=5, threshold=0.5 |
| mindur_w2 | Min duration | min_windows=2 |
| mindur_w3 | Min duration | min_windows=3 |
| mindur_w4 | Min duration | min_windows=4 |

## Results — Test Set

| Config | Sensitivity | Specificity | F2 | Changed |
|--------|-------------|-------------|------|---------|
| **Baseline** | **0.2335** | **0.9281** | **0.0682** | — |
| median_k3_t0.5 | 0.2166 | 0.9426 | 0.0746 | 2,679 |
| median_k5_t0.5 | 0.1996 | 0.9504 | 0.0762 | 3,694 |
| median_k7_t0.5 | 0.1890 | 0.9549 | 0.0770 | 4,078 |
| mavg_k3_t0.5 | 0.2166 | 0.9444 | 0.0764 | 3,265 |
| mavg_k5_t0.5 | 0.1890 | 0.9501 | 0.0720 | 3,983 |
| mindur_w2 | 0.1890 | 0.9511 | 0.0729 | 1,958 |
| mindur_w3 | 0.1550 | 0.9657 | 0.0752 | 3,212 |
| **mindur_w4** | **0.1359** | **0.9739** | **0.0771** | 3,911 |

## Key Findings

### 1. All strategies improve F2

Every post-processing configuration improved F2 over the baseline
(0.0682). The best improvement was mindur_w4 at F2=0.0771 (+13.0%).
This confirms that removing isolated false positives, even at the
cost of some sensitivity, improves the clinical utility metric.

### 2. Sensitivity-specificity trade-off is consistent

All strategies reduce sensitivity and increase specificity. This is
expected: filtering removes both true and false isolated detections.
The net effect on F2 is positive because the baseline had far more
false positives than true positives.

### 3. Minimum duration is the most effective strategy

The `min_duration` filter achieves the best specificity improvements
(+4.6% with mindur_w4) with the clearest clinical interpretation:
"a detection must persist for at least N consecutive windows."
With 5-second windows, mindur_w4 means "at least 20 seconds of
sustained activity" — a reasonable clinical criterion.

### 4. Median filter provides the best balance

Among smoothing strategies, median_k7 achieves the highest F2 on test
(0.0770), nearly matching mindur_w4 (0.0771). The median filter has
the advantage of working on probabilities rather than hard decisions,
making it composable with threshold tuning.

## Deliverables

- `src/neuro_eeg_cdss/postprocessing/__init__.py`
- `src/neuro_eeg_cdss/postprocessing/temporal.py` — core module
- `scripts/postprocessing/run_temporal_postprocessing.py` — experiment script
- `tests/test_temporal_postprocessing.py` — 52 tests
- `experiments/postprocessing/` — result artifacts

## Commit Message

```
Sprint 2A: Temporal post-processing — 3 strategies (median, moving avg, min duration) improve F2 by up to 13% on test set
```
