# Sprint 1C — Patient-Independent Split

## Status

Completed

---

## 1. Objective

Create a train/validation/test split at the subject level that prevents data leakage between sets. This ensures that the model is evaluated on patients it has never seen during training, which is the only valid evaluation protocol for patient-independent seizure detection.

---

## 2. Context

### Why patient-independent splitting is critical

EEG signals are highly patient-specific. Each individual has a distinctive background pattern that a model can learn to recognize. If segments from the same patient appear in both training and test sets, the model may achieve high accuracy by recognizing individual patients rather than by detecting seizure patterns that generalize across patients.

This form of data leakage inflates performance metrics and produces models that fail in clinical deployment, where every new patient is unseen. Patient-independent evaluation is a strict requirement for clinical credibility and for publication in the medical AI literature.

### Dataset characteristics

From the pre-split inspection (Sprint 1B / inspection script):

* 23 subjects, all with at least some positive (seizure) segments
* Total positive segments: 2,321 out of 707,524 (~0.3%)
* Positive segments are distributed unevenly: the top 3 subjects account for ~37.7% of all positives
* No methodological risk flags were raised
* Recommendation: dataset is compatible with a subject-level hold-out split

---

## 3. Splitting Strategy

### 3.1 Hold-out split

A fixed hold-out split was chosen over cross-validation for the baseline phase.

**Rationale:**

* Simpler to implement and reason about
* Sufficient for initial model development and debugging
* Cross-validation can be added later for final evaluation and paper results

### 3.2 Target ratios

* Train: 60%
* Validation: 20%
* Test: 20%

These ratios refer to the target distribution of **positive segments**, not subjects. This is more important than balancing subject counts because the number of seizure segments varies significantly across subjects.

### 3.3 Algorithm: Greedy Stratified Assignment

The splitting algorithm is fully deterministic and does not rely on random seeds.

**Steps:**

1. Compute per-subject statistics (number of positive segments)
2. Sort subjects by positive count in descending order (tie-break by subject ID)
3. For each subject, compute the "deficit ratio" for each split: how far below its target each split currently is
4. Assign the subject to the split with the largest deficit
5. When deficit ratios are tied, smaller-target splits are filled first (test > val > train)

**Properties:**

* Deterministic: same input always produces the same output
* Balanced: subjects with many positive segments are distributed across all splits
* Greedy: locally optimal at each step, produces near-optimal global distribution

---

## 4. Results

### 4.1 Split assignment

| Split | Subjects | Segments | Positive | Positive share | Target |
|-------|----------|----------|----------|----------------|--------|
| Train | 16 | 531,300 | 1,378 | 59.4% | 60% |
| Val | 3 | 91,327 | 472 | 20.3% | 20% |
| Test | 4 | 84,897 | 471 | 20.3% | 20% |

### 4.2 Subject assignments

**Train:** sub-01, sub-04, sub-05, sub-06, sub-07, sub-08, sub-10, sub-11, sub-13, sub-14, sub-18, sub-19, sub-20, sub-22, sub-23, sub-24

**Validation:** sub-09, sub-15, sub-16

**Test:** sub-02, sub-03, sub-12, sub-17

### 4.3 Observations

* Positive share is within 1% of target for all three splits
* All splits contain subjects with positive segments
* The validation set includes sub-15 (the subject with the most seizures, 399), which is appropriate because it provides a robust validation signal
* The test set includes sub-12 (294 positives), ensuring sufficient positive examples for evaluation

---

## 5. Validation

The following integrity checks are performed automatically:

* No subject appears in more than one split
* All 23 subjects are assigned to exactly one split
* Each split is non-empty
* Each split contains at least one positive segment

All checks passed.

---

## 6. Implementation

### 6.1 Core module

`src/neuro_eeg_cdss/data/splits.py`

Key components:

* `SplitConfig` — configurable target ratios
* `SplitAssignment` — immutable result of split creation
* `create_split()` — main entry point
* `validate_split()` — integrity checks
* `apply_split()` — filter a dataset into train/val/test DataFrames
* `save_split()` / `load_split()` — JSON serialization

### 6.2 Script

`scripts/splits/create_splits.py`

Usage:

```bash
python scripts/splits/create_splits.py
python scripts/splits/create_splits.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
```

### 6.3 Output files

```text
data/splits/
├── train_subjects.json
├── val_subjects.json
├── test_subjects.json
└── split_summary.json
```

### 6.4 Tests

`tests/test_splits.py` — 15 tests covering:

* Subject statistics computation
* No-overlap guarantee
* All-subjects-assigned guarantee
* Positive distribution in each split
* Determinism
* Edge cases (too few subjects, invalid config)
* Save/load round-trip
* Apply split with unknown subjects

---

## 7. Key Design Decisions

### 7.1 Positive-count balancing over subject-count balancing

The algorithm targets proportional distribution of positive segments rather than equal distribution of subjects. This is because subjects contribute very different amounts of seizure data.

### 7.2 Deterministic over random

The algorithm does not use a random seed. Given the same input data, it always produces the same split. This eliminates a source of variability and simplifies reproducibility.

### 7.3 Greedy assignment with priority tie-breaking

When multiple splits have the same deficit ratio (common at the start), smaller-target splits are filled first. Without this rule, the training set would absorb all early subjects due to its higher absolute target.

---

## 8. Limitations

* The hold-out split uses only one fixed partition. For final paper results, subject-level cross-validation should be considered.
* With only 3 subjects in validation and 4 in test, individual subject characteristics may have a strong influence on aggregate metrics.
* The split is optimized for positive segment distribution but does not consider total segment volume or recording duration.

---

## 9. Contribution to the Overall System

This sprint establishes the evaluation protocol for all subsequent modeling work. Without a valid patient-independent split, no downstream results would be methodologically sound.

The `apply_split()` function provides a direct interface for Sprint 1D (baseline model training) and Sprint 1E (clinical evaluation).

---

## 10. Next Steps

Sprint 1D — Baseline Models:

* Load features dataset
* Apply split
* Train Logistic Regression and Random Forest
* Save models and initial results
