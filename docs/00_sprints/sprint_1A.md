# Sprint 1A — Dataset Preparation (EEG Seizure Detection)

## 1. Objective

The goal of this sprint is to transform raw EEG recordings from the CHB-MIT dataset (BIDS format) into a structured and labeled dataset suitable for machine learning.

Specifically, the objective is to construct a dataset of fixed-length temporal segments, each associated with a binary label indicating the presence or absence of a seizure.

---

## 2. Dataset Description

* Dataset: CHB-MIT Scalp EEG Database (BIDS format)
* Number of subjects: 23
* Sampling frequency: 256 Hz
* Channels: 18 (bipolar montage, "double banana")
* Recording duration: ~1 hour per file
* Annotations: events.tsv files containing seizure intervals

Each recording is associated with an events file describing temporal events using the `eventType` field.

---

## 3. Pipeline Overview

The dataset construction pipeline consists of the following stages:

1. **Event Parsing**

   * Read `events.tsv` files
   * Extract seizure intervals based on `eventType`

2. **Temporal Segmentation**

   * Divide each recording into fixed-length windows

3. **Overlap Computation**

   * Compute temporal overlap between each window and seizure intervals

4. **Label Assignment**

   * Assign binary labels based on overlap ratio

5. **Dataset Assembly**

   * Aggregate all segments into a unified tabular dataset
   * Store as a parquet file for efficient downstream processing

---

## 4. Key Design Decisions

### 4.1 Window Size (5 seconds)

EEG recordings are segmented into fixed windows of 5 seconds.

**Rationale:**

* Provides a balance between temporal resolution and feature stability
* Short enough to capture seizure dynamics
* Long enough to compute meaningful statistical and spectral features
* Common choice in EEG-based seizure detection literature

---

### 4.2 Non-overlapping Windows

Stride is set equal to window size (5 seconds), resulting in non-overlapping segments.

**Rationale:**

* Simplifies the dataset and reduces redundancy
* Avoids temporal leakage between adjacent samples
* Provides a clean baseline for initial modeling

---

### 4.3 Use of Complete Windows Only

Only windows fully contained within the recording duration are considered.

**Rationale:**

* Ensures consistent input length across all samples
* Avoids complications in feature extraction
* Simplifies downstream modeling

---

### 4.4 Seizure Identification via `eventType`

Seizure events are identified using the `eventType` field in `events.tsv`.

* `bckg` → non-seizure
* values starting with `sz` → seizure

**Rationale:**

* Based on official dataset metadata (`events.json`)
* Avoids fragile text-based heuristics
* Ensures consistency with dataset annotation standards

---

### 4.5 Labeling Strategy

Each window is assigned a label based on its overlap with seizure intervals.

Let:

* `overlap_ratio = seizure_overlap_duration / window_duration`

Then:

* `label = 1` if overlap_ratio ≥ 0.5
* `label = 0` if overlap_ratio = 0
* windows with 0 < overlap_ratio < 0.5 are discarded

**Rationale:**

* Reduces label noise near seizure boundaries
* Ensures high confidence in positive samples
* Avoids ambiguous training examples
* Standard practice in event detection problems

---

## 5. Implementation

The pipeline is modular and implemented in the following components:

* `events.py` → extraction of seizure intervals
* `segmentation.py` → generation of time windows and overlap computation
* `labeling.py` → assignment of labels based on overlap ratio
* `dataset_builder.py` → full dataset construction from manifest

Unit tests are implemented for each module to ensure correctness and robustness.

---

## 6. Output Dataset

The final dataset is stored as:

```text
data/processed/segments.parquet
```

Each row corresponds to a temporal segment with the following fields:

* subject
* session
* run
* path
* recording_duration_sec
* start_sec
* end_sec
* window_size_sec
* stride_sec
* overlap_ratio
* label

---

## 7. Dataset Statistics

* Total segments: 707,524
* Positive (seizure) segments: 2,321
* Negative (non-seizure) segments: 705,203

Class distribution:

* ~0.3% positive
* ~99.7% negative

**Observation:**
The dataset is highly imbalanced, which is expected in seizure detection tasks, as seizures are rare events.

---

## 8. Validation Checks

The following sanity checks were performed:

* No missing values in critical fields
* All window durations are exactly 5 seconds
* Overlap ratio ∈ [0, 1]
* Labels are strictly binary (0 or 1)
* All 23 subjects are present

---

## 9. Limitations and Considerations

* Severe class imbalance will affect model training
* Temporal independence between windows is assumed
* No signal-level preprocessing (filtering, normalization) applied yet
* No feature extraction performed at this stage

---

## 10. Next Steps

The next phase (Sprint 1B) will focus on:

* Feature extraction (time and frequency domain)
* Patient-independent train/test split
* Baseline model training (Logistic Regression, Random Forest)
* Clinical evaluation metrics (recall, specificity, F1, AUROC)

---

## 11. Notes

This sprint establishes the foundation of the entire pipeline.
All subsequent modeling stages depend critically on the correctness and reproducibility of this dataset construction process.
