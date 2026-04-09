# Sprint 1B — Feature Engineering for EEG Seizure Detection

## Status

Completed

---

## 1. Objective

The goal of this sprint is to transform segmented EEG signals into a **fixed, consistent, and interpretable tabular representation** suitable for classical machine learning models.

This step constitutes a critical bridge between raw biomedical signals and downstream predictive modeling, ensuring that the resulting dataset is:

* Structurally consistent (no missing features)
* Scalable to large datasets
* Clinically interpretable
* Reproducible

---

## 2. Context

In Sprint 1A, raw EEG recordings were segmented into fixed-length windows (5 seconds) and labeled as:

* `1` → seizure
* `0` → non-seizure

This resulted in a dataset of ~707,000 labeled segments.

Sprint 1B builds on this by extracting **feature representations** from each segment.

---

## 3. Feature Design

Feature extraction is performed **per segment and per channel**, resulting in a structured feature vector.

### 3.1 Time-Domain Features (per channel)

* Mean
* Standard deviation
* Root Mean Square (RMS)
* Line Length

These features capture amplitude distribution and signal variability.

---

### 3.2 Frequency-Domain Features (per channel)

Bandpower is computed for the following frequency bands:

* Delta (0.5–4 Hz)
* Theta (4–8 Hz)
* Alpha (8–13 Hz)
* Beta (13–30 Hz)

These features capture spectral characteristics of EEG signals, which are highly relevant for seizure detection.

---

### 3.3 Total Feature Count

* 18 channels
* 8 features per channel

→ **144 features per segment**

---

## 4. Key Design Decisions

### 4.1 Per-Channel Feature Representation

Features are computed independently for each channel rather than aggregated across channels.

**Rationale:**

* Seizure activity is often spatially localized
* Preserving channel-level information improves interpretability
* Enables compatibility with future deep learning models

---

### 4.2 Fixed Feature Space (Critical Decision)

#### Initial Approach

Feature names were constructed using EEG channel labels from EDF files:

* `mean_FP1-F3`
* `delta_power_T6-O2`
* etc.

#### Problem Observed

This approach led to:

* Inconsistent feature columns across recordings
* Explosion in total number of columns (~240 instead of 144)
* Large number of missing values (`NaN`)
* Non-rectangular dataset unsuitable for ML

#### Root Cause

Channel naming and availability were not perfectly consistent across all recordings.

#### Final Solution

We enforced a **fixed positional channel indexing**:

* Channels renamed to:

  * `ch_01`, `ch_02`, ..., `ch_18`

Feature names now follow the pattern:

* `mean_ch_01`
* `delta_power_ch_18`
* etc.

#### Outcome

* Fully consistent feature schema
* Zero missing values
* Deterministic feature dimensionality

---

### 4.3 Window-Level Representation

Each EEG segment (5 seconds) is treated as an independent sample.

**Rationale:**

* Compatible with classical ML models
* Enables window-level evaluation
* Simplifies pipeline design

---

### 4.4 Separation of Concerns

Feature extraction is kept independent from:

* normalization
* scaling
* model-specific preprocessing

**Rationale:**

* Improves modularity
* Enables reuse across different models
* Avoids data leakage

---

## 5. Implementation

### 5.1 Architecture

Feature extraction pipeline:

```
segments.parquet
    → dataset_builder.py
        → extractors.py
            → features.parquet
```

---

### 5.2 Processing Strategy

To ensure scalability:

* Data is processed **per recording**, not per segment
* Each EDF file is loaded once
* All segments from that recording are extracted in memory

**Benefits:**

* Reduces I/O overhead
* Improves performance significantly

---

### 5.3 Core Modules

* `features/time_domain.py`
* `features/frequency_domain.py`
* `features/extractors.py`
* `features/dataset_builder.py`

Scripts:

* `scripts/features/build_features_dataset.py`
* `scripts/features/check_features.py`

---

## 6. Challenges and Resolutions

### 6.1 Inconsistent Feature Space

**Observed:**

* ~240 feature columns instead of expected 144
* Large number of `NaN` values

**Cause:**

* Channel-dependent feature naming

**Resolution:**

* Switch to positional channel indexing (`ch_01` ... `ch_18`)

---

### 6.2 Large-Scale Processing

**Observed:**

* Dataset size: ~707k segments
* Long execution time (~6 hours)

**Resolution:**

* Process per recording
* Avoid repeated EDF loading
* Validate pipeline on small subsets before full execution

---

### 6.3 Debugging Strategy

A two-stage approach was used:

1. Small-scale validation (`max_segments=10000`)
2. Full dataset execution after verification

This ensured correctness before committing to long runs.

---

## 7. Validation

Validation was performed using `check_features.py`.

### 7.1 Structural Validation

* No missing values (`NaN`)
* Fixed number of columns (151 total)
* Consistent feature dimensionality across all rows

---

### 7.2 Data Integrity

* Labels preserved correctly
* Segment alignment maintained
* All 23 subjects present

---

### 7.3 Feature Sanity

* All features numeric
* Bandpower values non-negative
* Statistical distributions reasonable

---

## 8. Final Dataset Schema

| Component | Count |
| --------- | ----- |
| Metadata  | 7     |
| Features  | 144   |
| **Total** | 151   |

Metadata fields:

* subject
* session
* run
* path
* start_sec
* end_sec
* label

---

## 9. Reproducibility

The entire feature extraction pipeline is:

* Deterministic
* Script-based
* Fully reproducible

Requirements:

* Same dataset
* Same configuration
* Same codebase

---

## 10. Limitations

* No feature normalization applied yet
* No artifact removal (e.g., noise, eye movements)
* Frequency bands fixed (not optimized)
* Computationally expensive for large datasets

These aspects will be addressed in later sprints.

---

## 11. Contribution to the Overall System

This sprint enables:

* Transition from raw signals → ML-ready data
* First interpretable baseline models
* Foundation for all subsequent modeling work

---

## 12. Contribution to Research / Paper

This stage establishes:

* A reproducible feature extraction pipeline
* A well-defined and consistent feature space
* A validated dataset suitable for experimentation

The resolution of feature inconsistency constitutes a **key methodological contribution**, as it ensures validity of downstream experiments.

---

## 13. Next Steps

Sprint 1C — Patient-Independent Split

Key goal:

* Prevent data leakage across subjects
* Establish a valid evaluation protocol

---

## 14. Final Validation Results

The full feature dataset was successfully generated and validated.

### Final dataset summary

- Total rows: **707,524**
- Total columns: **151**
- Metadata columns: **7**
- Feature columns: **144**
- Subjects represented: **23**

### Label distribution

- Non-seizure segments: **705,203**
- Seizure segments: **2,321**

### Validation checks

- No missing values detected
- All feature columns are numeric
- No negative bandpower values detected
- Window duration remains consistent at 5 seconds
- Feature dimensionality is identical across all samples

These checks confirm that the feature extraction pipeline is correct, stable, and ready for downstream patient-independent training and evaluation.
