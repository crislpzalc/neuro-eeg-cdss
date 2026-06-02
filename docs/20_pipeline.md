# Pipeline Architecture

This document describes the complete end-to-end pipeline from raw EEG recordings to clinical evaluation. Each stage is implemented as an independent module, enabling experimentation and future extensions.

---

## 1. Pipeline Overview

```text
Raw EEG (.edf, BIDS format)
        │
        ▼
┌─────────────────────┐
│  Event Parsing       │  ← events.py
│  (seizure intervals) │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Segmentation        │  ← segmentation.py
│  (5s windows)        │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Labeling            │  ← labeling.py
│  (binary: 0/1)       │
└────────┬────────────┘
         │
         ▼
   segments.parquet        ← Sprint 1A output
         │
         ▼
┌─────────────────────┐
│  Feature Extraction  │  ← extractors.py, dataset_builder.py
│  (144 features)      │
└────────┬────────────┘
         │
         ▼
   features.parquet        ← Sprint 1B output
         │
         ▼
┌─────────────────────┐
│  Patient-Independent │  ← splits.py
│  Split (60/20/20)    │
└────────┬────────────┘
         │
         ▼
   data/splits/*.json      ← Sprint 1C output
         │
         ▼
┌─────────────────────┐
│  Model Training      │  ← trainer.py
│  (LR, RF)            │
└────────┬────────────┘
         │
         ▼
   experiments/baseline/   ← Sprint 1D output
   (models + predictions)
         │
         ▼
┌─────────────────────┐
│  Clinical Evaluation │  ← metrics.py, plots.py
│  (19 metrics + plots)│
└─────────────────────┘   ← Sprint 1E output
```

---

## 2. Data Flow

### 2.1 Raw data → Segments (Sprint 1A)

**Input:** EEG recordings in BIDS format (`.edf` files + `events.tsv` annotations)

**Process:**
1. Parse `events.tsv` to extract seizure intervals (onset, duration)
2. Divide each recording into non-overlapping 5-second windows
3. Compute overlap ratio between each window and seizure intervals
4. Assign binary labels: ≥50% overlap → positive, 0% overlap → negative, partial → discarded

**Output:** `data/processed/segments.parquet` — 707,524 labeled segments

### 2.2 Segments → Features (Sprint 1B)

**Input:** Segment metadata + raw EEG signal (loaded per recording)

**Process:**
1. For each recording, load the full EEG signal once
2. For each segment in that recording, extract a 5-second slice
3. Compute 8 features per channel (4 time-domain + 4 frequency-domain)
4. Use positional channel names (ch_01...ch_18) for consistent schema

**Output:** `data/processed/features.parquet` — 707,524 rows × 151 columns (7 metadata + 144 features)

### 2.3 Features → Split (Sprint 1C)

**Input:** Feature dataset + per-subject statistics

**Process:**
1. Compute positive segment count per subject
2. Greedy stratified assignment: assign subjects to splits targeting 60/20/20 positive distribution
3. Save subject lists as JSON files

**Output:** `data/splits/` — JSON files mapping subjects to train/val/test

### 2.4 Split → Training (Sprint 1D)

**Input:** Feature dataset + split assignment

**Process:**
1. Apply split to get train/val/test DataFrames
2. Separate features from labels (auto-detect feature columns)
3. Fit StandardScaler on training features
4. Train model with `class_weight="balanced"`
5. Generate predictions (y_pred + y_proba) on all three splits
6. Save model, scaler, config, predictions

**Output:** `experiments/baseline/` — model artifacts + prediction files per split

### 2.5 Predictions → Evaluation (Sprint 1E)

**Input:** Saved prediction files (no re-training needed)

**Process:**
1. Load (y_true, y_pred, y_proba) per model per split
2. Compute 19 clinical metrics
3. Generate ROC/PR curves, confusion matrices, threshold analysis
4. Compare models side-by-side

**Output:** `experiments/baseline/evaluation/` — metrics JSON + publication-quality plots

---

## 3. Module Map

| Module | Location | Responsibility |
|--------|----------|----------------|
| Event parsing | `preprocessing/events.py` | Extract seizure intervals from BIDS annotations |
| Segmentation | `preprocessing/segmentation.py` | Generate time windows, compute overlap |
| Labeling | `preprocessing/labeling.py` | Assign binary labels based on overlap threshold |
| Dataset builder | `preprocessing/dataset_builder.py` | Orchestrate full segment dataset construction |
| Time features | `features/time_domain.py` | Mean, std, RMS, line length |
| Frequency features | `features/frequency_domain.py` | Delta, theta, alpha, beta bandpower |
| Feature extractors | `features/extractors.py` | Combine time + frequency per channel |
| Feature builder | `features/dataset_builder.py` | Build full feature dataset from segments |
| Splits | `data/splits.py` | Patient-independent split creation and application |
| Trainer | `training/trainer.py` | Model-agnostic training pipeline |
| Metrics | `evaluation/metrics.py` | 19 clinical evaluation metrics |
| Plots | `evaluation/plots.py` | Publication-quality visualization |

---

## 4. Key Design Principles

### 4.1 Each stage produces a persistent artifact

Every pipeline stage writes its output to disk (parquet, JSON, pkl). This means:
- No stage needs to re-run its predecessors
- Debugging is easy: inspect any intermediate file
- Evaluation can run without re-training

### 4.2 Metadata travels with the data

Every row in `segments.parquet` and `features.parquet` carries its full provenance: subject, session, run, path, time range. This makes it possible to trace any prediction back to the original recording.

### 4.3 Feature columns are auto-detected

The trainer does not hardcode feature names. It identifies them as "everything that is not metadata and not the label." This makes the pipeline robust to changes in the feature set.

### 4.4 Training is decoupled from evaluation

Sprint 1D saves raw predictions. Sprint 1E loads them and computes metrics. This separation means you can:
- Add new metrics without re-training
- Try different thresholds without re-training
- Compare models trained at different times

---

## 5. Reproducibility

The full pipeline can be reproduced with these commands (inside the devcontainer):

```bash
# 1. Download data
python scripts/download/download_chbmit_bids.py

# 2. Build manifest
python scripts/data_index/build_manifest.py

# 3. Build segments
python scripts/preprocessing/build_segments_dataset.py

# 4. Extract features (~6 hours)
python scripts/features/build_features_dataset.py

# 5. Create patient-independent split
python scripts/splits/create_splits.py

# 6. Train baseline models
python scripts/training/train_baseline.py

# 7. Clinical evaluation
python scripts/evaluation/evaluate_baseline.py
```

Each script is idempotent: running it again overwrites the previous output with identical results.
