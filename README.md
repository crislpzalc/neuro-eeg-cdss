# Neuro EEG CDSS

Intelligent system for automatic detection of epileptic seizures from EEG signals.

## Overview

This project aims to develop a modular and reproducible system for seizure detection using EEG recordings. The goal is not only to train a machine learning model, but to build a complete pipeline that includes data ingestion, preprocessing, modeling, evaluation, uncertainty estimation, explainability, and deployment.

The system is designed as a **Clinical Decision Support System (CDSS)** prototype for epilepsy monitoring.

## Main Components

The project includes the following modules:

- EEG data ingestion using the **BIDS standard**
- Signal preprocessing and segmentation
- Baseline machine learning models
- Deep learning models (1D CNN)
- Clinical evaluation metrics
- Model calibration
- Uncertainty estimation
- Model explainability
- Inference pipeline
- REST API for predictions

## Dataset

The initial dataset used in this project is:

**CHB-MIT EEG dataset (BIDS format)**

This dataset contains EEG recordings of pediatric subjects with epilepsy and annotated seizure intervals.

## Dataset setup

Download the dataset:
```bash
python scripts/download/download_chbmit_bids.py
```

Validate the BIDS structure:
```bash
python scripts/data_checks/check_bids_structure.py
```

Read one EEG recording:
```bash
python scripts/data_checks/read_one_recording.py
```

Inspect annotations:
```bash
python scripts/data_checks/inspect_annotations.py
```

Build the dataset manifest:
```bash
python scripts/data_index/build_manifest.py
```


## Project Status

### Completed

#### Sprint 0A — Environment & Setup
- Dev container configured
- Dependencies installed
- Project structure initialized

#### Sprint 0B — Data Ingestion
- CHB-MIT BIDS dataset downloaded
- BIDS structure validated
- EEG recordings successfully loaded with mne-bids
- Annotations inspected

#### Sprint 0C — Data Indexing
- Dataset indexed into `manifest.parquet`
- Reproducible data pipeline established

#### Sprint 1A — Dataset Preparation
- Label extraction (seizure vs non-seizure)
- Segmenting EEG into 5-second training windows
- Building ML-ready dataset (`segments.parquet`)

#### Sprint 1B — Feature Engineering
- Time-domain features (mean, std, RMS, line length)
- Frequency-domain features (delta, theta, alpha, beta bandpower)
- 144 features per segment (8 features × 18 channels)

#### Sprint 1C — Patient-Independent Split
- Stratified subject-level train/val/test split (60/20/20)
- Zero patient overlap between splits
- Balanced positive segment distribution across splits

#### Sprint 1D — Baseline Models
- Logistic Regression and Random Forest with balanced class weights
- Generic training pipeline (`trainer.py`) with seed control and artifact serialization
- Predictions saved per split for decoupled evaluation
- Patient-independent results: LR test recall ~23%, RF test recall ~0% (establishes performance floor)

#### Sprint 1E — Clinical Evaluation
- 19 clinical metrics including sensitivity, specificity, F2, AUROC, AUPRC
- Publication-quality plots: ROC curves, PR curves, confusion matrices, threshold analysis
- Model comparison: RF discriminates better (AUROC 0.77 vs 0.71) but both insufficient for clinical use
- Reusable evaluation framework for all future models

#### Sprint 1F — Labeling Strategy Experiments
- Systematic comparison of 6 labeling configurations (3 thresholds × 2 drop policies)
- Finding: labeling threshold has negligible impact with 5s windows (only 0.022% of windows have partial overlap)
- Validates default policy and establishes that the bottleneck is feature representation, not labeling

---

### Next Steps

- Temporal post-processing (smoothing, majority voting)
- Event-level evaluation (detection delay, false alarms per hour)
- Deep learning models (CNN, LSTM, Transformer)