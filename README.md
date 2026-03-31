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

---

### In Progress

#### Sprint 1 — Dataset Preparation
- Label extraction (seizure vs non-seizure)
- Segmenting EEG into training windows
- Building ML-ready dataset

---

### Next Steps

- Feature extraction
- Model training (baseline)
- Evaluation pipeline
- Explainability module