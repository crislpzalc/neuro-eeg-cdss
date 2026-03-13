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

## Project Structure

```
neuro-eeg-cdss
│
├─ configs
├─ data
│ ├─ raw
│ └─ processed
│
├─ experiments
├─ notebooks
├─ tests
│
├─ src
│ ├─ data
│ ├─ preprocessing
│ ├─ features
│ ├─ models
│ ├─ training
│ ├─ evaluation
│ ├─ calibration
│ ├─ uncertainty
│ ├─ explainability
│ ├─ inference
│ └─ api
│
└─ pyproject.toml
```

## Development Environment

The project uses:

- **Python 3.11**
- **Docker Dev Containers**
- **PyTorch**
- **MNE / MNE-BIDS**
- **FastAPI**

## Status

Sprint 0A – Project infrastructure setup