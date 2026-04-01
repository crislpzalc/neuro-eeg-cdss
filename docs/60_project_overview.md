# Project Overview — EEG-Based Seizure Detection System

## 1. Motivation

Epilepsy is a neurological disorder characterized by recurrent seizures, affecting millions of people worldwide. Continuous monitoring of brain activity using electroencephalography (EEG) is essential for diagnosis and clinical management. However, manual inspection of EEG recordings is time-consuming, subjective and requires highly specialized expertise.

This project is motivated by the potential of artificial intelligence to assist in the detection of epileptic events, improving both efficiency and reliability in clinical workflows. Early and accurate detection of seizures can have a significant impact on patient care, enabling better diagnosis, monitoring and potentially real-time intervention systems.

From a personal perspective, this project reflects a strong interest in the intersection between machine learning and medicine. The long-term goal is to develop AI systems that can operate in real-world clinical settings, combining technical rigor with meaningful medical impact.

---

## 2. Problem Definition

This project addresses the problem of automatic seizure detection in EEG recordings.

Given a continuous EEG signal:

> The objective is to identify temporal segments that contain seizure activity (positive class) versus normal background activity (negative class).

Key challenges include:

* Extreme class imbalance (seizures are rare events)
* Temporal structure of the data
* High inter-patient variability
* Presence of noise and artifacts in EEG signals

---

## 3. Project Objectives

### Primary Objective

To design and implement an end-to-end machine learning pipeline for seizure detection from EEG recordings, ensuring robustness, reproducibility and clinical relevance.

### Secondary Objectives

* Build a modular and extensible system architecture
* Establish a strong baseline using classical machine learning methods
* Explore more advanced modeling approaches (e.g., deep learning and sequence models)
* Evaluate performance using clinically meaningful metrics
* Maintain interpretability and transparency in early stages

---

## 4. System Overview

The system is structured as a modular pipeline:

```text
Raw EEG (BIDS)
      ↓
Event parsing
      ↓
Temporal segmentation
      ↓
Label assignment
      ↓
Feature extraction
      ↓
Model training
      ↓
Evaluation (clinical metrics)
```

Each stage is implemented independently, enabling flexibility, experimentation and future extensions.

---

## 5. Dataset

The system is developed using the CHB-MIT Scalp EEG Database in BIDS format.

* 23 subjects
* Sampling frequency: 256 Hz
* 18 EEG channels (bipolar montage)
* Long recordings (~1 hour)
* Annotated seizure intervals

This dataset provides a realistic clinical scenario, including sparse seizure events and significant variability across patients.

---

## 6. Methodological Approach

The project follows a structured and iterative development strategy:

### Phase 1 — Dataset Preparation

* Extraction of seizure intervals from annotations
* Segmentation into fixed-length temporal windows
* Label assignment based on overlap with seizure intervals

### Phase 2 — Feature Engineering (ongoing)

* Time-domain features (mean, standard deviation, RMS, line length)
* Frequency-domain features (bandpower across EEG bands)

### Phase 3 — Baseline Modeling

* Logistic Regression
* Random Forest
* Patient-independent evaluation

### Phase 4 — Advanced Modeling (future work)

* Deep learning models (CNNs for EEG)
* Temporal models (Transformers, sequence models)
* Uncertainty estimation and calibration
* Explainability methods

---

## 7. Key Design Principles

### 7.1 Reproducibility

All stages of the pipeline are implemented as code and can be executed from scratch without manual intervention.

---

### 7.2 Modularity

The system is organized into independent modules:

* preprocessing
* feature extraction
* modeling
* evaluation

This design enables rapid experimentation and scalability.

---

### 7.3 Clinical Relevance

The evaluation focuses on clinically meaningful metrics:

* Recall (sensitivity) for seizure detection
* Specificity
* Minimization of false negatives

In clinical settings, missing a seizure is significantly more critical than false alarms.

---

### 7.4 Interpretability

Initial modeling focuses on interpretable features and models to ensure transparency and facilitate understanding of system behavior.

---

## 8. Challenges

The problem presents several inherent challenges:

* Severe class imbalance (~0.3% positive samples)
* High variability across patients
* Ambiguity at seizure boundaries
* Noise and artifacts in EEG recordings

These challenges guide the design of the preprocessing and labeling strategies.

---

## 9. Current Status

At the current stage:

* A fully reproducible dataset preparation pipeline has been implemented
* EEG recordings have been segmented and labeled (~707k segments)
* Data integrity and consistency have been validated through systematic checks

---

## 10. Expected Contributions

This project aims to provide:

* A fully reproducible EEG seizure detection pipeline
* A strong and well-justified baseline model
* A modular framework for further research in medical AI
* A foundation for exploring advanced sequence models in EEG analysis

---

## 11. Future Work

Planned extensions include:

* Advanced feature engineering
* Deep learning models for EEG signals
* Transformer-based approaches for temporal modeling
* Model calibration and uncertainty estimation
* Explainability for clinical validation
* Potential deployment as an inference system

---

## 12. Scope

This project is an independent research-oriented initiative aimed at bridging machine learning and clinical applications. It is designed to meet the standards of applied AI in healthcare and to serve as a foundation for further academic research or publication.
