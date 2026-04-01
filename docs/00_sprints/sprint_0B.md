# Sprint 0B — Data Ingestion & Validation (EEG)

## 1. Objective

The goal of this sprint is to acquire, validate, and inspect the EEG dataset in BIDS format.

This step ensures that the data is correctly structured, accessible, and usable for downstream processing.

---

## 2. Dataset Description

- Dataset: CHB-MIT Scalp EEG Database (BIDS format)
- Source: Zenodo (BIDS-converted version)
- Number of subjects: 23
- Sampling frequency: 256 Hz
- Channels: 18 (bipolar montage)
- Recording duration: ~1 hour per file

The dataset includes:
- EEG signals (`.edf`)
- Metadata (`.json`, `.tsv`)
- Event annotations (`events.tsv`)

---

## 3. Data Acquisition

A reproducible download pipeline was implemented:

- Automated script to download dataset from Zenodo
- Integrity verification via checksum (MD5)
- Extraction and normalization of directory structure

**Rationale:**

- Avoid manual download errors
- Ensure reproducibility
- Enable easy setup on new environments

---

## 4. BIDS Structure Validation

A validation script was implemented to verify:

- Presence of `dataset_description.json`
- Presence of `participants.tsv`
- Existence of subject directories (`sub-*`)

---

## 5. EEG Data Loading

EEG recordings were successfully loaded using:

- `mne-bids`

This confirmed:
- Correct file structure
- Compatibility with standard neurophysiology tools

---

## 6. Annotation Inspection

Event annotations were accessed via:

- `events.tsv`

Key observations:

- Some recordings contain seizure events
- Others contain only background activity
- Annotations are time-based intervals

---

## 7. Observations

- Dataset is heterogeneous across subjects and sessions
- Not all recordings contain seizures
- EEG signals are stored efficiently (lazy loading)

---

## 8. Validation Checks

The following checks were performed:

- Successful loading of at least one EEG file
- Correct number of channels (18)
- Correct sampling frequency (256 Hz)
- Annotations accessible and interpretable

---

## 9. Limitations

- Some metadata fields are not fully mapped by `mne-bids`
- Channel metadata (`channels.tsv`) may be missing in some cases
- No preprocessing applied at this stage

---

## 10. Outcome

At the end of this sprint:

- Dataset is locally available and validated
- EEG recordings can be read programmatically
- Annotation system is understood

---

## 11. Next Steps

Sprint 0C will focus on:

- Dataset indexing
- Building a structured manifest
- Preparing data access for ML pipelines
```