# Sprint 0C — Dataset Indexing

## 1. Objective

The goal of this sprint is to transform the raw dataset into an indexed structure that enables efficient and scalable access.

This is achieved by constructing a manifest file that acts as a centralized index of all EEG recordings.

---

## 2. Motivation

Working directly with filesystem traversal is inefficient and error-prone.

The manifest provides:

- Fast access to data
- Structured metadata
- Foundation for ML pipelines

---

## 3. Manifest Construction

A script was implemented to:

- Traverse the dataset directory
- Identify all EEG files (`*_eeg.edf`)
- Extract relevant metadata from filenames
- Build a tabular dataset

---

## 4. Extracted Fields

The manifest includes:

- subject
- session
- task
- run
- filename
- path

---

## 5. Storage Format

The manifest is stored as:

```

data/manifests/manifest.parquet

```

**Rationale:**

- Efficient storage
- Fast read/write operations
- Compatible with ML workflows

---

## 6. Validation

The following checks were performed:

- All EEG files are indexed
- No missing paths
- Correct number of subjects (23)
- Data consistency across rows

---

## 7. Example Structure

| subject | session | task | run | path |
|--------|--------|------|-----|------|
| sub-01 | ses-01 | szMonitoring | 00 | ... |

---

## 8. Design Considerations

- Metadata extracted from filenames (BIDS standard)
- No signal-level processing at this stage
- Manifest designed to be extensible

---

## 9. Outcome

At the end of this sprint:

- Dataset is fully indexed
- Data access is abstracted from filesystem
- Pipeline is ready for dataset transformation

---

## 10. Importance

This step is critical because:

> All downstream processing (segmentation, labeling, training) depends on this index.

---

## 11. Next Steps

Sprint 1 will focus on:

- Extracting seizure intervals
- Segmenting EEG signals
- Building a labeled dataset for machine learning
```