# Key Design Decisions

This document tracks important methodological and architectural decisions made during the project. Each entry records the decision, its rationale, and known trade-offs.

---

## D1. Window size: 5 seconds, non-overlapping

**Sprint:** 1A

**Decision:** EEG recordings are segmented into fixed 5-second windows with stride equal to window size.

**Rationale:** Balances temporal resolution with feature stability. Common in seizure detection literature. Non-overlapping windows avoid temporal leakage between adjacent samples.

**Trade-off:** Shorter windows would improve localization but reduce feature quality. Overlapping windows would increase training data but introduce redundancy and potential leakage.

---

## D2. Labeling threshold: 50% overlap with ambiguous drop

**Sprint:** 1A

**Decision:** Windows with seizure overlap >= 50% are labeled positive. Windows with 0% overlap are negative. Windows with partial overlap (0% < overlap < 50%) are discarded.

**Rationale:** Reduces label noise near seizure boundaries by removing ambiguous examples. Ensures high confidence in both positive and negative labels.

**Trade-off:** Discarding partial-overlap windows reduces the positive class further (already ~0.3%). Future Sprint 1F will systematically evaluate alternative thresholds.

---

## D3. Positional channel indexing

**Sprint:** 1B

**Decision:** EEG channels are renamed to positional identifiers (`ch_01` ... `ch_18`) instead of using original electrode labels.

**Rationale:** Original channel names were inconsistent across recordings, causing ~240 feature columns instead of the expected 144, with many NaN values. Positional indexing guarantees a fixed feature schema.

**Trade-off:** Loses the semantic association between features and electrode locations. For clinical interpretability, a channel mapping table should be maintained separately.

---

## D4. Split optimization target: positive segment distribution

**Sprint:** 1C

**Decision:** The patient-independent split algorithm optimizes for proportional distribution of positive (seizure) segments across train/val/test, rather than balancing subject counts or total segment volumes.

**Rationale:** Positive segments are the critical resource for both training and evaluation. With only 2,321 positives across 23 subjects and extreme variability (sub-15 has 399, sub-06 has 31), balancing positives is more important than balancing the much larger pool of negative segments.

**Observed effect:** The training set contains 69.6% of subjects and 75.1% of total segments, but only 59.4% of positives (matching the 60% target). This volume asymmetry arises because some subjects contribute many recordings with very few seizures.

**Why acceptable:** For tabular ML with `class_weight="balanced"`, negative volume imbalance is handled by the loss function. The critical metric is whether each split has enough positive examples for reliable gradient estimation (train) and metric computation (val/test).

**When to revisit:** If moving to deep learning with batch-based training (Sprint 3), monitor whether the volume imbalance causes epoch-level training instability.

---
