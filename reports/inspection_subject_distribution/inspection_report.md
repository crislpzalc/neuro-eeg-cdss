# Pre-split subject-level inspection

## 1. Metadata

- Dataset: `data/processed/segments.parquet`
- Number of rows: 707524
- Columns: subject, session, run, path, recording_duration_sec, start_sec, end_sec, window_size_sec, stride_sec, overlap_ratio, label
- Recording column used: run

## 2. Global summary

- **total_segments**: 707524
- **total_positive**: 2321
- **total_negative**: 705203
- **global_positive_ratio**: 0.003280454090603287
- **total_subjects**: 23
- **subjects_with_positive**: 23
- **subjects_without_positive**: 0
- **subject_positive_ratio**: 1.0

## 3. Subject-level distribution statistics

### n_segments_per_subject

- **min**: 13674.0
- **p25**: 18914.5
- **median**: 25046.0
- **mean**: 30761.913043
- **p75**: 32399.5
- **max**: 112356.0
- **std**: 20905.100043

### n_positive_per_subject

- **min**: 18.0
- **p25**: 50.5
- **median**: 76.0
- **mean**: 100.913043
- **p75**: 109.5
- **max**: 399.0
- **std**: 87.020694

### n_negative_per_subject

- **min**: 13656.0
- **p25**: 18855.5
- **median**: 24886.0
- **mean**: 30661.0
- **p75**: 32156.0
- **max**: 112280.0
- **std**: 20913.599372

### positive_ratio_per_subject

- **min**: 0.000645
- **p25**: 0.001598
- **median**: 0.002495
- **mean**: 0.004313
- **p75**: 0.00445
- **max**: 0.017267
- **std**: 0.004338

## 4. Dataset concentration

- **top_1_positive_share**: 0.171909
- **top_3_positive_share**: 0.376993
- **top_5_positive_share**: 0.501508
- **top_1_segment_share**: 0.158802
- **top_3_segment_share**: 0.302525
- **top_5_segment_share**: 0.438652

## 5. Methodological risk flags

- **few_total_subjects**: False
- **very_few_total_subjects**: False
- **few_positive_subjects**: False
- **very_few_positive_subjects**: False
- **many_subjects_without_positive**: False
- **positives_highly_concentrated_top1**: False
- **positives_highly_concentrated_top3**: False
- **segments_highly_concentrated_top1**: False
- **segments_highly_concentrated_top3**: False
- **has_many_very_small_subjects**: False
- **has_zero_positive_subjects**: False
- **recommended_strategy**: The dataset seems compatible with a reproducible subject-level hold-out split. Still, explicitly verify that validation and test contain enough positive subjects.

## 6. Top subjects by positive segments

| subject   |   n_segments |   n_positive |   n_negative |   positive_ratio |   n_recordings | has_positive   |   positive_share_overall |   segment_share_overall |
|:----------|-------------:|-------------:|-------------:|-----------------:|---------------:|:---------------|-------------------------:|------------------------:|
| sub-15    |        28792 |          399 |        28393 |           0.0139 |             40 | True           |                   0.1719 |                  0.0407 |
| sub-12    |        17027 |          294 |        16733 |           0.0173 |             24 | True           |                   0.1267 |                  0.0241 |
| sub-08    |        14397 |          182 |        14215 |           0.0126 |             20 | True           |                   0.0784 |                  0.0203 |
| sub-11    |        25046 |          160 |        24886 |           0.0064 |             35 | True           |                   0.0689 |                  0.0354 |
| sub-01    |        52827 |          129 |        52698 |           0.0024 |             42 | True           |                   0.0556 |                  0.0747 |
| sub-05    |        28078 |          112 |        27966 |           0.004  |             39 | True           |                   0.0483 |                  0.0397 |
| sub-13    |        23748 |          107 |        23641 |           0.0045 |             33 | True           |                   0.0461 |                  0.0336 |
| sub-24    |        15321 |          102 |        15219 |           0.0067 |             22 | True           |                   0.0439 |                  0.0217 |
| sub-10    |        36007 |           88 |        35919 |           0.0024 |             25 | True           |                   0.0379 |                  0.0509 |
| sub-23    |        19115 |           84 |        19031 |           0.0044 |              9 | True           |                   0.0362 |                  0.027  |

## 7. Top subjects by total segments

| subject   |   n_segments |   n_positive |   n_negative |   positive_ratio |   n_recordings | has_positive   |   positive_share_overall |   segment_share_overall |
|:----------|-------------:|-------------:|-------------:|-----------------:|---------------:|:---------------|-------------------------:|------------------------:|
| sub-04    |       112356 |           76 |       112280 |           0.0007 |             42 | True           |                   0.0327 |                  0.1588 |
| sub-01    |        52827 |          129 |        52698 |           0.0024 |             42 | True           |                   0.0556 |                  0.0747 |
| sub-09    |        48861 |           55 |        48806 |           0.0011 |             19 | True           |                   0.0237 |                  0.0691 |
| sub-07    |        48273 |           64 |        48209 |           0.0013 |             19 | True           |                   0.0276 |                  0.0682 |
| sub-06    |        48040 |           31 |        48009 |           0.0006 |             18 | True           |                   0.0134 |                  0.0679 |
| sub-10    |        36007 |           88 |        35919 |           0.0024 |             25 | True           |                   0.0379 |                  0.0509 |
| sub-15    |        28792 |          399 |        28393 |           0.0139 |             40 | True           |                   0.1719 |                  0.0407 |
| sub-05    |        28078 |          112 |        27966 |           0.004  |             39 | True           |                   0.0483 |                  0.0397 |
| sub-03    |        27358 |           83 |        27275 |           0.003  |             38 | True           |                   0.0358 |                  0.0387 |
| sub-18    |        25652 |           64 |        25588 |           0.0025 |             36 | True           |                   0.0276 |                  0.0363 |

## 8. Subjects with zero positive segments

_No data_

## 9. Subject summary preview

| subject   |   n_segments |   n_positive |   n_negative |   positive_ratio |   n_recordings | has_positive   |   positive_share_overall |   segment_share_overall |
|:----------|-------------:|-------------:|-------------:|-----------------:|---------------:|:---------------|-------------------------:|------------------------:|
| sub-15    |        28792 |          399 |        28393 |           0.0139 |             40 | True           |                   0.1719 |                  0.0407 |
| sub-12    |        17027 |          294 |        16733 |           0.0173 |             24 | True           |                   0.1267 |                  0.0241 |
| sub-08    |        14397 |          182 |        14215 |           0.0126 |             20 | True           |                   0.0784 |                  0.0203 |
| sub-11    |        25046 |          160 |        24886 |           0.0064 |             35 | True           |                   0.0689 |                  0.0354 |
| sub-01    |        52827 |          129 |        52698 |           0.0024 |             42 | True           |                   0.0556 |                  0.0747 |
| sub-05    |        28078 |          112 |        27966 |           0.004  |             39 | True           |                   0.0483 |                  0.0397 |
| sub-13    |        23748 |          107 |        23641 |           0.0045 |             33 | True           |                   0.0461 |                  0.0336 |
| sub-24    |        15321 |          102 |        15219 |           0.0067 |             22 | True           |                   0.0439 |                  0.0217 |
| sub-10    |        36007 |           88 |        35919 |           0.0024 |             25 | True           |                   0.0379 |                  0.0509 |
| sub-23    |        19115 |           84 |        19031 |           0.0044 |              9 | True           |                   0.0362 |                  0.027  |
| sub-03    |        27358 |           83 |        27275 |           0.003  |             38 | True           |                   0.0358 |                  0.0387 |
| sub-04    |       112356 |           76 |       112280 |           0.0007 |             42 | True           |                   0.0327 |                  0.1588 |
| sub-07    |        48273 |           64 |        48209 |           0.0013 |             19 | True           |                   0.0276 |                  0.0682 |
| sub-18    |        25652 |           64 |        25588 |           0.0025 |             36 | True           |                   0.0276 |                  0.0363 |
| sub-17    |        15123 |           59 |        15064 |           0.0039 |             21 | True           |                   0.0254 |                  0.0214 |
| sub-20    |        19863 |           58 |        19805 |           0.0029 |             29 | True           |                   0.025  |                  0.0281 |
| sub-09    |        48861 |           55 |        48806 |           0.0011 |             19 | True           |                   0.0237 |                  0.0691 |
| sub-19    |        21543 |           46 |        21497 |           0.0021 |             30 | True           |                   0.0198 |                  0.0304 |
| sub-22    |        22320 |           41 |        22279 |           0.0018 |             31 | True           |                   0.0177 |                  0.0315 |
| sub-02    |        25389 |           35 |        25354 |           0.0014 |             36 | True           |                   0.0151 |                  0.0359 |
