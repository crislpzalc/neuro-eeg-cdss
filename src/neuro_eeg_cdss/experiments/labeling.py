"""
Labeling strategy experiments for seizure detection.

This module implements a systematic comparison of different labeling
policies by varying the positive overlap threshold and the partial-overlap
handling strategy (drop vs. keep as negative).

Experiment design
-----------------
Instead of rebuilding the entire dataset from raw EEG for each
configuration (expensive, requires mne and the full CHB-MIT dataset), we
JOIN the existing ``features.parquet`` with ``segments.parquet`` to
recover ``overlap_ratio``, then relabel windows in-memory for each
experiment.

The 6 canonical configurations are:

    3 thresholds (0.3, 0.5, 0.7) x 2 drop policies (drop, keep) = 6

Data completeness limitation
----------------------------
The original dataset was built with ``threshold=0.5`` and
``drop_partial_overlap=True``.  Windows with ``0 < overlap_ratio < 0.5``
were excluded during construction and their EEG features are unavailable.
Analysis of the seizure boundaries shows that an estimated ~74 windows
with ``overlap ~= 0.4`` and ~82 with ``overlap ~= 0.2`` are missing.

Impact by threshold:
- **threshold=0.3**: ~74 would-be positives missing (overlap 0.4 >= 0.3).
  Results are slightly pessimistic.  Documented as limitation.
- **threshold=0.5**: Complete (this is the original build config).
- **threshold=0.7**: Complete.  All affected windows (overlap=0.6) are
  present in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neuro_eeg_cdss.data.splits import SplitAssignment, apply_split, load_split
from neuro_eeg_cdss.evaluation.metrics import WindowMetrics, compute_window_metrics
from neuro_eeg_cdss.training.trainer import (
    TrainConfig,
    predict,
    separate_features_and_labels,
    train_model,
)


class ExperimentError(ValueError):
    """Raised when an experiment encounters invalid inputs or state."""


# ── Configuration ────────────────────────────────────────────────────


@dataclass(frozen=True)
class LabelingExperimentConfig:
    """
    Configuration for one labeling experiment.

    Attributes
    ----------
    threshold : float
        Overlap ratio at or above which a window is labeled positive.
        Must be in the interval (0, 1].
    drop_partial : bool
        If True, windows with ``0 < overlap < threshold`` are excluded
        from the dataset.  If False, they are included as negatives.
    """

    threshold: float
    drop_partial: bool

    @property
    def name(self) -> str:
        """Human-readable experiment identifier."""
        drop_str = "drop" if self.drop_partial else "keep"
        return f"thresh_{self.threshold}_{drop_str}"


# The 6 canonical configurations from the roadmap:
# 3 thresholds x 2 drop policies.
ALL_CONFIGS: list[LabelingExperimentConfig] = [
    LabelingExperimentConfig(threshold=0.3, drop_partial=True),
    LabelingExperimentConfig(threshold=0.3, drop_partial=False),
    LabelingExperimentConfig(threshold=0.5, drop_partial=True),
    LabelingExperimentConfig(threshold=0.5, drop_partial=False),
    LabelingExperimentConfig(threshold=0.7, drop_partial=True),
    LabelingExperimentConfig(threshold=0.7, drop_partial=False),
]


# ── Result container ─────────────────────────────────────────────────


@dataclass
class ExperimentResult:
    """
    Complete result for one labeling experiment.

    Attributes
    ----------
    config : LabelingExperimentConfig
        The labeling policy used.
    dataset_stats : dict
        Counts and prevalence for the relabeled dataset.
    metrics_by_split : dict[str, WindowMetrics]
        Clinical metrics per split (train, val, test).
    """

    config: LabelingExperimentConfig
    dataset_stats: dict[str, Any]
    metrics_by_split: dict[str, WindowMetrics]

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "config": {
                "threshold": self.config.threshold,
                "drop_partial": self.config.drop_partial,
                "name": self.config.name,
            },
            "dataset_stats": self.dataset_stats,
            "metrics": {split: m.to_dict() for split, m in self.metrics_by_split.items()},
        }


# ── Data preparation ─────────────────────────────────────────────────


def join_overlap_ratio(
    features_df: pd.DataFrame,
    segments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join features with segments to add the ``overlap_ratio`` column.

    The join is performed on the composite key
    ``(subject, path, start_sec, end_sec)`` which uniquely identifies
    each window in both DataFrames.

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature dataset (from ``features.parquet``).
    segments_df : pd.DataFrame
        Segment dataset (from ``segments.parquet``) containing the
        ``overlap_ratio`` column.

    Returns
    -------
    pd.DataFrame
        ``features_df`` with ``overlap_ratio`` appended.

    Raises
    ------
    ExperimentError
        If required columns are missing or the join is not 1:1.
    """
    join_keys = ["subject", "path", "start_sec", "end_sec"]

    for key in join_keys:
        if key not in features_df.columns:
            raise ExperimentError(f"Missing join key '{key}' in features DataFrame.")
        if key not in segments_df.columns:
            raise ExperimentError(f"Missing join key '{key}' in segments DataFrame.")

    if "overlap_ratio" not in segments_df.columns:
        raise ExperimentError("Missing 'overlap_ratio' column in segments DataFrame.")

    segments_subset = segments_df[join_keys + ["overlap_ratio"]].copy()
    merged = features_df.merge(segments_subset, on=join_keys, how="inner")

    if len(merged) != len(features_df):
        raise ExperimentError(
            f"Join changed row count: {len(features_df)} -> {len(merged)}. "
            "Features and segments are not aligned."
        )

    return merged


def relabel_dataset(
    df: pd.DataFrame,
    config: LabelingExperimentConfig,
) -> pd.DataFrame:
    """
    Relabel a dataset using a new threshold/drop policy.

    The function expects the DataFrame to contain an ``overlap_ratio``
    column (typically added by :func:`join_overlap_ratio`).  Labeling
    rules:

    - ``overlap_ratio >= threshold`` -> label = 1 (positive)
    - ``overlap_ratio == 0``         -> label = 0 (negative)
    - ``0 < overlap_ratio < threshold``:
        - ``drop_partial=True``  -> row excluded
        - ``drop_partial=False`` -> label = 0 (negative)

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``overlap_ratio`` column.
    config : LabelingExperimentConfig
        Labeling policy to apply.

    Returns
    -------
    pd.DataFrame
        Relabeled dataset.  May have fewer rows if ``drop_partial=True``
        and partial-overlap windows exist.  The ``overlap_ratio`` column
        is removed (it is not a model feature).

    Raises
    ------
    ExperimentError
        If the ``overlap_ratio`` column is missing.
    """
    if "overlap_ratio" not in df.columns:
        raise ExperimentError("DataFrame must contain 'overlap_ratio' for relabeling.")

    result = df.copy()
    overlap = result["overlap_ratio"].values

    # Vectorized labeling decisions
    is_positive = overlap >= config.threshold
    is_zero = overlap == 0.0
    is_partial = (~is_positive) & (~is_zero)

    # Assign labels: positive if >= threshold, else negative
    result["label"] = np.where(is_positive, 1, 0).astype(int)

    # Handle partial-overlap windows
    if config.drop_partial:
        result = result[~is_partial].copy()

    # overlap_ratio is not a model feature — remove it
    result = result.drop(columns=["overlap_ratio"])

    return result.reset_index(drop=True)


def compute_dataset_stats(
    original_df: pd.DataFrame,
    relabeled_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute statistics comparing original and relabeled datasets.

    Parameters
    ----------
    original_df : pd.DataFrame
        Dataset before relabeling (with overlap_ratio).
    relabeled_df : pd.DataFrame
        Dataset after relabeling and optional row dropping.

    Returns
    -------
    dict
        Counts and prevalence for both the original and relabeled
        datasets.
    """
    n_original = len(original_df)
    n_relabeled = len(relabeled_df)
    n_positive = int((relabeled_df["label"] == 1).sum())
    n_negative = int((relabeled_df["label"] == 0).sum())

    return {
        "n_original": n_original,
        "n_relabeled": n_relabeled,
        "n_dropped": n_original - n_relabeled,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "prevalence": round(n_positive / max(n_relabeled, 1), 6),
    }


def analyze_data_completeness(segments_df: pd.DataFrame) -> dict[str, Any]:
    """
    Analyze which overlap_ratio values exist in the dataset.

    This helps identify which experiment configurations have complete
    data and which are affected by missing windows (dropped during the
    original dataset construction with threshold=0.5, drop=True).

    Parameters
    ----------
    segments_df : pd.DataFrame
        Segment dataset containing ``overlap_ratio``.

    Returns
    -------
    dict
        Summary of overlap distribution and data completeness notes.
    """
    overlap = segments_df["overlap_ratio"]
    partial = overlap[(overlap > 0) & (overlap < 1)]

    return {
        "total_windows": len(segments_df),
        "zero_overlap": int((overlap == 0).sum()),
        "full_overlap": int((overlap == 1.0).sum()),
        "partial_overlap_count": len(partial),
        "partial_overlap_values": sorted(partial.unique().tolist()),
        "note": (
            "Windows with 0 < overlap < 0.5 were excluded during the "
            "original dataset construction (threshold=0.5, drop=True). "
            "This primarily affects threshold=0.3 experiments where "
            "~74 windows with overlap~=0.4 (should be positive) are "
            "missing from the feature set."
        ),
    }


# ── Single experiment ────────────────────────────────────────────────


def run_single_experiment(
    merged_df: pd.DataFrame,
    split_assignment: SplitAssignment,
    config: LabelingExperimentConfig,
) -> ExperimentResult:
    """
    Run one labeling experiment.

    Pipeline:

    1. Relabel the merged dataset using the experiment's policy
    2. Split into train/val/test using the patient-independent split
    3. Train Logistic Regression with ``class_weight="balanced"``
    4. Generate predictions on all splits
    5. Compute clinical metrics (19 metrics per split)

    Parameters
    ----------
    merged_df : pd.DataFrame
        Feature dataset with ``overlap_ratio`` column (from
        :func:`join_overlap_ratio`).
    split_assignment : SplitAssignment
        Patient-independent split assignment.
    config : LabelingExperimentConfig
        Labeling policy for this experiment.

    Returns
    -------
    ExperimentResult
        Complete experiment results including dataset statistics and
        per-split clinical metrics.

    Raises
    ------
    ExperimentError
        If any split has zero positive samples after relabeling.

    Notes
    -----
    Only Logistic Regression is trained because Random Forest was shown
    in Sprint 1E to overfit catastrophically (0% sensitivity on test),
    making it unsuitable for meaningful comparison across labeling
    strategies.
    """
    # 1. Relabel
    relabeled_df = relabel_dataset(merged_df, config)
    stats = compute_dataset_stats(merged_df, relabeled_df)

    # 2. Split
    train_df, val_df, test_df = apply_split(relabeled_df, split_assignment)

    # Verify each split has positives
    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        n_pos = int((split_df["label"] == 1).sum())
        if n_pos == 0:
            raise ExperimentError(
                f"Split '{split_name}' has zero positive samples for "
                f"config '{config.name}'. Cannot train or evaluate."
            )

    # 3. Train LR
    train_config = TrainConfig(
        model_type="logistic_regression",
        seed=42,
        scale_features=True,
    )
    train_result = train_model(train_df, config=train_config)

    # 4-5. Predict and evaluate on each split
    metrics_by_split: dict[str, WindowMetrics] = {}

    for split_name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        _, y_true, _ = separate_features_and_labels(split_df)
        y_pred, y_proba = predict(train_result, split_df)
        metrics = compute_window_metrics(y_true.values, y_pred, y_proba)
        metrics_by_split[split_name] = metrics

    return ExperimentResult(
        config=config,
        dataset_stats=stats,
        metrics_by_split=metrics_by_split,
    )


# ── Batch runner ─────────────────────────────────────────────────────


def run_all_experiments(
    features_path: str | Path,
    segments_path: str | Path,
    splits_dir: str | Path,
    configs: list[LabelingExperimentConfig] | None = None,
) -> list[ExperimentResult]:
    """
    Run all labeling experiments end-to-end.

    Loads the data once, then iterates over configurations.

    Parameters
    ----------
    features_path : str | Path
        Path to ``features.parquet``.
    segments_path : str | Path
        Path to ``segments.parquet``.
    splits_dir : str | Path
        Directory containing split JSON files.
    configs : list[LabelingExperimentConfig] | None
        Experiment configurations.  Defaults to :data:`ALL_CONFIGS`.

    Returns
    -------
    list[ExperimentResult]
        Results for all experiments, in config order.
    """
    if configs is None:
        configs = ALL_CONFIGS

    features_df = pd.read_parquet(features_path)
    segments_df = pd.read_parquet(segments_path)
    split_assignment = load_split(splits_dir)

    merged_df = join_overlap_ratio(features_df, segments_df)

    completeness = analyze_data_completeness(segments_df)
    print("\n  Data completeness analysis:")
    print(f"    Total windows:      {completeness['total_windows']:,}")
    print(f"    Zero overlap:       {completeness['zero_overlap']:,}")
    print(f"    Full overlap (1.0): {completeness['full_overlap']:,}")
    print(f"    Partial overlap:    {completeness['partial_overlap_count']}")
    print(f"    Partial values:     {completeness['partial_overlap_values']}")
    print(f"    Note: {completeness['note']}")

    results: list[ExperimentResult] = []

    for config in configs:
        print(f"\n  Running: {config.name} ...")
        result = run_single_experiment(merged_df, split_assignment, config)

        m_test = result.metrics_by_split["test"]
        print(
            f"    Dataset: {result.dataset_stats['n_relabeled']:,} windows "
            f"({result.dataset_stats['n_positive']:,} positive, "
            f"{result.dataset_stats['n_dropped']} dropped)"
        )
        print(
            f"    Test:  Sens={m_test.sensitivity:.4f}  "
            f"Spec={m_test.specificity:.4f}  "
            f"F2={m_test.f2:.4f}  "
            f"AUROC={m_test.auroc:.4f}  "
            f"AUPRC={m_test.auprc:.4f}"
        )

        results.append(result)

    return results


# ── Formatting ───────────────────────────────────────────────────────


def format_comparison_table(
    results: list[ExperimentResult],
    split_name: str = "test",
) -> str:
    """
    Format a comparison table of all experiment results.

    Parameters
    ----------
    results : list[ExperimentResult]
        Experiment results to compare.
    split_name : str
        Which split to show metrics for.

    Returns
    -------
    str
        Formatted comparison table ready for printing or saving.
    """
    lines: list[str] = []
    lines.append("=" * 110)
    lines.append(f"  LABELING EXPERIMENT COMPARISON — {split_name.upper()} SET")
    lines.append("=" * 110)

    header = (
        f"  {'Config':<25s}  {'Thresh':>6s}  {'Drop':>5s}  "
        f"{'N+':>6s}  {'Prev':>8s}  "
        f"{'Sens':>7s}  {'Spec':>7s}  {'F1':>7s}  "
        f"{'F2':>7s}  {'AUROC':>7s}  {'AUPRC':>7s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 106)

    for r in results:
        m = r.metrics_by_split[split_name]
        drop_str = "Yes" if r.config.drop_partial else "No"
        lines.append(
            f"  {r.config.name:<25s}  {r.config.threshold:>6.1f}  "
            f"{drop_str:>5s}  {r.dataset_stats['n_positive']:>6d}  "
            f"{r.dataset_stats['prevalence']:>8.4%}  "
            f"{m.sensitivity:>7.4f}  {m.specificity:>7.4f}  "
            f"{m.f1:>7.4f}  {m.f2:>7.4f}  "
            f"{m.auroc:>7.4f}  {m.auprc:>7.4f}"
        )

    lines.append("=" * 110)
    return "\n".join(lines)
