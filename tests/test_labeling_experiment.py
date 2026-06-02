"""Tests for the labeling experiment module (Sprint 1F)."""

import numpy as np
import pandas as pd
import pytest

from neuro_eeg_cdss.data.splits import SplitAssignment, SplitConfig
from neuro_eeg_cdss.experiments.labeling import (
    ALL_CONFIGS,
    ExperimentError,
    ExperimentResult,
    LabelingExperimentConfig,
    analyze_data_completeness,
    compute_dataset_stats,
    format_comparison_table,
    join_overlap_ratio,
    relabel_dataset,
    run_single_experiment,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_test_data(
    n_per_subject: int = 100,
    n_subjects: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create aligned features and segments DataFrames for testing.

    Each subject contributes ``n_per_subject`` windows.  The first three
    windows of each subject receive non-zero overlap values (1.0, 0.8,
    0.6) to exercise the labeling logic at every threshold.

    Returns
    -------
    features_df : pd.DataFrame
        Synthetic feature dataset (2 features per window).
    segments_df : pd.DataFrame
        Corresponding segment dataset with overlap_ratio.
    """
    rng = np.random.RandomState(seed)

    subjects = [f"sub-{i + 1:02d}" for i in range(n_subjects)]
    rows: list[dict] = []

    for subject in subjects:
        for j in range(n_per_subject):
            rows.append(
                {
                    "subject": subject,
                    "session": None,
                    "run": None,
                    "path": f"/data/{subject}_eeg.edf",
                    "start_sec": float(j * 5),
                    "end_sec": float((j + 1) * 5),
                    "label": 0,
                    "mean_ch_01": rng.normal(0, 1),
                    "std_ch_01": abs(rng.normal(1, 0.5)),
                }
            )

    features_df = pd.DataFrame(rows)

    # Build segments with overlap_ratio
    segments_df = features_df[["subject", "session", "run", "path", "start_sec", "end_sec"]].copy()
    segments_df["overlap_ratio"] = 0.0
    segments_df["label"] = 0
    segments_df["recording_duration_sec"] = 3600.0
    segments_df["window_size_sec"] = 5.0
    segments_df["stride_sec"] = 5.0

    # Assign overlap values to first 3 windows per subject
    for i in range(n_subjects):
        base = i * n_per_subject
        segments_df.loc[base, "overlap_ratio"] = 1.0
        segments_df.loc[base + 1, "overlap_ratio"] = 0.8
        segments_df.loc[base + 2, "overlap_ratio"] = 0.6

    return features_df, segments_df


def _make_split_assignment(n_subjects: int = 6) -> SplitAssignment:
    """
    Create a deterministic split for the synthetic test data.

    With 6 subjects: train=3, val=1, test=2.
    """
    subjects = [f"sub-{i + 1:02d}" for i in range(n_subjects)]
    train_end = max(1, int(n_subjects * 0.6))
    val_end = max(train_end + 1, int(n_subjects * 0.8))

    return SplitAssignment(
        train_subjects=tuple(subjects[:train_end]),
        val_subjects=tuple(subjects[train_end:val_end]),
        test_subjects=tuple(subjects[val_end:]),
        config=SplitConfig(),
    )


# ── LabelingExperimentConfig ────────────────────────────────────────


class TestLabelingExperimentConfig:
    def test_name_drop(self):
        c = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        assert c.name == "thresh_0.5_drop"

    def test_name_keep(self):
        c = LabelingExperimentConfig(threshold=0.3, drop_partial=False)
        assert c.name == "thresh_0.3_keep"

    def test_all_configs_count(self):
        assert len(ALL_CONFIGS) == 6

    def test_all_configs_unique_names(self):
        names = [c.name for c in ALL_CONFIGS]
        assert len(names) == len(set(names))


# ── join_overlap_ratio ───────────────────────────────────────────────


class TestJoinOverlapRatio:
    def test_adds_overlap_ratio_column(self):
        feat_df, seg_df = _make_test_data()
        merged = join_overlap_ratio(feat_df, seg_df)

        assert "overlap_ratio" in merged.columns
        assert len(merged) == len(feat_df)

    def test_preserves_all_feature_columns(self):
        feat_df, seg_df = _make_test_data()
        merged = join_overlap_ratio(feat_df, seg_df)

        for col in feat_df.columns:
            assert col in merged.columns

    def test_overlap_values_are_correct(self):
        feat_df, seg_df = _make_test_data(n_per_subject=10, n_subjects=2)
        merged = join_overlap_ratio(feat_df, seg_df)

        # First window of first subject should have overlap=1.0
        assert merged.iloc[0]["overlap_ratio"] == 1.0
        # Fourth window should have overlap=0
        assert merged.iloc[3]["overlap_ratio"] == 0.0

    def test_raises_on_missing_join_key(self):
        feat_df, seg_df = _make_test_data()
        feat_broken = feat_df.drop(columns=["subject"])

        with pytest.raises(ExperimentError, match="Missing join key"):
            join_overlap_ratio(feat_broken, seg_df)

    def test_raises_on_missing_overlap_ratio(self):
        feat_df, seg_df = _make_test_data()
        seg_broken = seg_df.drop(columns=["overlap_ratio"])

        with pytest.raises(ExperimentError, match="overlap_ratio"):
            join_overlap_ratio(feat_df, seg_broken)

    def test_raises_on_row_count_mismatch(self):
        feat_df, seg_df = _make_test_data()
        seg_short = seg_df.iloc[:10]

        with pytest.raises(ExperimentError, match="row count"):
            join_overlap_ratio(feat_df, seg_short)


# ── relabel_dataset ──────────────────────────────────────────────────


class TestRelabelDataset:
    def test_threshold_05_preserves_positives(self):
        """At threshold=0.5, windows with overlap 0.6, 0.8, 1.0 are positive."""
        feat_df, seg_df = _make_test_data(n_subjects=2)
        merged = join_overlap_ratio(feat_df, seg_df)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        relabeled = relabel_dataset(merged, config)

        # 3 positives per subject * 2 subjects = 6
        n_pos = int((relabeled["label"] == 1).sum())
        assert n_pos == 6

    def test_threshold_07_drop_removes_partial(self):
        """At threshold=0.7+drop, overlap=0.6 windows are dropped."""
        feat_df, seg_df = _make_test_data(n_subjects=2)
        merged = join_overlap_ratio(feat_df, seg_df)

        config = LabelingExperimentConfig(threshold=0.7, drop_partial=True)
        relabeled = relabel_dataset(merged, config)

        # 2 positives per subject (overlap 0.8, 1.0), 1 dropped (0.6)
        n_pos = int((relabeled["label"] == 1).sum())
        assert n_pos == 4
        assert len(relabeled) == len(merged) - 2  # 2 windows dropped

    def test_threshold_07_keep_relabels_partial_as_negative(self):
        """At threshold=0.7+keep, overlap=0.6 windows become negative."""
        feat_df, seg_df = _make_test_data(n_subjects=2)
        merged = join_overlap_ratio(feat_df, seg_df)

        config = LabelingExperimentConfig(threshold=0.7, drop_partial=False)
        relabeled = relabel_dataset(merged, config)

        # 2 positives per subject (overlap 0.8, 1.0)
        n_pos = int((relabeled["label"] == 1).sum())
        assert n_pos == 4
        # No rows dropped
        assert len(relabeled) == len(merged)

    def test_threshold_03_includes_all_nonzero(self):
        """At threshold=0.3, overlap 0.6, 0.8, 1.0 are all positive."""
        feat_df, seg_df = _make_test_data(n_subjects=2)
        merged = join_overlap_ratio(feat_df, seg_df)

        config = LabelingExperimentConfig(threshold=0.3, drop_partial=True)
        relabeled = relabel_dataset(merged, config)

        # All non-zero overlap windows are >= 0.3
        n_pos = int((relabeled["label"] == 1).sum())
        assert n_pos == 6  # Same as threshold=0.5

    def test_removes_overlap_ratio_column(self):
        feat_df, seg_df = _make_test_data()
        merged = join_overlap_ratio(feat_df, seg_df)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        relabeled = relabel_dataset(merged, config)

        assert "overlap_ratio" not in relabeled.columns

    def test_raises_without_overlap_ratio(self):
        feat_df, _ = _make_test_data()
        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)

        with pytest.raises(ExperimentError, match="overlap_ratio"):
            relabel_dataset(feat_df, config)

    def test_zero_overlap_always_negative(self):
        """Windows with overlap=0 are always labeled negative."""
        feat_df, seg_df = _make_test_data(n_subjects=1, n_per_subject=10)
        merged = join_overlap_ratio(feat_df, seg_df)

        for threshold in [0.3, 0.5, 0.7]:
            config = LabelingExperimentConfig(threshold=threshold, drop_partial=False)
            relabeled = relabel_dataset(merged, config)

            # Windows 3-9 have overlap=0.0 → always negative
            zero_overlap_labels = relabeled.iloc[3:]["label"]
            assert (zero_overlap_labels == 0).all()


# ── compute_dataset_stats ────────────────────────────────────────────


class TestComputeDatasetStats:
    def test_counts_are_consistent(self):
        feat_df, seg_df = _make_test_data()
        merged = join_overlap_ratio(feat_df, seg_df)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        relabeled = relabel_dataset(merged, config)

        stats = compute_dataset_stats(merged, relabeled)

        assert stats["n_positive"] + stats["n_negative"] == stats["n_relabeled"]
        assert stats["n_relabeled"] + stats["n_dropped"] == stats["n_original"]

    def test_drop_increases_dropped_count(self):
        feat_df, seg_df = _make_test_data()
        merged = join_overlap_ratio(feat_df, seg_df)

        config_drop = LabelingExperimentConfig(threshold=0.7, drop_partial=True)
        relabeled = relabel_dataset(merged, config_drop)
        stats = compute_dataset_stats(merged, relabeled)

        assert stats["n_dropped"] > 0

    def test_keep_has_zero_dropped(self):
        feat_df, seg_df = _make_test_data()
        merged = join_overlap_ratio(feat_df, seg_df)

        config_keep = LabelingExperimentConfig(threshold=0.7, drop_partial=False)
        relabeled = relabel_dataset(merged, config_keep)
        stats = compute_dataset_stats(merged, relabeled)

        assert stats["n_dropped"] == 0


# ── analyze_data_completeness ────────────────────────────────────────


class TestAnalyzeDataCompleteness:
    def test_returns_expected_keys(self):
        _, seg_df = _make_test_data()
        result = analyze_data_completeness(seg_df)

        assert "total_windows" in result
        assert "zero_overlap" in result
        assert "partial_overlap_values" in result

    def test_counts_match(self):
        _, seg_df = _make_test_data(n_subjects=2, n_per_subject=10)
        result = analyze_data_completeness(seg_df)

        assert result["total_windows"] == 20
        # Per subject: 1 full (1.0) + 2 partial (0.8, 0.6) + 7 zero
        assert result["full_overlap"] == 2  # 1 per subject
        assert result["partial_overlap_count"] == 4  # 2 per subject


# ── run_single_experiment ────────────────────────────────────────────


class TestRunSingleExperiment:
    def test_returns_experiment_result(self):
        feat_df, seg_df = _make_test_data(n_per_subject=200, n_subjects=6)
        merged = join_overlap_ratio(feat_df, seg_df)
        split = _make_split_assignment(n_subjects=6)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        result = run_single_experiment(merged, split, config)

        assert isinstance(result, ExperimentResult)
        assert "test" in result.metrics_by_split
        assert result.metrics_by_split["test"].n_samples > 0

    def test_all_splits_have_metrics(self):
        feat_df, seg_df = _make_test_data(n_per_subject=200, n_subjects=6)
        merged = join_overlap_ratio(feat_df, seg_df)
        split = _make_split_assignment(n_subjects=6)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        result = run_single_experiment(merged, split, config)

        for split_name in ["train", "val", "test"]:
            assert split_name in result.metrics_by_split
            m = result.metrics_by_split[split_name]
            assert 0.0 <= m.sensitivity <= 1.0
            assert 0.0 <= m.specificity <= 1.0
            assert 0.0 <= m.auroc <= 1.0

    def test_to_dict_is_json_serializable(self):
        import json

        feat_df, seg_df = _make_test_data(n_per_subject=200, n_subjects=6)
        merged = join_overlap_ratio(feat_df, seg_df)
        split = _make_split_assignment(n_subjects=6)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        result = run_single_experiment(merged, split, config)

        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_different_threshold_changes_metrics(self):
        """Threshold 0.7 should produce different metrics than 0.5."""
        feat_df, seg_df = _make_test_data(n_per_subject=200, n_subjects=6)
        merged = join_overlap_ratio(feat_df, seg_df)
        split = _make_split_assignment(n_subjects=6)

        config_05 = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        config_07 = LabelingExperimentConfig(threshold=0.7, drop_partial=True)

        result_05 = run_single_experiment(merged, split, config_05)
        result_07 = run_single_experiment(merged, split, config_07)

        # Different number of positives
        assert result_05.dataset_stats["n_positive"] != result_07.dataset_stats["n_positive"]


# ── format_comparison_table ──────────────────────────────────────────


class TestFormatComparisonTable:
    def test_contains_config_name(self):
        feat_df, seg_df = _make_test_data(n_per_subject=200, n_subjects=6)
        merged = join_overlap_ratio(feat_df, seg_df)
        split = _make_split_assignment(n_subjects=6)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        result = run_single_experiment(merged, split, config)

        table = format_comparison_table([result], split_name="test")

        assert "thresh_0.5_drop" in table
        assert "LABELING EXPERIMENT" in table

    def test_contains_metric_headers(self):
        feat_df, seg_df = _make_test_data(n_per_subject=200, n_subjects=6)
        merged = join_overlap_ratio(feat_df, seg_df)
        split = _make_split_assignment(n_subjects=6)

        config = LabelingExperimentConfig(threshold=0.5, drop_partial=True)
        result = run_single_experiment(merged, split, config)

        table = format_comparison_table([result], split_name="test")

        assert "Sens" in table
        assert "Spec" in table
        assert "AUROC" in table
        assert "AUPRC" in table
