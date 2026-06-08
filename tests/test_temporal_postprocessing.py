"""Tests for the temporal post-processing module (Sprint 2A)."""

import numpy as np
import pandas as pd
import pytest

from neuro_eeg_cdss.postprocessing.temporal import (
    PostprocessingError,
    TemporalConfig,
    _find_runs,
    apply_minimum_duration,
    compute_postprocessing_summary,
    enrich_predictions,
    median_filter_proba,
    moving_average_proba,
    postprocess_predictions,
    postprocess_recording,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_recording_proba() -> np.ndarray:
    """Simulated probability sequence with an isolated spike and a sustained event."""
    # Windows: 0    1     2    3    4     5     6     7    8    9
    return np.array([0.1, 0.8, 0.1, 0.1, 0.05, 0.85, 0.90, 0.88, 0.1, 0.05])
    #                     ^spike^         ^--- sustained seizure ---^


def _make_predictions_df() -> pd.DataFrame:
    """Create a predictions DataFrame with temporal metadata for 2 recordings."""
    # Recording 1: sub-01, path_a — isolated spike + sustained event
    rec1_proba = [0.1, 0.8, 0.1, 0.05, 0.85, 0.90, 0.88, 0.1]
    # Recording 2: sub-02, path_b — no seizure activity
    rec2_proba = [0.05, 0.1, 0.15, 0.05, 0.1]

    n1 = len(rec1_proba)
    n2 = len(rec2_proba)

    return pd.DataFrame(
        {
            "subject": ["sub-01"] * n1 + ["sub-02"] * n2,
            "path": ["path_a"] * n1 + ["path_b"] * n2,
            "start_sec": [i * 5.0 for i in range(n1)] + [i * 5.0 for i in range(n2)],
            "y_true": [0, 0, 0, 0, 1, 1, 1, 0] + [0, 0, 0, 0, 0],
            "y_pred": [0, 1, 0, 0, 1, 1, 1, 0] + [0, 0, 0, 0, 0],
            "y_proba": rec1_proba + rec2_proba,
        }
    )


# ── TestMedianFilterProba ────────────────────────────────────────────


class TestMedianFilterProba:
    """Tests for median_filter_proba."""

    def test_identity_with_kernel_1(self):
        proba = np.array([0.1, 0.8, 0.1])
        result = median_filter_proba(proba, kernel_size=1)
        np.testing.assert_array_equal(result, proba)

    def test_removes_isolated_spike(self):
        proba = np.array([0.1, 0.9, 0.1])
        result = median_filter_proba(proba, kernel_size=3)
        assert result[1] == pytest.approx(0.1)

    def test_preserves_sustained_high(self):
        proba = np.array([0.1, 0.9, 0.9, 0.9, 0.1])
        result = median_filter_proba(proba, kernel_size=3)
        assert result[2] == pytest.approx(0.9)

    def test_output_same_length(self):
        proba = _make_recording_proba()
        result = median_filter_proba(proba, kernel_size=5)
        assert len(result) == len(proba)

    def test_values_in_valid_range(self):
        proba = np.random.RandomState(42).uniform(0, 1, size=100)
        result = median_filter_proba(proba, kernel_size=5)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_rejects_even_kernel(self):
        with pytest.raises(PostprocessingError, match="odd"):
            median_filter_proba(np.array([0.1, 0.5, 0.9]), kernel_size=4)

    def test_rejects_non_1d(self):
        with pytest.raises(PostprocessingError, match="1D"):
            median_filter_proba(np.array([[0.1, 0.5]]), kernel_size=3)

    def test_empty_array(self):
        result = median_filter_proba(np.array([]), kernel_size=3)
        assert len(result) == 0

    def test_single_element(self):
        result = median_filter_proba(np.array([0.7]), kernel_size=3)
        assert result[0] == pytest.approx(0.7)


# ── TestMovingAverageProba ───────────────────────────────────────────


class TestMovingAverageProba:
    """Tests for moving_average_proba."""

    def test_identity_with_window_1(self):
        proba = np.array([0.1, 0.8, 0.1])
        result = moving_average_proba(proba, window_size=1)
        np.testing.assert_array_almost_equal(result, proba)

    def test_smooths_isolated_spike(self):
        proba = np.array([0.1, 0.9, 0.1])
        result = moving_average_proba(proba, window_size=3)
        expected_center = (0.1 + 0.9 + 0.1) / 3
        assert result[1] == pytest.approx(expected_center)

    def test_output_same_length(self):
        proba = np.array([0.1, 0.5, 0.9, 0.5, 0.1])
        result = moving_average_proba(proba, window_size=3)
        assert len(result) == len(proba)

    def test_values_in_valid_range(self):
        proba = np.random.RandomState(42).uniform(0, 1, size=100)
        result = moving_average_proba(proba, window_size=5)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_rejects_invalid_window(self):
        with pytest.raises(PostprocessingError, match="positive"):
            moving_average_proba(np.array([0.1, 0.5]), window_size=0)

    def test_edge_values_use_truncated_window(self):
        proba = np.array([0.2, 0.4, 0.6])
        result = moving_average_proba(proba, window_size=3)
        # First element: mean(0.2, 0.4) = 0.3 (truncated left)
        assert result[0] == pytest.approx(0.3)
        # Last element: mean(0.4, 0.6) = 0.5 (truncated right)
        assert result[2] == pytest.approx(0.5)


# ── TestApplyMinimumDuration ─────────────────────────────────────────


class TestApplyMinimumDuration:
    """Tests for apply_minimum_duration."""

    def test_removes_single_isolated_positive(self):
        y_pred = np.array([0, 1, 0, 0, 0])
        result = apply_minimum_duration(y_pred, min_windows=2)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])

    def test_preserves_long_run(self):
        y_pred = np.array([0, 1, 1, 1, 0])
        result = apply_minimum_duration(y_pred, min_windows=2)
        np.testing.assert_array_equal(result, [0, 1, 1, 1, 0])

    def test_mixed_runs(self):
        y_pred = np.array([0, 1, 0, 1, 1, 1, 0, 1, 1, 0])
        result = apply_minimum_duration(y_pred, min_windows=2)
        # Run of 1 at index 1: removed
        # Run of 3 at indices 3-5: kept
        # Run of 2 at indices 7-8: kept
        np.testing.assert_array_equal(result, [0, 0, 0, 1, 1, 1, 0, 1, 1, 0])

    def test_min_windows_1_is_identity(self):
        y_pred = np.array([0, 1, 0, 1, 0])
        result = apply_minimum_duration(y_pred, min_windows=1)
        np.testing.assert_array_equal(result, y_pred)

    def test_removes_all_when_min_exceeds_longest(self):
        y_pred = np.array([0, 1, 1, 0, 1, 0])
        result = apply_minimum_duration(y_pred, min_windows=3)
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 0])

    def test_empty_array(self):
        result = apply_minimum_duration(np.array([], dtype=int), min_windows=2)
        assert len(result) == 0

    def test_all_negative(self):
        y_pred = np.array([0, 0, 0, 0])
        result = apply_minimum_duration(y_pred, min_windows=2)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_all_positive(self):
        y_pred = np.array([1, 1, 1, 1])
        result = apply_minimum_duration(y_pred, min_windows=2)
        np.testing.assert_array_equal(result, [1, 1, 1, 1])

    def test_rejects_non_binary(self):
        with pytest.raises(PostprocessingError, match="0s and 1s"):
            apply_minimum_duration(np.array([0, 1, 2]), min_windows=2)

    def test_does_not_modify_original(self):
        y_pred = np.array([0, 1, 0])
        original = y_pred.copy()
        apply_minimum_duration(y_pred, min_windows=2)
        np.testing.assert_array_equal(y_pred, original)


# ── TestFindRuns ─────────────────────────────────────────────────────


class TestFindRuns:
    """Tests for _find_runs."""

    def test_single_run(self):
        runs = _find_runs(np.array([0, 1, 1, 1, 0]), value=1)
        assert runs == [(1, 3)]

    def test_multiple_runs(self):
        runs = _find_runs(np.array([1, 1, 0, 1, 0, 1, 1, 1]), value=1)
        assert runs == [(0, 2), (3, 1), (5, 3)]

    def test_no_runs(self):
        runs = _find_runs(np.array([0, 0, 0]), value=1)
        assert runs == []

    def test_empty_array(self):
        runs = _find_runs(np.array([], dtype=int), value=1)
        assert runs == []


# ── TestPostprocessRecording ─────────────────────────────────────────


class TestPostprocessRecording:
    """Tests for postprocess_recording."""

    def test_median_filter_strategy(self):
        proba = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.9])
        pred = (proba >= 0.5).astype(int)
        config = TemporalConfig(strategy="median_filter", kernel_size=3)

        y_pred_post, y_proba_post = postprocess_recording(proba, pred, config)

        # Isolated spike at index 1 should be suppressed
        assert y_pred_post[1] == 0
        # Sustained event at indices 3-5 should be preserved
        assert y_pred_post[4] == 1

    def test_moving_average_strategy(self):
        proba = np.array([0.1, 0.9, 0.1])
        pred = (proba >= 0.5).astype(int)
        config = TemporalConfig(strategy="moving_average", kernel_size=3)

        y_pred_post, y_proba_post = postprocess_recording(proba, pred, config)
        assert len(y_pred_post) == len(proba)

    def test_min_duration_strategy(self):
        proba = np.array([0.1, 0.9, 0.1, 0.9, 0.9, 0.9])
        pred = np.array([0, 1, 0, 1, 1, 1])
        config = TemporalConfig(strategy="min_duration", min_windows=2)

        y_pred_post, y_proba_post = postprocess_recording(proba, pred, config)

        # Single positive at index 1 removed
        assert y_pred_post[1] == 0
        # Run of 3 at indices 3-5 preserved
        np.testing.assert_array_equal(y_pred_post[3:6], [1, 1, 1])
        # Probabilities unchanged for min_duration
        np.testing.assert_array_equal(y_proba_post, proba)


# ── TestPostprocessPredictions ───────────────────────────────────────


class TestPostprocessPredictions:
    """Tests for postprocess_predictions (DataFrame-level)."""

    def test_adds_post_columns(self):
        df = _make_predictions_df()
        config = TemporalConfig(strategy="median_filter", kernel_size=3)
        result = postprocess_predictions(df, config)

        assert "y_pred_post" in result.columns
        assert "y_proba_post" in result.columns

    def test_preserves_original_columns(self):
        df = _make_predictions_df()
        config = TemporalConfig(strategy="median_filter", kernel_size=3)
        result = postprocess_predictions(df, config)

        assert "y_pred" in result.columns
        assert "y_proba" in result.columns
        assert "y_true" in result.columns

    def test_does_not_modify_input(self):
        df = _make_predictions_df()
        original_proba = df["y_proba"].copy()
        config = TemporalConfig(strategy="median_filter", kernel_size=3)
        postprocess_predictions(df, config)

        pd.testing.assert_series_equal(df["y_proba"], original_proba)

    def test_processes_recordings_independently(self):
        df = _make_predictions_df()
        config = TemporalConfig(strategy="min_duration", min_windows=2)
        result = postprocess_predictions(df, config)

        # Recording 2 (sub-02) has no positives; should remain all negative
        rec2 = result[result["subject"] == "sub-02"]
        assert (rec2["y_pred_post"] == 0).all()

    def test_removes_isolated_spike_in_recording(self):
        df = _make_predictions_df()
        config = TemporalConfig(strategy="min_duration", min_windows=2)
        result = postprocess_predictions(df, config)

        # Recording 1: isolated positive at start_sec=5.0 should be removed
        rec1 = result[result["subject"] == "sub-01"].sort_values("start_sec")
        isolated_pred = rec1[rec1["start_sec"] == 5.0]["y_pred_post"].iloc[0]
        assert isolated_pred == 0

    def test_preserves_sustained_event(self):
        df = _make_predictions_df()
        config = TemporalConfig(strategy="min_duration", min_windows=2)
        result = postprocess_predictions(df, config)

        # Recording 1: sustained event at start_sec 20, 25, 30 should survive
        rec1 = result[result["subject"] == "sub-01"].sort_values("start_sec")
        sustained = rec1[rec1["start_sec"].isin([20.0, 25.0, 30.0])]
        assert (sustained["y_pred_post"] == 1).all()

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"y_proba": [0.5], "y_true": [1]})
        config = TemporalConfig()
        with pytest.raises(PostprocessingError, match="missing"):
            postprocess_predictions(df, config)

    def test_rejects_invalid_strategy(self):
        df = _make_predictions_df()
        config = TemporalConfig(strategy="invalid")
        with pytest.raises(PostprocessingError, match="Unknown strategy"):
            postprocess_predictions(df, config)

    def test_output_same_length(self):
        df = _make_predictions_df()
        config = TemporalConfig(strategy="median_filter", kernel_size=3)
        result = postprocess_predictions(df, config)
        assert len(result) == len(df)


# ── TestEnrichPredictions ────────────────────────────────────────────


class TestEnrichPredictions:
    """Tests for enrich_predictions."""

    def test_adds_metadata_columns(self):
        preds = pd.DataFrame(
            {
                "y_true": [0, 1, 0],
                "y_pred": [0, 1, 0],
                "y_proba": [0.1, 0.9, 0.2],
            }
        )
        features = pd.DataFrame(
            {
                "subject": ["sub-01", "sub-01", "sub-01"],
                "path": ["p1", "p1", "p1"],
                "start_sec": [0.0, 5.0, 10.0],
                "end_sec": [5.0, 10.0, 15.0],
                "label": [0, 1, 0],
                "ch_01_mean": [1.0, 2.0, 3.0],
            }
        )

        result = enrich_predictions(preds, features)

        assert "subject" in result.columns
        assert "path" in result.columns
        assert "start_sec" in result.columns
        assert "end_sec" in result.columns

    def test_validates_alignment_with_labels(self):
        preds = pd.DataFrame(
            {
                "y_true": [0, 1, 0],
                "y_pred": [0, 1, 0],
                "y_proba": [0.1, 0.9, 0.2],
            }
        )
        features = pd.DataFrame(
            {
                "subject": ["sub-01", "sub-01", "sub-01"],
                "path": ["p1", "p1", "p1"],
                "start_sec": [0.0, 5.0, 10.0],
                "end_sec": [5.0, 10.0, 15.0],
                "label": [1, 0, 0],  # Mismatched!
            }
        )

        with pytest.raises(PostprocessingError, match="does not match"):
            enrich_predictions(preds, features)

    def test_rejects_length_mismatch(self):
        preds = pd.DataFrame({"y_true": [0, 1], "y_proba": [0.1, 0.9]})
        features = pd.DataFrame(
            {
                "subject": ["sub-01"],
                "path": ["p1"],
                "start_sec": [0.0],
                "end_sec": [5.0],
            }
        )

        with pytest.raises(PostprocessingError, match="Length mismatch"):
            enrich_predictions(preds, features)


# ── TestTemporalConfig ───────────────────────────────────────────────


class TestTemporalConfig:
    """Tests for TemporalConfig."""

    def test_default_values(self):
        config = TemporalConfig()
        assert config.strategy == "median_filter"
        assert config.kernel_size == 3
        assert config.min_windows == 2
        assert config.threshold == 0.5

    def test_name_median(self):
        config = TemporalConfig(strategy="median_filter", kernel_size=5, threshold=0.3)
        assert config.name == "median_k5_t0.3"

    def test_name_moving_average(self):
        config = TemporalConfig(strategy="moving_average", kernel_size=7, threshold=0.4)
        assert config.name == "mavg_k7_t0.4"

    def test_name_min_duration(self):
        config = TemporalConfig(strategy="min_duration", min_windows=3)
        assert config.name == "mindur_w3"

    def test_frozen(self):
        config = TemporalConfig()
        with pytest.raises(AttributeError):
            config.strategy = "other"


# ── TestComputePostprocessingSummary ─────────────────────────────────


class TestComputePostprocessingSummary:
    """Tests for compute_postprocessing_summary."""

    def test_counts_changes(self):
        df = pd.DataFrame(
            {
                "y_pred": [0, 1, 0, 1, 1],
                "y_pred_post": [0, 0, 0, 1, 1],
            }
        )
        config = TemporalConfig(strategy="min_duration", min_windows=2)
        summary = compute_postprocessing_summary(df, config)

        assert summary["n_predictions_changed"] == 1
        assert summary["positive_to_negative"] == 1
        assert summary["negative_to_positive"] == 0

    def test_no_changes(self):
        df = pd.DataFrame(
            {
                "y_pred": [0, 1, 1, 0],
                "y_pred_post": [0, 1, 1, 0],
            }
        )
        config = TemporalConfig()
        summary = compute_postprocessing_summary(df, config)

        assert summary["n_predictions_changed"] == 0
        assert summary["change_rate"] == 0.0

    def test_includes_config_info(self):
        df = pd.DataFrame(
            {
                "y_pred": [0, 1],
                "y_pred_post": [0, 1],
            }
        )
        config = TemporalConfig(strategy="median_filter", kernel_size=5)
        summary = compute_postprocessing_summary(df, config)

        assert summary["config"]["strategy"] == "median_filter"
        assert summary["config"]["kernel_size"] == 5
