"""Tests for the clinical evaluation metrics module."""

import numpy as np
import pytest

from neuro_eeg_cdss.evaluation.metrics import (
    EvaluationError,
    WindowMetrics,
    compute_pr_curve,
    compute_roc_curve,
    compute_threshold_analysis,
    compute_window_metrics,
    format_metrics_report,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_predictions(
    n_positive: int = 50,
    n_negative: int = 950,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic predictions with realistic separation."""
    rng = np.random.RandomState(seed)

    y_true = np.array([1] * n_positive + [0] * n_negative)

    # Probabilities: positives centered around 0.7, negatives around 0.2
    proba_pos = np.clip(rng.normal(0.7, 0.2, n_positive), 0, 1)
    proba_neg = np.clip(rng.normal(0.2, 0.15, n_negative), 0, 1)
    y_proba = np.concatenate([proba_pos, proba_neg])

    y_pred = (y_proba >= 0.5).astype(int)

    return y_true, y_pred, y_proba


def _make_perfect_predictions(
    n_positive: int = 50,
    n_negative: int = 950,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create perfect predictions."""
    y_true = np.array([1] * n_positive + [0] * n_negative)
    y_pred = y_true.copy()
    y_proba = y_true.astype(float)
    return y_true, y_pred, y_proba


# ── compute_window_metrics ───────────────────────────────────────────


class TestComputeWindowMetrics:
    def test_basic_computation(self):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        assert isinstance(m, WindowMetrics)
        assert m.n_samples == 1000
        assert m.n_positive == 50
        assert m.n_negative == 950

    def test_counts_sum_to_total(self):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        assert m.tp + m.fn + m.fp + m.tn == m.n_samples
        assert m.tp + m.fn == m.n_positive
        assert m.fp + m.tn == m.n_negative

    def test_metrics_in_valid_range(self):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        for metric in [
            m.sensitivity,
            m.specificity,
            m.precision,
            m.npv,
            m.f1,
            m.f2,
            m.accuracy,
            m.balanced_accuracy,
            m.fpr,
            m.auroc,
            m.auprc,
        ]:
            assert 0.0 <= metric <= 1.0, f"Metric out of range: {metric}"

    def test_fpr_complement_of_specificity(self):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        assert abs(m.fpr + m.specificity - 1.0) < 0.001

    def test_perfect_predictions(self):
        y_true, y_pred, y_proba = _make_perfect_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        assert m.sensitivity == 1.0
        assert m.specificity == 1.0
        assert m.precision == 1.0
        assert m.npv == 1.0
        assert m.f1 == 1.0
        assert m.f2 == 1.0
        assert m.accuracy == 1.0
        assert m.fpr == 0.0

    def test_prevalence_correct(self):
        y_true, y_pred, y_proba = _make_predictions(n_positive=30, n_negative=970)
        m = compute_window_metrics(y_true, y_pred, y_proba)

        assert m.prevalence == pytest.approx(0.03, abs=0.001)

    def test_empty_arrays_raises(self):
        with pytest.raises(EvaluationError, match="empty"):
            compute_window_metrics(np.array([]), np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(EvaluationError, match="mismatch"):
            compute_window_metrics(
                np.array([0, 1]),
                np.array([0]),
                np.array([0.1, 0.9]),
            )

    def test_no_positives_raises(self):
        with pytest.raises(EvaluationError, match="No positive"):
            compute_window_metrics(
                np.array([0, 0, 0]),
                np.array([0, 0, 0]),
                np.array([0.1, 0.2, 0.3]),
            )

    def test_no_negatives_raises(self):
        with pytest.raises(EvaluationError, match="No negative"):
            compute_window_metrics(
                np.array([1, 1, 1]),
                np.array([1, 1, 1]),
                np.array([0.8, 0.9, 0.7]),
            )

    def test_f2_favors_recall(self):
        """F2 should be closer to sensitivity than F1 is."""
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        if m.sensitivity != m.precision:
            dist_f1 = abs(m.f1 - m.sensitivity)
            dist_f2 = abs(m.f2 - m.sensitivity)
            assert dist_f2 <= dist_f1

    def test_imbalanced_dataset(self):
        """Simulate the real dataset's ~0.3% prevalence."""
        y_true, y_pred, y_proba = _make_predictions(n_positive=3, n_negative=997)
        m = compute_window_metrics(y_true, y_pred, y_proba)

        assert m.n_samples == 1000
        assert m.prevalence < 0.01
        # Balanced accuracy should differ from raw accuracy
        assert m.balanced_accuracy != m.accuracy

    def test_to_dict(self):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        d = m.to_dict()
        assert isinstance(d, dict)
        assert "sensitivity" in d
        assert "auroc" in d
        assert d["n_samples"] == 1000


# ── ROC and PR curves ────────────────────────────────────────────────


class TestCurves:
    def test_roc_curve_shape(self):
        y_true, _, y_proba = _make_predictions()
        fpr, tpr, thresholds = compute_roc_curve(y_true, y_proba)

        assert len(fpr) == len(tpr)
        assert len(thresholds) >= 2
        assert fpr[0] == 0.0
        assert tpr[0] == 0.0

    def test_pr_curve_shape(self):
        y_true, _, y_proba = _make_predictions()
        precision, recall, thresholds = compute_pr_curve(y_true, y_proba)

        assert len(precision) == len(recall) == len(thresholds) + 1

    def test_roc_perfect_model(self):
        y_true, _, y_proba = _make_perfect_predictions()
        fpr, tpr, _ = compute_roc_curve(y_true, y_proba)

        # Perfect model: TPR reaches 1.0 while FPR stays at 0.0
        assert 1.0 in tpr


# ── Threshold analysis ───────────────────────────────────────────────


class TestThresholdAnalysis:
    def test_default_thresholds(self):
        y_true, _, y_proba = _make_predictions()
        results = compute_threshold_analysis(y_true, y_proba)

        assert len(results) == 11  # 0.0 to 1.0 in steps of 0.1
        assert results[0]["threshold"] == 0.0
        assert results[-1]["threshold"] == 1.0

    def test_custom_thresholds(self):
        y_true, _, y_proba = _make_predictions()
        results = compute_threshold_analysis(y_true, y_proba, thresholds=[0.3, 0.5, 0.7])

        assert len(results) == 3
        assert results[0]["threshold"] == 0.3

    def test_threshold_zero_predicts_all_positive(self):
        y_true, _, y_proba = _make_predictions()
        results = compute_threshold_analysis(y_true, y_proba, thresholds=[0.0])

        assert results[0]["sensitivity"] == 1.0
        assert results[0]["fn"] == 0

    def test_threshold_one_predicts_all_negative(self):
        y_true, _, y_proba = _make_predictions()
        # Probabilities never reach exactly 1.0, so threshold=1.0 gives all negative
        results = compute_threshold_analysis(y_true, y_proba, thresholds=[1.0])

        assert results[0]["specificity"] == 1.0
        assert results[0]["fp"] == 0

    def test_each_result_has_required_keys(self):
        y_true, _, y_proba = _make_predictions()
        results = compute_threshold_analysis(y_true, y_proba, thresholds=[0.5])

        required_keys = {
            "threshold",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "f2",
            "tp",
            "fn",
            "fp",
            "tn",
        }
        assert required_keys.issubset(set(results[0].keys()))


# ── Format report ────────────────────────────────────────────────────


class TestFormatReport:
    def test_report_contains_key_metrics(self):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        report = format_metrics_report({"test": m}, model_name="Test Model")

        assert "Test Model" in report
        assert "Sensitivity" in report
        assert "Specificity" in report
        assert "AUROC" in report
        assert "AUPRC" in report
        assert "TEST" in report

    def test_report_multiple_splits(self):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        report = format_metrics_report(
            {"train": m, "val": m, "test": m},
            model_name="Multi-Split",
        )

        assert "TRAIN" in report
        assert "VAL" in report
        assert "TEST" in report
