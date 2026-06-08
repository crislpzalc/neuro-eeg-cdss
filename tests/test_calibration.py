"""
Tests for probability calibration module.

Covers:
- CalibrationConfig validation
- CalibrationMetrics container
- ReliabilityBin container
- Calibrator fitting (Platt and Isotonic)
- Probability calibration
- Reliability bin computation
- Calibration metrics computation
- Input validation
- Formatting
- Plots (smoke tests)
"""

from __future__ import annotations

import numpy as np
import pytest

from neuro_eeg_cdss.calibration.calibrator import (
    CalibrationConfig,
    CalibrationError,
    CalibrationMetrics,
    ReliabilityBin,
    calibrate_probabilities,
    compute_calibration_metrics,
    compute_reliability_bins,
    fit_calibrator,
    format_calibration_report,
)
from neuro_eeg_cdss.calibration.plots import (
    plot_calibration_comparison,
    plot_metrics_comparison,
    plot_reliability_diagram,
)

# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture()
def simple_data():
    """Simple balanced dataset for basic tests."""
    rng = np.random.RandomState(42)
    n = 1000
    y_true = np.array([0] * 500 + [1] * 500)
    # Probabilities with some correlation to labels
    y_proba = np.clip(y_true * 0.6 + rng.normal(0, 0.2, n), 0, 1)
    return y_true, y_proba


@pytest.fixture()
def imbalanced_data():
    """Imbalanced dataset mimicking seizure detection."""
    rng = np.random.RandomState(42)
    n_neg = 5000
    n_pos = 50
    y_true = np.array([0] * n_neg + [1] * n_pos)
    # Uncalibrated probabilities: most clustered near 0
    y_proba_neg = np.clip(rng.beta(1, 10, n_neg), 0, 1)
    y_proba_pos = np.clip(rng.beta(3, 3, n_pos), 0, 1)
    y_proba = np.concatenate([y_proba_neg, y_proba_pos])
    return y_true, y_proba


@pytest.fixture()
def perfectly_calibrated_data():
    """A dataset where probabilities are already well-calibrated."""
    rng = np.random.RandomState(123)
    n = 10000
    y_proba = rng.uniform(0, 1, n)
    y_true = (rng.uniform(0, 1, n) < y_proba).astype(int)
    return y_true, y_proba


# ── TestCalibrationConfig ────────────────────────────────────────────


class TestCalibrationConfig:
    def test_valid_platt(self):
        cfg = CalibrationConfig(method="platt", n_bins=10)
        assert cfg.method == "platt"
        assert cfg.n_bins == 10
        assert cfg.name == "platt_bins10"

    def test_valid_isotonic(self):
        cfg = CalibrationConfig(method="isotonic", n_bins=15)
        assert cfg.method == "isotonic"
        assert cfg.name == "isotonic_bins15"

    def test_default_n_bins(self):
        cfg = CalibrationConfig(method="platt")
        assert cfg.n_bins == 10

    def test_invalid_method(self):
        with pytest.raises(CalibrationError, match="Invalid method"):
            CalibrationConfig(method="invalid")

    def test_invalid_n_bins(self):
        with pytest.raises(CalibrationError, match="n_bins must be >= 2"):
            CalibrationConfig(method="platt", n_bins=1)


# ── TestCalibrationMetrics ───────────────────────────────────────────


class TestCalibrationMetrics:
    def test_to_dict(self):
        m = CalibrationMetrics(ece=0.05, mce=0.15, brier=0.2, log_loss_val=0.5, n_bins=10)
        d = m.to_dict()
        assert d["ece"] == 0.05
        assert d["mce"] == 0.15
        assert d["brier"] == 0.2
        assert d["log_loss_val"] == 0.5
        assert d["n_bins"] == 10

    def test_nan_to_none(self):
        m = CalibrationMetrics(ece=float("nan"), mce=0.1, brier=0.2, log_loss_val=0.3, n_bins=10)
        d = m.to_dict()
        assert d["ece"] is None
        assert d["mce"] == 0.1


# ── TestReliabilityBin ───────────────────────────────────────────────


class TestReliabilityBin:
    def test_to_dict(self):
        b = ReliabilityBin(
            bin_lower=0.0,
            bin_upper=0.1,
            bin_mid=0.05,
            avg_predicted=0.04,
            avg_observed=0.02,
            count=100,
            gap=0.02,
        )
        d = b.to_dict()
        assert d["count"] == 100
        assert d["gap"] == 0.02

    def test_frozen(self):
        b = ReliabilityBin(
            bin_lower=0.0,
            bin_upper=0.1,
            bin_mid=0.05,
            avg_predicted=0.04,
            avg_observed=0.02,
            count=100,
            gap=0.02,
        )
        with pytest.raises(AttributeError):
            b.count = 200  # type: ignore


# ── TestFitCalibrator ────────────────────────────────────────────────


class TestFitCalibrator:
    def test_platt_returns_logistic_regression(self, simple_data):
        y_true, y_proba = simple_data
        from sklearn.linear_model import LogisticRegression

        calibrator = fit_calibrator(y_true, y_proba, method="platt")
        assert isinstance(calibrator, LogisticRegression)

    def test_isotonic_returns_isotonic_regression(self, simple_data):
        y_true, y_proba = simple_data
        from sklearn.isotonic import IsotonicRegression

        calibrator = fit_calibrator(y_true, y_proba, method="isotonic")
        assert isinstance(calibrator, IsotonicRegression)

    def test_invalid_method_raises(self, simple_data):
        y_true, y_proba = simple_data
        with pytest.raises(CalibrationError, match="Invalid method"):
            fit_calibrator(y_true, y_proba, method="temperature")

    def test_empty_input_raises(self):
        with pytest.raises(CalibrationError, match="empty"):
            fit_calibrator(np.array([]), np.array([]), method="platt")

    def test_single_class_raises(self):
        with pytest.raises(CalibrationError, match="both classes"):
            fit_calibrator(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), method="platt")

    def test_nan_proba_raises(self):
        with pytest.raises(CalibrationError, match="NaN"):
            fit_calibrator(np.array([0, 1]), np.array([0.5, float("nan")]), method="platt")

    def test_out_of_range_proba_raises(self):
        with pytest.raises(CalibrationError, match="\\[0, 1\\]"):
            fit_calibrator(
                np.array([0, 1, 0, 1]),
                np.array([0.5, 1.5, 0.3, 0.7]),
                method="platt",
            )


# ── TestCalibrateProba ───────────────────────────────────────────────


class TestCalibrateProbabilities:
    def test_platt_output_in_range(self, simple_data):
        y_true, y_proba = simple_data
        calibrator = fit_calibrator(y_true, y_proba, method="platt")
        y_cal = calibrate_probabilities(calibrator, y_proba)
        assert y_cal.shape == y_proba.shape
        assert np.all(y_cal >= 0)
        assert np.all(y_cal <= 1)

    def test_isotonic_output_in_range(self, simple_data):
        y_true, y_proba = simple_data
        calibrator = fit_calibrator(y_true, y_proba, method="isotonic")
        y_cal = calibrate_probabilities(calibrator, y_proba)
        assert y_cal.shape == y_proba.shape
        assert np.all(y_cal >= 0)
        assert np.all(y_cal <= 1)

    def test_monotonic_platt(self, simple_data):
        """Platt scaling should be monotonically increasing."""
        y_true, y_proba = simple_data
        calibrator = fit_calibrator(y_true, y_proba, method="platt")
        test_proba = np.linspace(0, 1, 100)
        y_cal = calibrate_probabilities(calibrator, test_proba)
        # Should be monotonically non-decreasing
        assert np.all(np.diff(y_cal) >= -1e-10)

    def test_isotonic_monotonic(self, simple_data):
        """Isotonic regression is monotonically non-decreasing by definition."""
        y_true, y_proba = simple_data
        calibrator = fit_calibrator(y_true, y_proba, method="isotonic")
        test_proba = np.linspace(0, 1, 100)
        y_cal = calibrate_probabilities(calibrator, test_proba)
        assert np.all(np.diff(y_cal) >= -1e-10)

    def test_unknown_calibrator_raises(self):
        """Passing a wrong type should raise."""
        with pytest.raises(CalibrationError, match="Unknown calibrator"):
            calibrate_probabilities("not_a_calibrator", np.array([0.5]))

    def test_empty_input_raises(self, simple_data):
        y_true, y_proba = simple_data
        calibrator = fit_calibrator(y_true, y_proba, method="platt")
        with pytest.raises(CalibrationError, match="empty"):
            calibrate_probabilities(calibrator, np.array([]))


# ── TestComputeReliabilityBins ───────────────────────────────────────


class TestComputeReliabilityBins:
    def test_basic_shape(self, simple_data):
        y_true, y_proba = simple_data
        bins = compute_reliability_bins(y_true, y_proba, n_bins=10)
        assert len(bins) > 0
        assert all(isinstance(b, ReliabilityBin) for b in bins)

    def test_bins_cover_data(self, simple_data):
        y_true, y_proba = simple_data
        bins = compute_reliability_bins(y_true, y_proba, n_bins=10)
        total_count = sum(b.count for b in bins)
        assert total_count == len(y_true)

    def test_gap_is_abs_difference(self, simple_data):
        y_true, y_proba = simple_data
        bins = compute_reliability_bins(y_true, y_proba, n_bins=10)
        for b in bins:
            expected_gap = round(abs(b.avg_predicted - b.avg_observed), 6)
            assert abs(b.gap - expected_gap) < 1e-5

    def test_bin_edges(self, simple_data):
        y_true, y_proba = simple_data
        bins = compute_reliability_bins(y_true, y_proba, n_bins=5)
        for b in bins:
            assert b.bin_lower < b.bin_upper
            assert b.bin_lower >= 0.0
            assert b.bin_upper <= 1.0

    def test_invalid_n_bins(self, simple_data):
        y_true, y_proba = simple_data
        with pytest.raises(CalibrationError, match="n_bins must be >= 2"):
            compute_reliability_bins(y_true, y_proba, n_bins=1)

    def test_empty_bins_skipped(self):
        """If all probabilities are in one bin, other bins are empty."""
        y_true = np.array([0, 0, 1, 0, 0])
        y_proba = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        bins = compute_reliability_bins(y_true, y_proba, n_bins=10)
        # All samples should be in the first bin
        assert len(bins) == 1
        assert bins[0].count == 5

    def test_perfectly_calibrated(self, perfectly_calibrated_data):
        """A well-calibrated model should have small gaps."""
        y_true, y_proba = perfectly_calibrated_data
        bins = compute_reliability_bins(y_true, y_proba, n_bins=10)
        max_gap = max(b.gap for b in bins)
        # With 10K samples, each bin gap should be small (< 0.05)
        assert max_gap < 0.05


# ── TestComputeCalibrationMetrics ────────────────────────────────────


class TestComputeCalibrationMetrics:
    def test_returns_calibration_metrics(self, simple_data):
        y_true, y_proba = simple_data
        m = compute_calibration_metrics(y_true, y_proba, n_bins=10)
        assert isinstance(m, CalibrationMetrics)

    def test_ece_range(self, simple_data):
        y_true, y_proba = simple_data
        m = compute_calibration_metrics(y_true, y_proba)
        assert 0 <= m.ece <= 1

    def test_mce_gte_ece(self, simple_data):
        """MCE >= ECE always (max >= weighted average)."""
        y_true, y_proba = simple_data
        m = compute_calibration_metrics(y_true, y_proba)
        assert m.mce >= m.ece - 1e-10

    def test_brier_range(self, simple_data):
        y_true, y_proba = simple_data
        m = compute_calibration_metrics(y_true, y_proba)
        assert 0 <= m.brier <= 1

    def test_perfect_calibration_low_ece(self, perfectly_calibrated_data):
        """Well-calibrated data should have low ECE."""
        y_true, y_proba = perfectly_calibrated_data
        m = compute_calibration_metrics(y_true, y_proba, n_bins=10)
        assert m.ece < 0.05

    def test_constant_proba_high_ece(self):
        """A model that always predicts 0.5 on imbalanced data has high ECE."""
        y_true = np.array([0] * 900 + [1] * 100)
        y_proba = np.full(1000, 0.5)
        m = compute_calibration_metrics(y_true, y_proba, n_bins=10)
        # ECE should reflect the gap between 0.5 and 0.1 prevalence
        assert m.ece > 0.3

    def test_brier_perfect_model(self):
        """A perfect model has Brier score = 0."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.0, 0.0, 1.0, 1.0])
        m = compute_calibration_metrics(y_true, y_proba, n_bins=2)
        assert m.brier == 0.0

    def test_empty_raises(self):
        with pytest.raises(CalibrationError, match="empty"):
            compute_calibration_metrics(np.array([]), np.array([]))

    def test_length_mismatch_raises(self):
        with pytest.raises(CalibrationError, match="Length mismatch"):
            compute_calibration_metrics(np.array([0, 1]), np.array([0.5]))


# ── TestFormatReport ─────────────────────────────────────────────────


class TestFormatCalibrationReport:
    def test_basic_format(self):
        before = CalibrationMetrics(ece=0.15, mce=0.30, brier=0.25, log_loss_val=0.6, n_bins=10)
        after = CalibrationMetrics(ece=0.05, mce=0.12, brier=0.20, log_loss_val=0.45, n_bins=10)
        report = format_calibration_report(
            before, after, method="platt", split_name="test", model_name="LR"
        )
        assert "ECE" in report
        assert "MCE" in report
        assert "Brier" in report
        assert "Log Loss" in report
        assert "LR" in report
        assert "platt" in report

    def test_shows_improvement(self):
        before = CalibrationMetrics(ece=0.20, mce=0.40, brier=0.30, log_loss_val=0.7, n_bins=10)
        after = CalibrationMetrics(ece=0.05, mce=0.10, brier=0.20, log_loss_val=0.5, n_bins=10)
        report = format_calibration_report(before, after, method="isotonic", split_name="val")
        # Changes should be negative (improvement)
        assert "-" in report


# ── TestPlots (smoke tests) ──────────────────────────────────────────


class TestPlots:
    def test_reliability_diagram_runs(self, simple_data):
        y_true, y_proba = simple_data
        bins = compute_reliability_bins(y_true, y_proba, n_bins=10)
        fig = plot_reliability_diagram(
            bins, model_name="test_model", split_name="test", label="Uncalibrated"
        )
        assert fig is not None

    def test_calibration_comparison_runs(self, simple_data):
        y_true, y_proba = simple_data
        bins_before = compute_reliability_bins(y_true, y_proba, n_bins=10)
        # Simulate calibrated probabilities
        calibrator = fit_calibrator(y_true, y_proba, method="platt")
        y_cal = calibrate_probabilities(calibrator, y_proba)
        bins_after = compute_reliability_bins(y_true, y_cal, n_bins=10)
        fig = plot_calibration_comparison(
            bins_before,
            bins_after,
            method="platt",
            model_name="test_model",
            split_name="test",
        )
        assert fig is not None

    def test_metrics_comparison_runs(self, simple_data):
        y_true, y_proba = simple_data
        m1 = compute_calibration_metrics(y_true, y_proba)
        m2 = CalibrationMetrics(ece=0.02, mce=0.08, brier=0.15, log_loss_val=0.4, n_bins=10)
        fig = plot_metrics_comparison(
            {"uncalibrated": m1, "platt": m2},
            model_name="test_model",
            split_name="test",
        )
        assert fig is not None

    def test_reliability_diagram_saves(self, simple_data, tmp_path):
        y_true, y_proba = simple_data
        bins = compute_reliability_bins(y_true, y_proba, n_bins=10)
        out = tmp_path / "reliability.png"
        plot_reliability_diagram(
            bins,
            model_name="test",
            split_name="test",
            output_path=out,
        )
        assert out.exists()


# ── TestCalibrationEndToEnd ──────────────────────────────────────────


class TestCalibrationEndToEnd:
    """End-to-end tests: fit calibrator, calibrate, evaluate."""

    def test_platt_improves_calibration(self, simple_data):
        """Platt scaling should reduce ECE on held-out data."""
        y_true, y_proba = simple_data
        # Shuffle to ensure both classes in each split
        rng = np.random.RandomState(99)
        idx = rng.permutation(len(y_true))
        y_true, y_proba = y_true[idx], y_proba[idx]
        # Split into fit/eval
        y_true_fit, y_proba_fit = y_true[:700], y_proba[:700]
        y_true_eval, y_proba_eval = y_true[700:], y_proba[700:]

        calibrator = fit_calibrator(y_true_fit, y_proba_fit, method="platt")
        y_cal = calibrate_probabilities(calibrator, y_proba_eval)

        m_before = compute_calibration_metrics(y_true_eval, y_proba_eval)
        m_after = compute_calibration_metrics(y_true_eval, y_cal)

        # Calibrated ECE should be <= uncalibrated (on held-out data)
        # Allow small tolerance since it's not guaranteed on finite samples
        assert m_after.ece <= m_before.ece + 0.05

    def test_isotonic_improves_calibration(self, simple_data):
        """Isotonic regression should reduce ECE on held-out data."""
        y_true, y_proba = simple_data
        # Shuffle to ensure both classes in each split
        rng = np.random.RandomState(99)
        idx = rng.permutation(len(y_true))
        y_true, y_proba = y_true[idx], y_proba[idx]
        y_true_fit, y_proba_fit = y_true[:700], y_proba[:700]
        y_true_eval, y_proba_eval = y_true[700:], y_proba[700:]

        calibrator = fit_calibrator(y_true_fit, y_proba_fit, method="isotonic")
        y_cal = calibrate_probabilities(calibrator, y_proba_eval)

        m_before = compute_calibration_metrics(y_true_eval, y_proba_eval)
        m_after = compute_calibration_metrics(y_true_eval, y_cal)

        assert m_after.ece <= m_before.ece + 0.05

    def test_imbalanced_calibration(self, imbalanced_data):
        """Calibration should work with severely imbalanced data."""
        y_true, y_proba = imbalanced_data
        calibrator = fit_calibrator(y_true, y_proba, method="isotonic")
        y_cal = calibrate_probabilities(calibrator, y_proba)

        m = compute_calibration_metrics(y_true, y_cal)
        assert isinstance(m, CalibrationMetrics)
        assert m.ece >= 0
        assert m.brier >= 0

    def test_calibration_preserves_ranking(self, simple_data):
        """Isotonic calibration should preserve the ranking of probabilities."""
        y_true, y_proba = simple_data
        calibrator = fit_calibrator(y_true, y_proba, method="isotonic")
        y_cal = calibrate_probabilities(calibrator, y_proba)

        # For isotonic regression, if p1 > p2, then cal(p1) >= cal(p2)
        # Check on sorted unique values
        sorted_idx = np.argsort(y_proba)
        y_cal_sorted = y_cal[sorted_idx]
        assert np.all(np.diff(y_cal_sorted) >= -1e-10)
