"""Tests for the evaluation plots module."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for testing

from neuro_eeg_cdss.evaluation.metrics import compute_window_metrics
from neuro_eeg_cdss.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_analysis,
)


def _make_predictions(
    n_positive: int = 50,
    n_negative: int = 950,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic predictions."""
    rng = np.random.RandomState(seed)
    y_true = np.array([1] * n_positive + [0] * n_negative)
    proba_pos = np.clip(rng.normal(0.7, 0.2, n_positive), 0, 1)
    proba_neg = np.clip(rng.normal(0.2, 0.15, n_negative), 0, 1)
    y_proba = np.concatenate([proba_pos, proba_neg])
    y_pred = (y_proba >= 0.5).astype(int)
    return y_true, y_pred, y_proba


class TestPlotROC:
    def test_creates_figure(self):
        y_true, _, y_proba = _make_predictions()
        fig = plot_roc_curves(
            {"test_model": (y_true, y_proba, 0.85)},
            split_name="test",
        )
        assert fig is not None

    def test_saves_to_file(self, tmp_path):
        y_true, _, y_proba = _make_predictions()
        output = tmp_path / "roc.png"
        plot_roc_curves(
            {"test_model": (y_true, y_proba, 0.85)},
            split_name="test",
            output_path=output,
        )
        assert output.exists()
        assert output.stat().st_size > 0


class TestPlotPR:
    def test_creates_figure(self):
        y_true, _, y_proba = _make_predictions()
        fig = plot_pr_curves(
            {"test_model": (y_true, y_proba, 0.45)},
            split_name="test",
            prevalence=0.05,
        )
        assert fig is not None

    def test_saves_to_file(self, tmp_path):
        y_true, _, y_proba = _make_predictions()
        output = tmp_path / "pr.png"
        plot_pr_curves(
            {"test_model": (y_true, y_proba, 0.45)},
            split_name="test",
            prevalence=0.05,
            output_path=output,
        )
        assert output.exists()


class TestPlotConfusionMatrix:
    def test_absolute(self, tmp_path):
        y_true, y_pred, _ = _make_predictions()
        output = tmp_path / "cm.png"
        plot_confusion_matrix(
            y_true,
            y_pred,
            "test_model",
            "test",
            normalize=False,
            output_path=output,
        )
        assert output.exists()

    def test_normalized(self, tmp_path):
        y_true, y_pred, _ = _make_predictions()
        output = tmp_path / "cm_norm.png"
        plot_confusion_matrix(
            y_true,
            y_pred,
            "test_model",
            "test",
            normalize=True,
            output_path=output,
        )
        assert output.exists()


class TestPlotThresholdAnalysis:
    def test_creates_and_saves(self, tmp_path):
        y_true, _, y_proba = _make_predictions()
        output = tmp_path / "thresh.png"
        plot_threshold_analysis(
            y_true,
            y_proba,
            "test_model",
            "test",
            output_path=output,
        )
        assert output.exists()


class TestPlotModelComparison:
    def test_creates_and_saves(self, tmp_path):
        y_true, y_pred, y_proba = _make_predictions()
        m = compute_window_metrics(y_true, y_pred, y_proba)

        output = tmp_path / "comparison.png"
        plot_model_comparison(
            {"model_a": m, "model_b": m},
            split_name="test",
            output_path=output,
        )
        assert output.exists()
