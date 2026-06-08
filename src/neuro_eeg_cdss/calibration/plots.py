"""
Visualization module for probability calibration.

Generates publication-quality plots for calibration analysis:
- Reliability diagrams (calibration curves)
- Before/after calibration comparison
- Calibration metrics bar charts

All plots follow the same style conventions as the evaluation
module (``neuro_eeg_cdss.evaluation.plots``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neuro_eeg_cdss.calibration.calibrator import (
    CalibrationMetrics,
    ReliabilityBin,
)

# ── Style defaults ───────────────────────────────────────────────────

_COLORS = {
    "logistic_regression": "#2196F3",
    "random_forest": "#4CAF50",
    "uncalibrated": "#9E9E9E",
    "platt": "#E53935",
    "isotonic": "#FF9800",
}


def _get_color(key: str) -> str:
    return _COLORS.get(key, "#607D8B")


def _clean_model_name(name: str) -> str:
    return name.replace("_", " ").title()


# ── Reliability diagram ─────────────────────────────────────────────


def plot_reliability_diagram(
    bins_data: list[ReliabilityBin],
    model_name: str,
    split_name: str,
    label: str = "",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot a single reliability diagram.

    Parameters
    ----------
    bins_data : list[ReliabilityBin]
        Bin data from ``compute_reliability_bins()``.
    model_name : str
        Model name (for the title).
    split_name : str
        Split name (for the title).
    label : str
        Label for the calibration line (e.g., "Uncalibrated").
    output_path : Path | None
        If provided, saves the figure.

    Returns
    -------
    plt.Figure
    """
    fig, (ax_cal, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Calibration curve
    midpoints = [b.bin_mid for b in bins_data]
    observed = [b.avg_observed for b in bins_data]
    counts = [b.count for b in bins_data]

    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly calibrated")
    ax_cal.plot(
        midpoints,
        observed,
        "s-",
        color=_get_color("uncalibrated"),
        linewidth=2,
        markersize=6,
        label=label or "Model",
    )
    ax_cal.set_ylabel("Observed frequency (fraction of positives)", fontsize=11)
    ax_cal.set_title(
        f"Reliability Diagram — {_clean_model_name(model_name)}\n{split_name.capitalize()} Set",
        fontsize=13,
    )
    ax_cal.legend(loc="upper left", fontsize=10)
    ax_cal.set_ylim([-0.02, 1.02])
    ax_cal.grid(True, alpha=0.3)

    # Histogram of predictions
    ax_hist.bar(
        midpoints,
        counts,
        width=1.0 / len(bins_data) * 0.8,
        color=_get_color("uncalibrated"),
        alpha=0.7,
        edgecolor="gray",
    )
    ax_hist.set_xlabel("Mean predicted probability", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_xlim([0, 1])
    ax_hist.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_calibration_comparison(
    bins_before: list[ReliabilityBin],
    bins_after: list[ReliabilityBin],
    method: str,
    model_name: str,
    split_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot before/after calibration comparison on one diagram.

    Parameters
    ----------
    bins_before : list[ReliabilityBin]
        Bin data from uncalibrated probabilities.
    bins_after : list[ReliabilityBin]
        Bin data from calibrated probabilities.
    method : str
        Calibration method name (for the legend).
    model_name : str
        Model name (for the title).
    split_name : str
        Split name (for the title).
    output_path : Path | None
        If provided, saves the figure.

    Returns
    -------
    plt.Figure
    """
    fig, (ax_cal, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # Perfect calibration line
    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfectly calibrated")

    # Before
    mid_before = [b.bin_mid for b in bins_before]
    obs_before = [b.avg_observed for b in bins_before]
    ax_cal.plot(
        mid_before,
        obs_before,
        "s-",
        color=_get_color("uncalibrated"),
        linewidth=2,
        markersize=5,
        label="Uncalibrated",
        alpha=0.7,
    )

    # After
    mid_after = [b.bin_mid for b in bins_after]
    obs_after = [b.avg_observed for b in bins_after]
    ax_cal.plot(
        mid_after,
        obs_after,
        "o-",
        color=_get_color(method),
        linewidth=2,
        markersize=5,
        label=f"Calibrated ({_clean_model_name(method)})",
    )

    ax_cal.set_ylabel("Observed frequency (fraction of positives)", fontsize=11)
    ax_cal.set_title(
        f"Calibration Comparison — {_clean_model_name(model_name)}\n{split_name.capitalize()} Set",
        fontsize=13,
    )
    ax_cal.legend(loc="upper left", fontsize=10)
    ax_cal.set_ylim([-0.02, 1.02])
    ax_cal.grid(True, alpha=0.3)

    # Histograms — before and after side by side
    counts_before = [b.count for b in bins_before]
    counts_after = [b.count for b in bins_after]
    n_bins = max(len(bins_before), len(bins_after))
    bar_width = 1.0 / (n_bins + 1) * 0.35

    ax_hist.bar(
        [m - bar_width / 2 for m in mid_before],
        counts_before,
        width=bar_width,
        color=_get_color("uncalibrated"),
        alpha=0.6,
        label="Uncalibrated",
        edgecolor="gray",
    )
    ax_hist.bar(
        [m + bar_width / 2 for m in mid_after],
        counts_after,
        width=bar_width,
        color=_get_color(method),
        alpha=0.6,
        label=f"Calibrated ({_clean_model_name(method)})",
        edgecolor="gray",
    )
    ax_hist.set_xlabel("Mean predicted probability", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_xlim([0, 1])
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


def plot_metrics_comparison(
    metrics_by_config: dict[str, CalibrationMetrics],
    model_name: str,
    split_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Bar chart comparing calibration metrics across configurations.

    Parameters
    ----------
    metrics_by_config : dict[str, CalibrationMetrics]
        Mapping of config name (e.g., "uncalibrated", "platt") to
        calibration metrics.
    model_name : str
        Model name (for the title).
    split_name : str
        Split name (for the title).
    output_path : Path | None
        If provided, saves the figure.

    Returns
    -------
    plt.Figure
    """
    metric_names = ["ECE", "MCE", "Brier", "Log Loss"]
    metric_keys = ["ece", "mce", "brier", "log_loss_val"]

    config_names = list(metrics_by_config.keys())
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / len(config_names)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, config_name in enumerate(config_names):
        m = metrics_by_config[config_name]
        values = [getattr(m, key) for key in metric_keys]
        offset = (i - (len(config_names) - 1) / 2) * width
        color = _get_color(config_name)
        ax.bar(
            x + offset,
            values,
            width,
            label=_clean_model_name(config_name),
            color=color,
            alpha=0.85,
        )
        for j, v in enumerate(values):
            ax.text(
                x[j] + offset,
                v + 0.002,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_ylabel("Score (lower is better)", fontsize=12)
    ax.set_title(
        f"Calibration Metrics — {_clean_model_name(model_name)}\n{split_name.capitalize()} Set",
        fontsize=13,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig
