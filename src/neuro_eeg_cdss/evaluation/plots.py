"""
Visualization module for clinical evaluation.

Generates publication-quality plots for seizure detection evaluation:
- ROC curves (per model, with AUROC annotation)
- Precision-Recall curves (per model, with AUPRC annotation)
- Confusion matrices (normalized and absolute)
- Threshold analysis (sensitivity-specificity trade-off)

All plots use matplotlib with a clean, consistent style suitable
for academic papers.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from neuro_eeg_cdss.evaluation.metrics import (
    WindowMetrics,
    compute_pr_curve,
    compute_roc_curve,
    compute_threshold_analysis,
)

# ── Style defaults ───────────────────────────────────────────────────

_COLORS = {
    "logistic_regression": "#2196F3",
    "random_forest": "#4CAF50",
}
_DEFAULT_COLOR = "#FF9800"


def _get_color(model_name: str) -> str:
    return _COLORS.get(model_name, _DEFAULT_COLOR)


def _clean_model_name(name: str) -> str:
    return name.replace("_", " ").title()


# ── ROC curve ────────────────────────────────────────────────────────


def plot_roc_curves(
    models_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    split_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on one split.

    Parameters
    ----------
    models_data : dict[str, tuple[y_true, y_proba, auroc]]
        Mapping of model name to (y_true, y_proba, auroc).
    split_name : str
        Name of the split (for the title).
    output_path : Path | None
        If provided, saves the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (y_true, y_proba, auroc_val) in models_data.items():
        fpr, tpr, _ = compute_roc_curve(y_true, y_proba)
        color = _get_color(model_name)
        label = f"{_clean_model_name(model_name)} (AUROC = {auroc_val:.4f})"
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUROC = 0.5)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(f"ROC Curve — {split_name.capitalize()} Set", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── Precision-Recall curve ───────────────────────────────────────────


def plot_pr_curves(
    models_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
    split_name: str,
    prevalence: float,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models on one split.

    Parameters
    ----------
    models_data : dict[str, tuple[y_true, y_proba, auprc]]
        Mapping of model name to (y_true, y_proba, auprc).
    split_name : str
        Name of the split (for the title).
    prevalence : float
        Class prevalence (for the random baseline line).
    output_path : Path | None
        If provided, saves the figure to this path.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, (y_true, y_proba, auprc_val) in models_data.items():
        pr_prec, pr_rec, _ = compute_pr_curve(y_true, y_proba)
        color = _get_color(model_name)
        label = f"{_clean_model_name(model_name)} (AUPRC = {auprc_val:.4f})"
        ax.plot(pr_rec, pr_prec, color=color, linewidth=2, label=label)

    ax.axhline(
        y=prevalence,
        color="k",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label=f"Random (prevalence = {prevalence:.4f})",
    )
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12)
    ax.set_ylabel("Precision (PPV)", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {split_name.capitalize()} Set", fontsize=14)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── Confusion matrix ─────────────────────────────────────────────────


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    split_name: str,
    normalize: bool = False,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot a confusion matrix.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred : np.ndarray
        Binary predictions.
    model_name : str
        Model name (for the title).
    split_name : str
        Split name (for the title).
    normalize : bool
        If True, normalize by true class counts.
    output_path : Path | None
        If provided, saves the figure.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Non-seizure", "Seizure"],
    )

    if normalize:
        cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm_normalized,
            display_labels=["Non-seizure", "Seizure"],
        )

    disp.plot(ax=ax, cmap="Blues", values_format=".2%" if normalize else "d")
    ax.set_title(
        f"{'Normalized ' if normalize else ''}Confusion Matrix\n"
        f"{_clean_model_name(model_name)} — {split_name.capitalize()} Set",
        fontsize=12,
    )
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── Threshold analysis ───────────────────────────────────────────────


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    split_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot sensitivity and specificity as a function of the decision threshold.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_proba : np.ndarray
        Predicted probabilities.
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
    thresholds = [round(t, 2) for t in np.arange(0.0, 1.01, 0.05)]
    results = compute_threshold_analysis(y_true, y_proba, thresholds=thresholds)

    ts = [r["threshold"] for r in results]
    sens = [r["sensitivity"] for r in results]
    spec = [r["specificity"] for r in results]
    f2s = [r["f2"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, sens, color="#E53935", linewidth=2, label="Sensitivity")
    ax.plot(ts, spec, color="#1E88E5", linewidth=2, label="Specificity")
    ax.plot(ts, f2s, color="#43A047", linewidth=2, linestyle="--", label="F2 Score")

    # Mark the default threshold
    ax.axvline(x=0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="Default (0.5)")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(
        f"Threshold Analysis — {_clean_model_name(model_name)}\n{split_name.capitalize()} Set",
        fontsize=13,
    )
    ax.legend(loc="center right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig


# ── Model comparison table ───────────────────────────────────────────


def plot_model_comparison(
    metrics_by_model: dict[str, WindowMetrics],
    split_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """
    Plot a bar chart comparing key metrics across models.

    Parameters
    ----------
    metrics_by_model : dict[str, WindowMetrics]
        Mapping of model name to metrics.
    split_name : str
        Split name (for the title).
    output_path : Path | None
        If provided, saves the figure.

    Returns
    -------
    plt.Figure
    """
    metric_names = [
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1",
        "F2",
        "Bal. Accuracy",
        "AUROC",
        "AUPRC",
    ]
    metric_keys = [
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "f2",
        "balanced_accuracy",
        "auroc",
        "auprc",
    ]

    model_names = list(metrics_by_model.keys())
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model_name in enumerate(model_names):
        m = metrics_by_model[model_name]
        values = [getattr(m, key) for key in metric_keys]
        offset = (i - (len(model_names) - 1) / 2) * width
        color = _get_color(model_name)
        ax.bar(
            x + offset, values, width, label=_clean_model_name(model_name), color=color, alpha=0.85
        )

        # Add value labels
        for j, v in enumerate(values):
            ax.text(x[j] + offset, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Model Comparison — {split_name.capitalize()} Set", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)
    return fig
