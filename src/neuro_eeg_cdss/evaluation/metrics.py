"""
Clinical evaluation metrics for seizure detection.

This module computes window-level metrics that are clinically relevant
for EEG seizure detection. It is designed to work with the prediction
files saved by the training pipeline (Sprint 1D), decoupling evaluation
from training.

Key design principles
---------------------
- **Sensitivity over accuracy:** In seizure detection, missing a real
  seizure (false negative) is far more dangerous than a false alarm.
  Metrics like sensitivity, F2, and NPV are prioritized.
- **Threshold-independent metrics:** AUROC and AUPRC evaluate model
  quality independently of the decision threshold.
- **Imbalance awareness:** With ~0.3% prevalence, accuracy is misleading.
  Balanced accuracy and AUPRC provide better signal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


class EvaluationError(ValueError):
    """Raised when evaluation encounters invalid inputs."""


# ── Metric container ─────────────────────────────────────────────────


@dataclass(frozen=True)
class WindowMetrics:
    """
    All window-level clinical metrics for one model on one split.

    Attributes
    ----------
    tp, fn, fp, tn : int
        Confusion matrix counts.
    n_samples : int
        Total number of samples evaluated.
    n_positive, n_negative : int
        Class counts.
    prevalence : float
        Proportion of positive samples (n_positive / n_samples).

    sensitivity : float
        TP / (TP + FN). Also called recall. The most critical metric
        for seizure detection: proportion of real seizures detected.
    specificity : float
        TN / (TN + FP). Proportion of non-seizure windows correctly
        identified. Controls false alarm rate.
    precision : float
        TP / (TP + FP). Also called positive predictive value (PPV).
        Proportion of alarms that are real seizures.
    npv : float
        TN / (TN + FN). Negative predictive value. Proportion of
        non-alarm windows that are truly non-seizure.
    f1 : float
        Harmonic mean of precision and sensitivity.
    f2 : float
        Weighted harmonic mean with beta=2, favoring sensitivity.
        More appropriate than F1 for seizure detection.
    accuracy : float
        (TP + TN) / N. Misleading with extreme imbalance.
    balanced_accuracy : float
        Average of sensitivity and specificity. Not inflated by
        the majority class.
    fpr : float
        False positive rate = FP / (FP + TN) = 1 - specificity.
    auroc : float
        Area under the ROC curve. Threshold-independent measure
        of discrimination ability.
    auprc : float
        Area under the Precision-Recall curve. More informative
        than AUROC for imbalanced datasets.
    """

    # Counts
    tp: int
    fn: int
    fp: int
    tn: int
    n_samples: int
    n_positive: int
    n_negative: int
    prevalence: float

    # Core clinical metrics
    sensitivity: float
    specificity: float
    precision: float
    npv: float
    f1: float
    f2: float
    accuracy: float
    balanced_accuracy: float
    fpr: float

    # Threshold-independent
    auroc: float
    auprc: float

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return asdict(self)


# ── Metric computation ───────────────────────────────────────────────


def _safe_divide(numerator: int, denominator: int) -> float:
    """Divide with zero-safe fallback."""
    return float(numerator) / denominator if denominator > 0 else 0.0


def compute_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> WindowMetrics:
    """
    Compute all window-level clinical metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels (0 or 1).
    y_pred : np.ndarray
        Binary predictions (0 or 1).
    y_proba : np.ndarray
        Predicted probability of the positive class.

    Returns
    -------
    WindowMetrics
        Complete set of clinical metrics.

    Raises
    ------
    EvaluationError
        If inputs have mismatched lengths or are empty.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    if len(y_true) == 0:
        raise EvaluationError("Cannot evaluate empty arrays.")

    if not (len(y_true) == len(y_pred) == len(y_proba)):
        raise EvaluationError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, y_proba={len(y_proba)}"
        )

    n_positive = int(y_true.sum())
    n_negative = int(len(y_true) - n_positive)

    if n_positive == 0:
        raise EvaluationError("No positive samples in y_true. Cannot compute sensitivity.")

    if n_negative == 0:
        raise EvaluationError("No negative samples in y_true. Cannot compute specificity.")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Core metrics
    sensitivity = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    npv = _safe_divide(tn, tn + fn)
    fpr = _safe_divide(fp, fp + tn)
    accuracy = _safe_divide(tp + tn, len(y_true))
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    f1 = f1_score(y_true, y_pred, zero_division=0.0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0.0)

    # Threshold-independent metrics
    auroc = roc_auc_score(y_true, y_proba)

    pr_precision, pr_recall, _ = precision_recall_curve(y_true, y_proba)
    auprc = auc(pr_recall, pr_precision)

    return WindowMetrics(
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        tn=int(tn),
        n_samples=len(y_true),
        n_positive=n_positive,
        n_negative=n_negative,
        prevalence=round(n_positive / len(y_true), 6),
        sensitivity=round(sensitivity, 4),
        specificity=round(specificity, 4),
        precision=round(precision, 4),
        npv=round(npv, 4),
        f1=round(f1, 4),
        f2=round(f2, 4),
        accuracy=round(accuracy, 4),
        balanced_accuracy=round(bal_acc, 4),
        fpr=round(fpr, 4),
        auroc=round(auroc, 4),
        auprc=round(auprc, 4),
    )


# ── Curve data ───────────────────────────────────────────────────────


def compute_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the ROC curve.

    Returns
    -------
    fpr : np.ndarray
        False positive rates at each threshold.
    tpr : np.ndarray
        True positive rates (sensitivity) at each threshold.
    thresholds : np.ndarray
        Decision thresholds.
    """
    return roc_curve(y_true, y_proba)


def compute_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Precision-Recall curve.

    Returns
    -------
    precision : np.ndarray
        Precision values at each threshold.
    recall : np.ndarray
        Recall (sensitivity) values at each threshold.
    thresholds : np.ndarray
        Decision thresholds.
    """
    return precision_recall_curve(y_true, y_proba)


# ── Threshold analysis ───────────────────────────────────────────────


def compute_threshold_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict]:
    """
    Compute key metrics at multiple decision thresholds.

    This enables analyzing the sensitivity-specificity trade-off
    at thresholds other than the default 0.5.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_proba : np.ndarray
        Predicted probabilities.
    thresholds : list[float] | None
        Thresholds to evaluate. Defaults to 11 evenly-spaced values.

    Returns
    -------
    list[dict]
        One dictionary per threshold with keys: threshold, sensitivity,
        specificity, precision, f1, f2, tp, fn, fp, tn.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.0, 1.05, 0.1)]

    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    results = []
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        results.append(
            {
                "threshold": thresh,
                "sensitivity": round(_safe_divide(tp, tp + fn), 4),
                "specificity": round(_safe_divide(tn, tn + fp), 4),
                "precision": round(_safe_divide(tp, tp + fp), 4),
                "f1": round(float(f1_score(y_true, y_pred, zero_division=0.0)), 4),
                "f2": round(float(fbeta_score(y_true, y_pred, beta=2, zero_division=0.0)), 4),
                "tp": int(tp),
                "fn": int(fn),
                "fp": int(fp),
                "tn": int(tn),
            }
        )

    return results


# ── Formatting ───────────────────────────────────────────────────────


def format_metrics_report(
    metrics_by_split: dict[str, WindowMetrics],
    model_name: str = "",
) -> str:
    """
    Format a multi-split metrics report as a readable string.

    Parameters
    ----------
    metrics_by_split : dict[str, WindowMetrics]
        Mapping of split name to computed metrics.
    model_name : str
        Optional model name for the header.

    Returns
    -------
    str
        Formatted report.
    """
    lines = []
    header = f"  Clinical Evaluation: {model_name}" if model_name else "  Clinical Evaluation"
    lines.append("=" * 70)
    lines.append(header)
    lines.append("=" * 70)

    for split_name, m in metrics_by_split.items():
        lines.append(
            f"\n  [{split_name.upper()}]  ({m.n_samples:,} samples, prevalence={m.prevalence:.4%})"
        )
        lines.append(
            f"    Sensitivity (Recall) : {m.sensitivity:.4f}  "
            f"← {m.tp} of {m.n_positive} seizures detected"
        )
        lines.append(
            f"    Specificity          : {m.specificity:.4f}  "
            f"← {m.tn} of {m.n_negative} non-seizures correct"
        )
        lines.append(f"    Precision (PPV)      : {m.precision:.4f}")
        lines.append(f"    NPV                  : {m.npv:.4f}")
        lines.append(f"    F1                   : {m.f1:.4f}")
        lines.append(f"    F2                   : {m.f2:.4f}")
        lines.append(f"    Balanced Accuracy    : {m.balanced_accuracy:.4f}")
        lines.append(f"    AUROC                : {m.auroc:.4f}")
        lines.append(f"    AUPRC                : {m.auprc:.4f}")
        lines.append(f"    FPR                  : {m.fpr:.4f}")
        lines.append(f"    Confusion: TP={m.tp}  FN={m.fn}  FP={m.fp}  TN={m.tn}")

    return "\n".join(lines)
