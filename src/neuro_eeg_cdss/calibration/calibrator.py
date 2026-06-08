"""
Probability calibration for seizure detection models.

This module makes model output probabilities *interpretable*: when a
calibrated model says 0.3, roughly 30% of those windows are actual
seizures. Uncalibrated models (especially Random Forest) produce
probabilities that are internally ranked correctly (high AUROC) but
whose magnitudes are meaningless.

Two calibration methods are implemented:

- **Platt scaling**: Fits a logistic regression on uncalibrated
  probabilities. Works well when the calibration curve is sigmoid-
  shaped. Has only 2 parameters, so it is very resistant to
  overfitting even with small calibration sets.

- **Isotonic regression**: Non-parametric monotone fit. More flexible
  than Platt but can overfit if the calibration set is small. Our
  validation set (91K+ samples) is large enough.

Both calibrators are fitted on the **validation set** and evaluated on
the **test set** to prevent data leakage.

Calibration metrics
-------------------
- **ECE** (Expected Calibration Error): Weighted average of per-bin
  |predicted - observed| across probability bins. Lower is better.
- **MCE** (Maximum Calibration Error): Worst-case bin error.
- **Brier score**: Mean squared error of probabilities. Decomposes
  into calibration + discrimination + uncertainty.
- **Log loss**: Cross-entropy loss. Heavily penalises confident wrong
  predictions.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# ── Exceptions ───────────────────────────────────────────────────────


class CalibrationError(ValueError):
    """Raised when calibration encounters invalid inputs."""


# ── Constants ────────────────────────────────────────────────────────

VALID_METHODS = {"platt", "isotonic"}
DEFAULT_N_BINS = 10


# ── Config / containers ─────────────────────────────────────────────


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuration for a calibration experiment.

    Parameters
    ----------
    method : str
        Calibration method: ``"platt"`` or ``"isotonic"``.
    n_bins : int
        Number of bins for ECE / reliability diagram.
    """

    method: str
    n_bins: int = DEFAULT_N_BINS

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise CalibrationError(
                f"Invalid method '{self.method}'. Valid: {sorted(VALID_METHODS)}"
            )
        if self.n_bins < 2:
            raise CalibrationError(f"n_bins must be >= 2, got {self.n_bins}")

    @property
    def name(self) -> str:
        return f"{self.method}_bins{self.n_bins}"


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration quality metrics for one set of predictions.

    Attributes
    ----------
    ece : float
        Expected Calibration Error (weighted average per-bin error).
    mce : float
        Maximum Calibration Error (worst bin).
    brier : float
        Brier score = mean((y_proba - y_true)^2).
    log_loss_val : float
        Log loss (cross-entropy).
    n_bins : int
        Number of bins used for ECE / MCE computation.
    """

    ece: float
    mce: float
    brier: float
    log_loss_val: float
    n_bins: int

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        # Replace any NaN with None for JSON
        return {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in d.items()}


@dataclass(frozen=True)
class ReliabilityBin:
    """A single bin in the reliability diagram.

    Attributes
    ----------
    bin_lower : float
        Lower edge of the probability bin.
    bin_upper : float
        Upper edge of the probability bin.
    bin_mid : float
        Midpoint of the bin (for plotting).
    avg_predicted : float
        Mean predicted probability in this bin.
    avg_observed : float
        Observed fraction of positives in this bin.
    count : int
        Number of samples in this bin.
    gap : float
        |avg_predicted - avg_observed| — the calibration gap.
    """

    bin_lower: float
    bin_upper: float
    bin_mid: float
    avg_predicted: float
    avg_observed: float
    count: int
    gap: float

    def to_dict(self) -> dict:
        return asdict(self)


# ── Validation helpers ───────────────────────────────────────────────


def _validate_proba(y_proba: np.ndarray) -> None:
    """Validate a probability array."""
    if len(y_proba) == 0:
        raise CalibrationError("y_proba is empty.")
    if np.any(np.isnan(y_proba)):
        raise CalibrationError("y_proba contains NaN values.")
    if np.any(y_proba < 0) or np.any(y_proba > 1):
        raise CalibrationError("y_proba values must be in [0, 1].")


def _validate_labels(y_true: np.ndarray) -> None:
    """Validate a binary label array."""
    if len(y_true) == 0:
        raise CalibrationError("y_true is empty.")
    unique = set(np.unique(y_true))
    if not unique.issubset({0, 1}):
        raise CalibrationError(f"y_true must contain only 0 and 1, got {unique}")
    if len(unique) < 2:
        raise CalibrationError("y_true must contain both classes (0 and 1).")


def _validate_inputs(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Validate paired labels and probabilities."""
    _validate_labels(y_true)
    _validate_proba(y_proba)
    if len(y_true) != len(y_proba):
        raise CalibrationError(f"Length mismatch: y_true={len(y_true)}, y_proba={len(y_proba)}")


# ── Calibrator fitting ───────────────────────────────────────────────


def fit_calibrator(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: str,
) -> LogisticRegression | IsotonicRegression:
    """Fit a calibrator on validation set predictions.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels from the validation set.
    y_proba : np.ndarray
        Uncalibrated predicted probabilities from the validation set.
    method : str
        ``"platt"`` for Platt scaling (logistic regression on
        probabilities) or ``"isotonic"`` for isotonic regression.

    Returns
    -------
    LogisticRegression | IsotonicRegression
        Fitted calibrator ready for ``calibrate_probabilities()``.

    Raises
    ------
    CalibrationError
        If inputs are invalid or method is unknown.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_inputs(y_true, y_proba)

    if method not in VALID_METHODS:
        raise CalibrationError(f"Invalid method '{method}'. Valid: {sorted(VALID_METHODS)}")

    if method == "platt":
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(y_proba.reshape(-1, 1), y_true)
    else:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(y_proba, y_true)

    return calibrator


def calibrate_probabilities(
    calibrator: LogisticRegression | IsotonicRegression,
    y_proba: np.ndarray,
) -> np.ndarray:
    """Apply a fitted calibrator to produce calibrated probabilities.

    Parameters
    ----------
    calibrator : LogisticRegression | IsotonicRegression
        Fitted calibrator from ``fit_calibrator()``.
    y_proba : np.ndarray
        Uncalibrated probabilities to transform.

    Returns
    -------
    np.ndarray
        Calibrated probabilities in [0, 1].

    Raises
    ------
    CalibrationError
        If inputs are invalid or calibrator type is unrecognized.
    """
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_proba(y_proba)

    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
    elif isinstance(calibrator, IsotonicRegression):
        return calibrator.predict(y_proba)
    else:
        raise CalibrationError(f"Unknown calibrator type: {type(calibrator).__name__}")


# ── Calibration metrics ──────────────────────────────────────────────


def compute_reliability_bins(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> list[ReliabilityBin]:
    """Compute reliability diagram bin data.

    Divides [0, 1] into ``n_bins`` equal-width bins and computes the
    average predicted probability and observed frequency per bin.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_proba : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins (default: 10).

    Returns
    -------
    list[ReliabilityBin]
        One entry per non-empty bin.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_inputs(y_true, y_proba)

    if n_bins < 2:
        raise CalibrationError(f"n_bins must be >= 2, got {n_bins}")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Last bin includes right edge
        if i == n_bins - 1:
            mask = (y_proba >= lower) & (y_proba <= upper)
        else:
            mask = (y_proba >= lower) & (y_proba < upper)

        count = int(mask.sum())
        if count == 0:
            continue

        avg_pred = float(np.mean(y_proba[mask]))
        avg_obs = float(np.mean(y_true[mask]))

        bins.append(
            ReliabilityBin(
                bin_lower=round(lower, 4),
                bin_upper=round(upper, 4),
                bin_mid=round((lower + upper) / 2, 4),
                avg_predicted=round(avg_pred, 6),
                avg_observed=round(avg_obs, 6),
                count=count,
                gap=round(abs(avg_pred - avg_obs), 6),
            )
        )

    return bins


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = DEFAULT_N_BINS,
) -> CalibrationMetrics:
    """Compute calibration quality metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth binary labels.
    y_proba : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins for ECE/MCE computation.

    Returns
    -------
    CalibrationMetrics
        ECE, MCE, Brier score, and log loss.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    _validate_inputs(y_true, y_proba)

    # Reliability bins for ECE/MCE
    bins = compute_reliability_bins(y_true, y_proba, n_bins)
    total_samples = len(y_true)

    if len(bins) == 0:
        raise CalibrationError("No non-empty bins found. Cannot compute ECE/MCE.")

    # ECE = weighted average of bin gaps
    ece = sum(b.count / total_samples * b.gap for b in bins)

    # MCE = maximum bin gap
    mce = max(b.gap for b in bins)

    # Brier score
    brier = float(brier_score_loss(y_true, y_proba))

    # Log loss (with clipping to avoid log(0))
    y_proba_clipped = np.clip(y_proba, 1e-15, 1 - 1e-15)
    ll = float(log_loss(y_true, y_proba_clipped))

    return CalibrationMetrics(
        ece=round(ece, 6),
        mce=round(mce, 6),
        brier=round(brier, 6),
        log_loss_val=round(ll, 6),
        n_bins=n_bins,
    )


# ── Formatting ───────────────────────────────────────────────────────


def format_calibration_report(
    metrics_before: CalibrationMetrics,
    metrics_after: CalibrationMetrics,
    method: str,
    split_name: str,
    model_name: str = "",
) -> str:
    """Format a before/after calibration comparison report.

    Parameters
    ----------
    metrics_before : CalibrationMetrics
        Metrics computed on uncalibrated probabilities.
    metrics_after : CalibrationMetrics
        Metrics computed on calibrated probabilities.
    method : str
        Calibration method used.
    split_name : str
        Name of the evaluation split.
    model_name : str
        Optional model name for the header.

    Returns
    -------
    str
        Formatted report string.
    """
    header = f"  Calibration Report: {model_name}" if model_name else "  Calibration Report"
    lines = [
        "=" * 70,
        header,
        f"  Method: {method} | Split: {split_name}",
        "=" * 70,
        "",
        f"  {'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12}",
        f"  {'-' * 61}",
    ]

    metrics = [
        ("ECE", metrics_before.ece, metrics_after.ece),
        ("MCE", metrics_before.mce, metrics_after.mce),
        ("Brier Score", metrics_before.brier, metrics_after.brier),
        ("Log Loss", metrics_before.log_loss_val, metrics_after.log_loss_val),
    ]

    for name, before, after in metrics:
        change = after - before
        direction = "+" if change > 0 else ""
        lines.append(f"  {name:<25} {before:>12.6f} {after:>12.6f} {direction}{change:>11.6f}")

    lines.append("")
    lines.append("  (Lower is better for all metrics)")
    return "\n".join(lines)
