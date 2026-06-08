"""
Temporal post-processing for seizure detection predictions.

This module applies temporal smoothing and filtering to window-level
predictions, exploiting the fact that real seizures persist across
multiple consecutive EEG windows while false positives tend to be
isolated.

Design goals
------------
- Operate on temporally ordered predictions within each recording
- Provide multiple configurable strategies (median, moving average,
  minimum duration)
- Preserve the (y_true, y_pred, y_proba) interface so downstream
  evaluation works unchanged
- Never modify ground-truth labels — only predicted values change

Strategies
----------
1. **Median filter**: replaces each y_proba with the median of its
   local neighborhood, suppressing isolated spikes.
2. **Moving average**: replaces each y_proba with the mean of its
   local neighborhood, providing smoother transitions.
3. **Minimum duration**: removes positive detection runs shorter than
   a specified number of consecutive windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class PostprocessingError(ValueError):
    """Raised when post-processing encounters invalid input."""


# ── Configuration ────────────────────────────────────────────────────


VALID_STRATEGIES = {"median_filter", "moving_average", "min_duration"}


@dataclass(frozen=True)
class TemporalConfig:
    """
    Configuration for temporal post-processing.

    Attributes
    ----------
    strategy : str
        Post-processing strategy. One of: ``"median_filter"``,
        ``"moving_average"``, ``"min_duration"``.
    kernel_size : int
        Window size for median filter or moving average. Must be odd
        and >= 3 for median filter. Ignored for ``"min_duration"``.
    min_windows : int
        Minimum number of consecutive positive windows to retain a
        detection. Only used with ``"min_duration"`` strategy.
    threshold : float
        Probability threshold for converting smoothed y_proba to
        binary y_pred. Used by ``"median_filter"`` and
        ``"moving_average"``.

    Notes
    -----
    The threshold parameter controls the sensitivity-specificity
    trade-off after smoothing. Lower thresholds increase sensitivity;
    higher thresholds increase specificity.
    """

    strategy: str = "median_filter"
    kernel_size: int = 3
    min_windows: int = 2
    threshold: float = 0.5

    @property
    def name(self) -> str:
        """Short descriptive name for this configuration."""
        if self.strategy == "median_filter":
            return f"median_k{self.kernel_size}_t{self.threshold}"
        if self.strategy == "moving_average":
            return f"mavg_k{self.kernel_size}_t{self.threshold}"
        return f"mindur_w{self.min_windows}"


REQUIRED_COLUMNS = {"subject", "path", "start_sec", "y_true", "y_proba"}


# ── Low-level filters (operate on single 1D arrays) ──────────────────


def median_filter_proba(
    y_proba: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """
    Apply a median filter to a probability sequence.

    The median filter replaces each value with the median of its local
    neighborhood. This is effective at removing isolated spikes (single
    false positives) while preserving sustained positive runs (true
    seizures).

    Parameters
    ----------
    y_proba : np.ndarray
        1D array of predicted probabilities, ordered by time.
    kernel_size : int
        Size of the median filter window. Must be an odd integer >= 1.

    Returns
    -------
    np.ndarray
        Filtered probability sequence, same length as input.

    Raises
    ------
    PostprocessingError
        If kernel_size is invalid or y_proba is not 1D.

    Notes
    -----
    Edge handling: values near the boundaries are computed with a
    truncated window (no padding). This is implemented manually rather
    than using ``scipy.signal.medfilt`` to avoid a heavy dependency
    for a simple operation.
    """
    _validate_proba_array(y_proba)

    if kernel_size < 1 or kernel_size % 2 == 0:
        raise PostprocessingError(
            f"kernel_size must be a positive odd integer. Received: {kernel_size}"
        )

    if kernel_size == 1 or len(y_proba) <= 1:
        return y_proba.copy()

    n = len(y_proba)
    half = kernel_size // 2
    result = np.empty(n, dtype=y_proba.dtype)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.median(y_proba[lo:hi])

    return result


def moving_average_proba(
    y_proba: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """
    Apply a moving average to a probability sequence.

    Provides smoother transitions than the median filter but is less
    effective at completely eliminating isolated spikes.

    Parameters
    ----------
    y_proba : np.ndarray
        1D array of predicted probabilities, ordered by time.
    window_size : int
        Number of windows to average over. Must be >= 1.

    Returns
    -------
    np.ndarray
        Smoothed probability sequence, same length as input.

    Raises
    ------
    PostprocessingError
        If window_size is invalid or y_proba is not 1D.

    Notes
    -----
    Uses centered averaging with truncated windows at boundaries,
    matching the edge behavior of ``median_filter_proba``.
    """
    _validate_proba_array(y_proba)

    if window_size < 1:
        raise PostprocessingError(
            f"window_size must be a positive integer. Received: {window_size}"
        )

    if window_size == 1 or len(y_proba) <= 1:
        return y_proba.copy()

    n = len(y_proba)
    half = window_size // 2
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        result[i] = np.mean(y_proba[lo:hi])

    return result


def apply_minimum_duration(
    y_pred: np.ndarray,
    min_windows: int = 2,
) -> np.ndarray:
    """
    Remove positive detection runs shorter than a minimum duration.

    A "run" is a consecutive sequence of positive predictions (1s).
    If a run is shorter than ``min_windows``, all its values are set
    to 0 (negative). This eliminates brief, likely spurious detections.

    Parameters
    ----------
    y_pred : np.ndarray
        1D binary prediction array (0s and 1s), ordered by time.
    min_windows : int
        Minimum run length to retain. Must be >= 1.

    Returns
    -------
    np.ndarray
        Filtered prediction array with short runs removed.

    Raises
    ------
    PostprocessingError
        If min_windows is invalid or y_pred is not 1D binary.

    Examples
    --------
    >>> apply_minimum_duration(np.array([0, 1, 0, 1, 1, 1, 0]), min_windows=2)
    array([0, 0, 0, 1, 1, 1, 0])

    The isolated ``1`` at index 1 is removed because its run length (1)
    is below the minimum (2). The run at indices 3-5 has length 3 >= 2
    and is preserved.
    """
    _validate_pred_array(y_pred)

    if min_windows < 1:
        raise PostprocessingError(
            f"min_windows must be a positive integer. Received: {min_windows}"
        )

    if min_windows == 1 or len(y_pred) == 0:
        return y_pred.copy()

    result = y_pred.copy()
    runs = _find_runs(result, value=1)

    for start, length in runs:
        if length < min_windows:
            result[start : start + length] = 0

    return result


# ── Run detection utility ────────────────────────────────────────────


def _find_runs(
    arr: np.ndarray,
    value: int = 1,
) -> list[tuple[int, int]]:
    """
    Find consecutive runs of a given value in a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        1D array to scan.
    value : int
        Value to look for.

    Returns
    -------
    list[tuple[int, int]]
        Each element is ``(start_index, run_length)``.
    """
    runs: list[tuple[int, int]] = []
    n = len(arr)
    i = 0

    while i < n:
        if arr[i] == value:
            start = i
            while i < n and arr[i] == value:
                i += 1
            runs.append((start, i - start))
        else:
            i += 1

    return runs


# ── Validation helpers ───────────────────────────────────────────────


def _validate_proba_array(y_proba: np.ndarray) -> None:
    """Validate that y_proba is a 1D numeric array with values in [0, 1]."""
    if not isinstance(y_proba, np.ndarray):
        raise PostprocessingError(
            f"y_proba must be a numpy array. Received: {type(y_proba).__name__}"
        )
    if y_proba.ndim != 1:
        raise PostprocessingError(f"y_proba must be 1D. Received shape: {y_proba.shape}")
    if len(y_proba) > 0:
        if np.any(y_proba < 0) or np.any(y_proba > 1):
            raise PostprocessingError("y_proba values must be in [0, 1].")
        if np.any(np.isnan(y_proba)):
            raise PostprocessingError("y_proba contains NaN values.")


def _validate_pred_array(y_pred: np.ndarray) -> None:
    """Validate that y_pred is a 1D binary array (0s and 1s)."""
    if not isinstance(y_pred, np.ndarray):
        raise PostprocessingError(
            f"y_pred must be a numpy array. Received: {type(y_pred).__name__}"
        )
    if y_pred.ndim != 1:
        raise PostprocessingError(f"y_pred must be 1D. Received shape: {y_pred.shape}")
    if len(y_pred) > 0:
        unique = set(np.unique(y_pred))
        if not unique.issubset({0, 1}):
            raise PostprocessingError(
                f"y_pred must contain only 0s and 1s. Found: {sorted(unique)}"
            )


def _validate_config(config: TemporalConfig) -> None:
    """Validate a TemporalConfig."""
    if config.strategy not in VALID_STRATEGIES:
        raise PostprocessingError(
            f"Unknown strategy: '{config.strategy}'. Supported: {sorted(VALID_STRATEGIES)}"
        )

    if config.strategy == "median_filter" and (
        config.kernel_size < 1 or config.kernel_size % 2 == 0
    ):
        raise PostprocessingError(
            f"kernel_size must be a positive odd integer for median_filter. "
            f"Received: {config.kernel_size}"
        )

    if config.strategy == "moving_average" and config.kernel_size < 1:
        raise PostprocessingError(
            f"kernel_size must be >= 1 for moving_average. Received: {config.kernel_size}"
        )

    if config.strategy == "min_duration" and config.min_windows < 1:
        raise PostprocessingError(
            f"min_windows must be >= 1 for min_duration. Received: {config.min_windows}"
        )

    if not 0.0 <= config.threshold <= 1.0:
        raise PostprocessingError(f"threshold must be in [0, 1]. Received: {config.threshold}")


def _validate_predictions_df(df: pd.DataFrame) -> None:
    """Validate that a predictions DataFrame has the required columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise PostprocessingError(
            f"Predictions DataFrame missing required columns: {sorted(missing)}"
        )


# ── High-level API ───────────────────────────────────────────────────


def postprocess_recording(
    y_proba: np.ndarray,
    y_pred: np.ndarray,
    config: TemporalConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply temporal post-processing to a single recording's predictions.

    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities for one recording, ordered by time.
    y_pred : np.ndarray
        Binary predictions for one recording, ordered by time.
    config : TemporalConfig
        Post-processing configuration.

    Returns
    -------
    y_pred_post : np.ndarray
        Post-processed binary predictions.
    y_proba_post : np.ndarray
        Post-processed probabilities (unchanged for min_duration).
    """
    if config.strategy == "median_filter":
        y_proba_post = median_filter_proba(y_proba, config.kernel_size)
        y_pred_post = (y_proba_post >= config.threshold).astype(int)
        return y_pred_post, y_proba_post

    if config.strategy == "moving_average":
        y_proba_post = moving_average_proba(y_proba, config.kernel_size)
        y_pred_post = (y_proba_post >= config.threshold).astype(int)
        return y_pred_post, y_proba_post

    # min_duration: filter binary predictions, keep probabilities unchanged
    y_pred_post = apply_minimum_duration(y_pred, config.min_windows)
    return y_pred_post, y_proba.copy()


def postprocess_predictions(
    df: pd.DataFrame,
    config: TemporalConfig,
) -> pd.DataFrame:
    """
    Apply temporal post-processing to a full predictions DataFrame.

    Predictions are grouped by recording (subject + path), sorted by
    time within each recording, and filtered independently. This ensures
    temporal context never crosses recording boundaries.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions with columns: ``subject``, ``path``, ``start_sec``,
        ``y_true``, ``y_proba``, and optionally ``y_pred``.
    config : TemporalConfig
        Post-processing configuration.

    Returns
    -------
    pd.DataFrame
        Copy of input with added columns: ``y_pred_post`` and
        ``y_proba_post``. Original columns are preserved for
        comparison.

    Raises
    ------
    PostprocessingError
        If the DataFrame is missing required columns or the config
        is invalid.
    """
    _validate_config(config)
    _validate_predictions_df(df)

    result = df.copy()

    # Ensure y_pred exists (threshold raw probabilities if missing)
    if "y_pred" not in result.columns:
        result["y_pred"] = (result["y_proba"] >= 0.5).astype(int)

    # Initialize output columns
    result["y_pred_post"] = result["y_pred"].copy()
    result["y_proba_post"] = result["y_proba"].copy()

    # Group by recording and process each independently
    for (_subject, _path), group in result.groupby(["subject", "path"]):
        # Sort by time within this recording
        sorted_idx = group.sort_values("start_sec").index

        y_proba_rec = result.loc[sorted_idx, "y_proba"].values
        y_pred_rec = result.loc[sorted_idx, "y_pred"].values

        y_pred_post, y_proba_post = postprocess_recording(y_proba_rec, y_pred_rec, config)

        result.loc[sorted_idx, "y_pred_post"] = y_pred_post
        result.loc[sorted_idx, "y_proba_post"] = y_proba_post

    return result


def enrich_predictions(
    predictions_df: pd.DataFrame,
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add temporal metadata to a predictions DataFrame.

    The current prediction pipeline saves only ``(y_true, y_pred,
    y_proba)`` without temporal or patient metadata. This function
    restores that metadata by positional join with the corresponding
    feature DataFrame (which must be in the same row order).

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with columns ``y_true``, ``y_pred``, ``y_proba``.
    features_df : pd.DataFrame
        Feature data for the same split, in the same row order.
        Must contain ``subject``, ``path``, ``start_sec``, ``end_sec``.

    Returns
    -------
    pd.DataFrame
        Enriched predictions with metadata columns added.

    Raises
    ------
    PostprocessingError
        If lengths don't match or required columns are missing.
    """
    if len(predictions_df) != len(features_df):
        raise PostprocessingError(
            f"Length mismatch: predictions has {len(predictions_df)} rows, "
            f"features has {len(features_df)} rows."
        )

    pred_required = {"y_true", "y_proba"}
    pred_missing = pred_required - set(predictions_df.columns)
    if pred_missing:
        raise PostprocessingError(f"Predictions missing columns: {sorted(pred_missing)}")

    feat_required = {"subject", "path", "start_sec", "end_sec"}
    feat_missing = feat_required - set(features_df.columns)
    if feat_missing:
        raise PostprocessingError(f"Features missing columns: {sorted(feat_missing)}")

    result = predictions_df.copy()
    result = result.reset_index(drop=True)

    for col in ["subject", "path", "start_sec", "end_sec"]:
        result[col] = features_df[col].values

    # Validate alignment: y_true should match label
    if "label" in features_df.columns:
        labels = features_df["label"].values.astype(int)
        y_true = result["y_true"].values.astype(int)
        if not np.array_equal(y_true, labels):
            raise PostprocessingError(
                "y_true in predictions does not match labels in features. "
                "Row order may have changed between training and enrichment."
            )

    return result


def compute_postprocessing_summary(
    df: pd.DataFrame,
    config: TemporalConfig,
) -> dict[str, Any]:
    """
    Compute summary statistics comparing pre and post-processed predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``postprocess_predictions`` with both original and
        post-processed columns.
    config : TemporalConfig
        Configuration used for post-processing.

    Returns
    -------
    dict[str, Any]
        Summary including counts of changed predictions, flipped
        labels, and configuration details.
    """
    n_total = len(df)
    pred_changed = int((df["y_pred"] != df["y_pred_post"]).sum())
    pos_to_neg = int(((df["y_pred"] == 1) & (df["y_pred_post"] == 0)).sum())
    neg_to_pos = int(((df["y_pred"] == 0) & (df["y_pred_post"] == 1)).sum())

    return {
        "config": {
            "strategy": config.strategy,
            "kernel_size": config.kernel_size,
            "min_windows": config.min_windows,
            "threshold": config.threshold,
            "name": config.name,
        },
        "n_total": n_total,
        "n_predictions_changed": pred_changed,
        "change_rate": round(pred_changed / max(n_total, 1), 6),
        "positive_to_negative": pos_to_neg,
        "negative_to_positive": neg_to_pos,
        "original_positives": int((df["y_pred"] == 1).sum()),
        "postprocessed_positives": int((df["y_pred_post"] == 1).sum()),
    }
