"""
Utilities for segmenting EEG recordings into fixed-length time windows and
computing temporal overlap with annotated intervals.

This module provides the temporal backbone for downstream dataset generation.
It defines deterministic window generation and overlap computation rules that
are later used for seizure labeling.

Design goals
------------
- Produce reproducible fixed-length windows from continuous recordings
- Make overlap calculations explicit and auditable
- Enforce basic temporal consistency through early validation
"""

from __future__ import annotations

from dataclasses import dataclass


class SegmentationError(ValueError):
    """Raised when segmentation parameters or temporal boundaries are invalid."""


@dataclass(frozen=True)
class TimeWindow:
    """
    Fixed temporal window within an EEG recording.

    Attributes
    ----------
    start_sec : float
        Window start time in seconds.
    end_sec : float
        Window end time in seconds.

    Notes
    -----
    This object represents a half-open temporal span in recording-relative
    time, intended for deterministic segmentation of continuous EEG signals.
    Downstream code assumes that all times are expressed in seconds from the
    start of the recording.
    """

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        """Window duration in seconds."""
        return self.end_sec - self.start_sec


def _validate_positive(value: float, name: str) -> None:
    """
    Validate that a numeric value is strictly positive.
    """
    if value <= 0:
        raise SegmentationError(f"'{name}' must be > 0. Received value: {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """
    Validate that a numeric value is non-negative.
    """
    if value < 0:
        raise SegmentationError(f"'{name}' must be >= 0. Received value: {value}")


def generate_time_windows(
    recording_duration_sec: float,
    window_size_sec: float,
    stride_sec: float,
) -> list[TimeWindow]:
    """
    Generate complete fixed-length time windows within an EEG recording.

    Rule
    ----
    - Only complete windows are generated.
    - A window is included only if ``end_sec <= recording_duration_sec``.

    Parameters
    ----------
    recording_duration_sec : float
        Total recording duration in seconds.
    window_size_sec : float
        Length of each window in seconds.
    stride_sec : float
        Step between consecutive windows in seconds.

    Returns
    -------
    list[TimeWindow]
        Sorted list of temporal windows.

    Raises
    ------
    SegmentationError
        If the input parameters are invalid.

    Notes
    -----
    This function intentionally discards trailing incomplete segments. This
    keeps the window geometry uniform across the dataset and simplifies feature
    extraction and model input construction.
    """
    _validate_non_negative(recording_duration_sec, "recording_duration_sec")
    _validate_positive(window_size_sec, "window_size_sec")
    _validate_positive(stride_sec, "stride_sec")

    if recording_duration_sec < window_size_sec:
        return []

    windows: list[TimeWindow] = []
    start_sec = 0.0

    # Windows are generated sequentially from the start of the recording to
    # guarantee deterministic coverage under a fixed window/stride policy.
    while start_sec + window_size_sec <= recording_duration_sec:
        end_sec = start_sec + window_size_sec
        windows.append(TimeWindow(start_sec=start_sec, end_sec=end_sec))
        start_sec += stride_sec

    return windows


def compute_overlap_seconds(
    window_start_sec: float,
    window_end_sec: float,
    interval_start_sec: float,
    interval_end_sec: float,
) -> float:
    """
    Compute the temporal overlap in seconds between a window and an interval.

    Parameters
    ----------
    window_start_sec : float
        Window start time.
    window_end_sec : float
        Window end time.
    interval_start_sec : float
        Interval start time.
    interval_end_sec : float
        Interval end time.

    Returns
    -------
    float
        Overlap duration in seconds. Returns 0.0 if there is no overlap.

    Raises
    ------
    SegmentationError
        If the temporal boundaries are invalid.

    Notes
    -----
    The overlap is computed geometrically from interval intersections and does
    not assume any specific semantic meaning of the interval beyond its start
    and end boundaries.
    """
    if window_end_sec < window_start_sec:
        raise SegmentationError("Invalid window boundaries: end_sec < start_sec.")

    if interval_end_sec < interval_start_sec:
        raise SegmentationError("Invalid interval boundaries: end_sec < start_sec.")

    overlap_start = max(window_start_sec, interval_start_sec)
    overlap_end = min(window_end_sec, interval_end_sec)

    return max(0.0, overlap_end - overlap_start)


def compute_total_overlap_seconds(
    window: TimeWindow,
    intervals: list[tuple[float, float]],
) -> float:
    """
    Compute the total overlap in seconds between a window and multiple intervals.

    Parameters
    ----------
    window : TimeWindow
        Temporal window.
    intervals : list[tuple[float, float]]
        List of intervals as ``(start_sec, end_sec)``.

    Returns
    -------
    float
        Total overlap duration in seconds.

    Notes
    -----
    This function assumes that the provided intervals are already semantically
    meaningful for aggregation. If intervals overlap each other, the total may
    exceed the window duration due to double counting and should be validated
    upstream if that behavior is undesired.
    """
    total_overlap = 0.0

    for interval_start_sec, interval_end_sec in intervals:
        total_overlap += compute_overlap_seconds(
            window_start_sec=window.start_sec,
            window_end_sec=window.end_sec,
            interval_start_sec=interval_start_sec,
            interval_end_sec=interval_end_sec,
        )

    return total_overlap


def compute_overlap_ratio(
    window: TimeWindow,
    overlap_seconds: float,
) -> float:
    """
    Compute the overlap fraction for a time window.

    Parameters
    ----------
    window : TimeWindow
        Temporal window.
    overlap_seconds : float
        Total overlap in seconds.

    Returns
    -------
    float
        Overlap ratio in the interval [0, 1].

    Raises
    ------
    SegmentationError
        If the overlap value is invalid.

    Notes
    -----
    This normalization step converts an absolute overlap duration into a
    scale-independent quantity that can be used directly by downstream labeling
    policies.
    """
    _validate_non_negative(overlap_seconds, "overlap_seconds")

    if overlap_seconds > window.duration_sec:
        raise SegmentationError("Overlap cannot be greater than the window duration.")

    return overlap_seconds / window.duration_sec
