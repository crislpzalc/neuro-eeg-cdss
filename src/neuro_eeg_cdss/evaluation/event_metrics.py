"""
Event-level evaluation metrics for seizure detection.

Window-level metrics (Sprint 1E) treat each 5-second segment independently.
This module aggregates consecutive positive windows into *events* and
evaluates detection at the event level — which is how neurologists
actually think about seizure detection.

Key concepts
------------
- **Event**: a contiguous run of positive windows within a single
  recording, characterized by start time, end time, and duration.
- **Matching**: a ground-truth event is considered "detected" if at
  least one detected event overlaps with it in time.
- **False alarm**: a detected event that does not overlap with any
  ground-truth event.
- **Detection latency**: time from ground-truth onset to the first
  overlapping detected window.

Design goals
------------
- Produce clinically meaningful metrics (event recall, false alarms
  per hour, detection latency) that complement window-level metrics
- Process each recording independently — events never span recordings
- Support both raw and post-processed predictions via column selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class EventEvaluationError(ValueError):
    """Raised when event evaluation encounters invalid input."""


# ── Data structures ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Event:
    """
    A temporal event (seizure or detection) within a single recording.

    Attributes
    ----------
    start_sec : float
        Start time in seconds from the beginning of the recording.
    end_sec : float
        End time in seconds.
    n_windows : int
        Number of consecutive windows forming this event.
    """

    start_sec: float
    end_sec: float
    n_windows: int

    @property
    def duration_sec(self) -> float:
        """Duration of the event in seconds."""
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "n_windows": self.n_windows,
            "duration_sec": self.duration_sec,
        }


@dataclass(frozen=True)
class EventMetrics:
    """
    Event-level evaluation metrics aggregated across all recordings.

    Attributes
    ----------
    n_true_events : int
        Total number of ground-truth seizure events.
    n_detected_events : int
        Total number of detected events (positive prediction runs).
    n_true_positives : int
        Ground-truth events that were detected (at least one overlap).
    n_false_negatives : int
        Ground-truth events that were missed entirely.
    n_false_positives : int
        Detected events that overlap with no ground-truth event.
    event_sensitivity : float
        Fraction of ground-truth events detected. Also called event
        recall or event detection rate.
    event_precision : float
        Fraction of detected events that match a ground-truth event.
    event_f1 : float
        Harmonic mean of event sensitivity and event precision.
    event_f2 : float
        F-beta with beta=2, weighting sensitivity over precision.
    false_alarm_rate_per_hour : float
        Number of false positive events per hour of recording.
    mean_latency_sec : float
        Mean detection latency across detected true events (seconds).
        NaN if no events were detected.
    median_latency_sec : float
        Median detection latency. NaN if no events were detected.
    total_duration_hours : float
        Total monitoring duration in hours.
    """

    n_true_events: int
    n_detected_events: int
    n_true_positives: int
    n_false_negatives: int
    n_false_positives: int
    event_sensitivity: float
    event_precision: float
    event_f1: float
    event_f2: float
    false_alarm_rate_per_hour: float
    mean_latency_sec: float
    median_latency_sec: float
    total_duration_hours: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary, converting NaN to None for JSON."""
        d: dict[str, Any] = {}
        for key in self.__dataclass_fields__:
            val = getattr(self, key)
            d[key] = None if isinstance(val, float) and np.isnan(val) else val
        return d


# ── Event extraction ─────────────────────────────────────────────────


def extract_events(
    labels: np.ndarray,
    start_secs: np.ndarray,
    window_duration: float = 5.0,
) -> list[Event]:
    """
    Extract contiguous events from a sequence of binary labels.

    Scans a temporally ordered label array and groups consecutive
    positive windows (value=1) into events.

    Parameters
    ----------
    labels : np.ndarray
        1D binary array (0/1) ordered by time within a recording.
    start_secs : np.ndarray
        1D array of window start times (seconds), same length.
    window_duration : float
        Duration of each window in seconds. Used to compute the end
        time of the last window in each event.

    Returns
    -------
    list[Event]
        Detected events sorted by start time.

    Raises
    ------
    EventEvaluationError
        If inputs have mismatched lengths or invalid values.
    """
    if len(labels) != len(start_secs):
        raise EventEvaluationError(
            f"Length mismatch: labels={len(labels)}, start_secs={len(start_secs)}"
        )

    if len(labels) == 0:
        return []

    events: list[Event] = []
    n = len(labels)
    i = 0

    while i < n:
        if labels[i] == 1:
            run_start = i
            while i < n and labels[i] == 1:
                i += 1
            run_end = i  # exclusive
            n_windows = run_end - run_start
            event_start = float(start_secs[run_start])
            event_end = float(start_secs[run_end - 1]) + window_duration
            events.append(Event(event_start, event_end, n_windows))
        else:
            i += 1

    return events


# ── Event matching ───────────────────────────────────────────────────


def _events_overlap(a: Event, b: Event) -> bool:
    """Check if two events overlap in time (any overlap counts)."""
    return a.start_sec < b.end_sec and b.start_sec < a.end_sec


def match_events(
    true_events: list[Event],
    detected_events: list[Event],
) -> tuple[list[bool], list[bool], list[float]]:
    """
    Match detected events to ground-truth events by temporal overlap.

    A ground-truth event is "detected" if at least one detected event
    overlaps with it. A detected event is a "true positive" if it
    overlaps with at least one ground-truth event.

    Parameters
    ----------
    true_events : list[Event]
        Ground-truth seizure events.
    detected_events : list[Event]
        Predicted seizure events.

    Returns
    -------
    true_matched : list[bool]
        For each ground-truth event, whether it was detected.
    detected_matched : list[bool]
        For each detected event, whether it matches a ground-truth.
    latencies : list[float]
        For each detected ground-truth event, the detection latency
        in seconds (time from true onset to detected onset). Only
        includes entries for matched true events. Negative latency
        means early detection.
    """
    true_matched = [False] * len(true_events)
    detected_matched = [False] * len(detected_events)
    latencies: list[float] = []

    for i, true_ev in enumerate(true_events):
        best_latency = float("inf")

        for j, det_ev in enumerate(detected_events):
            if _events_overlap(true_ev, det_ev):
                true_matched[i] = True
                detected_matched[j] = True
                latency = det_ev.start_sec - true_ev.start_sec
                if abs(latency) < abs(best_latency):
                    best_latency = latency

        if true_matched[i]:
            latencies.append(best_latency)

    return true_matched, detected_matched, latencies


# ── Metric computation ───────────────────────────────────────────────


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide with zero-denominator protection."""
    return numerator / denominator if denominator > 0 else 0.0


def _compute_fbeta(precision: float, sensitivity: float, beta: float) -> float:
    """Compute F-beta score from precision and sensitivity."""
    beta_sq = beta * beta
    numerator = (1 + beta_sq) * precision * sensitivity
    denominator = beta_sq * precision + sensitivity
    return _safe_divide(numerator, denominator)


def compute_event_metrics(
    true_events: list[Event],
    detected_events: list[Event],
    total_duration_hours: float,
) -> EventMetrics:
    """
    Compute event-level metrics from matched events.

    Parameters
    ----------
    true_events : list[Event]
        Ground-truth seizure events.
    detected_events : list[Event]
        Predicted seizure events.
    total_duration_hours : float
        Total monitoring duration in hours, used for false alarm rate.

    Returns
    -------
    EventMetrics
        Complete event-level evaluation.
    """
    true_matched, detected_matched, latencies = match_events(true_events, detected_events)

    n_true = len(true_events)
    n_detected = len(detected_events)
    n_tp = sum(true_matched)
    n_fn = n_true - n_tp
    n_fp = sum(not m for m in detected_matched)

    sensitivity = _safe_divide(n_tp, n_true)
    precision = _safe_divide(n_tp, n_detected)
    f1 = _compute_fbeta(precision, sensitivity, beta=1.0)
    f2 = _compute_fbeta(precision, sensitivity, beta=2.0)
    fa_rate = _safe_divide(n_fp, total_duration_hours)

    mean_lat = float(np.mean(latencies)) if latencies else float("nan")
    median_lat = float(np.median(latencies)) if latencies else float("nan")

    return EventMetrics(
        n_true_events=n_true,
        n_detected_events=n_detected,
        n_true_positives=n_tp,
        n_false_negatives=n_fn,
        n_false_positives=n_fp,
        event_sensitivity=sensitivity,
        event_precision=precision,
        event_f1=f1,
        event_f2=f2,
        false_alarm_rate_per_hour=fa_rate,
        mean_latency_sec=mean_lat,
        median_latency_sec=median_lat,
        total_duration_hours=total_duration_hours,
    )


# ── DataFrame-level API ──────────────────────────────────────────────


REQUIRED_COLUMNS = {"subject", "path", "start_sec", "y_true"}


def compute_dataset_event_metrics(
    df: pd.DataFrame,
    pred_col: str = "y_pred",
    window_duration: float = 5.0,
) -> EventMetrics:
    """
    Compute event-level metrics from an enriched predictions DataFrame.

    Groups predictions by recording (subject + path), extracts events
    from each recording, aggregates all events across the dataset, and
    computes metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched predictions with columns: ``subject``, ``path``,
        ``start_sec``, ``y_true``, and the prediction column.
    pred_col : str
        Name of the column containing binary predictions. Use
        ``"y_pred"`` for raw or ``"y_pred_post"`` for post-processed.
    window_duration : float
        Duration of each window in seconds.

    Returns
    -------
    EventMetrics
        Aggregated event-level metrics across the full dataset.

    Raises
    ------
    EventEvaluationError
        If required columns are missing.
    """
    required = REQUIRED_COLUMNS | {pred_col}
    missing = required - set(df.columns)
    if missing:
        raise EventEvaluationError(f"Missing required columns: {sorted(missing)}")

    all_true_events: list[Event] = []
    all_detected_events: list[Event] = []
    total_windows = 0

    for (_subject, _path), group in df.groupby(["subject", "path"]):
        sorted_group = group.sort_values("start_sec")

        labels_true = sorted_group["y_true"].values.astype(int)
        labels_pred = sorted_group[pred_col].values.astype(int)
        start_secs = sorted_group["start_sec"].values.astype(float)

        true_events = extract_events(labels_true, start_secs, window_duration)
        detected_events = extract_events(labels_pred, start_secs, window_duration)

        all_true_events.extend(true_events)
        all_detected_events.extend(detected_events)
        total_windows += len(sorted_group)

    total_duration_hours = (total_windows * window_duration) / 3600.0

    return compute_event_metrics(all_true_events, all_detected_events, total_duration_hours)


def compute_per_recording_summary(
    df: pd.DataFrame,
    pred_col: str = "y_pred",
    window_duration: float = 5.0,
) -> list[dict[str, Any]]:
    """
    Compute event-level summary for each recording individually.

    Useful for understanding which recordings have the most false
    alarms or missed events.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched predictions DataFrame.
    pred_col : str
        Prediction column name.
    window_duration : float
        Window duration in seconds.

    Returns
    -------
    list[dict[str, Any]]
        One entry per recording with event counts and metrics.
    """
    required = REQUIRED_COLUMNS | {pred_col}
    missing = required - set(df.columns)
    if missing:
        raise EventEvaluationError(f"Missing required columns: {sorted(missing)}")

    summaries: list[dict[str, Any]] = []

    for (subject, path), group in df.groupby(["subject", "path"]):
        sorted_group = group.sort_values("start_sec")

        labels_true = sorted_group["y_true"].values.astype(int)
        labels_pred = sorted_group[pred_col].values.astype(int)
        start_secs = sorted_group["start_sec"].values.astype(float)

        true_events = extract_events(labels_true, start_secs, window_duration)
        detected_events = extract_events(labels_pred, start_secs, window_duration)

        n_windows = len(sorted_group)
        duration_hours = (n_windows * window_duration) / 3600.0

        metrics = compute_event_metrics(true_events, detected_events, duration_hours)

        summaries.append(
            {
                "subject": subject,
                "path": path,
                "n_windows": n_windows,
                "duration_hours": round(duration_hours, 4),
                "n_true_events": metrics.n_true_events,
                "n_detected_events": metrics.n_detected_events,
                "n_true_positives": metrics.n_true_positives,
                "n_false_positives": metrics.n_false_positives,
                "n_false_negatives": metrics.n_false_negatives,
                "event_sensitivity": metrics.event_sensitivity,
                "false_alarm_rate_per_hour": round(metrics.false_alarm_rate_per_hour, 2),
            }
        )

    return summaries
