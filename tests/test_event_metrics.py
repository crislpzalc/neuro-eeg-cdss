"""Tests for the event-level evaluation module (Sprint 2B)."""

import numpy as np
import pandas as pd
import pytest

from neuro_eeg_cdss.evaluation.event_metrics import (
    Event,
    EventEvaluationError,
    compute_dataset_event_metrics,
    compute_event_metrics,
    compute_per_recording_summary,
    extract_events,
    match_events,
)

# ── TestEvent ────────────────────────────────────────────────────────


class TestEvent:
    """Tests for the Event dataclass."""

    def test_duration(self):
        ev = Event(start_sec=10.0, end_sec=25.0, n_windows=3)
        assert ev.duration_sec == 15.0

    def test_to_dict(self):
        ev = Event(start_sec=0.0, end_sec=10.0, n_windows=2)
        d = ev.to_dict()
        assert d["start_sec"] == 0.0
        assert d["end_sec"] == 10.0
        assert d["n_windows"] == 2
        assert d["duration_sec"] == 10.0

    def test_frozen(self):
        ev = Event(start_sec=0.0, end_sec=5.0, n_windows=1)
        with pytest.raises(AttributeError):
            ev.start_sec = 1.0


# ── TestExtractEvents ────────────────────────────────────────────────


class TestExtractEvents:
    """Tests for extract_events."""

    def test_single_event(self):
        labels = np.array([0, 0, 1, 1, 1, 0, 0])
        start_secs = np.array([0, 5, 10, 15, 20, 25, 30], dtype=float)
        events = extract_events(labels, start_secs, window_duration=5.0)

        assert len(events) == 1
        assert events[0].start_sec == 10.0
        assert events[0].end_sec == 25.0  # 20 + 5
        assert events[0].n_windows == 3

    def test_multiple_events(self):
        labels = np.array([1, 1, 0, 0, 1, 0])
        start_secs = np.array([0, 5, 10, 15, 20, 25], dtype=float)
        events = extract_events(labels, start_secs, window_duration=5.0)

        assert len(events) == 2
        assert events[0].n_windows == 2
        assert events[1].n_windows == 1

    def test_no_events(self):
        labels = np.array([0, 0, 0, 0])
        start_secs = np.array([0, 5, 10, 15], dtype=float)
        events = extract_events(labels, start_secs, window_duration=5.0)

        assert events == []

    def test_all_positive(self):
        labels = np.array([1, 1, 1])
        start_secs = np.array([0, 5, 10], dtype=float)
        events = extract_events(labels, start_secs, window_duration=5.0)

        assert len(events) == 1
        assert events[0].start_sec == 0.0
        assert events[0].end_sec == 15.0
        assert events[0].n_windows == 3

    def test_empty_input(self):
        events = extract_events(np.array([], dtype=int), np.array([]), window_duration=5.0)
        assert events == []

    def test_single_window_event(self):
        labels = np.array([0, 1, 0])
        start_secs = np.array([0, 5, 10], dtype=float)
        events = extract_events(labels, start_secs, window_duration=5.0)

        assert len(events) == 1
        assert events[0].start_sec == 5.0
        assert events[0].end_sec == 10.0
        assert events[0].n_windows == 1

    def test_length_mismatch_raises(self):
        with pytest.raises(EventEvaluationError, match="Length mismatch"):
            extract_events(np.array([0, 1]), np.array([0.0]))

    def test_events_at_boundaries(self):
        labels = np.array([1, 0, 0, 1])
        start_secs = np.array([0, 5, 10, 15], dtype=float)
        events = extract_events(labels, start_secs, window_duration=5.0)

        assert len(events) == 2
        assert events[0].start_sec == 0.0
        assert events[1].start_sec == 15.0


# ── TestMatchEvents ──────────────────────────────────────────────────


class TestMatchEvents:
    """Tests for match_events."""

    def test_perfect_detection(self):
        true_ev = [Event(10.0, 25.0, 3)]
        det_ev = [Event(10.0, 25.0, 3)]

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert true_m == [True]
        assert det_m == [True]
        assert latencies == [0.0]

    def test_missed_event(self):
        true_ev = [Event(10.0, 25.0, 3)]
        det_ev = []

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert true_m == [False]
        assert latencies == []

    def test_false_alarm(self):
        true_ev = []
        det_ev = [Event(10.0, 15.0, 1)]

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert det_m == [False]
        assert latencies == []

    def test_partial_overlap(self):
        true_ev = [Event(10.0, 25.0, 3)]
        det_ev = [Event(20.0, 30.0, 2)]  # overlaps at 20-25

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert true_m == [True]
        assert det_m == [True]
        assert latencies[0] == pytest.approx(10.0)  # 20 - 10

    def test_early_detection(self):
        true_ev = [Event(10.0, 25.0, 3)]
        det_ev = [Event(5.0, 15.0, 2)]  # starts before true event

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert true_m == [True]
        assert latencies[0] == pytest.approx(-5.0)  # early detection

    def test_no_overlap(self):
        true_ev = [Event(10.0, 20.0, 2)]
        det_ev = [Event(25.0, 30.0, 1)]  # after true event

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert true_m == [False]
        assert det_m == [False]
        assert latencies == []

    def test_multiple_detections_for_one_true(self):
        true_ev = [Event(10.0, 30.0, 4)]
        det_ev = [Event(10.0, 15.0, 1), Event(25.0, 35.0, 2)]

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert true_m == [True]
        assert det_m == [True, True]
        # Latency is from closest detection: det[0] at 10.0 => latency 0.0
        assert latencies[0] == pytest.approx(0.0)

    def test_mixed_scenario(self):
        true_ev = [Event(10.0, 20.0, 2), Event(50.0, 60.0, 2)]
        det_ev = [Event(15.0, 25.0, 2), Event(35.0, 40.0, 1)]

        true_m, det_m, latencies = match_events(true_ev, det_ev)

        assert true_m == [True, False]  # first detected, second missed
        assert det_m == [True, False]  # first matches, second is FA
        assert len(latencies) == 1
        assert latencies[0] == pytest.approx(5.0)  # 15 - 10

    def test_empty_inputs(self):
        true_m, det_m, latencies = match_events([], [])
        assert true_m == []
        assert det_m == []
        assert latencies == []


# ── TestComputeEventMetrics ──────────────────────────────────────────


class TestComputeEventMetrics:
    """Tests for compute_event_metrics."""

    def test_perfect_detection(self):
        true_ev = [Event(10.0, 20.0, 2)]
        det_ev = [Event(10.0, 20.0, 2)]
        metrics = compute_event_metrics(true_ev, det_ev, total_duration_hours=1.0)

        assert metrics.event_sensitivity == 1.0
        assert metrics.event_precision == 1.0
        assert metrics.n_false_positives == 0
        assert metrics.false_alarm_rate_per_hour == 0.0

    def test_no_detections(self):
        true_ev = [Event(10.0, 20.0, 2)]
        det_ev = []
        metrics = compute_event_metrics(true_ev, det_ev, total_duration_hours=1.0)

        assert metrics.event_sensitivity == 0.0
        assert metrics.n_false_negatives == 1
        assert np.isnan(metrics.mean_latency_sec)

    def test_all_false_alarms(self):
        true_ev = []
        det_ev = [Event(10.0, 15.0, 1), Event(30.0, 35.0, 1)]
        metrics = compute_event_metrics(true_ev, det_ev, total_duration_hours=2.0)

        assert metrics.event_precision == 0.0
        assert metrics.n_false_positives == 2
        assert metrics.false_alarm_rate_per_hour == 1.0

    def test_f2_weights_sensitivity(self):
        # 1 true event detected + 3 false alarms
        true_ev = [Event(10.0, 20.0, 2)]
        det_ev = [
            Event(10.0, 20.0, 2),
            Event(30.0, 35.0, 1),
            Event(40.0, 45.0, 1),
            Event(50.0, 55.0, 1),
        ]
        metrics = compute_event_metrics(true_ev, det_ev, total_duration_hours=1.0)

        # Sensitivity = 1.0, Precision = 0.25
        assert metrics.event_sensitivity == 1.0
        assert metrics.event_precision == 0.25
        # F2 should be higher than F1 since sensitivity is perfect
        assert metrics.event_f2 > metrics.event_f1

    def test_no_events_at_all(self):
        metrics = compute_event_metrics([], [], total_duration_hours=1.0)

        assert metrics.n_true_events == 0
        assert metrics.n_detected_events == 0
        assert metrics.event_sensitivity == 0.0
        assert metrics.false_alarm_rate_per_hour == 0.0

    def test_to_dict_handles_nan(self):
        metrics = compute_event_metrics([Event(0.0, 10.0, 2)], [], total_duration_hours=1.0)
        d = metrics.to_dict()
        assert d["mean_latency_sec"] is None
        assert d["median_latency_sec"] is None
        assert d["n_true_events"] == 1


# ── TestComputeDatasetEventMetrics ───────────────────────────────────


class TestComputeDatasetEventMetrics:
    """Tests for compute_dataset_event_metrics (DataFrame-level)."""

    def _make_df(self) -> pd.DataFrame:
        """Two recordings: rec1 has a seizure, rec2 has none."""
        return pd.DataFrame(
            {
                "subject": (["sub-01"] * 8 + ["sub-02"] * 5),
                "path": (["rec_a"] * 8 + ["rec_b"] * 5),
                "start_sec": ([0, 5, 10, 15, 20, 25, 30, 35] + [0, 5, 10, 15, 20]),
                "y_true": ([0, 0, 1, 1, 1, 0, 0, 0] + [0, 0, 0, 0, 0]),
                "y_pred": (
                    [0, 0, 0, 1, 1, 0, 1, 0]  # late detection + isolated FA
                    + [0, 1, 0, 0, 0]  # isolated FA in rec2
                ),
            }
        )

    def test_basic_metrics(self):
        df = self._make_df()
        metrics = compute_dataset_event_metrics(df, pred_col="y_pred")

        assert metrics.n_true_events == 1  # one seizure in rec_a
        assert metrics.n_true_positives == 1  # detected (overlap at windows 15,20)
        assert metrics.n_false_positives == 2  # isolated det at window 30 + rec_b window 5

    def test_event_sensitivity(self):
        df = self._make_df()
        metrics = compute_dataset_event_metrics(df, pred_col="y_pred")

        assert metrics.event_sensitivity == 1.0  # the seizure was detected

    def test_false_alarm_rate(self):
        df = self._make_df()
        metrics = compute_dataset_event_metrics(df, pred_col="y_pred")

        # Total: 13 windows × 5s = 65s = 65/3600 hours
        expected_hours = 65.0 / 3600.0
        assert metrics.total_duration_hours == pytest.approx(expected_hours)
        assert metrics.n_false_positives == 2
        expected_fa_rate = 2.0 / expected_hours
        assert metrics.false_alarm_rate_per_hour == pytest.approx(expected_fa_rate)

    def test_detection_latency(self):
        df = self._make_df()
        metrics = compute_dataset_event_metrics(df, pred_col="y_pred")

        # True event: starts at 10.0
        # Detection: starts at 15.0 (first pred=1 overlapping)
        assert metrics.mean_latency_sec == pytest.approx(5.0)

    def test_supports_post_processed_column(self):
        df = self._make_df()
        df["y_pred_post"] = [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        metrics = compute_dataset_event_metrics(df, pred_col="y_pred_post")

        assert metrics.n_false_positives == 0  # isolated FAs removed
        assert metrics.n_true_positives == 1  # seizure still detected

    def test_missing_column_raises(self):
        df = pd.DataFrame({"subject": ["a"], "path": ["b"], "y_true": [0]})
        with pytest.raises(EventEvaluationError, match="Missing"):
            compute_dataset_event_metrics(df, pred_col="y_pred")

    def test_all_negative_recordings(self):
        df = pd.DataFrame(
            {
                "subject": ["sub-01"] * 4,
                "path": ["rec_a"] * 4,
                "start_sec": [0, 5, 10, 15],
                "y_true": [0, 0, 0, 0],
                "y_pred": [0, 0, 0, 0],
            }
        )
        metrics = compute_dataset_event_metrics(df, pred_col="y_pred")

        assert metrics.n_true_events == 0
        assert metrics.n_detected_events == 0
        assert metrics.false_alarm_rate_per_hour == 0.0


# ── TestComputePerRecordingSummary ───────────────────────────────────


class TestComputePerRecordingSummary:
    """Tests for compute_per_recording_summary."""

    def test_returns_one_entry_per_recording(self):
        df = pd.DataFrame(
            {
                "subject": ["sub-01"] * 4 + ["sub-02"] * 3,
                "path": ["rec_a"] * 4 + ["rec_b"] * 3,
                "start_sec": [0, 5, 10, 15, 0, 5, 10],
                "y_true": [0, 1, 1, 0, 0, 0, 0],
                "y_pred": [0, 1, 1, 0, 0, 0, 0],
            }
        )
        summaries = compute_per_recording_summary(df, pred_col="y_pred")

        assert len(summaries) == 2
        subjects = {s["subject"] for s in summaries}
        assert subjects == {"sub-01", "sub-02"}

    def test_includes_key_fields(self):
        df = pd.DataFrame(
            {
                "subject": ["sub-01"] * 3,
                "path": ["rec_a"] * 3,
                "start_sec": [0, 5, 10],
                "y_true": [1, 1, 0],
                "y_pred": [1, 0, 0],
            }
        )
        summaries = compute_per_recording_summary(df, pred_col="y_pred")

        s = summaries[0]
        assert "n_true_events" in s
        assert "n_false_positives" in s
        assert "event_sensitivity" in s
        assert "false_alarm_rate_per_hour" in s
        assert "duration_hours" in s


# ── TestEventMetrics ─────────────────────────────────────────────────


class TestEventMetrics:
    """Tests for the EventMetrics dataclass."""

    def test_frozen(self):
        metrics = compute_event_metrics([], [], total_duration_hours=1.0)
        with pytest.raises(AttributeError):
            metrics.n_true_events = 5

    def test_to_dict_complete(self):
        true_ev = [Event(0.0, 10.0, 2)]
        det_ev = [Event(0.0, 10.0, 2)]
        metrics = compute_event_metrics(true_ev, det_ev, total_duration_hours=1.0)
        d = metrics.to_dict()

        assert isinstance(d, dict)
        assert "event_sensitivity" in d
        assert "false_alarm_rate_per_hour" in d
        assert "mean_latency_sec" in d
