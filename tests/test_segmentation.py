from neuro_eeg_cdss.preprocessing.segmentation import (
    TimeWindow,
    compute_overlap_ratio,
    compute_overlap_seconds,
    compute_total_overlap_seconds,
    generate_time_windows,
)


def test_generate_time_windows_without_overlap():
    windows = generate_time_windows(
        recording_duration_sec=12.0,
        window_size_sec=5.0,
        stride_sec=5.0,
    )

    assert len(windows) == 2
    assert windows[0] == TimeWindow(start_sec=0.0, end_sec=5.0)
    assert windows[1] == TimeWindow(start_sec=5.0, end_sec=10.0)


def test_generate_time_windows_empty_when_recording_is_too_short():
    windows = generate_time_windows(
        recording_duration_sec=4.0,
        window_size_sec=5.0,
        stride_sec=5.0,
    )

    assert windows == []


def test_compute_overlap_seconds_partial_overlap():
    overlap = compute_overlap_seconds(
        window_start_sec=10.0,
        window_end_sec=15.0,
        interval_start_sec=12.0,
        interval_end_sec=18.0,
    )

    assert overlap == 3.0


def test_compute_overlap_seconds_no_overlap():
    overlap = compute_overlap_seconds(
        window_start_sec=0.0,
        window_end_sec=5.0,
        interval_start_sec=6.0,
        interval_end_sec=8.0,
    )

    assert overlap == 0.0


def test_compute_total_overlap_seconds():
    window = TimeWindow(start_sec=10.0, end_sec=15.0)
    intervals = [
        (12.0, 13.0),
        (14.0, 16.0),
    ]

    total_overlap = compute_total_overlap_seconds(window, intervals)

    assert total_overlap == 2.0


def test_compute_overlap_ratio():
    window = TimeWindow(start_sec=10.0, end_sec=15.0)
    ratio = compute_overlap_ratio(window=window, overlap_seconds=2.5)

    assert ratio == 0.5
