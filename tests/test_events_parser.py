from pathlib import Path

from neuro_eeg_cdss.preprocessing.events import read_seizure_intervals


def test_events_parser_runs():
    events_file = list(Path("data/raw/chbmit_bids").rglob("*events.tsv"))[0]

    intervals = read_seizure_intervals(events_file)

    assert isinstance(intervals, list)


def test_intervals_have_valid_times():
    events_file = list(Path("data/raw/chbmit_bids").rglob("*events.tsv"))[0]

    intervals = read_seizure_intervals(events_file)

    for interval in intervals:
        assert interval.onset_sec >= 0
        assert interval.duration_sec >= 0
        assert interval.end_sec >= interval.onset_sec
