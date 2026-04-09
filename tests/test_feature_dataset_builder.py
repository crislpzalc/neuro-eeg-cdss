from pathlib import Path

from neuro_eeg_cdss.features.dataset_builder import load_segments_dataset


def test_load_segments_dataset_runs():
    segments_path = Path("data/processed/segments.parquet")
    df = load_segments_dataset(segments_path)

    assert len(df) > 0
    assert "path" in df.columns
    assert "label" in df.columns
    assert "start_sec" in df.columns
    assert "end_sec" in df.columns
