"""
Utilities for building the tabular feature dataset from the segment-level EEG
dataset.

This module links segment metadata, raw EEG loading, temporal cropping, and
feature extraction into a reproducible pipeline that produces one feature row
per labeled segment.

Design goals
------------
- Preserve traceability from each feature row back to the original recording
  and segment boundaries
- Keep feature extraction deterministic and modular
- Avoid loading more signal data than required for each segment
- Support both in-memory dataset construction and chunked disk output
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neuro_eeg_cdss.features.extractors import extract_all_features_per_channel


class FeatureDatasetBuilderError(ValueError):
    """Raised when feature dataset construction fails."""


@dataclass(frozen=True)
class FeatureRecord:
    """
    Representation of one row in the final feature dataset.

    Attributes
    ----------
    subject : str
        Subject identifier.
    session : str | None
        Optional session identifier.
    run : str | None
        Optional run identifier.
    path : str
        Path to the source EEG file.
    start_sec : float
        Segment start time in seconds.
    end_sec : float
        Segment end time in seconds.
    label : int
        Binary label assigned to the segment.

    Notes
    -----
    This record stores the segment-level metadata that must remain attached to
    the extracted features for traceability, downstream joins, and evaluation.
    """

    subject: str
    session: str | None
    run: str | None
    path: str
    start_sec: float
    end_sec: float
    label: int


def load_segments_dataset(segments_path: str | Path) -> pd.DataFrame:
    """
    Load the segment-level dataset and validate the minimal required schema.

    Parameters
    ----------
    segments_path : str | Path
        Path to ``segments.parquet``.

    Returns
    -------
    pd.DataFrame
        Loaded segment-level dataset.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    FeatureDatasetBuilderError
        If required columns are missing.

    Notes
    -----
    Only the columns strictly required for feature extraction are enforced
    here. Additional columns may exist in the segment dataset and are
    intentionally ignored by this builder.
    """
    segments_path = Path(segments_path)

    if not segments_path.exists():
        raise FileNotFoundError(f"Segments dataset not found: {segments_path}")

    df = pd.read_parquet(segments_path)

    # Only the columns needed to locate each segment in the source EEG and to
    # preserve its downstream identity are required at this stage.
    required_columns = {
        "subject",
        "session",
        "run",
        "path",
        "start_sec",
        "end_sec",
        "label",
    }
    missing = required_columns - set(df.columns)

    if missing:
        raise FeatureDatasetBuilderError(
            f"Missing required columns in segments.parquet: {sorted(missing)}"
        )

    return df


def _safe_get_str(row: pd.Series, key: str) -> str | None:
    """
    Retrieve an optional string value from a row.

    Parameters
    ----------
    row : pd.Series
        Input row.
    key : str
        Column name to retrieve.

    Returns
    -------
    str | None
        String value if present and non-null, otherwise None.
    """
    if key not in row.index:
        return None

    value = row[key]
    if pd.isna(value):
        return None

    return str(value)


def load_raw_eeg(eeg_path: str | Path) -> mne.io.BaseRaw:
    """
    Load an EDF recording with MNE without full preload.

    Parameters
    ----------
    eeg_path : str | Path
        Path to the EDF file.

    Returns
    -------
    mne.io.BaseRaw
        MNE raw object for the EEG recording.

    Notes
    -----
    The file is opened with ``preload=False`` so this builder can access only
    the required segment portions without forcing the entire recording into
    memory.
    """
    eeg_path = Path(eeg_path)

    try:
        raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose="ERROR")
    except Exception as exc:
        raise FeatureDatasetBuilderError(f"Could not read EEG file: {eeg_path}") from exc

    return raw


def extract_segment_signal(
    raw: mne.io.BaseRaw,
    start_sec: float,
    end_sec: float,
) -> tuple[np.ndarray, float, list[str]]:
    """
    Extract the signal corresponding to a temporal segment.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        EEG recording.
    start_sec : float
        Segment start time in seconds.
    end_sec : float
        Segment end time in seconds.

    Returns
    -------
    signal : np.ndarray
        Signal array with shape ``(n_channels, n_samples)``.
    sfreq : float
        Sampling frequency.
    channel_names : list[str]
        Channel names associated with the signal.

    Notes
    -----
    Segment boundaries are converted from seconds to sample indices using the
    recording sampling frequency. The resulting crop is expected to preserve
    channel order exactly as stored in the source recording.
    """
    if end_sec <= start_sec:
        raise FeatureDatasetBuilderError(
            f"Invalid segment boundaries: start_sec={start_sec}, end_sec={end_sec}"
        )

    sfreq = float(raw.info["sfreq"])
    start_sample = int(round(start_sec * sfreq))
    end_sample = int(round(end_sec * sfreq))

    if end_sample <= start_sample:
        raise FeatureDatasetBuilderError(
            f"Invalid segment after conversion to samples: "
            f"start_sample={start_sample}, end_sample={end_sample}"
        )

    data = raw.get_data(start=start_sample, stop=end_sample)
    ch_names = list(raw.ch_names)

    if data.ndim != 2:
        raise FeatureDatasetBuilderError(
            f"Expected 2D signal array (n_channels, n_samples). shape={data.shape}"
        )

    return data, sfreq, ch_names


def build_features_for_single_recording(
    recording_segments_df: pd.DataFrame,
    relative_bandpower: bool = False,
) -> pd.DataFrame:
    """
    Build features for all segments belonging to a single EEG recording.

    Parameters
    ----------
    recording_segments_df : pd.DataFrame
        Sub-dataframe containing all segments for one ``path``.
    relative_bandpower : bool, default=False
        If True, compute relative bandpower instead of absolute bandpower.

    Returns
    -------
    pd.DataFrame
        Feature rows for all segments in the recording.

    Notes
    -----
    Grouping by recording allows the raw EEG file to be opened only once per
    source file, which improves efficiency while preserving deterministic
    segment-level feature extraction.
    """
    if recording_segments_df.empty:
        return pd.DataFrame()

    eeg_path = Path(str(recording_segments_df.iloc[0]["path"]))
    raw = load_raw_eeg(eeg_path)

    rows: list[dict] = []

    # Each segment is cropped independently so that feature extraction remains
    # fully traceable to the original segment boundaries.
    for _, row in recording_segments_df.iterrows():
        signal, sfreq, ch_names = extract_segment_signal(
            raw=raw,
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
        )

        feature_values = extract_all_features_per_channel(
            signal=signal,
            sfreq=sfreq,
            channel_names=None,
            relative_bandpower=relative_bandpower,
        )

        base_record = FeatureRecord(
            subject=str(row["subject"]),
            session=_safe_get_str(row, "session"),
            run=_safe_get_str(row, "run"),
            path=str(row["path"]),
            start_sec=float(row["start_sec"]),
            end_sec=float(row["end_sec"]),
            label=int(row["label"]),
        )

        # Metadata and extracted features are merged into one flat row so the
        # resulting table can be serialized and consumed directly by classical
        # ML pipelines.
        combined_row = {**asdict(base_record), **feature_values}
        rows.append(combined_row)

    return pd.DataFrame(rows)


def build_features_dataset(
    segments_path: str | Path,
    relative_bandpower: bool = False,
    max_segments: int | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build the full feature dataset from ``segments.parquet``.

    Parameters
    ----------
    segments_path : str | Path
        Path to ``segments.parquet``.
    relative_bandpower : bool, default=False
        If True, compute relative bandpower features.
    max_segments : int | None, default=None
        If provided, limit the number of segments for quick testing.
    output_path : str | Path | None, default=None
        If provided, write the dataset in per-recording parquet parts instead
        of returning a concatenated in-memory dataframe.

    Returns
    -------
    pd.DataFrame
        Full feature dataset. Returns an empty dataframe when ``output_path``
        is provided and results are written incrementally to disk.

    Notes
    -----
    This function supports two execution modes:
    - in-memory accumulation for smaller experiments
    - chunked on-disk writing for larger runs
    """
    segments_df = load_segments_dataset(segments_path)

    if max_segments is not None:
        if max_segments <= 0:
            raise FeatureDatasetBuilderError(
                f"'max_segments' must be > 0. Received value: {max_segments}"
            )
        segments_df = segments_df.head(max_segments).copy()

    grouped = segments_df.groupby("path", sort=True)
    total_recordings = len(grouped)

    all_feature_dfs: list[pd.DataFrame] = []

    # Processing is performed recording by recording to avoid reopening the
    # same EEG file for every segment.
    for i, (path, recording_segments_df) in enumerate(grouped):
        print(
            f"[{i + 1}/{total_recordings}] Processing recording: {path} "
            f"({len(recording_segments_df)} segments)"
        )

        feature_df = build_features_for_single_recording(
            recording_segments_df=recording_segments_df,
            relative_bandpower=relative_bandpower,
        )

        if output_path is not None:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            part_path = output_dir / f"features_part_{i:04d}.parquet"
            feature_df.to_parquet(part_path, index=False)
        else:
            all_feature_dfs.append(feature_df)

    if output_path is not None:
        print(f"Dataset saved in parts to: {Path(output_path).parent} (features_part_XXXX.parquet)")
        return pd.DataFrame()

    if not all_feature_dfs:
        return pd.DataFrame()

    features_df = pd.concat(all_feature_dfs, axis=0, ignore_index=True)

    return features_df.sort_values(
        by=["subject", "path", "start_sec"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def save_features_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save the feature dataset to parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Feature dataset.
    output_path : str | Path
        Output path.

    Notes
    -----
    Parent directories are created automatically to make serialization
    reproducible in clean environments.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
