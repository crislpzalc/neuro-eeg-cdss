"""
Utilities for constructing the segment-level dataset from indexed EEG
recordings and seizure annotations.

This module connects manifest-level metadata, event parsing, temporal
segmentation, and labeling into a reproducible dataset-building pipeline.
Its purpose is to produce a tabular representation of fixed-length EEG
segments that can be consumed by downstream feature extraction and modeling
stages.

Design goals
------------
- Preserve traceability from each segment back to its source recording
- Keep dataset construction deterministic and auditable
- Separate I/O, temporal logic, and labeling policy across modules
- Fail early when required metadata or source files are missing
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import pandas as pd

from neuro_eeg_cdss.preprocessing.events import read_seizure_intervals
from neuro_eeg_cdss.preprocessing.labeling import assign_label
from neuro_eeg_cdss.preprocessing.segmentation import (
    compute_overlap_ratio,
    compute_total_overlap_seconds,
    generate_time_windows,
)


class DatasetBuilderError(ValueError):
    """Raised when segment dataset construction fails."""


@dataclass(frozen=True)
class SegmentRecord:
    """
    Representation of one row in the final segment-level dataset.

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
    recording_duration_sec : float
        Total duration of the source recording in seconds.
    start_sec : float
        Segment start time in seconds.
    end_sec : float
        Segment end time in seconds.
    window_size_sec : float
        Fixed window size used to generate the segment.
    stride_sec : float
        Stride used between consecutive windows.
    overlap_ratio : float
        Fraction of the segment overlapping with annotated seizure activity.
    label : int
        Binary label assigned to the segment.

    Notes
    -----
    This record is intentionally flat and serialization-friendly so it can be
    written directly to parquet and joined later with downstream features or
    evaluation metadata.
    """

    subject: str
    session: str | None
    run: str | None
    path: str
    recording_duration_sec: float
    start_sec: float
    end_sec: float
    window_size_sec: float
    stride_sec: float
    overlap_ratio: float
    label: int


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """
    Load ``manifest.parquet``.

    Parameters
    ----------
    manifest_path : str | Path
        Path to the manifest file.

    Returns
    -------
    pd.DataFrame
        Loaded manifest.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    DatasetBuilderError
        If required columns are missing.

    Notes
    -----
    Only the minimal columns required for segment construction are enforced at
    this stage. Additional manifest metadata may be present and is preserved
    for optional downstream use.
    """
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    df = pd.read_parquet(path)

    # Only the essential columns needed to locate recordings and attach subject
    # identity are required here.
    required_columns = {"subject", "path"}
    missing = required_columns - set(df.columns)

    if missing:
        raise DatasetBuilderError(f"Missing required columns in manifest: {sorted(missing)}")

    return df


def derive_events_tsv_path_from_eeg_path(eeg_path: str | Path) -> Path:
    """
    Derive the corresponding ``events.tsv`` path from an EEG file path.

    Example
    -------
    ``sub-01_ses-01_task-xxx_eeg.edf`` ->
    ``sub-01_ses-01_task-xxx_events.tsv``

    Notes
    -----
    This derivation rule is intentionally naming-convention-based and assumes
    BIDS-like file naming. If recording filenames deviate from this pattern,
    the path resolution logic should be updated explicitly rather than guessed.
    """
    eeg_path = Path(eeg_path)

    if eeg_path.name.endswith("_eeg.edf"):
        return eeg_path.with_name(eeg_path.name.replace("_eeg.edf", "_events.tsv"))

    raise DatasetBuilderError(f"Could not derive events.tsv from EEG path: {eeg_path}")


def get_recording_duration_sec(eeg_path: str | Path) -> float:
    """
    Read the total EEG recording duration in seconds.

    Parameters
    ----------
    eeg_path : str | Path
        Path to the EDF file.

    Returns
    -------
    float
        Recording duration in seconds.

    Notes
    -----
    The EDF file is opened with ``preload=False`` because only metadata are
    required at this stage. This keeps dataset construction lightweight and
    avoids unnecessary signal loading during segment index generation.
    """
    eeg_path = Path(eeg_path)

    try:
        raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose="ERROR")
    except Exception as exc:
        raise DatasetBuilderError(
            f"Could not read EEG file to obtain recording duration: {eeg_path}"
        ) from exc

    return float(raw.n_times / raw.info["sfreq"])


def _safe_get_str(row: pd.Series, key: str) -> str | None:
    """
    Retrieve an optional string value from a manifest row.

    Parameters
    ----------
    row : pd.Series
        Manifest row.
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


def build_segments_for_recording(
    row: pd.Series,
    window_size_sec: float,
    stride_sec: float,
    positive_overlap_threshold: float = 0.5,
    drop_partial_overlap: bool = True,
) -> list[SegmentRecord]:
    """
    Build labeled segments for a single EEG recording.

    Parameters
    ----------
    row : pd.Series
        Manifest row describing one recording.
    window_size_sec : float
        Window size in seconds.
    stride_sec : float
        Stride between consecutive windows in seconds.
    positive_overlap_threshold : float, default=0.5
        Threshold above which a window is considered positive.
    drop_partial_overlap : bool, default=True
        If True, discard ambiguous windows with partial overlap below the
        threshold.

    Returns
    -------
    list[SegmentRecord]
        Valid segment records for the recording.

    Notes
    -----
    This function is the core integration point of the preprocessing pipeline:
    it combines recording metadata, event annotations, temporal segmentation,
    and label assignment into a single deterministic output.
    """
    eeg_path = Path(str(row["path"]))
    events_tsv_path = derive_events_tsv_path_from_eeg_path(eeg_path)

    recording_duration_sec = get_recording_duration_sec(eeg_path)
    seizure_intervals = read_seizure_intervals(events_tsv_path)

    # Convert structured seizure intervals into plain tuples to decouple this
    # module from the internal representation used by the events parser.
    interval_tuples = [(interval.onset_sec, interval.end_sec) for interval in seizure_intervals]

    windows = generate_time_windows(
        recording_duration_sec=recording_duration_sec,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
    )

    subject = str(row["subject"])
    session = _safe_get_str(row, "session")
    run = _safe_get_str(row, "run")

    records: list[SegmentRecord] = []

    # Each window is processed independently so that overlap computation and
    # labeling remain transparent and easy to audit.
    for window in windows:
        overlap_seconds = compute_total_overlap_seconds(window, interval_tuples)
        overlap_ratio = compute_overlap_ratio(window, overlap_seconds)

        decision = assign_label(
            overlap_ratio=overlap_ratio,
            positive_overlap_threshold=positive_overlap_threshold,
            drop_partial_overlap=drop_partial_overlap,
        )

        # Ambiguous windows may be intentionally excluded depending on the
        # configured labeling policy.
        if not decision.keep:
            continue

        records.append(
            SegmentRecord(
                subject=subject,
                session=session,
                run=run,
                path=str(eeg_path),
                recording_duration_sec=recording_duration_sec,
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                window_size_sec=window_size_sec,
                stride_sec=stride_sec,
                overlap_ratio=overlap_ratio,
                label=decision.label,
            )
        )

    return records


def build_segments_dataset(
    manifest_path: str | Path,
    window_size_sec: float,
    stride_sec: float,
    positive_overlap_threshold: float = 0.5,
    drop_partial_overlap: bool = True,
) -> pd.DataFrame:
    """
    Build the full segment-level dataset from ``manifest.parquet``.

    Parameters
    ----------
    manifest_path : str | Path
        Path to ``manifest.parquet``.
    window_size_sec : float
        Window size in seconds.
    stride_sec : float
        Stride between consecutive windows in seconds.
    positive_overlap_threshold : float, default=0.5
        Positivity threshold used by the labeling policy.
    drop_partial_overlap : bool, default=True
        If True, discard ambiguous windows.

    Returns
    -------
    pd.DataFrame
        Final segment-level dataset.

    Notes
    -----
    Output rows are sorted deterministically by subject, path, and start time
    to guarantee stable downstream serialization and testing behavior.
    """
    manifest_df = load_manifest(manifest_path)

    all_records: list[SegmentRecord] = []

    # Row-wise iteration is used for clarity and to preserve explicit
    # per-recording control over I/O, segmentation, and labeling.
    for _, row in manifest_df.iterrows():
        record_segments = build_segments_for_recording(
            row=row,
            window_size_sec=window_size_sec,
            stride_sec=stride_sec,
            positive_overlap_threshold=positive_overlap_threshold,
            drop_partial_overlap=drop_partial_overlap,
        )
        all_records.extend(record_segments)

    df = pd.DataFrame([asdict(record) for record in all_records])

    if df.empty:
        # Return an explicitly typed empty table with the expected schema so
        # downstream code can rely on stable column names even when no segments
        # are produced.
        return pd.DataFrame(
            columns=[
                "subject",
                "session",
                "run",
                "path",
                "recording_duration_sec",
                "start_sec",
                "end_sec",
                "window_size_sec",
                "stride_sec",
                "overlap_ratio",
                "label",
            ]
        )

    return df.sort_values(
        by=["subject", "path", "start_sec"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def save_segments_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save the segment-level dataset to parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Segment-level dataset.
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
