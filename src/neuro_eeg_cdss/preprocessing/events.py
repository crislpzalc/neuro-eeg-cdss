"""
Utilities for reading and parsing BIDS ``events.tsv`` files into structured
seizure intervals.

This module intentionally implements a conservative, dataset-specific parser
for CHB-MIT-style event annotations. It does not attempt to support arbitrary
BIDS event schemas beyond the columns required by this project.

Design goals
------------
- Fail early on malformed or ambiguous inputs
- Preserve deterministic behavior for downstream segmentation
- Make dataset assumptions explicit rather than implicit
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SeizureInterval:
    """
    Seizure interval in seconds relative to the start of the recording.

    Attributes
    ----------
    onset_sec : float
        Seizure onset time in seconds.
    duration_sec : float
        Seizure duration in seconds.
    end_sec : float
        Seizure end time in seconds.
    event_type : str
        Original event type from the events.tsv file.

    Notes
    -----
    This object represents an interval in recording-relative time, not in
    absolute clock time. Downstream code assumes that intervals are expressed
    in seconds from the beginning of the EEG recording.
    """

    onset_sec: float
    duration_sec: float
    end_sec: float
    event_type: str


class EventsFileError(ValueError):
    """Raised when an events.tsv file cannot be interpreted correctly."""


def read_events_tsv(events_tsv_path: str | Path) -> pd.DataFrame:
    """
    Read a BIDS events.tsv file and return it as a DataFrame.

    Notes
    -----
    Column names are stripped of surrounding whitespace, but their semantic
    meaning is not altered. This function performs lightweight I/O and basic
    normalization only; semantic validation is delegated to downstream steps.

    Parameters
    ----------
    events_tsv_path : str | Path
        Path to the events.tsv file.

    Returns
    -------
    pd.DataFrame
        Event table with normalized column names.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    EventsFileError
        If the file cannot be read.
    """
    path = Path(events_tsv_path)

    if not path.exists():
        raise FileNotFoundError(f"events.tsv not found: {path}")

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise EventsFileError(f"Could not read events.tsv file: {path}") from exc

    if df.empty:
        return df

    # Normalize column names defensively because tabular metadata exported from
    # heterogeneous sources may include trailing or leading whitespace.
    df.columns = [str(col).strip() for col in df.columns]
    return df


def validate_events_columns(df: pd.DataFrame) -> None:
    """
    Validate that the minimum required columns for seizure extraction exist.

    For this dataset, the required columns are:
    - onset
    - duration
    - eventType

    Parameters
    ----------
    df : pd.DataFrame
        Events DataFrame.

    Raises
    ------
    EventsFileError
        If required columns are missing.
    """
    # Only the minimal columns required for seizure interval extraction are
    # enforced here. Additional columns may exist in the original events.tsv
    # and are intentionally ignored by this parser.
    required_columns = {"onset", "duration", "eventType"}
    missing = required_columns - set(df.columns)

    if missing:
        raise EventsFileError(f"Missing required columns in events.tsv: {sorted(missing)}")


def _normalize_event_type(value: object) -> str:
    """
    Normalize the eventType value for robust comparison.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _is_seizure_event_type(event_type: str) -> bool:
    """
    Decide whether an eventType corresponds to a seizure.

    Dataset-specific rule:
    - 'bckg' = non-seizure
    - any code starting with 'sz' = seizure

    Notes
    -----
    This rule is intentionally dataset-specific. If event naming conventions
    change in another corpus, this function should be updated rather than
    silently reused.

    Examples
    --------
    'sz' -> True
    'sz_foc_a' -> True
    'sz_gen_m_tonicClonic' -> True
    'bckg' -> False
    """
    # TODO: consider logging or tracking unknown non-background event types to
    # detect annotation schema drift early.
    if event_type == "bckg":
        return False

    return event_type.startswith("sz")


def _coerce_non_negative_float(value: object, field_name: str) -> float:
    """
    Convert a value to a non-negative float.

    Parameters
    ----------
    value : object
        Value to convert.
    field_name : str
        Field name used in error messages.

    Returns
    -------
    float
        Converted numeric value.

    Raises
    ------
    EventsFileError
        If the value cannot be converted or is negative.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise EventsFileError(f"Non-numeric value for '{field_name}': {value!r}") from exc

    if numeric_value < 0:
        raise EventsFileError(f"Negative value for '{field_name}': {numeric_value}")

    return numeric_value


def extract_seizure_intervals(df: pd.DataFrame) -> list[SeizureInterval]:
    """
    Extract seizure intervals from an events DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame read from events.tsv.

    Returns
    -------
    list[SeizureInterval]
        Sorted list of seizure intervals.

    Raises
    ------
    EventsFileError
        If required columns are missing or values are invalid.
    """
    if df.empty:
        return []

    validate_events_columns(df)

    # TODO: add optional semantic validation for duplicated, overlapping, or
    # zero-duration seizure intervals if such cases appear in the source data.
    seizure_intervals: list[SeizureInterval] = []

    # Row-wise iteration is used here for clarity and strict per-event error
    # reporting. The expected event tables are small, so readability is favored
    # over vectorized micro-optimizations.
    for _, row in df.iterrows():
        event_type = _normalize_event_type(row["eventType"])

        if not _is_seizure_event_type(event_type):
            continue

        onset_sec = _coerce_non_negative_float(row["onset"], "onset")
        duration_sec = _coerce_non_negative_float(row["duration"], "duration")

        # End time is derived rather than read directly to ensure internal
        # consistency between onset and duration.
        end_sec = onset_sec + duration_sec

        seizure_intervals.append(
            SeizureInterval(
                onset_sec=onset_sec,
                duration_sec=duration_sec,
                end_sec=end_sec,
                event_type=event_type,
            )
        )

    # Sorting guarantees deterministic ordering even if the source file is not
    # strictly ordered, which simplifies downstream segmentation and testing.
    seizure_intervals.sort(key=lambda interval: interval.onset_sec)
    return seizure_intervals


def read_seizure_intervals(events_tsv_path: str | Path) -> list[SeizureInterval]:
    """
    Read an events.tsv file and directly return seizure intervals.

    Parameters
    ----------
    events_tsv_path : str | Path
        Path to the events.tsv file.

    Returns
    -------
    list[SeizureInterval]
        Seizure intervals sorted by onset time.
    """
    df = read_events_tsv(events_tsv_path)
    return extract_seizure_intervals(df)


def intervals_to_dataframe(intervals: Iterable[SeizureInterval]) -> pd.DataFrame:
    """
    Convert a sequence of seizure intervals into a tabular DataFrame.

    Parameters
    ----------
    intervals : Iterable[SeizureInterval]
        Seizure intervals.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - onset_sec
        - duration_sec
        - end_sec
        - event_type

    Notes
    -----
    The output schema is intentionally explicit to support stable serialization
    and downstream joins with segment-level metadata.
    """
    rows = [
        {
            "onset_sec": interval.onset_sec,
            "duration_sec": interval.duration_sec,
            "end_sec": interval.end_sec,
            "event_type": interval.event_type,
        }
        for interval in intervals
    ]

    return pd.DataFrame(
        rows,
        columns=["onset_sec", "duration_sec", "end_sec", "event_type"],
    )
