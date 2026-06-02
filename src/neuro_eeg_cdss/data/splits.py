"""
Patient-independent data splitting for seizure detection evaluation.

This module implements a stratified subject-level split that prevents data
leakage between train, validation and test sets. The splitting strategy
ensures that no patient's data appears in more than one split and that
the distribution of positive (seizure) segments is balanced across splits.

Design goals
------------
- Guarantee zero patient overlap between splits
- Distribute seizure segments proportionally across splits
- Produce fully deterministic, reproducible assignments
- Support both hold-out and future cross-validation extensions
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


class SplitError(ValueError):
    """Raised when split creation or validation fails."""


@dataclass(frozen=True)
class SplitConfig:
    """
    Configuration for patient-independent splitting.

    Attributes
    ----------
    train_ratio : float
        Target fraction of positive samples for the training set.
    val_ratio : float
        Target fraction of positive samples for the validation set.
    test_ratio : float
        Target fraction of positive samples for the test set.

    Notes
    -----
    Ratios must sum to 1.0 and define the target distribution of positive
    segments across splits. The actual subject assignment is performed by
    a greedy algorithm that approximates these targets as closely as
    possible given the discrete nature of subjects.
    """

    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2


@dataclass(frozen=True)
class SplitAssignment:
    """
    Result of assigning subjects to train/validation/test splits.

    Attributes
    ----------
    train_subjects : tuple[str, ...]
        Subject identifiers assigned to the training set.
    val_subjects : tuple[str, ...]
        Subject identifiers assigned to the validation set.
    test_subjects : tuple[str, ...]
        Subject identifiers assigned to the test set.
    config : SplitConfig
        Configuration used to generate this assignment.

    Notes
    -----
    Subjects are stored as sorted tuples for deterministic serialization
    and comparison.
    """

    train_subjects: tuple[str, ...]
    val_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    config: SplitConfig


def compute_subject_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-subject segment and label statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing at least ``subject`` and ``label`` columns.

    Returns
    -------
    pd.DataFrame
        One row per subject with columns: subject, n_segments, n_positive,
        n_negative. Sorted by n_positive descending.

    Raises
    ------
    SplitError
        If required columns are missing.
    """
    required = {"subject", "label"}
    missing = required - set(df.columns)
    if missing:
        raise SplitError(f"Missing required columns: {sorted(missing)}")

    stats = df.groupby("subject")["label"].agg(n_segments="size", n_positive="sum").reset_index()
    stats["n_negative"] = stats["n_segments"] - stats["n_positive"]

    return stats.sort_values(
        by=["n_positive", "subject"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _validate_config(config: SplitConfig) -> None:
    """
    Validate that split ratios are individually valid and sum to 1.

    Parameters
    ----------
    config : SplitConfig
        Split configuration to validate.

    Raises
    ------
    SplitError
        If any ratio is out of range or ratios do not sum to 1.
    """
    for name, value in [
        ("train_ratio", config.train_ratio),
        ("val_ratio", config.val_ratio),
        ("test_ratio", config.test_ratio),
    ]:
        if value <= 0.0 or value >= 1.0:
            raise SplitError(f"'{name}' must be in the interval (0, 1). Received: {value}")

    total = config.train_ratio + config.val_ratio + config.test_ratio
    if abs(total - 1.0) > 1e-9:
        raise SplitError(f"Split ratios must sum to 1.0. Received sum: {total}")


def _greedy_stratified_assignment(
    subject_stats: pd.DataFrame,
    config: SplitConfig,
) -> SplitAssignment:
    """
    Assign subjects to splits using a greedy stratified algorithm.

    The algorithm processes subjects in descending order of positive
    segment count. At each step, it assigns the current subject to the
    split whose positive count is furthest below its target proportion.

    This ensures that subjects with the most seizure segments are
    distributed across all splits, preventing concentration of positives
    in a single split.

    Parameters
    ----------
    subject_stats : pd.DataFrame
        Per-subject statistics from ``compute_subject_stats``.
    config : SplitConfig
        Target split ratios.

    Returns
    -------
    SplitAssignment
        Deterministic subject-to-split mapping.
    """
    total_positive = int(subject_stats["n_positive"].sum())

    targets = {
        "train": config.train_ratio * total_positive,
        "val": config.val_ratio * total_positive,
        "test": config.test_ratio * total_positive,
    }

    current: dict[str, float] = {"train": 0.0, "val": 0.0, "test": 0.0}
    assignments: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    # When deficit ratios are tied, smaller-target splits are filled first.
    # This prevents train from absorbing all subjects in early iterations
    # while val and test remain empty.
    priority = sorted(targets.keys(), key=lambda s: targets[s])

    sorted_subjects = subject_stats.sort_values(
        by=["n_positive", "subject"],
        ascending=[False, True],
    )

    for _, row in sorted_subjects.iterrows():
        subject = str(row["subject"])
        n_pos = int(row["n_positive"])

        best_split = None
        best_deficit = -float("inf")

        for split_name in priority:
            target = targets[split_name]
            if target > 0:
                deficit = (target - current[split_name]) / target
            else:
                deficit = 0.0

            if deficit > best_deficit:
                best_deficit = deficit
                best_split = split_name

        assignments[best_split].append(subject)
        current[best_split] += n_pos

    return SplitAssignment(
        train_subjects=tuple(sorted(assignments["train"])),
        val_subjects=tuple(sorted(assignments["val"])),
        test_subjects=tuple(sorted(assignments["test"])),
        config=config,
    )


def create_split(
    df: pd.DataFrame,
    config: SplitConfig | None = None,
) -> SplitAssignment:
    """
    Create a patient-independent split from a labeled dataset.

    This is the main entry point for split creation. It computes subject
    statistics, runs the greedy stratified assignment, and validates the
    result before returning.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with ``subject`` and ``label`` columns.
    config : SplitConfig | None
        Split configuration. Uses default ratios (60/20/20) if not provided.

    Returns
    -------
    SplitAssignment
        Subject-to-split mapping.

    Raises
    ------
    SplitError
        If the dataset or configuration is invalid, or if the resulting
        split violates integrity constraints.
    """
    if config is None:
        config = SplitConfig()

    _validate_config(config)

    subject_stats = compute_subject_stats(df)
    n_subjects = len(subject_stats)

    if n_subjects < 3:
        raise SplitError(f"At least 3 subjects are required for a 3-way split. Found: {n_subjects}")

    assignment = _greedy_stratified_assignment(subject_stats, config)
    validate_split(assignment, subject_stats)

    return assignment


def validate_split(
    assignment: SplitAssignment,
    subject_stats: pd.DataFrame,
) -> None:
    """
    Validate that a split assignment satisfies all integrity constraints.

    Checks performed:
    - No subject appears in more than one split
    - All subjects in the dataset are assigned to exactly one split
    - Each split contains at least one subject
    - Each split contains at least one positive segment

    Parameters
    ----------
    assignment : SplitAssignment
        Split to validate.
    subject_stats : pd.DataFrame
        Per-subject statistics.

    Raises
    ------
    SplitError
        If any constraint is violated.
    """
    train = set(assignment.train_subjects)
    val = set(assignment.val_subjects)
    test = set(assignment.test_subjects)

    if train & val:
        raise SplitError(f"Overlap between train and val: {train & val}")
    if train & test:
        raise SplitError(f"Overlap between train and test: {train & test}")
    if val & test:
        raise SplitError(f"Overlap between val and test: {val & test}")

    all_subjects = set(subject_stats["subject"].astype(str))
    assigned = train | val | test
    if assigned != all_subjects:
        missing = all_subjects - assigned
        extra = assigned - all_subjects
        raise SplitError(f"Subject mismatch. Missing: {missing}, Extra: {extra}")

    if not train:
        raise SplitError("Training split is empty.")
    if not val:
        raise SplitError("Validation split is empty.")
    if not test:
        raise SplitError("Test split is empty.")

    for split_name, subjects in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        split_stats = subject_stats[subject_stats["subject"].isin(subjects)]
        n_positive = int(split_stats["n_positive"].sum())
        if n_positive == 0:
            raise SplitError(f"Split '{split_name}' contains zero positive segments.")


def compute_split_summary(
    assignment: SplitAssignment,
    subject_stats: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute summary statistics for a split assignment.

    Parameters
    ----------
    assignment : SplitAssignment
        Split assignment.
    subject_stats : pd.DataFrame
        Per-subject statistics.

    Returns
    -------
    dict[str, Any]
        Summary including per-split subject counts, segment counts,
        and positive segment distribution.
    """
    total_positive = int(subject_stats["n_positive"].sum())
    total_segments = int(subject_stats["n_segments"].sum())

    summary: dict[str, Any] = {
        "total_subjects": len(subject_stats),
        "total_segments": total_segments,
        "total_positive": total_positive,
        "config": {
            "train_ratio": assignment.config.train_ratio,
            "val_ratio": assignment.config.val_ratio,
            "test_ratio": assignment.config.test_ratio,
        },
        "splits": {},
    }

    for split_name, subjects in [
        ("train", assignment.train_subjects),
        ("val", assignment.val_subjects),
        ("test", assignment.test_subjects),
    ]:
        split_stats = subject_stats[subject_stats["subject"].isin(set(subjects))]
        n_subjects = len(subjects)
        n_segments = int(split_stats["n_segments"].sum())
        n_positive = int(split_stats["n_positive"].sum())
        n_negative = int(split_stats["n_negative"].sum())

        summary["splits"][split_name] = {
            "n_subjects": n_subjects,
            "n_segments": n_segments,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "positive_ratio": round(n_positive / max(n_segments, 1), 6),
            "positive_share": round(n_positive / max(total_positive, 1), 4),
            "segment_share": round(n_segments / max(total_segments, 1), 4),
            "subjects": list(subjects),
        }

    return summary


def save_split(
    assignment: SplitAssignment,
    output_dir: str | Path,
    subject_stats: pd.DataFrame | None = None,
) -> None:
    """
    Save split assignment to disk as JSON files.

    Creates three per-split subject list files and, if subject statistics
    are provided, a combined summary file with distribution metrics.

    Parameters
    ----------
    assignment : SplitAssignment
        Split assignment to save.
    output_dir : str | Path
        Output directory.
    subject_stats : pd.DataFrame | None
        If provided, a summary with statistics is also saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, subjects in [
        ("train", assignment.train_subjects),
        ("val", assignment.val_subjects),
        ("test", assignment.test_subjects),
    ]:
        path = output_dir / f"{split_name}_subjects.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(list(subjects), f, indent=2, ensure_ascii=False)

    if subject_stats is not None:
        summary = compute_split_summary(assignment, subject_stats)
        summary_path = output_dir / "split_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


def load_split(split_dir: str | Path) -> SplitAssignment:
    """
    Load a split assignment from previously saved JSON files.

    Parameters
    ----------
    split_dir : str | Path
        Directory containing ``train_subjects.json``, ``val_subjects.json``
        and ``test_subjects.json``.

    Returns
    -------
    SplitAssignment
        Loaded split assignment.

    Raises
    ------
    SplitError
        If any required file is missing or malformed.
    """
    split_dir = Path(split_dir)

    subjects: dict[str, tuple[str, ...]] = {}

    for split_name in ("train", "val", "test"):
        path = split_dir / f"{split_name}_subjects.json"
        if not path.exists():
            raise SplitError(f"Missing split file: {path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise SplitError(f"Expected a JSON list in {path}")

        subjects[split_name] = tuple(sorted(str(s) for s in data))

    config = SplitConfig()

    summary_path = split_dir / "split_summary.json"
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        if "config" in summary:
            config = SplitConfig(
                train_ratio=summary["config"].get("train_ratio", 0.6),
                val_ratio=summary["config"].get("val_ratio", 0.2),
                test_ratio=summary["config"].get("test_ratio", 0.2),
            )

    return SplitAssignment(
        train_subjects=subjects["train"],
        val_subjects=subjects["val"],
        test_subjects=subjects["test"],
        config=config,
    )


def apply_split(
    df: pd.DataFrame,
    assignment: SplitAssignment,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter a dataset into train, validation and test subsets.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a ``subject`` column.
    assignment : SplitAssignment
        Subject-to-split mapping.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ``(train_df, val_df, test_df)``

    Raises
    ------
    SplitError
        If the dataset contains subjects not present in the assignment.
    """
    if "subject" not in df.columns:
        raise SplitError("Dataset must contain a 'subject' column.")

    df = df.copy()
    df["subject"] = df["subject"].astype(str)

    all_assigned = (
        set(assignment.train_subjects)
        | set(assignment.val_subjects)
        | set(assignment.test_subjects)
    )
    dataset_subjects = set(df["subject"].unique())
    unassigned = dataset_subjects - all_assigned

    if unassigned:
        raise SplitError(
            f"Dataset contains subjects not present in the split: {sorted(unassigned)}"
        )

    train_df = df[df["subject"].isin(set(assignment.train_subjects))].copy()
    val_df = df[df["subject"].isin(set(assignment.val_subjects))].copy()
    test_df = df[df["subject"].isin(set(assignment.test_subjects))].copy()

    return train_df, val_df, test_df
