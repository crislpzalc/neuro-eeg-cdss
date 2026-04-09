"""
Utilities for assigning segment-level labels based on seizure overlap.

This module implements a deterministic labeling policy that converts the
fraction of seizure overlap within a fixed-length window into a binary label
or a discard decision.

Design goals
------------
- Make labeling rules explicit and reproducible
- Prevent ambiguous window assignments from silently entering the dataset
- Keep the policy simple enough to support baseline modeling and ablation
"""

from __future__ import annotations

from dataclasses import dataclass


class LabelingError(ValueError):
    """Raised when labeling parameters or overlap values are invalid."""


@dataclass(frozen=True)
class LabelingDecision:
    """
    Result of labeling a single window.

    Attributes
    ----------
    label : int | None
        Assigned label:
        - 1 for seizure
        - 0 for non-seizure
        - None if the window should be discarded
    keep : bool
        Whether the window is retained in the final dataset.
    reason : str
        Reason describing the labeling decision.

    Notes
    -----
    This object separates the semantic label from the dataset retention
    decision. This makes the labeling policy explicit and easier to audit in
    downstream dataset construction.
    """

    label: int | None
    keep: bool
    reason: str


def _validate_overlap_ratio(overlap_ratio: float) -> None:
    """
    Validate that overlap_ratio lies within the closed interval [0, 1].
    """
    if overlap_ratio < 0.0 or overlap_ratio > 1.0:
        raise LabelingError(
            f"'overlap_ratio' must be between 0 and 1. Received value: {overlap_ratio}"
        )


def assign_label(
    overlap_ratio: float,
    positive_overlap_threshold: float = 0.5,
    drop_partial_overlap: bool = True,
) -> LabelingDecision:
    """
    Assign a label to a window based on its seizure overlap ratio.

    Parameters
    ----------
    overlap_ratio : float
        Fraction of the window that overlaps with seizure activity.
    positive_overlap_threshold : float, default=0.5
        Threshold above which a window is considered positive.
    drop_partial_overlap : bool, default=True
        If True, windows with partial overlap below the threshold are dropped.
        If False, those windows are assigned as negative.

    Returns
    -------
    LabelingDecision
        Labeling decision for the window.

    Raises
    ------
    LabelingError
        If the input parameters are invalid.

    Notes
    -----
    The labeling rule implemented here is intentionally simple and designed for
    fixed-window baseline experiments. More refined policies, such as
    multi-class labels or pre-ictal handling, can be introduced in later
    stages without changing the basic interface.
    """
    _validate_overlap_ratio(overlap_ratio)

    if positive_overlap_threshold <= 0.0 or positive_overlap_threshold > 1.0:
        raise LabelingError("'positive_overlap_threshold' must lie in the interval (0, 1].")

    # Windows whose seizure overlap meets or exceeds the configured threshold
    # are treated as positive examples.
    if overlap_ratio >= positive_overlap_threshold:
        return LabelingDecision(
            label=1,
            keep=True,
            reason="positive_overlap_threshold_reached",
        )

    # Windows with no seizure overlap are unambiguously negative.
    if overlap_ratio == 0.0:
        return LabelingDecision(
            label=0,
            keep=True,
            reason="no_overlap",
        )

    # Partial-overlap windows below the positive threshold are the main source
    # of ambiguity in binary fixed-window labeling.
    if drop_partial_overlap:
        return LabelingDecision(
            label=None,
            keep=False,
            reason="partial_overlap_dropped",
        )

    # If ambiguous windows are not discarded, they are explicitly assigned to
    # the negative class according to the configured policy.
    return LabelingDecision(
        label=0,
        keep=True,
        reason="partial_overlap_assigned_negative",
    )
