import pytest

from neuro_eeg_cdss.preprocessing.labeling import (
    LabelingError,
    assign_label,
)


def test_assign_label_positive():
    decision = assign_label(overlap_ratio=0.7)

    assert decision.label == 1
    assert decision.keep is True
    assert decision.reason == "positive_overlap_threshold_reached"


def test_assign_label_negative():
    decision = assign_label(overlap_ratio=0.0)

    assert decision.label == 0
    assert decision.keep is True
    assert decision.reason == "no_overlap"


def test_assign_label_partial_overlap_dropped():
    decision = assign_label(overlap_ratio=0.2)

    assert decision.label is None
    assert decision.keep is False
    assert decision.reason == "partial_overlap_dropped"


def test_assign_label_partial_overlap_as_negative_when_configured():
    decision = assign_label(
        overlap_ratio=0.2,
        drop_partial_overlap=False,
    )

    assert decision.label == 0
    assert decision.keep is True
    assert decision.reason == "partial_overlap_assigned_negative"


def test_assign_label_threshold_boundary_is_positive():
    decision = assign_label(overlap_ratio=0.5)

    assert decision.label == 1
    assert decision.keep is True


def test_assign_label_raises_for_invalid_overlap_ratio():
    with pytest.raises(LabelingError):
        assign_label(overlap_ratio=1.2)


def test_assign_label_raises_for_invalid_threshold():
    with pytest.raises(LabelingError):
        assign_label(overlap_ratio=0.0, positive_overlap_threshold=0.0)
