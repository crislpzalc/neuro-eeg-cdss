import pandas as pd
import pytest

from neuro_eeg_cdss.data.splits import (
    SplitAssignment,
    SplitConfig,
    SplitError,
    apply_split,
    compute_subject_stats,
    create_split,
    load_split,
    save_split,
)


def _make_dataset(
    subject_positive_counts: dict[str, int],
    negatives_per_subject: int = 100,
) -> pd.DataFrame:
    """Create a minimal dataset for testing."""
    rows = []
    for subject, n_positive in subject_positive_counts.items():
        for _ in range(n_positive):
            rows.append({"subject": subject, "label": 1})
        for _ in range(negatives_per_subject):
            rows.append({"subject": subject, "label": 0})
    return pd.DataFrame(rows)


# --- compute_subject_stats ---


def test_subject_stats_basic():
    df = _make_dataset({"A": 10, "B": 5, "C": 20})
    stats = compute_subject_stats(df)

    assert len(stats) == 3
    assert set(stats.columns) >= {"subject", "n_segments", "n_positive", "n_negative"}
    assert stats.iloc[0]["subject"] == "C"


def test_subject_stats_missing_columns():
    df = pd.DataFrame({"subject": ["A"], "value": [1]})

    with pytest.raises(SplitError):
        compute_subject_stats(df)


# --- create_split ---


def test_split_no_overlap():
    df = _make_dataset({f"sub-{i:02d}": (i + 1) * 5 for i in range(10)})
    assignment = create_split(df)

    train = set(assignment.train_subjects)
    val = set(assignment.val_subjects)
    test = set(assignment.test_subjects)

    assert train & val == set()
    assert train & test == set()
    assert val & test == set()


def test_split_all_subjects_assigned():
    subjects = {f"sub-{i:02d}": (i + 1) * 5 for i in range(10)}
    df = _make_dataset(subjects)
    assignment = create_split(df)

    all_assigned = (
        set(assignment.train_subjects)
        | set(assignment.val_subjects)
        | set(assignment.test_subjects)
    )
    assert all_assigned == set(subjects.keys())


def test_split_all_splits_non_empty():
    df = _make_dataset({f"sub-{i:02d}": (i + 1) * 5 for i in range(10)})
    assignment = create_split(df)

    assert len(assignment.train_subjects) > 0
    assert len(assignment.val_subjects) > 0
    assert len(assignment.test_subjects) > 0


def test_split_each_split_has_positives():
    df = _make_dataset({f"sub-{i:02d}": (i + 1) * 5 for i in range(10)})
    assignment = create_split(df)

    stats = compute_subject_stats(df)
    for split_name, subjects in [
        ("train", assignment.train_subjects),
        ("val", assignment.val_subjects),
        ("test", assignment.test_subjects),
    ]:
        split_stats = stats[stats["subject"].isin(set(subjects))]
        assert split_stats["n_positive"].sum() > 0, f"{split_name} has no positives"


def test_split_deterministic():
    df = _make_dataset({f"sub-{i:02d}": (i + 1) * 5 for i in range(10)})
    a1 = create_split(df)
    a2 = create_split(df)

    assert a1.train_subjects == a2.train_subjects
    assert a1.val_subjects == a2.val_subjects
    assert a1.test_subjects == a2.test_subjects


def test_split_too_few_subjects():
    df = _make_dataset({"A": 10, "B": 5})

    with pytest.raises(SplitError):
        create_split(df)


def test_split_custom_config():
    df = _make_dataset({f"sub-{i:02d}": (i + 1) * 5 for i in range(10)})
    config = SplitConfig(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assignment = create_split(df, config=config)

    assert len(assignment.train_subjects) > 0
    assert len(assignment.val_subjects) > 0
    assert len(assignment.test_subjects) > 0


def test_split_invalid_config_sum():
    df = _make_dataset({f"sub-{i:02d}": 10 for i in range(5)})
    config = SplitConfig(train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)

    with pytest.raises(SplitError):
        create_split(df, config=config)


def test_split_positive_distribution_is_proportional():
    df = _make_dataset({f"sub-{i:02d}": (i + 1) * 10 for i in range(20)})
    assignment = create_split(df)

    stats = compute_subject_stats(df)
    total_positive = int(stats["n_positive"].sum())

    for split_name, subjects, target_ratio in [
        ("train", assignment.train_subjects, 0.6),
        ("val", assignment.val_subjects, 0.2),
        ("test", assignment.test_subjects, 0.2),
    ]:
        split_stats = stats[stats["subject"].isin(set(subjects))]
        actual_share = int(split_stats["n_positive"].sum()) / total_positive
        assert abs(actual_share - target_ratio) < 0.15, (
            f"{split_name}: expected ~{target_ratio:.0%}, got {actual_share:.2%}"
        )


# --- save / load round-trip ---


def test_save_and_load_round_trip(tmp_path):
    df = _make_dataset({f"sub-{i:02d}": (i + 1) * 5 for i in range(10)})
    assignment = create_split(df)
    stats = compute_subject_stats(df)

    save_split(assignment, tmp_path, subject_stats=stats)
    loaded = load_split(tmp_path)

    assert loaded.train_subjects == assignment.train_subjects
    assert loaded.val_subjects == assignment.val_subjects
    assert loaded.test_subjects == assignment.test_subjects


def test_load_split_missing_file(tmp_path):
    with pytest.raises(SplitError):
        load_split(tmp_path)


# --- apply_split ---


def test_apply_split_basic():
    df = _make_dataset({f"sub-{i:02d}": 10 for i in range(6)})
    assignment = create_split(df)
    train_df, val_df, test_df = apply_split(df, assignment)

    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    train_subjects = set(train_df["subject"].unique())
    val_subjects = set(val_df["subject"].unique())
    test_subjects = set(test_df["subject"].unique())

    assert train_subjects & val_subjects == set()
    assert train_subjects & test_subjects == set()
    assert val_subjects & test_subjects == set()


def test_apply_split_unknown_subject():
    assignment = SplitAssignment(
        train_subjects=("A", "B"),
        val_subjects=("C",),
        test_subjects=("D",),
        config=SplitConfig(),
    )
    df = pd.DataFrame({"subject": ["A", "B", "C", "D", "E"], "label": [0, 0, 0, 0, 0]})

    with pytest.raises(SplitError):
        apply_split(df, assignment)
