from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DatasetInspectionError(ValueError):
    """Raised when the dataset is not valid for split inspection."""


# Minimum columns needed to reason about subject-level splitting.
REQUIRED_COLUMNS = {"subject", "label"}

# Possible column names that may identify recordings/files/sessions.
# We try to infer one automatically because different pipelines may use different names.
OPTIONAL_RECORDING_COLUMNS = [
    "recording_id",
    "recording",
    "file_path",
    "edf_path",
    "signal_path",
    "run",
    "session",
    "filename",
]


@dataclass(frozen=True)
class InspectionConfig:
    """Configuration for the inspection script."""

    input_path: Path
    output_dir: Path
    top_k: int = 10
    positive_label: int = 1


def parse_args() -> InspectionConfig:
    """Parse command-line arguments and return a typed configuration object."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a segmented EEG dataset at the global and subject level "
            "to support methodological decisions before implementing splits.py."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the segmented dataset in .parquet or .csv format.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/inspection_subject_distribution"),
        help="Directory where inspection outputs will be saved.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top subjects to display in rankings.",
    )
    parser.add_argument(
        "--positive-label",
        type=int,
        default=1,
        help="Value used for the positive class label (default: 1).",
    )

    args = parser.parse_args()

    return InspectionConfig(
        input_path=args.input,
        output_dir=args.output_dir,
        top_k=args.top_k,
        positive_label=args.positive_label,
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load the dataset from disk.

    We support parquet and csv because parquet is common in ML pipelines,
    but csv is still convenient in early-stage experiments.
    """
    if not path.exists():
        raise DatasetInspectionError(f"Input file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise DatasetInspectionError(f"Unsupported file format: {suffix}. Use .parquet or .csv.")

    if df.empty:
        raise DatasetInspectionError("The input dataset is empty.")

    return df


def validate_dataset(df: pd.DataFrame, positive_label: int) -> pd.DataFrame:
    """
    Validate the minimum structure required for subject-level inspection.

    This step is important because a split analysis is only meaningful if
    subject identifiers and labels are clean and consistent.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise DatasetInspectionError(f"Missing required columns: {sorted(missing)}")

    validated = df.copy()

    # Normalize subject identifiers to strings and strip whitespace.
    validated["subject"] = validated["subject"].astype(str).str.strip()

    if validated["subject"].isna().any():
        raise DatasetInspectionError("Column 'subject' contains null values.")

    if (validated["subject"] == "").any():
        raise DatasetInspectionError("Column 'subject' contains empty identifiers.")

    if validated["label"].isna().any():
        raise DatasetInspectionError("Column 'label' contains null values.")

    unique_labels = set(pd.unique(validated["label"]))
    if not unique_labels.issubset({0, 1, positive_label}):
        raise DatasetInspectionError(
            f"Column 'label' contains unexpected values. Found values: {sorted(unique_labels)}"
        )

    # Convert labels to a clean binary representation.
    validated["label"] = (validated["label"] == positive_label).astype(int)

    return validated


def infer_recording_column(df: pd.DataFrame) -> str | None:
    """
    Try to identify a recording-related column automatically.

    This is useful because the number of recordings per subject can help us
    understand how heterogeneous each subject contribution is.
    """
    for col in OPTIONAL_RECORDING_COLUMNS:
        if col in df.columns:
            return col
    return None


def build_subject_summary(df: pd.DataFrame, recording_col: str | None) -> pd.DataFrame:
    """
    Build the key subject-level summary table.

    Each row corresponds to one subject, which is the correct unit for
    patient-independent splitting decisions.
    """
    grouped = df.groupby("subject", dropna=False)

    summary = (
        grouped["label"]
        .agg(
            n_segments="size",
            n_positive="sum",
        )
        .reset_index()
    )

    summary["n_negative"] = summary["n_segments"] - summary["n_positive"]
    summary["positive_ratio"] = summary["n_positive"] / summary["n_segments"]

    # If we found a recording column, count how many unique recordings each subject has.
    if recording_col is not None:
        recordings = (
            df.groupby("subject", dropna=False)[recording_col]
            .nunique(dropna=True)
            .reset_index(name="n_recordings")
        )
        summary = summary.merge(recordings, on="subject", how="left")
    else:
        summary["n_recordings"] = np.nan

    summary["has_positive"] = summary["n_positive"] > 0

    total_positives = max(int(summary["n_positive"].sum()), 1)
    total_segments = int(summary["n_segments"].sum())

    # Contribution shares help detect whether a few subjects dominate the dataset.
    summary["positive_share_overall"] = summary["n_positive"] / total_positives
    summary["segment_share_overall"] = summary["n_segments"] / total_segments

    summary = summary.sort_values(
        by=["n_positive", "n_segments", "subject"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return summary


def compute_global_summary(
    df: pd.DataFrame,
    subject_summary: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute global dataset statistics.

    These metrics provide the first high-level view before looking at
    the subject-by-subject breakdown.
    """
    total_segments = int(len(df))
    total_positive = int(df["label"].sum())
    total_negative = int(total_segments - total_positive)
    total_subjects = int(subject_summary["subject"].nunique())
    subjects_with_positive = int(subject_summary["has_positive"].sum())
    subjects_without_positive = int(total_subjects - subjects_with_positive)

    return {
        "total_segments": total_segments,
        "total_positive": total_positive,
        "total_negative": total_negative,
        "global_positive_ratio": (total_positive / total_segments if total_segments > 0 else 0.0),
        "total_subjects": total_subjects,
        "subjects_with_positive": subjects_with_positive,
        "subjects_without_positive": subjects_without_positive,
        "subject_positive_ratio": (
            subjects_with_positive / total_subjects if total_subjects > 0 else 0.0
        ),
    }


def compute_distribution_stats(subject_summary: pd.DataFrame) -> dict[str, Any]:
    """
    Compute descriptive statistics across subjects.

    This helps detect imbalance, outliers, and whether the dataset is
    reasonably homogeneous or dominated by a few subjects.
    """

    def stats(series: pd.Series) -> dict[str, float]:
        return {
            "min": float(series.min()),
            "p25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "p75": float(series.quantile(0.75)),
            "max": float(series.max()),
            "std": float(series.std(ddof=0)),
        }

    return {
        "n_segments_per_subject": stats(subject_summary["n_segments"]),
        "n_positive_per_subject": stats(subject_summary["n_positive"]),
        "n_negative_per_subject": stats(subject_summary["n_negative"]),
        "positive_ratio_per_subject": stats(subject_summary["positive_ratio"]),
    }


def compute_concentration_flags(subject_summary: pd.DataFrame) -> dict[str, Any]:
    """
    Measure how concentrated positives and total segments are.

    These concentration metrics are very important because a patient-independent
    split may still be weak if almost all positives come from a few subjects.
    """
    positive_sorted = subject_summary.sort_values(
        by="n_positive",
        ascending=False,
    ).reset_index(drop=True)

    segment_sorted = subject_summary.sort_values(
        by="n_segments",
        ascending=False,
    ).reset_index(drop=True)

    top_1_positive_share = float(positive_sorted["positive_share_overall"].head(1).sum())
    top_3_positive_share = float(positive_sorted["positive_share_overall"].head(3).sum())
    top_5_positive_share = float(positive_sorted["positive_share_overall"].head(5).sum())

    top_1_segment_share = float(segment_sorted["segment_share_overall"].head(1).sum())
    top_3_segment_share = float(segment_sorted["segment_share_overall"].head(3).sum())
    top_5_segment_share = float(segment_sorted["segment_share_overall"].head(5).sum())

    return {
        "top_1_positive_share": top_1_positive_share,
        "top_3_positive_share": top_3_positive_share,
        "top_5_positive_share": top_5_positive_share,
        "top_1_segment_share": top_1_segment_share,
        "top_3_segment_share": top_3_segment_share,
        "top_5_segment_share": top_5_segment_share,
    }


def recommend_strategy(
    global_summary: dict[str, Any],
    concentration: dict[str, Any],
    subject_summary: pd.DataFrame,
) -> str:
    """
    Produce a simple recommendation based on the inspected structure.

    This is not meant to replace judgment, but to summarize the main
    methodological implication of the observed dataset shape.
    """
    total_subjects = global_summary["total_subjects"]
    positive_subjects = global_summary["subjects_with_positive"]

    if total_subjects < 10:
        return (
            "Very few subjects: start with a subject-level hold-out split to move "
            "forward, but strongly consider subject-level cross-validation in the final evaluation."
        )

    if positive_subjects < 4:
        return (
            "Very few positive subjects: do not use a purely random subject split. "
            "Assign positive subjects in a controlled way to guarantee positives in validation and test."
        )

    if concentration["top_3_positive_share"] >= 0.75:
        return (
            "Positives are highly concentrated in a few subjects: use a subject-level "
            "hold-out split with controlled assignment of positive subjects before filling with negatives."
        )

    if concentration["top_3_segment_share"] >= 0.50:
        return (
            "Total segment volume is highly concentrated: use a subject-level hold-out "
            "split while also monitoring the total size of each split."
        )

    return (
        "The dataset seems compatible with a reproducible subject-level hold-out split. "
        "Still, explicitly verify that validation and test contain enough positive subjects."
    )


def build_risk_flags(
    global_summary: dict[str, Any],
    subject_summary: pd.DataFrame,
    concentration: dict[str, Any],
) -> dict[str, Any]:
    """
    Create methodological warning flags.

    These flags help translate raw dataset statistics into practical concerns
    for the future implementation of splits.py.
    """
    flags: dict[str, Any] = {}

    total_subjects = global_summary["total_subjects"]
    subjects_with_positive = global_summary["subjects_with_positive"]

    flags["few_total_subjects"] = total_subjects < 15
    flags["very_few_total_subjects"] = total_subjects < 10

    flags["few_positive_subjects"] = subjects_with_positive < 6
    flags["very_few_positive_subjects"] = subjects_with_positive < 4

    flags["many_subjects_without_positive"] = global_summary["subject_positive_ratio"] < 0.5

    flags["positives_highly_concentrated_top1"] = concentration["top_1_positive_share"] >= 0.40
    flags["positives_highly_concentrated_top3"] = concentration["top_3_positive_share"] >= 0.75

    flags["segments_highly_concentrated_top1"] = concentration["top_1_segment_share"] >= 0.25
    flags["segments_highly_concentrated_top3"] = concentration["top_3_segment_share"] >= 0.50

    small_subjects = subject_summary["n_segments"] < 50
    flags["has_many_very_small_subjects"] = float(small_subjects.mean()) >= 0.20

    zero_positive_subjects = subject_summary["n_positive"] == 0
    flags["has_zero_positive_subjects"] = bool(zero_positive_subjects.any())

    flags["recommended_strategy"] = recommend_strategy(
        global_summary=global_summary,
        concentration=concentration,
        subject_summary=subject_summary,
    )

    return flags


def create_top_tables(
    subject_summary: pd.DataFrame,
    top_k: int,
) -> dict[str, pd.DataFrame]:
    """
    Create ranked views of the most relevant subject subsets.

    These tables are useful for quickly spotting dominant subjects and
    for documenting the inspection results.
    """
    top_by_positive = (
        subject_summary.sort_values(
            by=["n_positive", "n_segments", "subject"],
            ascending=[False, False, True],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    top_by_segments = (
        subject_summary.sort_values(
            by=["n_segments", "n_positive", "subject"],
            ascending=[False, False, True],
        )
        .head(top_k)
        .reset_index(drop=True)
    )

    zero_positive_subjects = (
        subject_summary.loc[subject_summary["n_positive"] == 0]
        .sort_values(by=["n_segments", "subject"], ascending=[False, True])
        .reset_index(drop=True)
    )

    return {
        "top_by_positive": top_by_positive,
        "top_by_segments": top_by_segments,
        "zero_positive_subjects": zero_positive_subjects,
    }


def format_table_for_markdown(df: pd.DataFrame, n_rows: int = 10) -> str:
    """
    Convert a dataframe preview into a markdown table.

    The report only needs a preview, not the full table, because the full
    CSV is already saved separately.
    """
    if df.empty:
        return "_No data_\n"

    preview = df.head(n_rows).copy()

    float_cols = preview.select_dtypes(include=["float64", "float32"]).columns
    for col in float_cols:
        preview[col] = preview[col].map(lambda x: round(float(x), 4))

    return preview.to_markdown(index=False) + "\n"


def build_markdown_report(
    metadata: dict[str, Any],
    global_summary: dict[str, Any],
    distribution_stats: dict[str, Any],
    concentration: dict[str, Any],
    risk_flags: dict[str, Any],
    subject_summary: pd.DataFrame,
    top_tables: dict[str, pd.DataFrame],
) -> str:
    """
    Build a human-readable markdown report.

    This report is useful for sprint documentation and for justifying the
    design choices that will later appear in splits.py.
    """
    lines: list[str] = []

    lines.append("# Pre-split subject-level inspection\n")
    lines.append("## 1. Metadata\n")
    lines.append(f"- Dataset: `{metadata['input_path']}`")
    lines.append(f"- Number of rows: {metadata['n_rows_dataset']}")
    lines.append(f"- Columns: {', '.join(metadata['columns'])}")
    lines.append(f"- Recording column used: {metadata['recording_column_used']}\n")

    lines.append("## 2. Global summary\n")
    for key, value in global_summary.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## 3. Subject-level distribution statistics\n")
    for block_name, stats in distribution_stats.items():
        lines.append(f"### {block_name}\n")
        for key, value in stats.items():
            lines.append(f"- **{key}**: {round(value, 6)}")
        lines.append("")

    lines.append("## 4. Dataset concentration\n")
    for key, value in concentration.items():
        lines.append(f"- **{key}**: {round(value, 6)}")
    lines.append("")

    lines.append("## 5. Methodological risk flags\n")
    for key, value in risk_flags.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")

    lines.append("## 6. Top subjects by positive segments\n")
    lines.append(format_table_for_markdown(top_tables["top_by_positive"]))

    lines.append("## 7. Top subjects by total segments\n")
    lines.append(format_table_for_markdown(top_tables["top_by_segments"]))

    lines.append("## 8. Subjects with zero positive segments\n")
    lines.append(format_table_for_markdown(top_tables["zero_positive_subjects"]))

    lines.append("## 9. Subject summary preview\n")
    lines.append(format_table_for_markdown(subject_summary, n_rows=20))

    return "\n".join(lines)


def save_outputs(
    config: InspectionConfig,
    dataset_df: pd.DataFrame,
    subject_summary: pd.DataFrame,
    global_summary: dict[str, Any],
    distribution_stats: dict[str, Any],
    concentration: dict[str, Any],
    risk_flags: dict[str, Any],
    top_tables: dict[str, pd.DataFrame],
    recording_col: str | None,
) -> None:
    """
    Save all inspection outputs to disk.

    Persisting the outputs is better than relying only on terminal prints,
    because these artifacts can later support documentation and reproducibility.
    """
    config.output_dir.mkdir(parents=True, exist_ok=True)

    subject_summary.to_csv(config.output_dir / "subject_summary.csv", index=False)

    for name, table in top_tables.items():
        table.to_csv(config.output_dir / f"{name}.csv", index=False)

    metadata = {
        "input_path": str(config.input_path),
        "n_rows_dataset": int(len(dataset_df)),
        "columns": list(dataset_df.columns),
        "recording_column_used": recording_col,
    }

    payload = {
        "metadata": metadata,
        "global_summary": global_summary,
        "distribution_stats": distribution_stats,
        "concentration": concentration,
        "risk_flags": risk_flags,
    }

    with open(config.output_dir / "inspection_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    report_md = build_markdown_report(
        metadata=metadata,
        global_summary=global_summary,
        distribution_stats=distribution_stats,
        concentration=concentration,
        risk_flags=risk_flags,
        subject_summary=subject_summary,
        top_tables=top_tables,
    )

    with open(config.output_dir / "inspection_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)


def print_console_summary(
    global_summary: dict[str, Any],
    concentration: dict[str, Any],
    risk_flags: dict[str, Any],
    top_tables: dict[str, pd.DataFrame],
    top_k: int,
    output_dir: Path,
) -> None:
    """
    Print a concise console summary so the user can immediately inspect the results.
    """
    print("\n" + "=" * 80)
    print("PRE-SPLIT SUBJECT-LEVEL INSPECTION")
    print("=" * 80)

    print("\n[Global summary]")
    for key, value in global_summary.items():
        print(f"- {key}: {value}")

    print("\n[Concentration]")
    for key, value in concentration.items():
        print(f"- {key}: {round(value, 6)}")

    print("\n[Methodological risk flags]")
    for key, value in risk_flags.items():
        print(f"- {key}: {value}")

    print(f"\n[Top {top_k} subjects by positive segments]")
    print(top_tables["top_by_positive"].head(top_k).to_string(index=False))

    print(f"\n[Top {top_k} subjects by total segments]")
    print(top_tables["top_by_segments"].head(top_k).to_string(index=False))

    print(f"\nResults saved to: {output_dir.resolve()}\n")


def main() -> None:
    """
    Main execution flow.

    The order matters:
    1. load the dataset
    2. validate structure
    3. summarize by subject
    4. compute global and methodological indicators
    5. save everything for later analysis
    """
    config = parse_args()

    dataset_df = load_dataset(config.input_path)
    dataset_df = validate_dataset(dataset_df, positive_label=config.positive_label)

    recording_col = infer_recording_column(dataset_df)

    subject_summary = build_subject_summary(
        dataset_df,
        recording_col=recording_col,
    )
    global_summary = compute_global_summary(dataset_df, subject_summary)
    distribution_stats = compute_distribution_stats(subject_summary)
    concentration = compute_concentration_flags(subject_summary)
    risk_flags = build_risk_flags(
        global_summary=global_summary,
        subject_summary=subject_summary,
        concentration=concentration,
    )
    top_tables = create_top_tables(subject_summary, top_k=config.top_k)

    save_outputs(
        config=config,
        dataset_df=dataset_df,
        subject_summary=subject_summary,
        global_summary=global_summary,
        distribution_stats=distribution_stats,
        concentration=concentration,
        risk_flags=risk_flags,
        top_tables=top_tables,
        recording_col=recording_col,
    )

    print_console_summary(
        global_summary=global_summary,
        concentration=concentration,
        risk_flags=risk_flags,
        top_tables=top_tables,
        top_k=config.top_k,
        output_dir=config.output_dir,
    )


if __name__ == "__main__":
    main()
