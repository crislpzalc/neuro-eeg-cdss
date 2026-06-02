"""
Create a patient-independent train/validation/test split.

Usage:
    python scripts/splits/create_splits.py
    python scripts/splits/create_splits.py --input data/processed/segments.parquet
    python scripts/splits/create_splits.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15

This script:
1. Loads the segments dataset
2. Creates a stratified subject-level split
3. Validates the split integrity
4. Saves JSON files for reproducible use in downstream pipelines
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from neuro_eeg_cdss.data.splits import (
    SplitConfig,
    compute_split_summary,
    compute_subject_stats,
    create_split,
    save_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a patient-independent train/validation/test split."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/segments.parquet"),
        help="Path to the segments or features dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory where split files will be saved.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading dataset from: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Rows: {len(df):,}")
    print(f"  Subjects: {df['subject'].nunique()}")

    config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    print(
        f"\nCreating split with target ratios: "
        f"{config.train_ratio}/{config.val_ratio}/{config.test_ratio}"
    )
    assignment = create_split(df, config=config)

    subject_stats = compute_subject_stats(df)
    summary = compute_split_summary(assignment, subject_stats)

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    for split_name in ("train", "val", "test"):
        s = summary["splits"][split_name]
        print(f"\n  [{split_name.upper()}]")
        print(f"    Subjects:       {s['n_subjects']}")
        print(f"    Segments:       {s['n_segments']:,}")
        print(f"    Positive:       {s['n_positive']:,} ({s['positive_share']:.1%} of total)")
        print(f"    Negative:       {s['n_negative']:,}")
        print(f"    Positive ratio: {s['positive_ratio']:.4%}")
        print(f"    Subject list:   {', '.join(s['subjects'])}")

    save_split(assignment, args.output_dir, subject_stats=subject_stats)
    print(f"\nSplit saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
