"""
Run all labeling strategy experiments for Sprint 1F.

This script compares 6 labeling configurations (3 thresholds x 2 drop
policies) to study how the positive overlap threshold and partial-overlap
handling affect seizure detection performance.

Usage:
    python scripts/experiments/run_labeling_experiments.py
    python scripts/experiments/run_labeling_experiments.py --output-dir experiments/labeling

Requirements:
    - data/processed/features.parquet   (from Sprint 1B)
    - data/processed/segments.parquet   (from Sprint 1A)
    - data/splits/*.json                (from Sprint 1C)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuro_eeg_cdss.experiments.labeling import (
    ALL_CONFIGS,
    format_comparison_table,
    run_all_experiments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run labeling strategy experiments.")
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Path to features.parquet.",
    )
    parser.add_argument(
        "--segments-path",
        type=Path,
        default=Path("data/processed/segments.parquet"),
        help="Path to segments.parquet.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing split JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/labeling"),
        help="Directory for experiment outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Sprint 1F — Labeling Strategy Experiments")
    print("=" * 70)
    print(f"  Features: {args.features_path}")
    print(f"  Segments: {args.segments_path}")
    print(f"  Splits:   {args.splits_dir}")
    print(f"  Output:   {args.output_dir}")
    print(f"  Configs:  {len(ALL_CONFIGS)}")

    # ── Run experiments ──────────────────────────────────────────────

    results = run_all_experiments(
        features_path=args.features_path,
        segments_path=args.segments_path,
        splits_dir=args.splits_dir,
    )

    # ── Save per-config results ──────────────────────────────────────

    all_results = {}
    for r in results:
        result_dict = r.to_dict()
        all_results[r.config.name] = result_dict

        config_dir = args.output_dir / r.config.name
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

    # ── Save combined results ────────────────────────────────────────

    with open(args.output_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Print and save comparison tables ─────────────────────────────

    for split_name in ["val", "test"]:
        table = format_comparison_table(results, split_name=split_name)
        print(f"\n{table}")

        with open(
            args.output_dir / f"comparison_{split_name}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(table)

    # ── Analysis notes ───────────────────────────────────────────────

    print(f"\n{'=' * 70}")
    print("  KEY OBSERVATIONS")
    print(f"{'=' * 70}")

    # Check if configs 1-4 produced identical results
    test_aurocs = [r.metrics_by_split["test"].auroc for r in results]
    first_four_identical = len(set(test_aurocs[:4])) == 1
    if first_four_identical:
        print("  - Configs with threshold 0.3 and 0.5 produce IDENTICAL results.")
        print("    This is expected: no windows with 0 < overlap < 0.5 exist in the dataset.")
        print("    (They were dropped during the original build with threshold=0.5, drop=True.)")

    # Check threshold=0.7 impact
    if len(results) == 6:
        baseline_sens = results[2].metrics_by_split["test"].sensitivity
        t07_drop_sens = results[4].metrics_by_split["test"].sensitivity
        t07_keep_sens = results[5].metrics_by_split["test"].sensitivity
        print("\n  - Threshold 0.7 vs baseline (0.5):")
        print(f"    Baseline sensitivity:  {baseline_sens:.4f}")
        print(f"    Thresh 0.7 (drop):     {t07_drop_sens:.4f}")
        print(f"    Thresh 0.7 (keep):     {t07_keep_sens:.4f}")

    print(f"\n  All results saved to: {args.output_dir.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
