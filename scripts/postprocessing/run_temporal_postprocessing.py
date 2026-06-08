"""
Run temporal post-processing experiments on baseline predictions.

This script loads saved predictions, enriches them with temporal metadata,
applies multiple post-processing strategies, and compares metrics before
and after. Results are saved for documentation and further analysis.

Usage
-----
    python scripts/postprocessing/run_temporal_postprocessing.py

Requirements
------------
- Trained baseline predictions in experiments/baseline/logistic_regression/
- Feature data in data/processed/features.parquet
- Split assignments in data/processed/splits/
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neuro_eeg_cdss.data.splits import apply_split, load_split
from neuro_eeg_cdss.evaluation.metrics import compute_window_metrics
from neuro_eeg_cdss.postprocessing.temporal import (
    TemporalConfig,
    compute_postprocessing_summary,
    enrich_predictions,
    postprocess_predictions,
)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PREDICTIONS_DIR = PROJECT_ROOT / "experiments" / "baseline" / "logistic_regression"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "postprocessing"

# ── Configurations to test ───────────────────────────────────────────

CONFIGS = [
    # Median filters with different kernel sizes
    TemporalConfig(strategy="median_filter", kernel_size=3, threshold=0.5),
    TemporalConfig(strategy="median_filter", kernel_size=5, threshold=0.5),
    TemporalConfig(strategy="median_filter", kernel_size=7, threshold=0.5),
    # Moving averages
    TemporalConfig(strategy="moving_average", kernel_size=3, threshold=0.5),
    TemporalConfig(strategy="moving_average", kernel_size=5, threshold=0.5),
    # Minimum duration filters
    TemporalConfig(strategy="min_duration", min_windows=2),
    TemporalConfig(strategy="min_duration", min_windows=3),
    TemporalConfig(strategy="min_duration", min_windows=4),
]


def load_and_enrich(split_name: str, features_df: pd.DataFrame) -> pd.DataFrame:
    """Load predictions for a split and enrich with temporal metadata."""
    pred_path = PREDICTIONS_DIR / f"predictions_{split_name}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")

    predictions_df = pd.read_parquet(pred_path)
    return enrich_predictions(predictions_df, features_df)


def evaluate_config(
    enriched_df: pd.DataFrame,
    config: TemporalConfig,
) -> dict:
    """Apply post-processing and compute before/after metrics."""
    result_df = postprocess_predictions(enriched_df, config)

    # Before: original predictions
    y_true = result_df["y_true"].values
    y_pred_before = result_df["y_pred"].values
    y_proba_before = result_df["y_proba"].values

    # After: post-processed predictions
    y_pred_after = result_df["y_pred_post"].values
    y_proba_after = result_df["y_proba_post"].values

    metrics_before = compute_window_metrics(y_true, y_pred_before, y_proba_before)
    metrics_after = compute_window_metrics(y_true, y_pred_after, y_proba_after)

    summary = compute_postprocessing_summary(result_df, config)

    return {
        "config": summary["config"],
        "changes": {
            "n_predictions_changed": summary["n_predictions_changed"],
            "change_rate": summary["change_rate"],
            "positive_to_negative": summary["positive_to_negative"],
            "negative_to_positive": summary["negative_to_positive"],
        },
        "before": metrics_before.to_dict(),
        "after": metrics_after.to_dict(),
        "deltas": {
            "sensitivity": metrics_after.sensitivity - metrics_before.sensitivity,
            "specificity": metrics_after.specificity - metrics_before.specificity,
            "f2": metrics_after.f2 - metrics_before.f2,
            "precision": metrics_after.precision - metrics_before.precision,
            "auroc": metrics_after.auroc - metrics_before.auroc,
            "auprc": metrics_after.auprc - metrics_before.auprc,
        },
    }


def print_comparison_table(results: list[dict], split_name: str) -> str:
    """Format a comparison table for multiple configs."""
    lines = [
        f"\n{'=' * 90}",
        f"  TEMPORAL POST-PROCESSING COMPARISON — {split_name.upper()} SET",
        f"{'=' * 90}",
        "",
        f"{'Config':<22} {'Sens_bef':>9} {'Sens_aft':>9} {'Spec_bef':>9} "
        f"{'Spec_aft':>9} {'F2_bef':>8} {'F2_aft':>8} {'Changed':>8}",
        f"{'-' * 22} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 9} {'-' * 8} {'-' * 8} {'-' * 8}",
    ]

    for r in results:
        name = r["config"]["name"]
        sb = r["before"]["sensitivity"]
        sa = r["after"]["sensitivity"]
        spb = r["before"]["specificity"]
        spa = r["after"]["specificity"]
        fb = r["before"]["f2"]
        fa = r["after"]["f2"]
        ch = r["changes"]["n_predictions_changed"]

        lines.append(
            f"{name:<22} {sb:>9.4f} {sa:>9.4f} {spb:>9.4f} "
            f"{spa:>9.4f} {fb:>8.4f} {fa:>8.4f} {ch:>8d}"
        )

    lines.extend(
        [
            "",
            "KEY OBSERVATIONS:",
        ]
    )

    # Auto-detect best config by F2 improvement
    best_f2 = max(results, key=lambda r: r["deltas"]["f2"])
    best_spec = max(results, key=lambda r: r["deltas"]["specificity"])

    if best_f2["deltas"]["f2"] > 0:
        lines.append(
            f"  - Best F2 improvement: {best_f2['config']['name']} (+{best_f2['deltas']['f2']:.4f})"
        )
    else:
        lines.append("  - No config improved F2 over baseline")

    if best_spec["deltas"]["specificity"] > 0:
        lines.append(
            f"  - Best specificity improvement: {best_spec['config']['name']} "
            f"(+{best_spec['deltas']['specificity']:.4f})"
        )

    text = "\n".join(lines)
    return text


def main() -> int:
    """Run all temporal post-processing experiments."""
    print("[INFO] Loading features...")
    features = pd.read_parquet(FEATURES_PATH)

    print("[INFO] Loading split assignment...")
    assignment = load_split(SPLITS_DIR)

    print("[INFO] Applying split...")
    _, val_df, test_df = apply_split(features, assignment)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        print(f"\n[INFO] Processing {split_name} set...")

        print(f"[INFO] Enriching {split_name} predictions with temporal metadata...")
        enriched = load_and_enrich(split_name, split_df)
        print(f"[OK] Enriched: {len(enriched)} windows")

        split_results = []
        for config in CONFIGS:
            print(f"  [INFO] Evaluating {config.name}...")
            result = evaluate_config(enriched, config)
            split_results.append(result)

            # Save per-config result
            config_dir = OUTPUT_DIR / config.name
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(config_dir / f"results_{split_name}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        all_results[split_name] = split_results

        # Print comparison table
        table = print_comparison_table(split_results, split_name)
        print(table)

        # Save comparison table
        with open(OUTPUT_DIR / f"comparison_{split_name}.txt", "w", encoding="utf-8") as f:
            f.write(table)

    # Save combined results
    with open(OUTPUT_DIR / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Results saved to {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user.")
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1) from exc
