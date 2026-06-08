"""
Run event-level evaluation on baseline predictions.

This script computes event-level metrics (event recall, false alarms
per hour, detection latency) for both raw and post-processed predictions,
providing a clinically meaningful complement to window-level metrics.

Usage
-----
    python scripts/evaluation/run_event_evaluation.py

Requirements
------------
- Trained baseline predictions in experiments/baseline/logistic_regression/
- Feature data in data/processed/features.parquet
- Split assignments in data/splits/
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neuro_eeg_cdss.data.splits import apply_split, load_split
from neuro_eeg_cdss.evaluation.event_metrics import (
    compute_dataset_event_metrics,
    compute_per_recording_summary,
)
from neuro_eeg_cdss.postprocessing.temporal import (
    TemporalConfig,
    enrich_predictions,
    postprocess_predictions,
)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
PREDICTIONS_DIR = PROJECT_ROOT / "experiments" / "baseline" / "logistic_regression"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "event_evaluation"

# Post-processing configs to compare
POST_CONFIGS = [
    TemporalConfig(strategy="median_filter", kernel_size=7, threshold=0.5),
    TemporalConfig(strategy="min_duration", min_windows=3),
    TemporalConfig(strategy="min_duration", min_windows=4),
]


def load_and_enrich(split_name: str, features_df: pd.DataFrame) -> pd.DataFrame:
    """Load predictions and add temporal metadata."""
    pred_path = PREDICTIONS_DIR / f"predictions_{split_name}.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions not found: {pred_path}")
    predictions_df = pd.read_parquet(pred_path)
    return enrich_predictions(predictions_df, features_df)


def print_event_metrics(label: str, metrics) -> None:
    """Print event metrics in a readable format."""
    print(f"\n  {label}:")
    print(f"    True events:     {metrics.n_true_events}")
    print(f"    Detected events: {metrics.n_detected_events}")
    print(f"    True positives:  {metrics.n_true_positives}")
    print(f"    False negatives: {metrics.n_false_negatives}")
    print(f"    False positives: {metrics.n_false_positives}")
    print(f"    Event sensitivity:  {metrics.event_sensitivity:.4f}")
    print(f"    Event precision:    {metrics.event_precision:.4f}")
    print(f"    Event F1:           {metrics.event_f1:.4f}")
    print(f"    Event F2:           {metrics.event_f2:.4f}")
    print(f"    FA/hour:            {metrics.false_alarm_rate_per_hour:.2f}")
    if metrics.n_true_positives > 0:
        print(f"    Mean latency:       {metrics.mean_latency_sec:.1f}s")
        print(f"    Median latency:     {metrics.median_latency_sec:.1f}s")
    print(f"    Total hours:        {metrics.total_duration_hours:.2f}")


def main() -> int:
    """Run event-level evaluation."""
    print("[INFO] Loading features...")
    features = pd.read_parquet(FEATURES_PATH)

    print("[INFO] Loading split assignment...")
    assignment = load_split(SPLITS_DIR)

    print("[INFO] Applying split...")
    _, val_df, test_df = apply_split(features, assignment)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict = {}

    for split_name, split_df in [("val", val_df), ("test", test_df)]:
        print(f"\n{'=' * 70}")
        print(f"  EVENT-LEVEL EVALUATION — {split_name.upper()} SET")
        print(f"{'=' * 70}")

        enriched = load_and_enrich(split_name, split_df)
        print(f"[OK] Enriched: {len(enriched)} windows")

        split_results: dict = {}

        # --- Baseline (raw predictions) ---
        baseline_metrics = compute_dataset_event_metrics(enriched, pred_col="y_pred")
        print_event_metrics("Baseline (raw y_pred)", baseline_metrics)
        split_results["baseline"] = baseline_metrics.to_dict()

        # --- Post-processed variants ---
        for config in POST_CONFIGS:
            post_df = postprocess_predictions(enriched, config)
            post_metrics = compute_dataset_event_metrics(post_df, pred_col="y_pred_post")
            print_event_metrics(f"Post-processed ({config.name})", post_metrics)
            split_results[config.name] = post_metrics.to_dict()

        # --- Per-recording breakdown (baseline only) ---
        per_rec = compute_per_recording_summary(enriched, pred_col="y_pred")
        split_results["per_recording"] = per_rec

        # Show top false-alarm recordings
        recs_with_fa = [r for r in per_rec if r["n_false_positives"] > 0]
        recs_with_fa.sort(key=lambda r: r["n_false_positives"], reverse=True)
        if recs_with_fa:
            print(f"\n  Top false-alarm recordings ({split_name}):")
            for r in recs_with_fa[:5]:
                print(
                    f"    {r['subject']} | "
                    f"FP events: {r['n_false_positives']} | "
                    f"FA/h: {r['false_alarm_rate_per_hour']:.1f} | "
                    f"True events: {r['n_true_events']}"
                )

        all_results[split_name] = split_results

        # Save per-split results
        output_path = OUTPUT_DIR / f"event_metrics_{split_name}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(split_results, f, indent=2, ensure_ascii=False)

    # Save combined
    with open(OUTPUT_DIR / "all_event_results.json", "w", encoding="utf-8") as f:
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
