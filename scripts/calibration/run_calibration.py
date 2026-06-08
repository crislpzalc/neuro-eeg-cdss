"""
Sprint 2C — Probability Calibration Experiment.

This script:
1. Loads prediction parquets for LR and RF (val + test)
2. Measures initial (uncalibrated) calibration metrics
3. Fits calibrators (Platt + Isotonic) on validation predictions
4. Applies calibrators to test predictions
5. Computes calibration metrics before/after
6. Generates reliability diagrams and comparison plots
7. Saves all results to experiments/calibration/

Usage
-----
    python scripts/calibration/run_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neuro_eeg_cdss.calibration.calibrator import (
    calibrate_probabilities,
    compute_calibration_metrics,
    compute_reliability_bins,
    fit_calibrator,
    format_calibration_report,
)
from neuro_eeg_cdss.calibration.plots import (
    plot_calibration_comparison,
    plot_metrics_comparison,
    plot_reliability_diagram,
)

# ── Paths ────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "experiments" / "baseline"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "calibration"

MODELS = ["logistic_regression", "random_forest"]
METHODS = ["platt", "isotonic"]
N_BINS = 10


def load_predictions(model_name: str, split: str) -> pd.DataFrame:
    """Load prediction parquet for a model and split."""
    path = BASELINE_DIR / model_name / f"predictions_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return pd.read_parquet(path)


def run_calibration_experiment() -> None:
    """Run the full calibration experiment."""
    print("=" * 70)
    print("  Sprint 2C — Probability Calibration Experiment")
    print("=" * 70)

    all_results: dict = {}

    for model_name in MODELS:
        print(f"\n{'─' * 60}")
        print(f"  Model: {model_name}")
        print(f"{'─' * 60}")

        # Load predictions
        try:
            df_val = load_predictions(model_name, "val")
            df_test = load_predictions(model_name, "test")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        y_true_val = df_val["y_true"].values
        y_proba_val = df_val["y_proba"].values
        y_true_test = df_test["y_true"].values
        y_proba_test = df_test["y_proba"].values

        print(f"  Val:  {len(y_true_val):,} samples, {int(y_true_val.sum()):,} positive")
        print(f"  Test: {len(y_true_test):,} samples, {int(y_true_test.sum()):,} positive")

        # Create output directory
        model_dir = OUTPUT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 1. Uncalibrated metrics
        print("\n  [Uncalibrated]")
        m_val_uncal = compute_calibration_metrics(y_true_val, y_proba_val, N_BINS)
        m_test_uncal = compute_calibration_metrics(y_true_test, y_proba_test, N_BINS)
        print(f"    Val  — ECE: {m_val_uncal.ece:.6f}, Brier: {m_val_uncal.brier:.6f}")
        print(f"    Test — ECE: {m_test_uncal.ece:.6f}, Brier: {m_test_uncal.brier:.6f}")

        # Reliability diagram (uncalibrated)
        bins_val_uncal = compute_reliability_bins(y_true_val, y_proba_val, N_BINS)
        bins_test_uncal = compute_reliability_bins(y_true_test, y_proba_test, N_BINS)

        plot_reliability_diagram(
            bins_test_uncal,
            model_name=model_name,
            split_name="test",
            label="Uncalibrated",
            output_path=model_dir / "reliability_uncalibrated_test.png",
        )
        plot_reliability_diagram(
            bins_val_uncal,
            model_name=model_name,
            split_name="val",
            label="Uncalibrated",
            output_path=model_dir / "reliability_uncalibrated_val.png",
        )

        model_results: dict = {
            "uncalibrated": {
                "val": m_val_uncal.to_dict(),
                "test": m_test_uncal.to_dict(),
            }
        }

        # 2. Calibrate with each method
        for method in METHODS:
            print(f"\n  [{method.capitalize()}]")

            # Fit on validation set
            calibrator = fit_calibrator(y_true_val, y_proba_val, method)

            # Apply to both splits
            y_cal_val = calibrate_probabilities(calibrator, y_proba_val)
            y_cal_test = calibrate_probabilities(calibrator, y_proba_test)

            # Compute metrics
            m_val_cal = compute_calibration_metrics(y_true_val, y_cal_val, N_BINS)
            m_test_cal = compute_calibration_metrics(y_true_test, y_cal_test, N_BINS)

            print(f"    Val  — ECE: {m_val_cal.ece:.6f}, Brier: {m_val_cal.brier:.6f}")
            print(f"    Test — ECE: {m_test_cal.ece:.6f}, Brier: {m_test_cal.brier:.6f}")

            # Report
            report = format_calibration_report(
                m_test_uncal,
                m_test_cal,
                method=method,
                split_name="test",
                model_name=model_name,
            )
            print(f"\n{report}")

            # Reliability bins (calibrated)
            bins_val_cal = compute_reliability_bins(y_true_val, y_cal_val, N_BINS)
            bins_test_cal = compute_reliability_bins(y_true_test, y_cal_test, N_BINS)

            # Comparison plots
            plot_calibration_comparison(
                bins_test_uncal,
                bins_test_cal,
                method=method,
                model_name=model_name,
                split_name="test",
                output_path=model_dir / f"comparison_{method}_test.png",
            )
            plot_calibration_comparison(
                bins_val_uncal,
                bins_val_cal,
                method=method,
                model_name=model_name,
                split_name="val",
                output_path=model_dir / f"comparison_{method}_val.png",
            )

            # Save calibrated predictions
            df_test_cal = df_test.copy()
            df_test_cal["y_proba_calibrated"] = y_cal_test
            df_test_cal["y_pred_calibrated"] = (y_cal_test >= 0.5).astype(int)
            df_test_cal.to_parquet(model_dir / f"predictions_test_{method}.parquet", index=False)

            df_val_cal = df_val.copy()
            df_val_cal["y_proba_calibrated"] = y_cal_val
            df_val_cal["y_pred_calibrated"] = (y_cal_val >= 0.5).astype(int)
            df_val_cal.to_parquet(model_dir / f"predictions_val_{method}.parquet", index=False)

            model_results[method] = {
                "val": m_val_cal.to_dict(),
                "test": m_test_cal.to_dict(),
            }

            # Save reliability bin data
            bins_data = {
                "val": {
                    "uncalibrated": [b.to_dict() for b in bins_val_uncal],
                    method: [b.to_dict() for b in bins_val_cal],
                },
                "test": {
                    "uncalibrated": [b.to_dict() for b in bins_test_uncal],
                    method: [b.to_dict() for b in bins_test_cal],
                },
            }
            with open(model_dir / f"reliability_bins_{method}.json", "w") as f:
                json.dump(bins_data, f, indent=2)

        # 3. Summary metrics comparison chart
        metrics_for_chart = {
            "uncalibrated": m_test_uncal,
        }
        for method in METHODS:
            m = compute_calibration_metrics(
                y_true_test,
                calibrate_probabilities(
                    fit_calibrator(y_true_val, y_proba_val, method),
                    y_proba_test,
                ),
                N_BINS,
            )
            metrics_for_chart[method] = m

        plot_metrics_comparison(
            metrics_for_chart,
            model_name=model_name,
            split_name="test",
            output_path=model_dir / "metrics_comparison_test.png",
        )

        # Save model results
        with open(model_dir / "results.json", "w") as f:
            json.dump(model_results, f, indent=2)

        all_results[model_name] = model_results

    # 4. Save combined results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 5. Generate comparison table
    _write_comparison_table(all_results)

    print(f"\n{'=' * 70}")
    print(f"  Results saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


def _write_comparison_table(all_results: dict) -> None:
    """Write a formatted comparison table to a text file."""
    lines = [
        "Calibration Results — Test Set Comparison",
        "=" * 80,
        "",
        f"{'Model':<25} {'Config':<15} {'ECE':>10} {'MCE':>10} {'Brier':>10} {'Log Loss':>10}",
        "-" * 80,
    ]

    for model_name, model_results in all_results.items():
        for config_name, split_results in model_results.items():
            if "test" in split_results:
                m = split_results["test"]
                ece_str = f"{m['ece']:.6f}" if m["ece"] is not None else "N/A"
                mce_str = f"{m['mce']:.6f}" if m["mce"] is not None else "N/A"
                brier_str = f"{m['brier']:.6f}" if m["brier"] is not None else "N/A"
                ll_str = f"{m['log_loss_val']:.6f}" if m["log_loss_val"] is not None else "N/A"
                lines.append(
                    f"{model_name:<25} {config_name:<15} "
                    f"{ece_str:>10} {mce_str:>10} {brier_str:>10} {ll_str:>10}"
                )

    with open(OUTPUT_DIR / "comparison_test.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Comparison table saved to: {OUTPUT_DIR / 'comparison_test.txt'}")


if __name__ == "__main__":
    run_calibration_experiment()
