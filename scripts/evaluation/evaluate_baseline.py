"""
Clinical evaluation of baseline seizure detection models.

Usage:
    python scripts/evaluation/evaluate_baseline.py
    python scripts/evaluation/evaluate_baseline.py --experiments-dir experiments/baseline
    python scripts/evaluation/evaluate_baseline.py --no-plots

This script:
1. Loads saved predictions from Sprint 1D (no re-training needed)
2. Computes full clinical metrics per model per split
3. Generates publication-quality plots (ROC, PR, confusion matrices, etc.)
4. Performs threshold analysis
5. Saves all results for documentation and downstream use
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neuro_eeg_cdss.evaluation.metrics import (
    compute_threshold_analysis,
    compute_window_metrics,
    format_metrics_report,
)
from neuro_eeg_cdss.evaluation.plots import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pr_curves,
    plot_roc_curves,
    plot_threshold_analysis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline seizure detection models.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments/baseline"),
        help="Root directory containing model prediction files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/baseline/evaluation"),
        help="Directory for evaluation outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (metrics only).",
    )
    return parser.parse_args()


def _load_predictions(model_dir: Path, split_name: str) -> pd.DataFrame:
    """Load predictions for a model and split."""
    path = model_dir / f"predictions_{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    return pd.read_parquet(path)


def evaluate_model(
    model_name: str,
    model_dir: Path,
    output_dir: Path,
    generate_plots: bool,
) -> dict[str, dict]:
    """Evaluate one model across all splits."""
    print(f"\n{'=' * 70}")
    print(f"  Evaluating: {model_name}")
    print(f"{'=' * 70}")

    splits = ["train", "val", "test"]
    metrics_by_split = {}
    predictions_by_split = {}

    for split_name in splits:
        df = _load_predictions(model_dir, split_name)
        y_true = df["y_true"].values
        y_pred = df["y_pred"].values
        y_proba = df["y_proba"].values

        metrics = compute_window_metrics(y_true, y_pred, y_proba)
        metrics_by_split[split_name] = metrics
        predictions_by_split[split_name] = (y_true, y_pred, y_proba)

    # Print report
    report = format_metrics_report(metrics_by_split, model_name=model_name)
    print(report)

    # Save metrics JSON
    model_eval_dir = output_dir / model_name
    model_eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_dict = {split: m.to_dict() for split, m in metrics_by_split.items()}
    with open(model_eval_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2, ensure_ascii=False)

    # Threshold analysis (val and test only)
    for split_name in ["val", "test"]:
        y_true, _, y_proba = predictions_by_split[split_name]
        thresh_results = compute_threshold_analysis(y_true, y_proba)
        with open(
            model_eval_dir / f"threshold_analysis_{split_name}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(thresh_results, f, indent=2, ensure_ascii=False)

    # Generate plots
    if generate_plots:
        plots_dir = model_eval_dir / "plots"

        for split_name in splits:
            y_true, y_pred, y_proba = predictions_by_split[split_name]

            # Confusion matrix (absolute)
            plot_confusion_matrix(
                y_true,
                y_pred,
                model_name,
                split_name,
                normalize=False,
                output_path=plots_dir / f"confusion_matrix_{split_name}.png",
            )

            # Confusion matrix (normalized)
            plot_confusion_matrix(
                y_true,
                y_pred,
                model_name,
                split_name,
                normalize=True,
                output_path=plots_dir / f"confusion_matrix_normalized_{split_name}.png",
            )

        # Threshold analysis plots (val and test)
        for split_name in ["val", "test"]:
            y_true, _, y_proba = predictions_by_split[split_name]
            plot_threshold_analysis(
                y_true,
                y_proba,
                model_name,
                split_name,
                output_path=plots_dir / f"threshold_analysis_{split_name}.png",
            )

        print(f"  Plots saved to: {plots_dir}")

    return metrics_dict


def generate_comparison_plots(
    all_predictions: dict[str, dict[str, tuple]],
    all_metrics: dict[str, dict[str, dict]],
    output_dir: Path,
) -> None:
    """Generate cross-model comparison plots."""
    comparison_dir = output_dir / "comparison"

    for split_name in ["val", "test"]:
        # ROC curves
        roc_data = {}
        pr_data = {}
        metrics_for_comparison = {}

        for model_name in all_predictions:
            y_true, _, y_proba = all_predictions[model_name][split_name]
            auroc = all_metrics[model_name][split_name]["auroc"]
            auprc = all_metrics[model_name][split_name]["auprc"]
            roc_data[model_name] = (y_true, y_proba, auroc)
            pr_data[model_name] = (y_true, y_proba, auprc)

            # Reconstruct WindowMetrics for comparison plot
            from neuro_eeg_cdss.evaluation.metrics import WindowMetrics

            metrics_for_comparison[model_name] = WindowMetrics(
                **all_metrics[model_name][split_name]
            )

        prevalence = all_metrics[list(all_predictions.keys())[0]][split_name]["prevalence"]

        plot_roc_curves(
            roc_data,
            split_name,
            output_path=comparison_dir / f"roc_comparison_{split_name}.png",
        )

        plot_pr_curves(
            pr_data,
            split_name,
            prevalence=prevalence,
            output_path=comparison_dir / f"pr_comparison_{split_name}.png",
        )

        plot_model_comparison(
            metrics_for_comparison,
            split_name,
            output_path=comparison_dir / f"model_comparison_{split_name}.png",
        )

    print(f"\n  Comparison plots saved to: {comparison_dir}")


def main() -> None:
    args = parse_args()

    models = ["logistic_regression", "random_forest"]
    all_metrics = {}
    all_predictions = {}

    for model_name in models:
        model_dir = args.experiments_dir / model_name
        if not model_dir.exists():
            print(f"  WARNING: Model directory not found: {model_dir}, skipping.")
            continue

        metrics_dict = evaluate_model(
            model_name=model_name,
            model_dir=model_dir,
            output_dir=args.output_dir,
            generate_plots=not args.no_plots,
        )
        all_metrics[model_name] = metrics_dict

        # Cache predictions for comparison plots
        predictions = {}
        for split_name in ["train", "val", "test"]:
            df = _load_predictions(model_dir, split_name)
            predictions[split_name] = (
                df["y_true"].values,
                df["y_pred"].values,
                df["y_proba"].values,
            )
        all_predictions[model_name] = predictions

    # Combined results
    combined_path = args.output_dir / "all_metrics.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    # Comparison plots
    if not args.no_plots and len(all_predictions) > 1:
        generate_comparison_plots(all_predictions, all_metrics, args.output_dir)

    # Print summary comparison table
    print(f"\n{'=' * 70}")
    print("  SUMMARY COMPARISON (Test Set)")
    print(f"{'=' * 70}")
    print(f"  {'Metric':<22s}", end="")
    for model_name in all_metrics:
        print(f"  {model_name.replace('_', ' ').title():>22s}", end="")
    print()
    print(f"  {'-' * 22}", end="")
    for _ in all_metrics:
        print(f"  {'-' * 22}", end="")
    print()

    key_metrics = [
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
        ("Precision", "precision"),
        ("F1", "f1"),
        ("F2", "f2"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("AUROC", "auroc"),
        ("AUPRC", "auprc"),
    ]

    for display_name, key in key_metrics:
        print(f"  {display_name:<22s}", end="")
        for model_name in all_metrics:
            val = all_metrics[model_name]["test"][key]
            print(f"  {val:>22.4f}", end="")
        print()

    print(f"\n  All evaluation results saved to: {args.output_dir.resolve()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
