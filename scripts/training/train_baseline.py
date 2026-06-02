"""
Train baseline models for seizure detection.

Usage:
    python scripts/training/train_baseline.py
    python scripts/training/train_baseline.py --features data/processed/features.parquet
    python scripts/training/train_baseline.py --seed 123

This script:
1. Loads the features dataset
2. Loads the patient-independent split
3. Trains Logistic Regression and Random Forest on the training set
4. Generates predictions on train/val/test for later evaluation
5. Saves all artifacts for reproducibility
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_eeg_cdss.data.splits import apply_split, load_split
from neuro_eeg_cdss.training.trainer import (
    TrainConfig,
    predict,
    save_train_result,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline seizure detection models.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features.parquet"),
        help="Path to the feature dataset.",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing split JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/baseline"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def _print_split_info(name: str, df: pd.DataFrame) -> None:
    """Print a concise summary of a split."""
    n_pos = int(df["label"].sum())
    n_neg = len(df) - n_pos
    print(f"  {name:>5s}: {len(df):>8,} samples ({n_pos:,} pos / {n_neg:,} neg)")


def _save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Save predictions to parquet for downstream evaluation."""
    pred_df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(output_path, index=False)


def train_and_evaluate_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> dict:
    """Train one model and save all artifacts."""
    model_dir = output_dir / model_type
    print(f"\n{'=' * 60}")
    print(f"  Training: {model_type}")
    print(f"{'=' * 60}")

    config = TrainConfig(
        model_type=model_type,
        seed=seed,
        scale_features=True,
    )

    result = train_model(train_df, config=config)
    print(f"  Model fitted: {result.train_shape[0]:,} samples, {result.train_shape[1]} features")

    # Save model artifacts
    save_train_result(result, model_dir)
    print(f"  Model saved to: {model_dir}")

    # Predict on all splits
    summary = {}
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        y_true = split_df["label"].values.astype(int)
        y_pred, y_proba = predict(result, split_df)

        _save_predictions(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            output_path=model_dir / f"predictions_{split_name}.parquet",
        )

        n_correct = int((y_pred == y_true).sum())
        accuracy = n_correct / len(y_true)
        n_tp = int(((y_pred == 1) & (y_true == 1)).sum())
        n_fn = int(((y_pred == 0) & (y_true == 1)).sum())
        n_fp = int(((y_pred == 1) & (y_true == 0)).sum())
        n_tn = int(((y_pred == 0) & (y_true == 0)).sum())

        n_pos = int(y_true.sum())
        recall = n_tp / max(n_pos, 1)
        precision = n_tp / max(n_tp + n_fp, 1)

        split_summary = {
            "n_samples": len(y_true),
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "tp": n_tp,
            "fn": n_fn,
            "fp": n_fp,
            "tn": n_tn,
        }
        summary[split_name] = split_summary

        print(
            f"  [{split_name:>5s}] "
            f"acc={accuracy:.4f}  "
            f"recall={recall:.4f}  "
            f"precision={precision:.4f}  "
            f"(TP={n_tp} FN={n_fn} FP={n_fp} TN={n_tn})"
        )

    # Save quick summary
    with open(model_dir / "quick_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    args = parse_args()

    print(f"Loading features from: {args.features}")
    features_df = pd.read_parquet(args.features)
    print(f"  Total: {len(features_df):,} rows, {len(features_df.columns)} columns")

    print(f"\nLoading split from: {args.split_dir}")
    split = load_split(args.split_dir)

    train_df, val_df, test_df = apply_split(features_df, split)
    _print_split_info("train", train_df)
    _print_split_info("val", val_df)
    _print_split_info("test", test_df)

    models = ["logistic_regression", "random_forest"]
    all_results = {}

    for model_type in models:
        summary = train_and_evaluate_model(
            model_type=model_type,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            output_dir=args.output_dir,
            seed=args.seed,
        )
        all_results[model_type] = summary

    # Save combined summary
    combined_path = args.output_dir / "all_results.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  All results saved to: {args.output_dir.resolve()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
