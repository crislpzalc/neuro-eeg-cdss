from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    features_path = Path("data/processed/features.parquet")

    if not features_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {features_path}")

    df = pd.read_parquet(features_path)

    print("Shape:", df.shape)

    print("\nLabels:")
    print(df["label"].value_counts(dropna=False))

    print("\nNulls per column:")
    null_counts = df.isna().sum()
    print(null_counts[null_counts > 0] if (null_counts > 0).any() else "No nulls found.")

    print("\nMetadata columns:")
    metadata_cols = ["subject", "session", "run", "path", "start_sec", "end_sec", "label"]
    present_metadata_cols = [col for col in metadata_cols if col in df.columns]
    print(present_metadata_cols)

    feature_cols = [col for col in df.columns if col not in metadata_cols]

    print("\nNumber of feature columns:", len(feature_cols))

    print("\nSample feature columns:")
    print(feature_cols[:20])

    print("\nSummary statistics for first 10 feature columns:")
    if feature_cols:
        print(df[feature_cols[:10]].describe().T)
    else:
        print("No feature columns found.")

    non_numeric_cols = df[feature_cols].select_dtypes(exclude=["number"]).columns.tolist()
    print("\nNon-numeric feature columns:")
    print(non_numeric_cols if non_numeric_cols else "All feature columns are numeric.")

    negative_bandpower_cols = [
        col for col in feature_cols if "power_" in col and (df[col] < 0).any()
    ]
    print("\nBandpower columns with negative values:")
    print(negative_bandpower_cols if negative_bandpower_cols else "None")

    print("\nWindow durations:")
    print((df["end_sec"] - df["start_sec"]).value_counts().head())

    print("\nSubjects:", df["subject"].nunique())

    print("\nDone.")


if __name__ == "__main__":
    main()
