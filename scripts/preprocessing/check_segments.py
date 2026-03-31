import pandas as pd

df = pd.read_parquet("data/processed/segments.parquet")

print("Shape:", df.shape)
print("\nLabels:")
print(df["label"].value_counts(dropna=False))

print("\nOverlap ratio summary:")
print(df["overlap_ratio"].describe())

print("\nNulls:")
print(df.isna().sum())

print("\nWindow duration unique values:")
print((df["end_sec"] - df["start_sec"]).value_counts().head())

print("\nWindow size unique:")
print(df["window_size_sec"].unique())

print("\nStride unique:")
print(df["stride_sec"].unique())

print("\nSubjects:", df["subject"].nunique())
