from neuro_eeg_cdss.preprocessing.dataset_builder import (
    build_segments_dataset,
    save_segments_dataset,
)


def main() -> None:
    manifest_path = "data/manifests/manifest.parquet"
    output_path = "data/processed/segments.parquet"

    df = build_segments_dataset(
        manifest_path=manifest_path,
        window_size_sec=5.0,
        stride_sec=5.0,
        positive_overlap_threshold=0.5,
        drop_partial_overlap=True,
    )

    save_segments_dataset(df, output_path)

    print(f"Dataset de segmentos guardado en: {output_path}")
    print(f"Número de segmentos: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
