from neuro_eeg_cdss.features.dataset_builder import (
    build_features_dataset,
    save_features_dataset,
)


def main() -> None:
    segments_path = "data/processed/segments.parquet"
    output_path = "data/processed/features.parquet"

    df = build_features_dataset(
        segments_path=segments_path,
        relative_bandpower=False,
        max_segments=None,  # cambiar o quitar cuando ya esté validado
    )

    save_features_dataset(df, output_path)

    print(f"Dataset de features guardado en: {output_path}")
    print(f"Número de filas: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
