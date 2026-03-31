from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import mne
import pandas as pd

from neuro_eeg_cdss.preprocessing.events import read_seizure_intervals
from neuro_eeg_cdss.preprocessing.labeling import assign_label
from neuro_eeg_cdss.preprocessing.segmentation import (
    compute_overlap_ratio,
    compute_total_overlap_seconds,
    generate_time_windows,
)


class DatasetBuilderError(ValueError):
    """Error lanzado cuando falla la construcción del dataset de segmentos."""


@dataclass(frozen=True)
class SegmentRecord:
    """
    Representa una fila del dataset final de segmentos.
    """

    subject: str
    session: str | None
    run: str | None
    path: str
    recording_duration_sec: float
    start_sec: float
    end_sec: float
    window_size_sec: float
    stride_sec: float
    overlap_ratio: float
    label: int


def load_manifest(manifest_path: str | Path) -> pd.DataFrame:
    """
    Carga el manifest.parquet.

    Parameters
    ----------
    manifest_path : str | Path
        Ruta al manifest.

    Returns
    -------
    pd.DataFrame
        Manifest cargado.

    Raises
    ------
    FileNotFoundError
        Si no existe el archivo.
    DatasetBuilderError
        Si faltan columnas mínimas.
    """
    path = Path(manifest_path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest no encontrado: {path}")

    df = pd.read_parquet(path)

    required_columns = {"subject", "path"}
    missing = required_columns - set(df.columns)

    if missing:
        raise DatasetBuilderError(f"Faltan columnas obligatorias en el manifest: {sorted(missing)}")

    return df


def derive_events_tsv_path_from_eeg_path(eeg_path: str | Path) -> Path:
    """
    Deriva la ruta al events.tsv a partir de la ruta del archivo EEG.

    Ejemplo:
    sub-01_ses-01_task-xxx_eeg.edf -> sub-01_ses-01_task-xxx_events.tsv
    """
    eeg_path = Path(eeg_path)

    if eeg_path.name.endswith("_eeg.edf"):
        return eeg_path.with_name(eeg_path.name.replace("_eeg.edf", "_events.tsv"))

    raise DatasetBuilderError(f"No se pudo derivar events.tsv desde el path EEG: {eeg_path}")


def get_recording_duration_sec(eeg_path: str | Path) -> float:
    """
    Lee la duración total del registro EEG en segundos.

    Parameters
    ----------
    eeg_path : str | Path
        Ruta al archivo EDF.

    Returns
    -------
    float
        Duración del registro en segundos.
    """
    eeg_path = Path(eeg_path)

    try:
        raw = mne.io.read_raw_edf(eeg_path, preload=False, verbose="ERROR")
    except Exception as exc:
        raise DatasetBuilderError(
            f"No se pudo leer el EEG para obtener duración: {eeg_path}"
        ) from exc

    return float(raw.n_times / raw.info["sfreq"])


def _safe_get_str(row: pd.Series, key: str) -> str | None:
    """
    Recupera un valor string opcional del manifest.
    """
    if key not in row.index:
        return None

    value = row[key]
    if pd.isna(value):
        return None

    return str(value)


def build_segments_for_recording(
    row: pd.Series,
    window_size_sec: float,
    stride_sec: float,
    positive_overlap_threshold: float = 0.5,
    drop_partial_overlap: bool = True,
) -> list[SegmentRecord]:
    """
    Construye los segmentos etiquetados para un único registro EEG.

    Parameters
    ----------
    row : pd.Series
        Fila del manifest.
    window_size_sec : float
        Tamaño de ventana.
    stride_sec : float
        Stride entre ventanas.
    positive_overlap_threshold : float, default=0.5
        Umbral para considerar una ventana como positiva.
    drop_partial_overlap : bool, default=True
        Si True, descarta ventanas ambiguas con solape parcial inferior al umbral.

    Returns
    -------
    list[SegmentRecord]
        Segmentos válidos para ese registro.
    """
    eeg_path = Path(str(row["path"]))
    events_tsv_path = derive_events_tsv_path_from_eeg_path(eeg_path)

    recording_duration_sec = get_recording_duration_sec(eeg_path)
    seizure_intervals = read_seizure_intervals(events_tsv_path)

    interval_tuples = [(interval.onset_sec, interval.end_sec) for interval in seizure_intervals]

    windows = generate_time_windows(
        recording_duration_sec=recording_duration_sec,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
    )

    subject = str(row["subject"])
    session = _safe_get_str(row, "session")
    run = _safe_get_str(row, "run")

    records: list[SegmentRecord] = []

    for window in windows:
        overlap_seconds = compute_total_overlap_seconds(window, interval_tuples)
        overlap_ratio = compute_overlap_ratio(window, overlap_seconds)

        decision = assign_label(
            overlap_ratio=overlap_ratio,
            positive_overlap_threshold=positive_overlap_threshold,
            drop_partial_overlap=drop_partial_overlap,
        )

        if not decision.keep:
            continue

        records.append(
            SegmentRecord(
                subject=subject,
                session=session,
                run=run,
                path=str(eeg_path),
                recording_duration_sec=recording_duration_sec,
                start_sec=window.start_sec,
                end_sec=window.end_sec,
                window_size_sec=window_size_sec,
                stride_sec=stride_sec,
                overlap_ratio=overlap_ratio,
                label=decision.label,
            )
        )

    return records


def build_segments_dataset(
    manifest_path: str | Path,
    window_size_sec: float,
    stride_sec: float,
    positive_overlap_threshold: float = 0.5,
    drop_partial_overlap: bool = True,
) -> pd.DataFrame:
    """
    Construye el dataset completo de segmentos a partir del manifest.

    Parameters
    ----------
    manifest_path : str | Path
        Ruta al manifest.parquet.
    window_size_sec : float
        Tamaño de ventana.
    stride_sec : float
        Stride entre ventanas.
    positive_overlap_threshold : float, default=0.5
        Umbral de positividad.
    drop_partial_overlap : bool, default=True
        Si True, descarta ventanas ambiguas.

    Returns
    -------
    pd.DataFrame
        Dataset final de segmentos.
    """
    manifest_df = load_manifest(manifest_path)

    all_records: list[SegmentRecord] = []

    for _, row in manifest_df.iterrows():
        record_segments = build_segments_for_recording(
            row=row,
            window_size_sec=window_size_sec,
            stride_sec=stride_sec,
            positive_overlap_threshold=positive_overlap_threshold,
            drop_partial_overlap=drop_partial_overlap,
        )
        all_records.extend(record_segments)

    df = pd.DataFrame([asdict(record) for record in all_records])

    if df.empty:
        return pd.DataFrame(
            columns=[
                "subject",
                "session",
                "run",
                "path",
                "recording_duration_sec",
                "start_sec",
                "end_sec",
                "window_size_sec",
                "stride_sec",
                "overlap_ratio",
                "label",
            ]
        )

    return df.sort_values(
        by=["subject", "path", "start_sec"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def save_segments_dataset(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Guarda el dataset de segmentos en parquet.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset de segmentos.
    output_path : str | Path
        Ruta de salida.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
