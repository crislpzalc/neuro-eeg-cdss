from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class SeizureInterval:
    """
    Intervalo de crisis epiléptica en segundos relativo al inicio del registro.

    Attributes
    ----------
    onset_sec : float
        Segundo de inicio de la crisis.
    duration_sec : float
        Duración de la crisis en segundos.
    end_sec : float
        Segundo de fin de la crisis.
    event_type : str
        Tipo de evento original en el events.tsv.
    """

    onset_sec: float
    duration_sec: float
    end_sec: float
    event_type: str


class EventsFileError(ValueError):
    """Error lanzado cuando un events.tsv no puede interpretarse correctamente."""


def read_events_tsv(events_tsv_path: str | Path) -> pd.DataFrame:
    """
    Lee un archivo BIDS events.tsv y devuelve un DataFrame.

    Parameters
    ----------
    events_tsv_path : str | Path
        Ruta al archivo events.tsv.

    Returns
    -------
    pd.DataFrame
        Tabla de eventos con nombres de columnas normalizados.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe.
    EventsFileError
        Si el archivo no puede leerse.
    """
    path = Path(events_tsv_path)

    if not path.exists():
        raise FileNotFoundError(f"events.tsv no encontrado: {path}")

    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise EventsFileError(f"No se pudo leer el archivo events.tsv: {path}") from exc

    if df.empty:
        return df

    df.columns = [str(col).strip() for col in df.columns]
    return df


def validate_events_columns(df: pd.DataFrame) -> None:
    """
    Valida que existan las columnas mínimas necesarias para extraer crisis.

    En este dataset necesitamos:
    - onset
    - duration
    - eventType

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de eventos.

    Raises
    ------
    EventsFileError
        Si faltan columnas obligatorias.
    """
    required_columns = {"onset", "duration", "eventType"}
    missing = required_columns - set(df.columns)

    if missing:
        raise EventsFileError(f"Faltan columnas obligatorias en events.tsv: {sorted(missing)}")


def _normalize_event_type(value: object) -> str:
    """
    Normaliza el valor de eventType para comparación robusta.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def _is_seizure_event_type(event_type: str) -> bool:
    """
    Decide si un eventType corresponde a una crisis.

    Regla específica para este dataset:
    - 'bckg' = no crisis
    - cualquier código que empiece por 'sz' = crisis

    Examples
    --------
    'sz' -> True
    'sz_foc_a' -> True
    'sz_gen_m_tonicClonic' -> True
    'bckg' -> False
    """
    if event_type == "bckg":
        return False

    return event_type.startswith("sz")


def _coerce_non_negative_float(value: object, field_name: str) -> float:
    """
    Convierte un valor a float no negativo.

    Parameters
    ----------
    value : object
        Valor a convertir.
    field_name : str
        Nombre del campo para mensajes de error.

    Returns
    -------
    float
        Valor convertido.

    Raises
    ------
    EventsFileError
        Si no puede convertirse o es negativo.
    """
    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise EventsFileError(f"Valor no numérico para '{field_name}': {value!r}") from exc

    if numeric_value < 0:
        raise EventsFileError(f"Valor negativo para '{field_name}': {numeric_value}")

    return numeric_value


def extract_seizure_intervals(df: pd.DataFrame) -> list[SeizureInterval]:
    """
    Extrae los intervalos de crisis a partir de un DataFrame de eventos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame leído desde events.tsv.

    Returns
    -------
    list[SeizureInterval]
        Lista ordenada de intervalos de crisis.

    Raises
    ------
    EventsFileError
        Si faltan columnas o hay valores inválidos.
    """
    if df.empty:
        return []

    validate_events_columns(df)

    seizure_intervals: list[SeizureInterval] = []

    for _, row in df.iterrows():
        event_type = _normalize_event_type(row["eventType"])

        if not _is_seizure_event_type(event_type):
            continue

        onset_sec = _coerce_non_negative_float(row["onset"], "onset")
        duration_sec = _coerce_non_negative_float(row["duration"], "duration")
        end_sec = onset_sec + duration_sec

        seizure_intervals.append(
            SeizureInterval(
                onset_sec=onset_sec,
                duration_sec=duration_sec,
                end_sec=end_sec,
                event_type=event_type,
            )
        )

    seizure_intervals.sort(key=lambda interval: interval.onset_sec)
    return seizure_intervals


def read_seizure_intervals(events_tsv_path: str | Path) -> list[SeizureInterval]:
    """
    Lee un events.tsv y devuelve directamente los intervalos de crisis.

    Parameters
    ----------
    events_tsv_path : str | Path
        Ruta al archivo events.tsv.

    Returns
    -------
    list[SeizureInterval]
        Intervalos de crisis ordenados por onset.
    """
    df = read_events_tsv(events_tsv_path)
    return extract_seizure_intervals(df)


def intervals_to_dataframe(intervals: Iterable[SeizureInterval]) -> pd.DataFrame:
    """
    Convierte una lista de intervalos a un DataFrame tabular.

    Parameters
    ----------
    intervals : Iterable[SeizureInterval]
        Intervalos de crisis.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas:
        - onset_sec
        - duration_sec
        - end_sec
        - event_type
    """
    rows = [
        {
            "onset_sec": interval.onset_sec,
            "duration_sec": interval.duration_sec,
            "end_sec": interval.end_sec,
            "event_type": interval.event_type,
        }
        for interval in intervals
    ]

    return pd.DataFrame(
        rows,
        columns=["onset_sec", "duration_sec", "end_sec", "event_type"],
    )
