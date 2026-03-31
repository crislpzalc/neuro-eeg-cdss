from __future__ import annotations

from dataclasses import dataclass


class SegmentationError(ValueError):
    """Error lanzado cuando la configuración o los tiempos de segmentación no son válidos."""


@dataclass(frozen=True)
class TimeWindow:
    """
    Ventana temporal fija dentro de un registro EEG.

    Attributes
    ----------
    start_sec : float
        Tiempo de inicio de la ventana en segundos.
    end_sec : float
        Tiempo de fin de la ventana en segundos.
    """

    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        """Duración de la ventana en segundos."""
        return self.end_sec - self.start_sec


def _validate_positive(value: float, name: str) -> None:
    """
    Valida que un valor numérico sea estrictamente positivo.
    """
    if value <= 0:
        raise SegmentationError(f"'{name}' debe ser > 0. Valor recibido: {value}")


def _validate_non_negative(value: float, name: str) -> None:
    """
    Valida que un valor numérico sea no negativo.
    """
    if value < 0:
        raise SegmentationError(f"'{name}' debe ser >= 0. Valor recibido: {value}")


def generate_time_windows(
    recording_duration_sec: float,
    window_size_sec: float,
    stride_sec: float,
) -> list[TimeWindow]:
    """
    Genera ventanas temporales completas dentro de un registro EEG.

    Regla:
    - Solo se generan ventanas completas.
    - Una ventana se incluye solo si end_sec <= recording_duration_sec.

    Parameters
    ----------
    recording_duration_sec : float
        Duración total del registro en segundos.
    window_size_sec : float
        Tamaño de cada ventana en segundos.
    stride_sec : float
        Desplazamiento entre ventanas consecutivas en segundos.

    Returns
    -------
    list[TimeWindow]
        Lista ordenada de ventanas temporales.

    Raises
    ------
    SegmentationError
        Si los parámetros son inválidos.
    """
    _validate_non_negative(recording_duration_sec, "recording_duration_sec")
    _validate_positive(window_size_sec, "window_size_sec")
    _validate_positive(stride_sec, "stride_sec")

    if recording_duration_sec < window_size_sec:
        return []

    windows: list[TimeWindow] = []
    start_sec = 0.0

    while start_sec + window_size_sec <= recording_duration_sec:
        end_sec = start_sec + window_size_sec
        windows.append(TimeWindow(start_sec=start_sec, end_sec=end_sec))
        start_sec += stride_sec

    return windows


def compute_overlap_seconds(
    window_start_sec: float,
    window_end_sec: float,
    interval_start_sec: float,
    interval_end_sec: float,
) -> float:
    """
    Calcula el solape temporal en segundos entre una ventana y un intervalo.

    Parameters
    ----------
    window_start_sec : float
        Inicio de la ventana.
    window_end_sec : float
        Fin de la ventana.
    interval_start_sec : float
        Inicio del intervalo.
    interval_end_sec : float
        Fin del intervalo.

    Returns
    -------
    float
        Segundos de solape. Si no hay solape, devuelve 0.0.

    Raises
    ------
    SegmentationError
        Si los tiempos son inválidos.
    """
    if window_end_sec < window_start_sec:
        raise SegmentationError("La ventana tiene tiempos inválidos: end_sec < start_sec.")

    if interval_end_sec < interval_start_sec:
        raise SegmentationError("El intervalo tiene tiempos inválidos: end_sec < start_sec.")

    overlap_start = max(window_start_sec, interval_start_sec)
    overlap_end = min(window_end_sec, interval_end_sec)

    return max(0.0, overlap_end - overlap_start)


def compute_total_overlap_seconds(
    window: TimeWindow,
    intervals: list[tuple[float, float]],
) -> float:
    """
    Calcula el solape total en segundos entre una ventana y una lista de intervalos.

    Parameters
    ----------
    window : TimeWindow
        Ventana temporal.
    intervals : list[tuple[float, float]]
        Lista de intervalos (start_sec, end_sec).

    Returns
    -------
    float
        Solape total en segundos.
    """
    total_overlap = 0.0

    for interval_start_sec, interval_end_sec in intervals:
        total_overlap += compute_overlap_seconds(
            window_start_sec=window.start_sec,
            window_end_sec=window.end_sec,
            interval_start_sec=interval_start_sec,
            interval_end_sec=interval_end_sec,
        )

    return total_overlap


def compute_overlap_ratio(
    window: TimeWindow,
    overlap_seconds: float,
) -> float:
    """
    Calcula la fracción de solape de una ventana.

    Parameters
    ----------
    window : TimeWindow
        Ventana temporal.
    overlap_seconds : float
        Solape total en segundos.

    Returns
    -------
    float
        Ratio de solape en [0, 1].

    Raises
    ------
    SegmentationError
        Si el solape es inválido.
    """
    _validate_non_negative(overlap_seconds, "overlap_seconds")

    if overlap_seconds > window.duration_sec:
        raise SegmentationError("El solape no puede ser mayor que la duración de la ventana.")

    return overlap_seconds / window.duration_sec
