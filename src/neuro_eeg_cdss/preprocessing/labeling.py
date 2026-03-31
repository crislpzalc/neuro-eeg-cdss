from __future__ import annotations

from dataclasses import dataclass


class LabelingError(ValueError):
    """Error lanzado cuando la configuración o el solape no son válidos."""


@dataclass(frozen=True)
class LabelingDecision:
    """
    Resultado del etiquetado de una ventana.

    Attributes
    ----------
    label : int | None
        Etiqueta asignada:
        - 1 para seizure
        - 0 para non-seizure
        - None si la ventana debe descartarse
    keep : bool
        Indica si la ventana se conserva en el dataset final.
    reason : str
        Motivo de la decisión tomada.
    """

    label: int | None
    keep: bool
    reason: str


def _validate_overlap_ratio(overlap_ratio: float) -> None:
    """
    Valida que el overlap_ratio esté en el rango [0, 1].
    """
    if overlap_ratio < 0.0 or overlap_ratio > 1.0:
        raise LabelingError(
            f"'overlap_ratio' debe estar entre 0 y 1. Valor recibido: {overlap_ratio}"
        )


def assign_label(
    overlap_ratio: float,
    positive_overlap_threshold: float = 0.5,
    drop_partial_overlap: bool = True,
) -> LabelingDecision:
    """
    Asigna una etiqueta a una ventana a partir de su ratio de solape con crisis.

    Parameters
    ----------
    overlap_ratio : float
        Fracción de la ventana que solapa con crisis.
    positive_overlap_threshold : float, default=0.5
        Umbral a partir del cual una ventana se considera positiva.
    drop_partial_overlap : bool, default=True
        Si es True, las ventanas con solape parcial pero menor al umbral se descartan.
        Si es False, esas ventanas se asignan como negativas.

    Returns
    -------
    LabelingDecision
        Decisión de etiquetado.

    Raises
    ------
    LabelingError
        Si los parámetros son inválidos.
    """
    _validate_overlap_ratio(overlap_ratio)

    if positive_overlap_threshold <= 0.0 or positive_overlap_threshold > 1.0:
        raise LabelingError("'positive_overlap_threshold' debe estar en el rango (0, 1].")

    if overlap_ratio >= positive_overlap_threshold:
        return LabelingDecision(
            label=1,
            keep=True,
            reason="positive_overlap_threshold_reached",
        )

    if overlap_ratio == 0.0:
        return LabelingDecision(
            label=0,
            keep=True,
            reason="no_overlap",
        )

    if drop_partial_overlap:
        return LabelingDecision(
            label=None,
            keep=False,
            reason="partial_overlap_dropped",
        )

    return LabelingDecision(
        label=0,
        keep=True,
        reason="partial_overlap_assigned_negative",
    )
