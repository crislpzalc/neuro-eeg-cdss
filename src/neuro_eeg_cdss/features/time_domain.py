from __future__ import annotations
import numpy as np

"""
Time-domain EEG feature extraction utilities.

This module provides simple, interpretable descriptors computed from
single-channel EEG signals. These features form part of the classical
baseline and are designed to capture basic statistical and morphological
properties of the signal.

Design goals
------------
- Enforce a strict 1D signal contract for all feature computations
- Provide numerically stable and easily interpretable features
- Keep implementations simple and reproducible for baseline experiments
"""


class FeatureComputationError(ValueError):
    """Raised when a signal is invalid for feature computation."""


def _validate_1d_signal(signal: np.ndarray) -> np.ndarray:
    """
    Validate that the signal is a non-empty 1D array and return it as float.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.

    Returns
    -------
    np.ndarray
        Validated 1D signal converted to floating-point numpy array.

    Raises
    ------
    FeatureComputationError
        If the signal is not 1D or is empty.

    Notes
    -----
    All time-domain features in this module operate on a single channel at a
    time, so a strict 1D input contract is enforced here.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise FeatureComputationError(f"Signal must be 1D. Received shape: {signal.shape}")

    if signal.size == 0:
        raise FeatureComputationError("Signal cannot be empty.")

    return signal


def compute_mean(signal: np.ndarray) -> float:
    """
    Compute the mean value of the signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional signal.

    Returns
    -------
    float
        Mean value of the signal.

    Notes
    -----
    The mean reflects the average signal level and can capture baseline shifts
    or DC offsets in the EEG.
    """
    signal = _validate_1d_signal(signal)
    return float(np.mean(signal))


def compute_std(signal: np.ndarray) -> float:
    """
    Compute the standard deviation of the signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional signal.

    Returns
    -------
    float
        Standard deviation of the signal.

    Notes
    -----
    Standard deviation captures signal variability and is often correlated with
    amplitude changes during abnormal activity such as seizures.
    """
    signal = _validate_1d_signal(signal)
    return float(np.std(signal))


def compute_rms(signal: np.ndarray) -> float:
    """
    Compute the root mean square (RMS) of the signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional signal.

    Returns
    -------
    float
        RMS value of the signal.

    Notes
    -----
    RMS reflects the energy content of the signal and is particularly useful
    for capturing sustained increases in amplitude.
    """
    signal = _validate_1d_signal(signal)
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_line_length(signal: np.ndarray) -> float:
    """
    Compute the line length of the signal.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional signal.

    Returns
    -------
    float
        Line length value.

    Notes
    -----
    Line length is defined as the sum of absolute differences between
    consecutive samples. It is a widely used feature in seizure detection as it
    captures both amplitude and frequency-related changes in the waveform.
    """
    signal = _validate_1d_signal(signal)
    return float(np.sum(np.abs(np.diff(signal))))
