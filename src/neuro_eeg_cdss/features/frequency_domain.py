from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

"""
Frequency-domain EEG feature extraction utilities.

This module provides Welch-based bandpower computation for 1D EEG signals.
It is intentionally focused on simple, interpretable spectral features for
the classical baseline.

Design goals
------------
- Enforce a strict and explicit 1D signal contract
- Use deterministic spectral estimation suitable for baseline experiments
- Keep feature definitions aligned with standard EEG frequency bands
- Return numerically stable outputs for downstream tabular pipelines
"""


class FrequencyFeatureError(ValueError):
    """Raised when a frequency-domain feature cannot be computed."""


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
    FrequencyFeatureError
        If the signal is not 1D or is empty.

    Notes
    -----
    Frequency-domain features in this module operate on a single channel at a
    time, so a strict 1D input contract is enforced here.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 1:
        raise FrequencyFeatureError(f"Signal must be 1D. Received shape: {signal.shape}")

    if signal.size == 0:
        raise FrequencyFeatureError("Signal cannot be empty.")

    return signal


def _validate_sampling_frequency(sfreq: float) -> float:
    """
    Validate that the sampling frequency is positive.

    Parameters
    ----------
    sfreq : float
        Sampling frequency.

    Returns
    -------
    float
        Validated sampling frequency as float.

    Raises
    ------
    FrequencyFeatureError
        If the sampling frequency is not strictly positive.
    """
    if sfreq <= 0:
        raise FrequencyFeatureError(f"'sfreq' must be > 0. Received value: {sfreq}")
    return float(sfreq)


def compute_bandpower(
    signal: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    relative: bool = False,
) -> float:
    """
    Compute bandpower in a frequency interval using Welch PSD estimation.

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional signal.
    sfreq : float
        Sampling frequency.
    fmin : float
        Lower bound of the frequency band.
    fmax : float
        Upper bound of the frequency band.
    relative : bool, default=False
        If True, return relative bandpower with respect to total power.

    Returns
    -------
    float
        Bandpower value for the requested interval.

    Raises
    ------
    FrequencyFeatureError
        If the input parameters are invalid.

    Notes
    -----
    Power is estimated from the Welch power spectral density and integrated
    over the requested frequency range using the trapezoidal rule. Relative
    bandpower is defined as band power divided by total PSD power.
    """
    signal = _validate_1d_signal(signal)
    sfreq = _validate_sampling_frequency(sfreq)

    if fmin < 0 or fmax <= fmin:
        raise FrequencyFeatureError(f"Invalid frequency band: fmin={fmin}, fmax={fmax}")

    nyquist = sfreq / 2.0
    if fmax > nyquist:
        raise FrequencyFeatureError(f"fmax={fmax} exceeds Nyquist frequency={nyquist}")

    # Welch segment length is capped at two seconds of data to provide a simple
    # and reproducible baseline setting while adapting safely to short signals.
    nperseg = min(len(signal), int(sfreq * 2))
    if nperseg < 2:
        raise FrequencyFeatureError("Signal is too short to compute Welch PSD reliably.")

    freqs, psd = welch(signal, fs=sfreq, nperseg=nperseg)

    band_mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(band_mask):
        return 0.0

    band_power = trapezoid(psd[band_mask], freqs[band_mask])

    if not relative:
        return float(band_power)

    total_power = trapezoid(psd, freqs)
    if total_power <= 0:
        return 0.0

    return float(band_power / total_power)


def compute_standard_bandpowers(
    signal: np.ndarray,
    sfreq: float,
    relative: bool = False,
) -> dict[str, float]:
    """
    Compute the standard EEG bandpowers.

    Bands
    -----
    delta : 0.5-4 Hz
    theta : 4-8 Hz
    alpha : 8-13 Hz
    beta  : 13-30 Hz

    Parameters
    ----------
    signal : np.ndarray
        One-dimensional signal.
    sfreq : float
        Sampling frequency.
    relative : bool, default=False
        If True, compute relative rather than absolute bandpower.

    Returns
    -------
    dict[str, float]
        Mapping from feature name to bandpower value.

    Notes
    -----
    These band definitions are intentionally fixed to provide a standard,
    interpretable baseline feature set for seizure detection experiments.
    """
    return {
        "delta_power": compute_bandpower(signal, sfreq, 0.5, 4.0, relative=relative),
        "theta_power": compute_bandpower(signal, sfreq, 4.0, 8.0, relative=relative),
        "alpha_power": compute_bandpower(signal, sfreq, 8.0, 13.0, relative=relative),
        "beta_power": compute_bandpower(signal, sfreq, 13.0, 30.0, relative=relative),
    }
