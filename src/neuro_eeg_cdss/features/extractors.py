"""
Utilities for extracting handcrafted baseline features from multichannel EEG
segments.

This module provides channel-wise feature extraction for the classical
baseline, combining time-domain descriptors and frequency-domain bandpower
features into a flat tabular representation suitable for downstream machine
learning.

Design goals
------------
- Enforce a consistent multichannel signal shape across all feature extractors
- Keep per-channel feature naming deterministic and serialization-friendly
- Make the baseline feature set modular and easy to extend
- Preserve a clear separation between signal validation, time-domain features,
  and frequency-domain features
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neuro_eeg_cdss.features.frequency_domain import compute_standard_bandpowers
from neuro_eeg_cdss.features.time_domain import (
    compute_line_length,
    compute_mean,
    compute_rms,
    compute_std,
)


class FeatureExtractionError(ValueError):
    """Raised when features cannot be extracted from an EEG window."""


def _validate_multichannel_signal(signal: np.ndarray) -> np.ndarray:
    """
    Validate that the signal has shape ``(n_channels, n_samples)``.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.

    Returns
    -------
    np.ndarray
        Validated signal converted to floating-point numpy array.

    Raises
    ------
    FeatureExtractionError
        If the signal does not have the expected shape or is empty.

    Notes
    -----
    All downstream feature extractors assume a 2D array where rows correspond
    to channels and columns correspond to temporal samples. This validation is
    centralized here to enforce a single input contract.
    """
    signal = np.asarray(signal, dtype=float)

    if signal.ndim != 2:
        raise FeatureExtractionError(
            f"Multichannel signal must be 2D with shape "
            f"(n_channels, n_samples). Received shape: {signal.shape}"
        )

    n_channels, n_samples = signal.shape

    if n_channels == 0:
        raise FeatureExtractionError("Signal must contain at least one channel.")

    if n_samples == 0:
        raise FeatureExtractionError("Signal must contain at least one sample per channel.")

    return signal


def _default_channel_names(n_channels: int) -> list[str]:
    """
    Generate default channel names: ``ch_01``, ``ch_02``, ..., ``ch_N``.

    Parameters
    ----------
    n_channels : int
        Number of channels.

    Returns
    -------
    list[str]
        Deterministic default channel names.

    Notes
    -----
    Default names are used only when explicit channel names are not provided.
    This keeps feature column generation stable even in generic test settings.
    """
    return [f"ch_{idx:02d}" for idx in range(1, n_channels + 1)]


def extract_time_domain_features_per_channel(
    signal: np.ndarray,
    channel_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """
    Extract time-domain features for each channel.

    Parameters
    ----------
    signal : np.ndarray
        Signal with shape ``(n_channels, n_samples)``.
    channel_names : Sequence[str] | None
        Channel names. If not provided, they are generated automatically.

    Returns
    -------
    dict[str, float]
        Dictionary of per-channel time-domain features.

    Notes
    -----
    The current baseline includes simple descriptive statistics and line
    length, which together provide a compact characterization of signal level,
    variability, energy, and waveform complexity.
    """
    signal = _validate_multichannel_signal(signal)
    n_channels, _ = signal.shape

    if channel_names is None:
        channel_names = _default_channel_names(n_channels)

    if len(channel_names) != n_channels:
        raise FeatureExtractionError("Number of channel names does not match number of channels.")

    features: dict[str, float] = {}

    # Features are computed independently for each channel to preserve the
    # spatial granularity of the original montage in the tabular baseline.
    for ch_idx, ch_name in enumerate(channel_names):
        channel_signal = signal[ch_idx]

        features[f"mean_{ch_name}"] = compute_mean(channel_signal)
        features[f"std_{ch_name}"] = compute_std(channel_signal)
        features[f"rms_{ch_name}"] = compute_rms(channel_signal)
        features[f"line_length_{ch_name}"] = compute_line_length(channel_signal)

    return features


def extract_frequency_domain_features_per_channel(
    signal: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] | None = None,
    relative_bandpower: bool = False,
) -> dict[str, float]:
    """
    Extract standard bandpower features for each channel.

    Parameters
    ----------
    signal : np.ndarray
        Signal with shape ``(n_channels, n_samples)``.
    sfreq : float
        Sampling frequency.
    channel_names : Sequence[str] | None
        Channel names.
    relative_bandpower : bool, default=False
        If True, compute relative power in each band.

    Returns
    -------
    dict[str, float]
        Dictionary of per-channel frequency-domain features.

    Notes
    -----
    Frequency features are computed independently per channel using the
    standard EEG bands returned by ``compute_standard_bandpowers``. The naming
    convention is kept flat to facilitate direct export to parquet and use in
    classical ML pipelines.
    """
    signal = _validate_multichannel_signal(signal)
    n_channels, _ = signal.shape

    if channel_names is None:
        channel_names = _default_channel_names(n_channels)

    if len(channel_names) != n_channels:
        raise FeatureExtractionError("Number of channel names does not match number of channels.")

    features: dict[str, float] = {}

    # Bandpower features are computed channel by channel so that each output
    # column remains directly attributable to one spatial source.
    for ch_idx, ch_name in enumerate(channel_names):
        channel_signal = signal[ch_idx]

        bandpowers = compute_standard_bandpowers(
            channel_signal,
            sfreq=sfreq,
            relative=relative_bandpower,
        )

        for band_name, value in bandpowers.items():
            features[f"{band_name}_{ch_name}"] = value

    return features


def extract_all_features_per_channel(
    signal: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str] | None = None,
    relative_bandpower: bool = False,
) -> dict[str, float]:
    """
    Extract all baseline features per channel:
    - mean
    - std
    - rms
    - line length
    - delta/theta/alpha/beta bandpower

    Parameters
    ----------
    signal : np.ndarray
        Signal with shape ``(n_channels, n_samples)``.
    sfreq : float
        Sampling frequency.
    channel_names : Sequence[str] | None
        Channel names.
    relative_bandpower : bool, default=False
        If True, compute relative bandpower values.

    Returns
    -------
    dict[str, float]
        Combined feature dictionary.

    Notes
    -----
    This function is the main entry point for handcrafted baseline extraction.
    It merges time-domain and frequency-domain descriptors into a single flat
    feature mapping with deterministic column names.
    """
    time_features = extract_time_domain_features_per_channel(
        signal=signal,
        channel_names=channel_names,
    )

    frequency_features = extract_frequency_domain_features_per_channel(
        signal=signal,
        sfreq=sfreq,
        channel_names=channel_names,
        relative_bandpower=relative_bandpower,
    )

    # The final feature representation is intentionally flat so it can be used
    # directly by tabular modeling pipelines without additional reshaping.
    return {**time_features, **frequency_features}
