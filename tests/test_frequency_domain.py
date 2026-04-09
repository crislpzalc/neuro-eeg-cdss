import numpy as np
import pytest

from neuro_eeg_cdss.features.frequency_domain import (
    FrequencyFeatureError,
    compute_bandpower,
    compute_standard_bandpowers,
)


def test_compute_bandpower_returns_non_negative_value():
    sfreq = 100.0
    t = np.arange(0, 2, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)

    power = compute_bandpower(signal, sfreq=sfreq, fmin=8.0, fmax=13.0)

    assert power >= 0.0


def test_alpha_power_is_higher_for_10hz_signal_than_delta_power():
    sfreq = 100.0
    t = np.arange(0, 2, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)

    alpha_power = compute_bandpower(signal, sfreq=sfreq, fmin=8.0, fmax=13.0)
    delta_power = compute_bandpower(signal, sfreq=sfreq, fmin=0.5, fmax=4.0)

    assert alpha_power > delta_power


def test_compute_standard_bandpowers_returns_expected_keys():
    sfreq = 100.0
    t = np.arange(0, 2, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)

    bandpowers = compute_standard_bandpowers(signal, sfreq=sfreq)

    assert set(bandpowers.keys()) == {
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
    }


def test_compute_bandpower_relative_is_between_zero_and_one():
    sfreq = 100.0
    t = np.arange(0, 2, 1 / sfreq)
    signal = np.sin(2 * np.pi * 10 * t)

    power = compute_bandpower(
        signal,
        sfreq=sfreq,
        fmin=8.0,
        fmax=13.0,
        relative=True,
    )

    assert 0.0 <= power <= 1.0


def test_raises_for_invalid_sampling_frequency():
    with pytest.raises(FrequencyFeatureError):
        compute_bandpower(
            np.array([1.0, 2.0, 3.0]),
            sfreq=0.0,
            fmin=8.0,
            fmax=13.0,
        )


def test_raises_for_invalid_band():
    with pytest.raises(FrequencyFeatureError):
        compute_bandpower(
            np.array([1.0, 2.0, 3.0]),
            sfreq=100.0,
            fmin=13.0,
            fmax=8.0,
        )
