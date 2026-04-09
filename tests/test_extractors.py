import numpy as np
import pytest

from neuro_eeg_cdss.features.extractors import (
    FeatureExtractionError,
    extract_all_features_per_channel,
    extract_frequency_domain_features_per_channel,
    extract_time_domain_features_per_channel,
)


def test_extract_time_domain_features_per_channel_returns_expected_keys():
    signal = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    features = extract_time_domain_features_per_channel(signal)

    expected_keys = {
        "mean_ch_01",
        "std_ch_01",
        "rms_ch_01",
        "line_length_ch_01",
        "mean_ch_02",
        "std_ch_02",
        "rms_ch_02",
        "line_length_ch_02",
    }

    assert set(features.keys()) == expected_keys


def test_extract_frequency_domain_features_per_channel_returns_expected_keys():
    sfreq = 100.0
    t = np.arange(0, 2, 1 / sfreq)
    signal = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 5 * t),
        ]
    )

    features = extract_frequency_domain_features_per_channel(signal, sfreq=sfreq)

    expected_keys = {
        "delta_power_ch_01",
        "theta_power_ch_01",
        "alpha_power_ch_01",
        "beta_power_ch_01",
        "delta_power_ch_02",
        "theta_power_ch_02",
        "alpha_power_ch_02",
        "beta_power_ch_02",
    }

    assert set(features.keys()) == expected_keys


def test_extract_all_features_per_channel_combines_time_and_frequency():
    sfreq = 100.0
    t = np.arange(0, 2, 1 / sfreq)
    signal = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 5 * t),
        ]
    )

    features = extract_all_features_per_channel(signal, sfreq=sfreq)

    assert "mean_ch_01" in features
    assert "line_length_ch_02" in features
    assert "alpha_power_ch_01" in features
    assert "theta_power_ch_02" in features


def test_extractors_raise_for_non_2d_signal():
    with pytest.raises(FeatureExtractionError):
        extract_all_features_per_channel(
            signal=np.array([1.0, 2.0, 3.0]),
            sfreq=100.0,
        )


def test_extractors_raise_if_channel_names_do_not_match():
    signal = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )

    with pytest.raises(FeatureExtractionError):
        extract_time_domain_features_per_channel(
            signal=signal,
            channel_names=["Fp1-F7"],
        )
