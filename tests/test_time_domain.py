import numpy as np
import pytest

from neuro_eeg_cdss.features.time_domain import (
    FeatureComputationError,
    compute_line_length,
    compute_mean,
    compute_rms,
    compute_std,
)


def test_compute_mean():
    signal = np.array([1.0, 2.0, 3.0])
    assert compute_mean(signal) == 2.0


def test_compute_std():
    signal = np.array([1.0, 2.0, 3.0])
    assert np.isclose(compute_std(signal), np.std(signal))


def test_compute_rms():
    signal = np.array([3.0, 4.0])
    expected = np.sqrt((9.0 + 16.0) / 2.0)
    assert np.isclose(compute_rms(signal), expected)


def test_compute_line_length():
    signal = np.array([1.0, 3.0, 2.0, 5.0])
    expected = abs(3.0 - 1.0) + abs(2.0 - 3.0) + abs(5.0 - 2.0)
    assert np.isclose(compute_line_length(signal), expected)


def test_raises_for_empty_signal():
    with pytest.raises(FeatureComputationError):
        compute_mean(np.array([]))


def test_raises_for_non_1d_signal():
    with pytest.raises(FeatureComputationError):
        compute_mean(np.array([[1.0, 2.0], [3.0, 4.0]]))
