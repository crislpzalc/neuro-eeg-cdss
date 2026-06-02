import numpy as np
import pandas as pd
import pytest

from neuro_eeg_cdss.training.trainer import (
    TrainConfig,
    TrainerError,
    load_train_result,
    predict,
    save_train_result,
    separate_features_and_labels,
    train_model,
)


def _make_train_df(
    n_positive: int = 50, n_negative: int = 500, n_features: int = 10
) -> pd.DataFrame:
    """Create a synthetic training DataFrame with metadata and features."""
    rng = np.random.RandomState(42)
    n_total = n_positive + n_negative

    data: dict = {
        "subject": [f"sub-{i % 5:02d}" for i in range(n_total)],
        "session": ["ses-01"] * n_total,
        "run": ["run-00"] * n_total,
        "path": ["fake/path.edf"] * n_total,
        "start_sec": [float(i * 5) for i in range(n_total)],
        "end_sec": [float(i * 5 + 5) for i in range(n_total)],
        "label": [1] * n_positive + [0] * n_negative,
    }

    # Positive samples: higher feature values to make the problem learnable.
    for j in range(n_features):
        pos_values = rng.normal(loc=2.0, scale=1.0, size=n_positive)
        neg_values = rng.normal(loc=0.0, scale=1.0, size=n_negative)
        data[f"feat_{j:02d}"] = np.concatenate([pos_values, neg_values])

    return pd.DataFrame(data)


# --- separate_features_and_labels ---


def test_separate_features_basic():
    df = _make_train_df()
    X, y, feature_names = separate_features_and_labels(df)

    assert len(feature_names) == 10
    assert all(col.startswith("feat_") for col in feature_names)
    assert len(X) == len(y) == len(df)
    assert set(y.unique()) == {0, 1}


def test_separate_features_missing_label():
    df = pd.DataFrame({"subject": ["A"], "feat_0": [1.0]})

    with pytest.raises(TrainerError):
        separate_features_and_labels(df)


def test_separate_features_no_features():
    df = pd.DataFrame({"subject": ["A"], "label": [0]})

    with pytest.raises(TrainerError):
        separate_features_and_labels(df)


# --- train_model ---


def test_train_logistic_regression():
    df = _make_train_df()
    config = TrainConfig(model_type="logistic_regression", seed=42)
    result = train_model(df, config=config)

    assert result.model is not None
    assert result.scaler is not None
    assert result.train_shape == (550, 10)
    assert result.train_positive_count == 50
    assert result.train_negative_count == 500


def test_train_random_forest():
    df = _make_train_df()
    config = TrainConfig(model_type="random_forest", seed=42)
    result = train_model(df, config=config)

    assert result.model is not None
    assert result.train_shape == (550, 10)


def test_train_unsupported_model():
    df = _make_train_df()
    config = TrainConfig(model_type="xgboost", seed=42)

    with pytest.raises(TrainerError):
        train_model(df, config=config)


def test_train_empty_dataset():
    df = _make_train_df().head(0)

    with pytest.raises(TrainerError):
        train_model(df)


def test_train_no_positives():
    df = _make_train_df(n_positive=0, n_negative=100)

    with pytest.raises(TrainerError):
        train_model(df)


def test_train_deterministic():
    df = _make_train_df()
    config = TrainConfig(model_type="logistic_regression", seed=42)

    r1 = train_model(df, config=config)
    r2 = train_model(df, config=config)

    y1, p1 = predict(r1, df)
    y2, p2 = predict(r2, df)

    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_almost_equal(p1, p2)


def test_train_without_scaling():
    df = _make_train_df()
    config = TrainConfig(scale_features=False, seed=42)
    result = train_model(df, config=config)

    assert result.scaler is None
    assert result.model is not None


# --- predict ---


def test_predict_basic():
    df = _make_train_df()
    result = train_model(df)

    y_pred, y_proba = predict(result, df)

    assert len(y_pred) == len(df)
    assert len(y_proba) == len(df)
    assert set(y_pred).issubset({0, 1})
    assert all(0.0 <= p <= 1.0 for p in y_proba)


def test_predict_missing_features():
    df = _make_train_df()
    result = train_model(df)

    # Remove a feature column
    df_bad = df.drop(columns=["feat_00"])

    with pytest.raises(TrainerError):
        predict(result, df_bad)


# --- save / load round-trip ---


def test_save_and_load_round_trip(tmp_path):
    df = _make_train_df()
    config = TrainConfig(model_type="random_forest", seed=42)
    result = train_model(df, config=config)

    save_train_result(result, tmp_path)
    loaded = load_train_result(tmp_path)

    assert loaded.config.model_type == "random_forest"
    assert loaded.config.seed == 42
    assert loaded.feature_names == result.feature_names
    assert loaded.train_positive_count == result.train_positive_count

    # Predictions should match
    y_orig, p_orig = predict(result, df)
    y_load, p_load = predict(loaded, df)

    np.testing.assert_array_equal(y_orig, y_load)
    np.testing.assert_array_almost_equal(p_orig, p_load)


def test_load_missing_model(tmp_path):
    with pytest.raises(TrainerError):
        load_train_result(tmp_path)
