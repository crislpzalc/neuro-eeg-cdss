"""
Baseline model training pipeline for seizure detection.

This module provides a reproducible training pipeline for classical machine
learning models. It handles feature/label separation, optional scaling,
model fitting, and artifact serialization.

Design goals
------------
- Keep the training interface model-agnostic
- Enforce reproducibility through explicit seed control
- Serialize all artifacts needed to reproduce or deploy a trained model
- Separate training logic from evaluation (Sprint 1E)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class TrainerError(ValueError):
    """Raised when training pipeline encounters an error."""


# ── Metadata columns ──────────────────────────────────────────────────

# These columns are preserved for traceability but excluded from the
# feature matrix. Any column that is not a feature and not the label
# must appear here.
METADATA_COLUMNS = {"subject", "session", "run", "path", "start_sec", "end_sec"}
LABEL_COLUMN = "label"


# ── Configuration ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainConfig:
    """
    Configuration for a single training run.

    Attributes
    ----------
    model_type : str
        Model identifier. Supported: ``"logistic_regression"``,
        ``"random_forest"``.
    seed : int
        Random seed for full reproducibility.
    scale_features : bool
        Whether to apply standard scaling before fitting.
    model_params : dict[str, Any]
        Additional keyword arguments forwarded to the sklearn constructor.
        ``class_weight`` defaults to ``"balanced"`` and ``random_state``
        is injected automatically from ``seed``.

    Notes
    -----
    ``class_weight="balanced"`` is critical for this dataset because
    seizure segments represent ~0.3% of total samples. Without it the
    model would learn to always predict the majority class and achieve
    99.7% accuracy while detecting zero seizures.
    """

    model_type: str = "logistic_regression"
    seed: int = 42
    scale_features: bool = True
    model_params: dict[str, Any] = field(default_factory=dict)


# ── Training result ───────────────────────────────────────────────────


@dataclass
class TrainResult:
    """
    Container for all artifacts produced by a training run.

    Attributes
    ----------
    model : Any
        Fitted sklearn estimator.
    scaler : StandardScaler | None
        Fitted scaler, or None if scaling was disabled.
    config : TrainConfig
        Configuration used for this run.
    feature_names : list[str]
        Ordered feature column names the model expects at inference.
    train_shape : tuple[int, int]
        Shape of the training feature matrix ``(n_samples, n_features)``.
    train_positive_count : int
        Number of positive samples in training data.
    train_negative_count : int
        Number of negative samples in training data.
    """

    model: Any
    scaler: StandardScaler | None
    config: TrainConfig
    feature_names: list[str]
    train_shape: tuple[int, int]
    train_positive_count: int
    train_negative_count: int


# ── Feature / label separation ────────────────────────────────────────


def separate_features_and_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Separate a feature dataset into X (features) and y (labels).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing metadata columns, feature columns and a
        ``label`` column.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary labels.
    feature_names : list[str]
        Ordered list of feature column names.

    Raises
    ------
    TrainerError
        If the label column is missing or the feature set is empty.
    """
    if LABEL_COLUMN not in df.columns:
        raise TrainerError(f"Missing '{LABEL_COLUMN}' column in dataset.")

    y = df[LABEL_COLUMN].astype(int)

    # Feature columns are everything except metadata and label.
    non_feature_cols = METADATA_COLUMNS | {LABEL_COLUMN}
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    if not feature_cols:
        raise TrainerError("No feature columns found after removing metadata and label.")

    X = df[feature_cols]
    return X, y, feature_cols


# ── Model factory ─────────────────────────────────────────────────────

_SUPPORTED_MODELS = {"logistic_regression", "random_forest"}


def _build_model(config: TrainConfig) -> Any:
    """
    Instantiate a sklearn estimator from configuration.

    Parameters
    ----------
    config : TrainConfig
        Training configuration.

    Returns
    -------
    sklearn estimator
        Unfitted model ready for ``.fit()``.

    Raises
    ------
    TrainerError
        If the model type is not supported.

    Notes
    -----
    ``class_weight`` and ``random_state`` are injected automatically.
    User-provided ``model_params`` can override ``class_weight`` but
    not ``random_state`` (which is always derived from the config seed
    for reproducibility).
    """
    if config.model_type not in _SUPPORTED_MODELS:
        raise TrainerError(
            f"Unsupported model type: '{config.model_type}'. Supported: {sorted(_SUPPORTED_MODELS)}"
        )

    # Start with balanced class weight, allow user override.
    params: dict[str, Any] = {"class_weight": "balanced"}
    params.update(config.model_params)

    # Seed is always injected for reproducibility and cannot be overridden.
    params["random_state"] = config.seed

    if config.model_type == "logistic_regression":
        params.setdefault("max_iter", 1000)
        params.setdefault("solver", "lbfgs")
        return LogisticRegression(**params)

    if config.model_type == "random_forest":
        params.setdefault("n_estimators", 200)
        params.setdefault("n_jobs", -1)
        return RandomForestClassifier(**params)

    # Unreachable due to the check above, but kept for safety.
    raise TrainerError(f"Unhandled model type: '{config.model_type}'")


# ── Training ──────────────────────────────────────────────────────────


def train_model(
    train_df: pd.DataFrame,
    config: TrainConfig | None = None,
) -> TrainResult:
    """
    Train a baseline model on a feature dataset.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with metadata, features and label columns.
    config : TrainConfig | None
        Training configuration. Uses defaults if not provided.

    Returns
    -------
    TrainResult
        All artifacts needed to evaluate or deploy the trained model.

    Raises
    ------
    TrainerError
        If the dataset is invalid or training fails.

    Notes
    -----
    The pipeline applies the following steps:

    1. Separate features and labels
    2. (Optional) Fit and apply standard scaling
    3. Set random seed for numpy (global) and sklearn (per-estimator)
    4. Fit the model
    5. Package all artifacts into a ``TrainResult``
    """
    if config is None:
        config = TrainConfig()

    X, y, feature_names = separate_features_and_labels(train_df)

    if len(X) == 0:
        raise TrainerError("Training dataset is empty.")

    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)

    if n_positive == 0:
        raise TrainerError("Training dataset contains no positive samples.")

    # Global seed for any numpy-dependent randomness during fit.
    np.random.seed(config.seed)

    scaler: StandardScaler | None = None
    X_train: np.ndarray | pd.DataFrame = X

    if config.scale_features:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X)

    model = _build_model(config)
    model.fit(X_train, y)

    return TrainResult(
        model=model,
        scaler=scaler,
        config=config,
        feature_names=feature_names,
        train_shape=(len(X), len(feature_names)),
        train_positive_count=n_positive,
        train_negative_count=n_negative,
    )


# ── Prediction ────────────────────────────────────────────────────────


def predict(
    result: TrainResult,
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions and probabilities from a trained model.

    Parameters
    ----------
    result : TrainResult
        Output of ``train_model``.
    df : pd.DataFrame
        Dataset with the same feature columns used during training.

    Returns
    -------
    y_pred : np.ndarray
        Binary predictions.
    y_proba : np.ndarray
        Probability of the positive class.

    Raises
    ------
    TrainerError
        If feature columns do not match the training schema.
    """
    missing = set(result.feature_names) - set(df.columns)
    if missing:
        raise TrainerError(f"Missing feature columns for prediction: {sorted(missing)}")

    X = df[result.feature_names]

    if result.scaler is not None:
        X = result.scaler.transform(X)

    y_pred = result.model.predict(X)
    y_proba = result.model.predict_proba(X)[:, 1]

    return y_pred, y_proba


# ── Serialization ─────────────────────────────────────────────────────


def save_train_result(result: TrainResult, output_dir: str | Path) -> None:
    """
    Save all training artifacts to disk.

    Creates:
    - ``model.pkl`` — fitted sklearn estimator
    - ``scaler.pkl`` — fitted scaler (if scaling was enabled)
    - ``train_config.json`` — full training configuration
    - ``feature_names.json`` — ordered feature column list
    - ``train_metadata.json`` — training dataset statistics

    Parameters
    ----------
    result : TrainResult
        Training result to save.
    output_dir : str | Path
        Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "model.pkl", "wb") as f:
        pickle.dump(result.model, f)

    if result.scaler is not None:
        with open(output_dir / "scaler.pkl", "wb") as f:
            pickle.dump(result.scaler, f)

    config_dict = asdict(result.config)
    with open(output_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    with open(output_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(result.feature_names, f, indent=2, ensure_ascii=False)

    metadata = {
        "train_shape": list(result.train_shape),
        "train_positive_count": result.train_positive_count,
        "train_negative_count": result.train_negative_count,
        "model_type": result.config.model_type,
        "seed": result.config.seed,
        "scale_features": result.config.scale_features,
    }
    with open(output_dir / "train_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_train_result(model_dir: str | Path) -> TrainResult:
    """
    Load a previously saved training result from disk.

    Parameters
    ----------
    model_dir : str | Path
        Directory containing saved artifacts.

    Returns
    -------
    TrainResult
        Restored training result.

    Raises
    ------
    TrainerError
        If required files are missing.
    """
    model_dir = Path(model_dir)

    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise TrainerError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)  # noqa: S301

    scaler = None
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)  # noqa: S301

    config_path = model_dir / "train_config.json"
    if not config_path.exists():
        raise TrainerError(f"Config file not found: {config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_dict = json.load(f)
    config = TrainConfig(**config_dict)

    feature_names_path = model_dir / "feature_names.json"
    if not feature_names_path.exists():
        raise TrainerError(f"Feature names file not found: {feature_names_path}")

    with open(feature_names_path, encoding="utf-8") as f:
        feature_names = json.load(f)

    metadata_path = model_dir / "train_metadata.json"
    train_shape = (0, 0)
    train_positive_count = 0
    train_negative_count = 0
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        train_shape = tuple(metadata.get("train_shape", [0, 0]))
        train_positive_count = metadata.get("train_positive_count", 0)
        train_negative_count = metadata.get("train_negative_count", 0)

    return TrainResult(
        model=model,
        scaler=scaler,
        config=config,
        feature_names=feature_names,
        train_shape=train_shape,
        train_positive_count=train_positive_count,
        train_negative_count=train_negative_count,
    )
