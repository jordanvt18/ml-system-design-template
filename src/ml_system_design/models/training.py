"""Utilities for training and evaluating supervised models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class EvaluationReport:
    """Container for model evaluation metrics.

    Attributes:
        mae: Mean absolute error on the held-out set.
        rmse: Root mean squared error on the held-out set.
        r2: Coefficient of determination on the held-out set.
        cv_scores: R2 per fold from cross-validation.
    """

    mae: float
    rmse: float
    r2: float
    cv_scores: list[float]


def build_pipeline(
    categorical_features: list[str], numeric_features: list[str]
) -> Pipeline:
    """Build a preprocessing and regression pipeline.

    Args:
        categorical_features: Columns to one-hot encode.
        numeric_features: Columns to standardize.

    Returns:
        A scikit-learn ``Pipeline`` ready for fitting and prediction.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate_model(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationReport:
    """Compute held-out metrics and cross-validated stability.

    Args:
        model: A fitted pipeline.
        X_train: Training features.
        y_train: Training target.
        X_test: Held-out features.
        y_test: Held-out target.

    Returns:
        An :class:`EvaluationReport` with point metrics and CV scores.
    """
    y_pred = model.predict(X_test)
    cv = cross_validate(model, X_train, y_train, cv=5, scoring="r2", n_jobs=-1)
    return EvaluationReport(
        mae=float(mean_absolute_error(y_test, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
        r2=float(r2_score(y_test, y_pred)),
        cv_scores=[float(score) for score in cv["test_score"]],
    )
