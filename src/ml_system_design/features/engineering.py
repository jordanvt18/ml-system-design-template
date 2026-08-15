"""Feature engineering utilities for tabular ML pipelines.

This module provides reusable scikit-learn-compatible transformers that can
be composed inside a ``Pipeline`` for production deployments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract cyclical and elapsed-time features from a datetime column.

    Args:
        date_column: Name of the datetime column to transform.
        normalize_days: Whether to normalize ``days_elapsed`` to [0, 1].

    Attributes:
        feature_names_: Names of the features generated during ``fit``.
    """

    def __init__(self, date_column: str, normalize_days: bool = True) -> None:
        """Initialize the transformer with the target datetime column."""
        self.date_column = date_column
        self.normalize_days = normalize_days
        self.feature_names_: list[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> DateFeatureExtractor:
        """Validate the input and precompute feature names.

        Args:
            X: Input DataFrame that must contain ``date_column``.
            y: Ignored; kept for scikit-learn API compatibility.

        Returns:
            The fitted transformer.

        Raises:
            ValueError: If ``date_column`` is not present in ``X``.
        """
        if self.date_column not in X.columns:
            raise ValueError(f"Column '{self.date_column}' not found in input.")
        self.feature_names_ = self._feature_names()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Append cyclical and elapsed-time features to ``X``.

        Args:
            X: Input DataFrame.

        Returns:
            A copy of ``X`` with the engineered features appended.
        """
        dates = pd.to_datetime(X[self.date_column])
        days_elapsed = (dates - dates.min()).dt.days.astype(float)
        if self.normalize_days and days_elapsed.max() > 0:
            days_elapsed = days_elapsed / days_elapsed.max()

        out = X.copy()
        out["day_sin"] = np.sin(2.0 * np.pi * dates.dt.dayofweek / 7.0)
        out["day_cos"] = np.cos(2.0 * np.pi * dates.dt.dayofweek / 7.0)
        out["month_sin"] = np.sin(2.0 * np.pi * dates.dt.month / 12.0)
        out["month_cos"] = np.cos(2.0 * np.pi * dates.dt.month / 12.0)
        out["days_elapsed"] = days_elapsed
        return out

    def _feature_names(self) -> list[str]:
        """Build the list of generated feature names."""
        return ["day_sin", "day_cos", "month_sin", "month_cos", "days_elapsed"]
