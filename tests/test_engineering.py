"""Tests for the feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_system_design.features.engineering import DateFeatureExtractor


@pytest.fixture
def sample_dates() -> pd.DataFrame:
    """Return a DataFrame with a date column spanning 60 days."""
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60, freq="D")})


def test_date_extractor_adds_expected_features(sample_dates: pd.DataFrame) -> None:
    """Transform must append exactly five engineered columns."""
    transformer = DateFeatureExtractor("date").fit(sample_dates)
    result = transformer.transform(sample_dates)

    assert len(transformer.feature_names_) == 5
    for column in transformer.feature_names_:
        assert column in result.columns


def test_cyclical_features_are_bounded(sample_dates: pd.DataFrame) -> None:
    """Cyclical features must live in the [-1, 1] interval."""
    result = DateFeatureExtractor("date").fit_transform(sample_dates)

    for column in ("day_sin", "day_cos", "month_sin", "month_cos"):
        assert np.all((result[column] >= -1.0) & (result[column] <= 1.0))


def test_days_elapsed_is_normalized(sample_dates: pd.DataFrame) -> None:
    """Normalized elapsed days must be in [0, 1] with max equal to 1."""
    result = DateFeatureExtractor("date").fit_transform(sample_dates)

    assert result["days_elapsed"].min() >= 0.0
    assert result["days_elapsed"].max() == pytest.approx(1.0)


def test_date_extractor_rejects_missing_column() -> None:
    """A missing date column must raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        DateFeatureExtractor("date").fit(pd.DataFrame({"other": [1, 2, 3]}))
