"""Validation utilities for model serving and drift monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Compute the Population Stability Index (PSI).

    The PSI quantifies how much the distribution of a score or feature has
    shifted between a reference (expected) and a current (actual) window.
    Rule of thumb: <0.1 stable, 0.1-0.25 moderate drift, >0.25 significant.

    Args:
        expected: Reference distribution (e.g., training window).
        actual: Current distribution (e.g., production window).
        bins: Number of buckets used to discretize both distributions.

    Returns:
        The PSI value.

    Raises:
        ValueError: If either series is empty or contains NaN values.
    """
    if expected.empty or actual.empty:
        raise ValueError("expected and actual must be non-empty.")
    if expected.isna().any() or actual.isna().any():
        raise ValueError("NaN values are not allowed; impute before calling psi.")

    edges = np.quantile(expected, np.linspace(0.0, 1.0, bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)

    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def validate_schema(df: pd.DataFrame, required_columns: set[str]) -> list[str]:
    """Return the required columns missing from ``df``.

    Args:
        df: DataFrame to validate against the model contract.
        required_columns: Columns the model contract requires.

    Returns:
        List of missing column names; empty when the schema is complete.
    """
    return sorted(required_columns - set(df.columns))
