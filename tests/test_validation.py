"""Tests for serving and validation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml_system_design.serving.validation import psi, validate_schema


def test_psi_is_zero_for_identical_distributions() -> None:
    """PSI must be ~0 when both distributions are identical."""
    rng = np.random.default_rng(42)
    data = pd.Series(rng.normal(0.0, 1.0, 10_000))

    assert psi(data, data) < 1e-6


def test_psi_grows_with_distribution_shift() -> None:
    """A shifted distribution must produce a larger PSI than an identical one."""
    rng = np.random.default_rng(7)
    base = pd.Series(rng.normal(0.0, 1.0, 10_000))
    shifted = pd.Series(rng.normal(2.0, 1.0, 10_000))

    assert psi(base, shifted) > psi(base, base)


def test_psi_rejects_nan_values() -> None:
    """NaN input must raise ValueError."""
    with pytest.raises(ValueError, match="NaN"):
        psi(pd.Series([1.0, np.nan]), pd.Series([1.0, 2.0]))


def test_validate_schema_reports_missing_columns() -> None:
    """Missing required columns must be reported."""
    df = pd.DataFrame({"a": [1], "b": [2]})

    assert validate_schema(df, {"a", "b", "c"}) == ["c"]
    assert validate_schema(df, {"a", "b"}) == []
