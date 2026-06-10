"""Lightweight data validation helpers.

These checks guard against common data problems: missing columns, malformed
probability values, and accidental target leakage in feature sets. Each helper
raises a clear ``ValueError`` so failures are easy to diagnose.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Ensure a DataFrame contains all required columns.

    Args:
        df: The DataFrame to check.
        required_columns: Columns that must be present.
        dataset_name: Human-readable dataset name used in error messages.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"[{dataset_name}] is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def validate_probability_column(df: pd.DataFrame, column_name: str) -> None:
    """Validate that a column contains probabilities in the range [0, 1].

    Args:
        df: The DataFrame containing the column.
        column_name: Name of the probability column to validate.

    Raises:
        ValueError: If the column is missing, non-numeric, contains nulls, or
            has values outside the inclusive range [0, 1].
    """
    if column_name not in df.columns:
        raise ValueError(f"Probability column '{column_name}' not found.")

    series = pd.to_numeric(df[column_name], errors="coerce")

    if series.isna().any():
        raise ValueError(
            f"Probability column '{column_name}' contains null or "
            f"non-numeric values."
        )

    if (series < 0).any() or (series > 1).any():
        out_of_range = series[(series < 0) | (series > 1)]
        raise ValueError(
            f"Probability column '{column_name}' has values outside [0, 1]: "
            f"{out_of_range.tolist()}"
        )


def validate_prediction_probabilities(probabilities: Sequence[float]) -> None:
    """Ensure model output probabilities are valid for scoring.

    Args:
        probabilities: Predicted home-win probabilities (one per row).

    Raises:
        ValueError: If any value is null/non-numeric or outside ``[0, 1]``.
    """
    series = pd.to_numeric(pd.Series(probabilities), errors="coerce")
    if series.isna().any():
        raise ValueError(
            "Prediction probabilities contain null or non-numeric values."
        )
    if (series < 0).any() or (series > 1).any():
        bad = series[(series < 0) | (series > 1)]
        raise ValueError(
            f"Prediction probabilities outside [0, 1]: {bad.tolist()[:5]}"
        )


def validate_no_target_leakage(
    feature_columns: Iterable[str],
    forbidden_columns: Iterable[str],
) -> None:
    """Ensure feature columns do not include any forbidden (leaky) columns.

    Args:
        feature_columns: Columns intended to be used as model inputs.
        forbidden_columns: Columns that would leak the target (e.g. final
            scores or the outcome label).

    Raises:
        ValueError: If any forbidden column appears in the feature set.
    """
    feature_set = set(feature_columns)
    forbidden_set = set(forbidden_columns)
    leaks = sorted(feature_set & forbidden_set)
    if leaks:
        raise ValueError(
            f"Potential target leakage detected. Forbidden columns present in "
            f"features: {leaks}"
        )
