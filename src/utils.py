"""Reusable helper functions for the NBA Win Probability Engine.

This module keeps small, dependency-light utilities for working with the
project's folder structure and CSV files. Heavier logic (feature engineering,
modeling) lives elsewhere in later builds.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd

from . import config


def ensure_directories() -> None:
    """Create every directory the project relies on.

    Safe to call repeatedly; existing folders are left untouched. This is the
    single source of truth for "the project layout exists on disk".
    """
    for directory in config.ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def load_csv(
    path: str | Path,
    required_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load a CSV file into a DataFrame with friendly error handling.

    Args:
        path: Path to the CSV file.
        required_columns: Optional list of columns that must be present. When
            provided, a ``ValueError`` is raised if any are missing.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"CSV file is empty: {path}") from exc

    if required_columns is not None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {path} is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a DataFrame to CSV, creating parent folders as needed.

    Args:
        df: The DataFrame to write.
        path: Destination CSV path.

    Returns:
        The path that was written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def append_or_update_csv(
    df: pd.DataFrame,
    path: str | Path,
    key_columns: Iterable[str],
) -> pd.DataFrame:
    """Append new rows and update existing rows in a CSV based on key columns.

    Rows in ``df`` whose key values already exist in the file replace the old
    rows; rows with new key values are appended. If the file does not exist yet,
    it is created from ``df``.

    Args:
        df: New or updated rows.
        path: CSV file to append to or update.
        key_columns: Columns that together uniquely identify a row.

    Returns:
        The combined DataFrame that was written to disk.

    Raises:
        ValueError: If any key column is missing from ``df``.
    """
    path = Path(path)
    key_columns = list(key_columns)

    missing_keys = [col for col in key_columns if col not in df.columns]
    if missing_keys:
        raise ValueError(
            f"Key columns missing from new data: {missing_keys}. "
            f"Available columns: {list(df.columns)}"
        )

    if path.exists():
        # Read key columns as strings so identifiers keep a stable, comparable
        # form. Without this, an id like "0022400001" is re-read from CSV as the
        # integer 22400001, which would never match the freshly-built string id
        # and would defeat de-duplication (creating duplicate rows).
        existing = pd.read_csv(path, dtype={col: str for col in key_columns})
        df = df.copy()
        for col in key_columns:
            df[col] = df[col].astype(str)
        combined = pd.concat([existing, df], ignore_index=True)
        # Keep the last occurrence so the incoming rows win on conflicts.
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    else:
        combined = df.copy()

    return_df = combined.reset_index(drop=True)
    save_csv(return_df, path)
    return return_df


def log_api_attempt(
    game_id: str,
    endpoint: str,
    status: str,
    error_message: str = "",
) -> None:
    """Record a single data-refresh / API attempt to the refresh log CSV.

    The log lives at ``data/logs/data_refresh_log.csv`` and is appended to on
    every call. Useful later when the NBA API is wired in, but available now so
    the contract is stable.

    Args:
        game_id: Identifier for the game involved (may be empty for batch jobs).
        endpoint: Name of the endpoint or operation attempted.
        status: Outcome string, e.g. ``"success"`` or ``"error"``.
        error_message: Optional error detail when the attempt failed.
    """
    entry = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "game_id": game_id,
                "endpoint": endpoint,
                "status": status,
                "error_message": error_message,
            }
        ]
    )

    path = config.DATA_REFRESH_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    header = not path.exists()
    entry.to_csv(path, mode="a", header=header, index=False)


def compute_winner(home_score: int, away_score: int) -> str:
    """Determine the winner of a game from its final scores.

    Args:
        home_score: Final score for the home team.
        away_score: Final score for the away team.

    Returns:
        ``"home"`` if the home team scored more, ``"away"`` if the away team
        scored more, or ``"tie"`` when the scores are equal.
    """
    if home_score == away_score:
        return "tie"
    return "home" if home_score > away_score else "away"


def safe_sleep(seconds: float) -> None:
    """Sleep for ``seconds``, ignoring non-positive values.

    A thin wrapper around :func:`time.sleep` that protects against negative or
    ``None``-like values so rate-limiting code stays simple at call sites.

    Args:
        seconds: Number of seconds to sleep. Values <= 0 are skipped.
    """
    if seconds and seconds > 0:
        time.sleep(seconds)
