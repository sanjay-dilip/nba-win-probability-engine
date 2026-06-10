"""Manual post-game result override helpers (Build 12).

Manual results are stored in ``data/manual/postgame_results.csv`` for tracking,
evaluation, and future safe enrichment. They are **not** merged into model
feature files in this build.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd

from . import config
from .utils import save_csv

# Allowed source values for manual post-game results.
VALID_SOURCES = frozenset({"manual", "corrected_manual", "api"})

REQUIRED_POSTGAME_RESULT_COLUMNS = [
    "game_id",
    "home_score",
    "away_score",
    "winner",
    "source",
    "confirmed_at",
    "notes",
]

ComparisonStatus = Literal[
    "match",
    "mismatch",
    "no_official_result_available",
    "no_manual_result_available",
]


def normalize_source(source: str) -> str:
    """Return a trimmed, lower-case source string."""
    return str(source).strip().lower()


def validate_source(source: str) -> str:
    """Validate and return a normalized source value.

    Raises:
        ValueError: If ``source`` is not one of the allowed values.
    """
    normalized = normalize_source(source)
    if normalized not in VALID_SOURCES:
        allowed = ", ".join(sorted(VALID_SOURCES))
        raise ValueError(
            f"Invalid source '{source}'. Allowed values: {allowed}."
        )
    return normalized


def validate_scores(home_score: int, away_score: int) -> None:
    """Ensure final scores are valid non-negative integers with no tie.

    Raises:
        ValueError: If scores are invalid or tied.
        TypeError: If scores are not integers.
    """
    for label, score in (("home_score", home_score), ("away_score", away_score)):
        if not isinstance(score, int):
            raise TypeError(f"{label} must be an integer, got {type(score).__name__}.")
        if score < 0:
            raise ValueError(f"{label} must be non-negative, got {score}.")
    if home_score == away_score:
        raise ValueError(
            f"Tie scores are invalid for NBA final results ({home_score}-{away_score})."
        )


def compute_winner_from_scores(
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
) -> str:
    """Return the winning team name from scores and team labels."""
    validate_scores(home_score, away_score)
    return home_team if home_score > away_score else away_team


def _ensure_notes_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add an empty ``notes`` column when loading legacy CSVs."""
    out = df.copy()
    if "notes" not in out.columns:
        out["notes"] = ""
    return out


def validate_postgame_results_dataframe(df: pd.DataFrame) -> None:
    """Ensure a manual-results DataFrame has all required columns.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = [col for col in REQUIRED_POSTGAME_RESULT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"postgame_results.csv is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )


def load_postgame_results(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load manual post-game results, keeping ``game_id`` as a string.

    Returns an empty DataFrame with the expected columns when the file is
    missing. Legacy files without ``notes`` get an empty notes column added
    in memory (the file on disk is not modified until the next save).
    """
    csv_path = Path(path) if path is not None else config.POSTGAME_RESULTS_PATH
    empty = pd.DataFrame(columns=REQUIRED_POSTGAME_RESULT_COLUMNS)

    if not csv_path.exists():
        return empty

    try:
        df = pd.read_csv(csv_path, dtype={"game_id": str})
    except pd.errors.EmptyDataError:
        return empty

    if df.empty:
        return empty

    df = _ensure_notes_column(df)
    validate_postgame_results_dataframe(df)
    return df.reset_index(drop=True)


def get_existing_result(
    game_id: str,
    path: Optional[Union[str, Path]] = None,
) -> Optional[pd.Series]:
    """Return the manual result row for ``game_id``, or ``None`` if absent."""
    df = load_postgame_results(path)
    if df.empty:
        return None
    match = df.loc[df["game_id"] == str(game_id)]
    if match.empty:
        return None
    return match.iloc[0]


def build_manual_result_record(
    game_id: str,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    source: str,
    notes: str = "",
    confirmed_at: Optional[str] = None,
) -> dict:
    """Build one validated manual result record ready for upsert."""
    validate_scores(home_score, away_score)
    normalized_source = validate_source(source)
    winner = compute_winner_from_scores(home_team, away_team, home_score, away_score)
    timestamp = confirmed_at or datetime.now(timezone.utc).isoformat()

    return {
        "game_id": str(game_id),
        "home_score": int(home_score),
        "away_score": int(away_score),
        "winner": winner,
        "source": normalized_source,
        "confirmed_at": timestamp,
        "notes": str(notes).strip(),
    }


def upsert_manual_result(
    record: dict,
    path: Optional[Union[str, Path]] = None,
    allow_overwrite: bool = False,
) -> pd.DataFrame:
    """Insert or update one manual result in ``postgame_results.csv``.

    Args:
        record: Row dict from :func:`build_manual_result_record`.
        path: Destination CSV path (defaults to ``config.POSTGAME_RESULTS_PATH``).
        allow_overwrite: When ``False`` (default), refuse to replace an existing
            row for the same ``game_id``.

    Returns:
        The full DataFrame written to disk.

    Raises:
        ValueError: If overwrite is refused or the record is invalid.
    """
    csv_path = Path(path) if path is not None else config.POSTGAME_RESULTS_PATH
    row_df = pd.DataFrame([record])
    validate_postgame_results_dataframe(row_df)

    game_id = str(record["game_id"])
    existing = load_postgame_results(csv_path)

    if not existing.empty and game_id in existing["game_id"].values:
        if not allow_overwrite:
            raise ValueError(
                f"Manual result for game_id '{game_id}' already exists. "
                "Enable allow overwrite to replace it."
            )
        updated = existing.copy()
        updated = updated.loc[updated["game_id"] != game_id]
        combined = pd.concat([updated, row_df], ignore_index=True)
    else:
        combined = pd.concat([existing, row_df], ignore_index=True) if not existing.empty else row_df

    combined = combined[REQUIRED_POSTGAME_RESULT_COLUMNS].reset_index(drop=True)
    save_csv(combined, csv_path)
    return combined


def _scores_match(
    manual_home: object,
    manual_away: object,
    official_home: object,
    official_away: object,
) -> bool:
    """Compare manual and official scores as integers."""
    return int(manual_home) == int(float(official_home)) and int(manual_away) == int(
        float(official_away)
    )


def compare_result_for_game(
    game_id: str,
    manual_row: Optional[pd.Series],
    official_row: Optional[pd.Series],
) -> ComparisonStatus:
    """Compare one manual result against the play-by-play-derived official result."""
    if manual_row is None:
        return "no_manual_result_available"
    if official_row is None:
        return "no_official_result_available"

    if not _scores_match(
        manual_row["home_score"],
        manual_row["away_score"],
        official_row["home_score"],
        official_row["away_score"],
    ):
        return "mismatch"

    manual_winner = str(manual_row.get("winner", "")).strip()
    official_winner = str(official_row.get("winner", "")).strip()
    if manual_winner and official_winner and manual_winner != official_winner:
        return "mismatch"

    return "match"


def compare_manual_to_game_results(
    manual_df: pd.DataFrame,
    game_results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every manual result against ``game_results.csv``.

    Returns a DataFrame with columns ``game_id`` and ``comparison_status``.
    Games present in only one source receive the appropriate missing status.
    """
    manual_df = manual_df.copy()
    game_results_df = game_results_df.copy()

    if not manual_df.empty:
        manual_df["game_id"] = manual_df["game_id"].astype(str)
    if not game_results_df.empty:
        game_results_df["game_id"] = game_results_df["game_id"].astype(str)

    manual_ids = set(manual_df["game_id"]) if not manual_df.empty else set()
    official_ids = set(game_results_df["game_id"]) if not game_results_df.empty else set()
    all_ids = sorted(manual_ids | official_ids)

    manual_by_id = (
        manual_df.set_index("game_id") if not manual_df.empty else pd.DataFrame()
    )
    official_by_id = (
        game_results_df.set_index("game_id") if not game_results_df.empty else pd.DataFrame()
    )

    rows: list[dict] = []
    for game_id in all_ids:
        manual_row = manual_by_id.loc[game_id] if game_id in manual_ids else None
        official_row = official_by_id.loc[game_id] if game_id in official_ids else None
        if manual_row is not None and isinstance(manual_row, pd.DataFrame):
            manual_row = manual_row.iloc[0]
        if official_row is not None and isinstance(official_row, pd.DataFrame):
            official_row = official_row.iloc[0]

        status = compare_result_for_game(game_id, manual_row, official_row)
        rows.append({"game_id": game_id, "comparison_status": status})

    return pd.DataFrame(rows)
