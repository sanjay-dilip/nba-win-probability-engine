"""Collect raw team-level season statistics and save them to ``data/raw/team_stats.csv``.

This is the data-ingestion layer for Build 4.  It uses ``nba_api``'s
``LeagueDashTeamStats`` endpoint to pull one row of season-level statistics per
team per season, normalizes the NBA API's uppercase column names into the
project's snake_case schema, and appends the results to a master team-stats file.

Design notes:
* ``nba_api`` is imported lazily inside :func:`fetch_team_stats_for_season` so
  every helper in this module stays importable and unit-testable offline.
* ``team_id`` is preserved as a string (like ``game_id`` elsewhere) so it keeps
  a stable, comparable form across CSV round-trips and de-duplication.
* Every season-level API attempt is logged to ``data/logs/data_refresh_log.csv``.
* De-duplication is by ``season`` + ``season_type`` + ``team_id`` (latest wins).
* This build collects **raw season statistics only**.  It does NOT build
  pre-game features, does NOT build live features, does NOT train models, does
  NOT create target variables, does NOT compute ``home_team_won``, and does NOT
  merge team stats into games.

LEAKAGE WARNING (important for later builds):
    These are full-season aggregates.  Later pre-game feature engineering must
    only use statistics that were available *before* each game's date.  Do NOT
    use these full-season stats directly for historical pre-game predictions,
    because doing so leaks information from the future (the rest of the season)
    into a prediction made before tip-off.

Run directly:
    python src/collect_team_stats.py
    python src/collect_team_stats.py --seasons 2023-24 2024-25
    python src/collect_team_stats.py --season-type "Regular Season"
    python src/collect_team_stats.py --season-type "Playoffs"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

# Allow running this file directly (``python src/collect_team_stats.py``) by
# making the project root importable — same pattern as the other collectors.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import validate_required_columns  # noqa: E402
from src.utils import (  # noqa: E402
    append_or_update_csv,
    ensure_directories,
    log_api_attempt,
    safe_sleep,
    save_csv,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENDPOINT_NAME = "LeagueDashTeamStats"

# Used when NBA_SEASONS is not set in the environment / .env file.
DEFAULT_SEASONS = ["2024-25"]

# Accepted season types and their canonical (nba_api) spelling. Keys are
# lower-cased aliases so user input is forgiving; values are what we send to the
# API and store in the season_type column.
SEASON_TYPE_ALIASES = {
    "regular season": "Regular Season",
    "regular_season": "Regular Season",
    "regular": "Regular Season",
    "playoffs": "Playoffs",
    "playoff": "Playoffs",
}

# Maps raw NBA API (uppercase) column names to our required snake_case columns.
# Only the columns listed here are treated as "main" stats; every one of these
# whose source column is present is renamed into the output.
BASE_COLUMN_RENAME = {
    "TEAM_ID": "team_id",
    "TEAM_NAME": "team_name",
    "GP": "games_played",
    "W": "wins",
    "L": "losses",
    "W_PCT": "win_pct",
    "MIN": "minutes",
    "PTS": "points",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG_PCT": "field_goal_pct",
    "FG3M": "three_pointers_made",
    "FG3A": "three_pointers_attempted",
    "FG3_PCT": "three_point_pct",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT_PCT": "free_throw_pct",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "rebounds",
    "AST": "assists",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "PF": "personal_fouls",
    "PLUS_MINUS": "plus_minus",
}

# Optional advanced columns. Included only when the endpoint response provides
# them — never forced (a Base measure-type response will not contain these).
ADVANCED_COLUMN_RENAME = {
    "OFF_RATING": "off_rating",
    "DEF_RATING": "def_rating",
    "NET_RATING": "net_rating",
    "PACE": "pace",
    "TS_PCT": "ts_pct",
    "EFG_PCT": "efg_pct",
    "AST_PCT": "ast_pct",
    "REB_PCT": "reb_pct",
    "TM_TOV_PCT": "tm_tov_pct",
}

# Source columns we cannot build a usable team-stats row without.
REQUIRED_SOURCE_COLUMNS = ["TEAM_ID", "TEAM_NAME", "GP"]

# Columns the final team_stats.csv must always contain (season + season_type are
# added by us; the rest come from BASE_COLUMN_RENAME).
REQUIRED_TEAM_STATS_COLUMNS = ["season", "season_type"] + list(BASE_COLUMN_RENAME.values())

# Columns that must never be null in a valid team-stats row.
NON_NULL_TEAM_STATS_COLUMNS = [
    "season",
    "season_type",
    "team_id",
    "team_name",
    "games_played",
]

# De-duplication key for the master output file.
DEDUP_KEY_COLUMNS = ["season", "season_type", "team_id"]

# Columns for the per-season failure report.
FAILURE_REPORT_COLUMNS = [
    "season",
    "season_type",
    "endpoint",
    "reason_failed",
    "error_message",
]


# ---------------------------------------------------------------------------
# Small, testable helpers (pure logic — no network, safe to import in tests)
# ---------------------------------------------------------------------------

def parse_seasons(raw_seasons: str | Sequence[str]) -> List[str]:
    """Normalize seasons from a comma string or a list into a clean list.

    Args:
        raw_seasons: Either ``"2023-24,2024-25"`` or ``["2023-24", "2024-25"]``.

    Returns:
        A list of trimmed, non-empty season strings.
    """
    if isinstance(raw_seasons, str):
        parts = raw_seasons.split(",")
    else:
        parts = list(raw_seasons)
    return [season.strip() for season in parts if str(season).strip()]


def get_default_seasons() -> List[str]:
    """Return seasons from the ``NBA_SEASONS`` env var, or a sensible default."""
    raw = os.getenv("NBA_SEASONS", "")
    seasons = parse_seasons(raw)
    return seasons or list(DEFAULT_SEASONS)


def normalize_season_type(season_type: str) -> str:
    """Validate and canonicalize a season type string.

    Args:
        season_type: User-supplied season type, e.g. ``"Regular Season"``,
            ``"regular_season"`` or ``"playoffs"`` (case-insensitive).

    Returns:
        The canonical nba_api spelling: ``"Regular Season"`` or ``"Playoffs"``.

    Raises:
        ValueError: If the season type is not a recognised value.
    """
    if not isinstance(season_type, str):
        raise ValueError(f"season_type must be a string, got: {season_type!r}")

    key = season_type.strip().lower()
    if key not in SEASON_TYPE_ALIASES:
        raise ValueError(
            f"Unsupported season type: {season_type!r}. "
            "Use 'Regular Season' or 'Playoffs'."
        )
    return SEASON_TYPE_ALIASES[key]


def get_rate_limit_seconds() -> float:
    """Return ``RATE_LIMIT_SECONDS`` from the environment (default 1.5)."""
    raw = os.getenv("RATE_LIMIT_SECONDS", "1.5")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.5


def normalize_team_stats_dataframe(
    raw_df: pd.DataFrame,
    season: str,
    season_type: str,
) -> pd.DataFrame:
    """Map raw ``LeagueDashTeamStats`` columns to the required output schema.

    The NBA API returns uppercase column names (``TEAM_ID``, ``GP``, ``PTS`` …).
    This renames the main stat columns to snake_case, stamps ``season`` and
    ``season_type`` onto every row, preserves ``team_id`` as a string, and keeps
    any optional advanced columns (``off_rating``, ``pace`` …) that happen to be
    present.

    Args:
        raw_df: DataFrame from ``LeagueDashTeamStats.get_data_frames()[0]``.
        season: Season label to stamp on every row (e.g. ``"2024-25"``).
        season_type: Canonical season type (e.g. ``"Regular Season"``).

    Returns:
        A DataFrame with at least :data:`REQUIRED_TEAM_STATS_COLUMNS`, plus any
        optional advanced columns that were available in ``raw_df``.

    Raises:
        ValueError: If a required source column (see
            :data:`REQUIRED_SOURCE_COLUMNS`) is missing from ``raw_df``.
    """
    missing_source = [col for col in REQUIRED_SOURCE_COLUMNS if col not in raw_df.columns]
    if missing_source:
        raise ValueError(
            f"Raw team stats are missing required source columns: {missing_source}. "
            f"Found columns: {list(raw_df.columns)}"
        )

    out = pd.DataFrame(index=raw_df.index)

    # Season metadata that the API response does not carry.
    out["season"] = season
    out["season_type"] = season_type

    # Map the main stat columns that are present.
    for raw_col, snake_col in BASE_COLUMN_RENAME.items():
        if raw_col in raw_df.columns:
            out[snake_col] = raw_df[raw_col].values

    # Keep optional advanced columns only when the endpoint provided them.
    for raw_col, snake_col in ADVANCED_COLUMN_RENAME.items():
        if raw_col in raw_df.columns:
            out[snake_col] = raw_df[raw_col].values

    # Preserve team_id as a string so it stays stable across CSV round-trips and
    # matches the project-wide "identifiers are strings" convention.
    if "team_id" in out.columns:
        out["team_id"] = out["team_id"].astype(str)

    return out.reset_index(drop=True)


def validate_team_stats_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a team-stats DataFrame into valid and invalid rows.

    Confirms the required columns exist, then flags any row with a null in a key
    column (see :data:`NON_NULL_TEAM_STATS_COLUMNS`).

    Args:
        df: The team-stats DataFrame to validate.

    Returns:
        A ``(valid_df, invalid_df)`` tuple.

    Raises:
        ValueError: If a required column is missing entirely.
    """
    validate_required_columns(df, REQUIRED_TEAM_STATS_COLUMNS, "team_stats")

    if df.empty:
        return df.copy(), df.copy()

    null_mask = df[NON_NULL_TEAM_STATS_COLUMNS].isnull().any(axis=1)
    valid_df = df[~null_mask].copy()
    invalid_df = df[null_mask].copy()
    return valid_df, invalid_df


def build_failure_record(
    season: str,
    season_type: str,
    endpoint: str,
    reason_failed: str,
    error_message: str,
) -> dict:
    """Build one row for the team-stats collection failures report.

    Args:
        season: Season label, e.g. ``"2024-25"``.
        season_type: Canonical season type, e.g. ``"Regular Season"``.
        endpoint: Name of the NBA API endpoint attempted.
        reason_failed: Short reason label (e.g. ``"empty_response"``,
            ``"api_error"``, ``"normalize_error"``).
        error_message: Full error detail string.

    Returns:
        A dict with exactly :data:`FAILURE_REPORT_COLUMNS` keys.
    """
    return {
        "season": season,
        "season_type": season_type,
        "endpoint": endpoint,
        "reason_failed": reason_failed,
        "error_message": error_message,
    }


# ---------------------------------------------------------------------------
# NBA API access — lazily imported so helpers stay testable offline
# ---------------------------------------------------------------------------

def fetch_team_stats_for_season(season: str, season_type: str) -> pd.DataFrame:
    """Fetch raw team-level season stats for one season from the NBA API.

    The import of ``nba_api`` is deferred to here so all the pure helpers in
    this module can be imported and unit-tested without a network connection.

    Args:
        season: Season label, e.g. ``"2024-25"``.
        season_type: Canonical nba_api season type, e.g. ``"Regular Season"``.

    Returns:
        The raw DataFrame from ``LeagueDashTeamStats`` (one row per team).
    """
    from nba_api.stats.endpoints import leaguedashteamstats  # lazy import

    endpoint = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
    )
    return endpoint.get_data_frames()[0]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def collect_team_stats(
    seasons: Optional[Sequence[str]] = None,
    season_type: str = "Regular Season",
    output_path: Optional[Path] = None,
) -> int:
    """Collect raw team season stats and append them to the master output file.

    Args:
        seasons: Seasons to collect. Defaults to ``NBA_SEASONS`` (or a built-in
            default if that is unset).
        season_type: Season type to collect. Defaults to ``"Regular Season"``.
        output_path: Where to write the output. Defaults to
            :data:`config.RAW_TEAM_STATS_PATH` (``data/raw/team_stats.csv``).

    Returns:
        Process exit code: ``0`` on success, ``1`` if nothing valid was
        collected and the output file does not yet exist.
    """
    ensure_directories()

    if seasons is None:
        seasons = get_default_seasons()
    seasons = parse_seasons(seasons)
    season_type = normalize_season_type(season_type)

    if output_path is None:
        output_path = config.RAW_TEAM_STATS_PATH

    rate_limit = get_rate_limit_seconds()

    print(f"Collecting NBA team stats ({season_type}) for seasons: {', '.join(seasons)}")

    collected_frames: List[pd.DataFrame] = []
    failure_records: List[dict] = []
    success_count = 0
    fail_count = 0

    for season in seasons:
        print(f"  Fetching {season} ({season_type})...", end=" ")

        try:
            raw = fetch_team_stats_for_season(season, season_type)
        except Exception as exc:  # noqa: BLE001 - log and continue on any failure
            msg = str(exc)
            print(f"[failed] {msg[:80]}")
            log_api_attempt(game_id=season, endpoint=ENDPOINT_NAME, status="failed", error_message=msg)
            failure_records.append(
                build_failure_record(season, season_type, ENDPOINT_NAME, "api_error", msg)
            )
            fail_count += 1
            safe_sleep(rate_limit)
            continue

        if raw is None or raw.empty:
            msg = "API returned zero rows."
            print("[empty]")
            log_api_attempt(game_id=season, endpoint=ENDPOINT_NAME, status="failed", error_message=msg)
            failure_records.append(
                build_failure_record(season, season_type, ENDPOINT_NAME, "empty_response", msg)
            )
            fail_count += 1
            safe_sleep(rate_limit)
            continue

        try:
            normalized = normalize_team_stats_dataframe(raw, season, season_type)
        except Exception as exc:  # noqa: BLE001 - bad schema should not abort the run
            msg = str(exc)
            print(f"[failed] {msg[:80]}")
            log_api_attempt(game_id=season, endpoint=ENDPOINT_NAME, status="failed", error_message=msg)
            failure_records.append(
                build_failure_record(season, season_type, ENDPOINT_NAME, "normalize_error", msg)
            )
            fail_count += 1
            safe_sleep(rate_limit)
            continue

        collected_frames.append(normalized)
        log_api_attempt(game_id=season, endpoint=ENDPOINT_NAME, status="success")
        success_count += 1
        print(f"[ok] {len(normalized)} teams")
        safe_sleep(rate_limit)

    # --- combine and validate -----------------------------------------------
    if collected_frames:
        new_df = pd.concat(collected_frames, ignore_index=True)
    else:
        new_df = pd.DataFrame(columns=REQUIRED_TEAM_STATS_COLUMNS)

    valid_new, invalid_new = validate_team_stats_dataframe(new_df)

    invalid_count = len(invalid_new)
    if invalid_count > 0:
        save_csv(invalid_new, config.INVALID_TEAM_STATS_REPORT_PATH)
        print(f"  [warning] {invalid_count} invalid row(s) -> {config.INVALID_TEAM_STATS_REPORT_PATH}")

    # --- save failures report -----------------------------------------------
    if failure_records:
        failures_df = pd.DataFrame(failure_records, columns=FAILURE_REPORT_COLUMNS)
        save_csv(failures_df, config.TEAM_STATS_FAILURES_REPORT_PATH)

    # --- append to master output --------------------------------------------
    if valid_new.empty and not output_path.exists():
        print("  [error] No valid team stats collected and no existing output file.")
        return 1

    if not valid_new.empty:
        final_df = append_or_update_csv(
            valid_new,
            output_path,
            key_columns=DEDUP_KEY_COLUMNS,
        )
    else:
        final_df = pd.read_csv(output_path, dtype={"team_id": str})

    # --- summary ------------------------------------------------------------
    total_rows = len(final_df)
    new_rows = len(valid_new)

    print("\nDone.")
    print(f"  Seasons requested:         {', '.join(seasons)}")
    print(f"  Season type:               {season_type}")
    print(f"  Successful season pulls:   {success_count}")
    print(f"  Failed / empty pulls:      {fail_count}")
    print(f"  New rows appended:         {new_rows}")
    print(f"  Total rows in output file: {total_rows}")
    print(f"  Invalid rows:              {invalid_count}")
    print(f"  Output path:               {output_path}")
    if failure_records:
        print(f"  Failure report:            {config.TEAM_STATS_FAILURES_REPORT_PATH}")
    else:
        print("  Failure report:            none (all seasons succeeded)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run team-stats collection."""
    parser = argparse.ArgumentParser(
        description="Collect raw NBA team season stats into data/raw/team_stats.csv."
    )
    parser.add_argument(
        "--seasons",
        nargs="*",
        default=None,
        help="Seasons to collect, e.g. --seasons 2023-24 2024-25. "
        "Defaults to the NBA_SEASONS environment variable.",
    )
    parser.add_argument(
        "--season-type",
        default="Regular Season",
        help='Season type to collect: "Regular Season" or "Playoffs" '
        '(default: "Regular Season").',
    )
    args = parser.parse_args(argv)

    print("Collecting NBA team stats...")
    return collect_team_stats(seasons=args.seasons, season_type=args.season_type)


if __name__ == "__main__":
    sys.exit(main())
