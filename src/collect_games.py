"""Collect the NBA game schedule and save it to ``data/raw/games.csv``.

This is the schedule data-ingestion layer. It uses ``nba_api``'s
``LeagueGameFinder`` endpoint to pull games for one or more seasons, transforms
the team-level rows it returns (one row per team per game) into clean
game-level rows (one row per game), and writes a master schedule file.

Design notes:
* The ``nba_api`` import is done lazily inside the fetch function so the small
  helper functions in this module can be imported and unit-tested without the
  network or the ``nba_api`` package.
* Pure, testable helpers: ``parse_matchup_location``,
  ``build_game_rows_from_team_rows``, ``validate_games_dataframe``,
  ``summarize_raw_counts``.
* Any game group that cannot be safely converted into a single home/away row is
  recorded in ``outputs/reports/skipped_games_report.csv`` rather than being
  dropped silently, so missing games are always diagnosable.

Run directly:
    python src/collect_games.py
    python src/collect_games.py --seasons 2023-24 2024-25
    python src/collect_games.py --season-type "Regular Season"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

# Allow running this file directly (``python src/collect_games.py``) by making
# the project root importable, exactly like the Streamlit pages do.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import validate_required_columns  # noqa: E402
from src.utils import (  # noqa: E402
    append_or_update_csv,
    ensure_directories,
    load_csv,
    log_api_attempt,
    safe_sleep,
    save_csv,
)

# Columns the final games.csv must always contain.
REQUIRED_GAME_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "status",
    "game_type",
]

# Columns that must never be null in a valid game row.
NON_NULL_COLUMNS = [
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
]

# Columns for the skipped-games diagnostic report. Every game group that cannot
# be safely converted into a single home/away row is recorded here so data loss
# is never silent.
SKIPPED_REPORT_COLUMNS = [
    "game_id",
    "season",
    "game_type",
    "raw_row_count",
    "home_row_count",
    "away_row_count",
    "team_names",
    "team_ids",
    "matchup_values",
    "game_dates",
    "reason_skipped",
]

# Separator used when packing multiple team values into one report cell. A pipe
# keeps cells readable and avoids confusion with CSV commas.
LIST_SEP = " | "

# Used when NBA_SEASONS is not set in the environment / .env file.
DEFAULT_SEASONS = ["2024-25"]

# Maps an nba_api season type to our simple game_type label.
SEASON_TYPE_TO_GAME_TYPE = {
    "Regular Season": "regular_season",
    "Playoffs": "playoffs",
}

ENDPOINT_NAME = "LeagueGameFinder"


# ---------------------------------------------------------------------------
# Small, testable helpers
# ---------------------------------------------------------------------------
def parse_matchup_location(matchup: str) -> str:
    """Determine whether a team was home or away from its MATCHUP string.

    ``LeagueGameFinder`` returns a MATCHUP like ``"BOS vs. MIA"`` (home) or
    ``"MIA @ BOS"`` (away).

    Args:
        matchup: The MATCHUP value for a single team row.

    Returns:
        ``"home"`` if the matchup contains ``"vs."``, ``"away"`` if it
        contains ``"@"``.

    Raises:
        ValueError: If the matchup is not a string or the location cannot be
            determined.
    """
    if not isinstance(matchup, str):
        raise ValueError(f"MATCHUP must be a string, got: {matchup!r}")

    if "vs." in matchup:
        return "home"
    if "@" in matchup:
        return "away"

    raise ValueError(f"Could not determine home/away from MATCHUP: {matchup!r}")


def season_type_to_game_type(season_type: str) -> str:
    """Translate an nba_api season type into our game_type label.

    Args:
        season_type: e.g. ``"Regular Season"`` or ``"Playoffs"``.

    Returns:
        ``"regular_season"`` or ``"playoffs"``. Unknown values fall back to
        ``"regular_season"`` so the pipeline stays robust.
    """
    return SEASON_TYPE_TO_GAME_TYPE.get(season_type, "regular_season")


def _join_group_values(group: pd.DataFrame, column: str) -> str:
    """Join all values of a column in a game group into one report cell."""
    return LIST_SEP.join(str(value) for value in group[column].tolist())


def _build_skipped_record(
    game_id: object,
    group: pd.DataFrame,
    season: str,
    game_type: str,
    home_count: int,
    away_count: int,
    reasons: Sequence[str],
) -> dict:
    """Build one row for the skipped-games diagnostic report."""
    return {
        "game_id": str(game_id),
        "season": season,
        "game_type": game_type,
        "raw_row_count": len(group),
        "home_row_count": home_count,
        "away_row_count": away_count,
        "team_names": _join_group_values(group, "TEAM_NAME"),
        "team_ids": _join_group_values(group, "TEAM_ID"),
        "matchup_values": _join_group_values(group, "MATCHUP"),
        "game_dates": _join_group_values(group, "GAME_DATE"),
        "reason_skipped": ";".join(reasons),
    }


def build_game_rows_from_team_rows(
    df: pd.DataFrame,
    season: str,
    game_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse team-level rows into one clean row per game, plus diagnostics.

    ``LeagueGameFinder`` returns one row per team per game. This groups by
    ``GAME_ID`` and uses each row's MATCHUP to decide which team is home and
    which is away. A game group is only turned into a final row when it has
    *exactly one* home row and *exactly one* away row with no parsing problems.
    Any group that cannot be safely converted is recorded in a skipped report
    instead of being dropped silently.

    Args:
        df: Raw team-level rows. Must contain ``GAME_ID``, ``TEAM_ID``,
            ``TEAM_NAME``, ``GAME_DATE`` and ``MATCHUP``.
        season: Season label to stamp on every row (e.g. ``"2023-24"``).
        game_type: Game type label (e.g. ``"regular_season"``).

    Returns:
        A ``(games_df, skipped_df)`` tuple. ``games_df`` has exactly
        :data:`REQUIRED_GAME_COLUMNS`; ``skipped_df`` has exactly
        :data:`SKIPPED_REPORT_COLUMNS`.
    """
    source_columns = {"GAME_ID", "TEAM_ID", "TEAM_NAME", "GAME_DATE", "MATCHUP"}
    missing = source_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Raw games data is missing expected columns: {sorted(missing)}. "
            f"Found: {list(df.columns)}"
        )

    game_rows: List[dict] = []
    skipped_rows: List[dict] = []

    for game_id, group in df.groupby("GAME_ID"):
        home_rows = []
        away_rows = []
        unparseable_count = 0
        for _, row in group.iterrows():
            try:
                location = parse_matchup_location(row["MATCHUP"])
            except ValueError:
                unparseable_count += 1
                continue
            if location == "home":
                home_rows.append(row)
            else:
                away_rows.append(row)

        # Collect every reason this group is not a clean 1-home / 1-away game.
        reasons: List[str] = []
        if unparseable_count > 0:
            reasons.append("invalid_matchup_format")
        if len(home_rows) == 0:
            reasons.append("missing_home_row")
        if len(away_rows) == 0:
            reasons.append("missing_away_row")
        if len(home_rows) > 1:
            reasons.append("multiple_home_rows")
        if len(away_rows) > 1:
            reasons.append("multiple_away_rows")

        candidate = None
        if not reasons:
            # Exactly one home and one away row, no parsing issues.
            home_row = home_rows[0]
            away_row = away_rows[0]
            candidate = {
                "game_id": str(game_id),
                "season": season,
                "game_date": home_row["GAME_DATE"],
                "home_team": home_row["TEAM_NAME"],
                "away_team": away_row["TEAM_NAME"],
                "home_team_id": home_row["TEAM_ID"],
                "away_team_id": away_row["TEAM_ID"],
                "status": "final",
                "game_type": game_type,
            }
            # Even a well-paired game is unusable if a required field is missing.
            if any(pd.isna(candidate[col]) for col in NON_NULL_COLUMNS):
                reasons.append("missing_required_fields")
                candidate = None

        if candidate is not None:
            game_rows.append(candidate)
        else:
            if not reasons:
                reasons = ["unknown"]
            skipped_rows.append(
                _build_skipped_record(
                    game_id,
                    group,
                    season,
                    game_type,
                    home_count=len(home_rows),
                    away_count=len(away_rows),
                    reasons=reasons,
                )
            )

    games_df = pd.DataFrame(game_rows, columns=REQUIRED_GAME_COLUMNS)
    skipped_df = pd.DataFrame(skipped_rows, columns=SKIPPED_REPORT_COLUMNS)
    return games_df, skipped_df


def summarize_raw_counts(df: pd.DataFrame) -> dict:
    """Summarize raw row counts to sanity-check the transformation.

    Args:
        df: Raw team-level rows from ``LeagueGameFinder``.

    Returns:
        A dict with ``raw_row_count``, ``unique_game_id_count`` and
        ``expected_game_count_from_raw_pairs`` (raw rows // 2, since a complete
        game has two team rows).
    """
    raw_row_count = len(df)
    unique_game_id_count = int(df["GAME_ID"].nunique()) if "GAME_ID" in df.columns else 0
    return {
        "raw_row_count": raw_row_count,
        "unique_game_id_count": unique_game_id_count,
        "expected_game_count_from_raw_pairs": raw_row_count // 2,
    }


def validate_games_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a games DataFrame into valid and invalid rows.

    Confirms the required columns exist, then flags any row that has a null in a
    key column (see :data:`NON_NULL_COLUMNS`).

    Args:
        df: The games DataFrame to validate.

    Returns:
        A ``(valid_df, invalid_df)`` tuple.

    Raises:
        ValueError: If a required column is missing entirely.
    """
    validate_required_columns(df, REQUIRED_GAME_COLUMNS, "games")

    if df.empty:
        return df.copy(), df.copy()

    null_mask = df[NON_NULL_COLUMNS].isnull().any(axis=1)
    invalid_df = df[null_mask].copy()
    valid_df = df[~null_mask].copy()
    return valid_df, invalid_df


def parse_seasons(raw_seasons: str | Sequence[str]) -> List[str]:
    """Normalize seasons from a comma string or a list into a clean list.

    Args:
        raw_seasons: Either ``"2021-22,2022-23"`` or ``["2021-22", "2022-23"]``.

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


def get_rate_limit_seconds() -> float:
    """Return ``RATE_LIMIT_SECONDS`` from the environment (default 1.5)."""
    raw = os.getenv("RATE_LIMIT_SECONDS", "1.5")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.5


# ---------------------------------------------------------------------------
# NBA API access (network) — kept thin and lazily imported
# ---------------------------------------------------------------------------
def fetch_raw_games_for_season(season: str, season_type: str) -> pd.DataFrame:
    """Fetch raw team-level game rows for a single season from the NBA API.

    Logs every attempt to ``data/logs/data_refresh_log.csv`` with a status of
    ``"success"`` or ``"failed"``. Network/endpoint errors are caught and an
    empty DataFrame is returned so a single bad season does not abort the run.

    Args:
        season: Season label, e.g. ``"2023-24"``.
        season_type: nba_api season type, e.g. ``"Regular Season"``.

    Returns:
        The raw DataFrame from ``LeagueGameFinder`` (possibly empty).
    """
    # Lazy import so this module's helpers stay importable without nba_api.
    from nba_api.stats.endpoints import leaguegamefinder

    try:
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season,
            season_type_nullable=season_type,
            league_id_nullable="00",  # "00" = NBA (excludes G-League, etc.)
        )
        raw = finder.get_data_frames()[0]
        log_api_attempt(game_id=season, endpoint=ENDPOINT_NAME, status="success")
        return raw
    except Exception as exc:  # noqa: BLE001 - we log and continue on any failure
        log_api_attempt(
            game_id=season,
            endpoint=ENDPOINT_NAME,
            status="failed",
            error_message=str(exc),
        )
        print(f"  [failed] Season {season}: {exc}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def collect_games(
    seasons: Optional[Sequence[str]] = None,
    season_type: str = "Regular Season",
    output_path: Optional[Path] = None,
) -> int:
    """Collect the NBA schedule for the given seasons and save games.csv.

    Args:
        seasons: Seasons to collect. Defaults to ``NBA_SEASONS`` (or a built-in
            default if that is unset).
        season_type: nba_api season type. Defaults to ``"Regular Season"``.
        output_path: Where to write the master schedule. Defaults to
            ``config.RAW_GAMES_PATH`` (``data/raw/games.csv``).

    Returns:
        Process exit code: ``0`` on success, ``1`` if no usable games resulted.
    """
    ensure_directories()

    if seasons is None:
        seasons = get_default_seasons()
    seasons = parse_seasons(seasons)
    if output_path is None:
        output_path = config.RAW_GAMES_PATH

    game_type = season_type_to_game_type(season_type)
    rate_limit_seconds = get_rate_limit_seconds()

    print(f"Collecting NBA games ({season_type}) for seasons: {', '.join(seasons)}")

    collected_frames: List[pd.DataFrame] = []
    skipped_frames: List[pd.DataFrame] = []
    total_raw_rows = 0
    total_unique_game_ids = 0
    for season in seasons:
        print(f"  Fetching season {season}...")
        raw = fetch_raw_games_for_season(season, season_type)
        total_raw_rows += len(raw)
        if not raw.empty:
            counts = summarize_raw_counts(raw)
            # GAME_IDs do not overlap across seasons, so summing is safe here.
            total_unique_game_ids += counts["unique_game_id_count"]
            game_rows, skipped_rows = build_game_rows_from_team_rows(
                raw, season, game_type
            )
            collected_frames.append(game_rows)
            if not skipped_rows.empty:
                skipped_frames.append(skipped_rows)
            print(
                f"    -> {len(game_rows)} games, {len(skipped_rows)} skipped "
                f"from {len(raw)} raw rows ({counts['unique_game_id_count']} game IDs)"
            )
        safe_sleep(rate_limit_seconds)

    if collected_frames:
        new_games = pd.concat(collected_frames, ignore_index=True)
    else:
        new_games = pd.DataFrame(columns=REQUIRED_GAME_COLUMNS)

    if skipped_frames:
        skipped_df = pd.concat(skipped_frames, ignore_index=True)
    else:
        skipped_df = pd.DataFrame(columns=SKIPPED_REPORT_COLUMNS)

    valid_new, invalid_new = validate_games_dataframe(new_games)

    if not invalid_new.empty:
        save_csv(invalid_new, config.INVALID_GAMES_REPORT_PATH)
        print(
            f"  [warning] {len(invalid_new)} invalid row(s) saved to "
            f"{config.INVALID_GAMES_REPORT_PATH}"
        )

    # Skipped-games report policy: only write the file when there is something
    # to report. When nothing was skipped we skip the file and say so, keeping
    # outputs/reports/ clean.
    if not skipped_df.empty:
        save_csv(skipped_df, config.SKIPPED_GAMES_REPORT_PATH)

    # If we collected nothing valid AND there is no existing file, the output
    # would be empty — treat that as a failure.
    if valid_new.empty and not output_path.exists():
        print("  [error] No valid games collected and no existing games.csv. Nothing saved.")
        return 1

    # Append + de-duplicate on game_id (incoming rows win on conflicts).
    if not valid_new.empty:
        final_df = append_or_update_csv(valid_new, output_path, key_columns=["game_id"])
    else:
        final_df = load_csv(output_path)

    print("\nDone.")
    print(f"  Seasons collected:        {', '.join(seasons)}")
    print(f"  Raw rows returned:        {total_raw_rows}")
    print(f"  Unique GAME_IDs (raw):    {total_unique_game_ids}")
    print(f"  Final game rows saved:    {len(final_df)}")
    print(f"  Skipped game groups:      {len(skipped_df)}")
    print(f"  Invalid rows:             {len(invalid_new)}")
    print(f"  Output path:              {output_path}")
    if not skipped_df.empty:
        print(f"  Skipped report:           {config.SKIPPED_GAMES_REPORT_PATH}")
    else:
        print("  Skipped report:           none (no skipped game groups)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run schedule collection."""
    parser = argparse.ArgumentParser(
        description="Collect NBA game schedule data into data/raw/games.csv."
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
        help='Season type to collect (default: "Regular Season").',
    )
    args = parser.parse_args(argv)

    return collect_games(seasons=args.seasons, season_type=args.season_type)


if __name__ == "__main__":
    sys.exit(main())
