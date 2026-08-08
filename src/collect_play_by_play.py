"""Collect raw play-by-play event data and save it to ``data/raw/play_by_play.csv``.

This is the play-by-play data-ingestion layer.  It reads completed games from
``data/raw/games.csv``, fetches play-by-play data for each game from the NBA
API, and appends the results to a master play-by-play file.

**Important — endpoint note:**
The spec lists ``PlayByPlayV2`` as the preferred endpoint.  That endpoint is
deprecated and the NBA API no longer returns data for it (it raises
``KeyError: 'resultSet'``).  This module uses ``PlayByPlayV3`` directly.
A natural fallback point is marked with ``# FUTURE FALLBACK:`` comments so a
secondary endpoint can be dropped in later if needed.

Design notes:
* ``nba_api`` is imported lazily inside ``fetch_play_by_play()`` so all helper
  functions remain importable and testable without a network connection.
* Every API attempt is logged to ``data/logs/data_refresh_log.csv``.
* ``game_id`` is preserved as a string with leading zeros throughout.
* The script is intentionally limited to ``--limit`` games per run (default 10)
  to prevent accidental full-season pulls during development.
* Feature engineering columns (``seconds_remaining_game``, ``score_margin_home``,
  ``home_team_won``, etc.) are NOT computed here.  Raw event data only.

Run directly:
    python src/collect_play_by_play.py
    python src/collect_play_by_play.py --limit 5
    python src/collect_play_by_play.py --season 2024-25 --limit 20
    python src/collect_play_by_play.py --game-id 0022400001
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple

import pandas as pd

# Allow running this file directly (``python src/collect_play_by_play.py``)
# by making the project root importable — same pattern as the other collectors.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import validate_required_columns  # noqa: E402
from src.utils import (  # noqa: E402
    append_or_update_csv,
    ensure_directories,
    get_rate_limit_seconds,
    log_api_attempt,
    safe_sleep,
    save_csv,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that the final play_by_play.csv must always contain.
REQUIRED_PBP_COLUMNS = [
    "game_id",
    "event_num",
    "event_msg_type",
    "event_msg_action_type",
    "period",
    "pctimestring",
    "home_description",
    "neutral_description",
    "visitor_description",
    "score",
    "score_margin",
    "home_team",
    "away_team",
    "season",
    "game_date",
]

# Columns that must not be null in a valid play-by-play row.
NON_NULL_PBP_COLUMNS = ["game_id", "event_num", "period"]

# Columns for the per-game failure report.
FAILURE_REPORT_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "endpoint",
    "reason_failed",
    "error_message",
]

# Columns for the data-coverage report (one row per season label per run).
COVERAGE_REPORT_COLUMNS = [
    "season",
    "eligible_games",
    "collected_games",
    "remaining_games",
    "coverage_pct",
    "total_event_rows",
    "generated_at",
]

ENDPOINT_NAME = "PlayByPlayV3"
DEFAULT_LIMIT = 10
# For full-season collection, save progress to disk after every this-many games
# so an API failure partway through never discards already-collected games.
DEFAULT_BATCH_SIZE = 50


# ---------------------------------------------------------------------------
# Helper functions (pure logic — no network, safe to import in tests)
# ---------------------------------------------------------------------------

def load_eligible_games(
    games_path: str | Path,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """Load completed games from games.csv, preserving game_id as a string.

    Applies baseline quality filters:

    * ``status == "final"``
    * ``game_id``, ``home_team``, and ``away_team`` must not be null

    Args:
        games_path: Path to the games schedule CSV.
        season: If given, only return games for this season label (e.g.
            ``"2024-25"``).

    Returns:
        A filtered DataFrame with ``game_id`` stored as ``str``.

    Raises:
        FileNotFoundError: If ``games_path`` does not exist.
    """
    games_path = Path(games_path)
    if not games_path.exists():
        raise FileNotFoundError(
            f"Games schedule not found: {games_path}. "
            "Run 'python run_pipeline.py --mode collect_games' first."
        )

    df = pd.read_csv(games_path, dtype={"game_id": str})
    df = df[df["status"] == "final"].copy()
    df = df[df["game_id"].notna()].copy()
    df = df[df["home_team"].notna()].copy()
    df = df[df["away_team"].notna()].copy()

    if season is not None:
        df = df[df["season"] == season].copy()

    return df.reset_index(drop=True)


def filter_games_for_collection(
    games_df: pd.DataFrame,
    already_collected: Optional[Set[str]] = None,
    game_id: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
) -> pd.DataFrame:
    """Narrow the eligible game list down to what this run should collect.

    Applies, in order:

    1. ``game_id`` filter (single-game mode).
    2. Skip games already present in the output file.
    3. ``limit`` cap.

    Args:
        games_df: Eligible games from :func:`load_eligible_games`.
        already_collected: Set of ``game_id`` strings already in the output
            file.  Pass an empty set or ``None`` to collect everything.
        game_id: If given, restrict to this single game.
        limit: Maximum number of games to return.  ``0`` or ``None`` means no
            cap (use with care — the full season is 1 200+ games).

    Returns:
        A DataFrame containing only the games that should be fetched this run.
    """
    df = games_df.copy()

    if game_id is not None:
        df = df[df["game_id"] == str(game_id)]

    if already_collected:
        df = df[~df["game_id"].isin(already_collected)]

    if limit and limit > 0:
        df = df.head(limit)

    return df.reset_index(drop=True)


def get_already_collected_game_ids(pbp_path: str | Path) -> Set[str]:
    """Return the set of ``game_id`` values already saved in the output file.

    Args:
        pbp_path: Path to ``data/raw/play_by_play.csv``.

    Returns:
        A set of string game IDs.  Empty set if the file does not exist.
    """
    pbp_path = Path(pbp_path)
    if not pbp_path.exists():
        return set()

    df = pd.read_csv(pbp_path, usecols=["game_id"], dtype={"game_id": str})
    return set(df["game_id"].dropna().unique())


def normalize_play_by_play_dataframe(
    raw_df: pd.DataFrame,
    game_row: pd.Series,
) -> pd.DataFrame:
    """Map raw ``PlayByPlayV3`` columns to the required output schema.

    ``PlayByPlayV3`` returns one row per action.  V2 (now defunct) had
    side-by-side ``HOMEDESCRIPTION``, ``VISITORDESCRIPTION``, and
    ``NEUTRALDESCRIPTION`` columns.  V3 stores a single ``description`` field
    plus a ``location`` field (``"h"`` / ``"v"`` / ``""``).  This function
    reconstructs the three-column layout from those two fields.

    Args:
        raw_df: DataFrame returned by ``PlayByPlayV3.get_data_frames()[0]``.
        game_row: A single row from games.csv as a pandas Series (provides
            ``home_team``, ``away_team``, ``season``, ``game_date``).

    Returns:
        A DataFrame with :data:`REQUIRED_PBP_COLUMNS` plus extra V3 columns
        (``teamId``, ``teamTricode``, ``personId``, ``playerName``,
        ``location``, ``scoreHome``, ``scoreAway``).
    """
    out = pd.DataFrame(index=raw_df.index)

    # --- required columns ---------------------------------------------------
    out["game_id"] = raw_df["gameId"].astype(str)
    out["event_num"] = raw_df["actionNumber"]
    out["event_msg_type"] = raw_df["actionType"]
    out["event_msg_action_type"] = raw_df["subType"]
    out["period"] = raw_df["period"]
    out["pctimestring"] = raw_df["clock"]

    # V3 encodes "which team's description" via the location field.
    # Reconstruct the three-column layout that V2 used.
    loc = raw_df["location"].fillna("")
    desc = raw_df["description"].fillna("")
    out["home_description"] = desc.where(loc == "h", "")
    out["visitor_description"] = desc.where(loc == "v", "")
    out["neutral_description"] = desc.where(loc == "", "")

    # Score: build "X - Y" string from the running scoreHome / scoreAway fields.
    sh_raw = raw_df["scoreHome"].fillna("").astype(str).str.strip()
    sa_raw = raw_df["scoreAway"].fillna("").astype(str).str.strip()
    has_score = (sh_raw != "") & (sa_raw != "") & (sh_raw != "nan") & (sa_raw != "nan")
    score_series = pd.Series("", index=raw_df.index)
    score_series.loc[has_score] = sh_raw[has_score] + " - " + sa_raw[has_score]
    out["score"] = score_series

    # Score margin: numeric difference when both scores are available.
    sh_num = pd.to_numeric(sh_raw, errors="coerce")
    sa_num = pd.to_numeric(sa_raw, errors="coerce")
    valid_num = has_score & sh_num.notna() & sa_num.notna()
    margin_series = pd.Series("", index=raw_df.index, dtype=object)
    margin_series.loc[valid_num] = (sh_num - sa_num).loc[valid_num].astype(int)
    out["score_margin"] = margin_series

    # Game-level metadata from the games.csv row.
    out["home_team"] = game_row["home_team"]
    out["away_team"] = game_row["away_team"]
    out["season"] = game_row["season"]
    out["game_date"] = game_row["game_date"]

    # --- keep useful extra V3 columns ---------------------------------------
    extra_v3_cols = [
        "teamId", "teamTricode", "personId", "playerName",
        "location", "scoreHome", "scoreAway",
    ]
    for col in extra_v3_cols:
        if col in raw_df.columns:
            out[col] = raw_df[col].values

    return out


def validate_play_by_play_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a play-by-play DataFrame into valid and invalid rows.

    Args:
        df: The DataFrame to validate.

    Returns:
        A ``(valid_df, invalid_df)`` tuple.

    Raises:
        ValueError: If a required column is missing entirely.
    """
    validate_required_columns(df, REQUIRED_PBP_COLUMNS, "play_by_play")

    if df.empty:
        return df.copy(), df.copy()

    null_mask = df[NON_NULL_PBP_COLUMNS].isnull().any(axis=1)
    return df[~null_mask].copy(), df[null_mask].copy()


def build_failure_record(
    game_row: pd.Series,
    endpoint: str,
    reason_failed: str,
    error_message: str,
) -> dict:
    """Build one row for the play-by-play collection failures report.

    Args:
        game_row: A single row from games.csv as a pandas Series.
        endpoint: Name of the NBA API endpoint attempted.
        reason_failed: Short reason label (e.g. ``"empty_response"``,
            ``"api_error"``).
        error_message: Full error detail string.

    Returns:
        A dict with exactly :data:`FAILURE_REPORT_COLUMNS` keys.
    """
    return {
        "game_id": str(game_row["game_id"]),
        "season": game_row["season"],
        "game_date": game_row["game_date"],
        "home_team": game_row["home_team"],
        "away_team": game_row["away_team"],
        "endpoint": endpoint,
        "reason_failed": reason_failed,
        "error_message": error_message,
    }


def build_coverage_record(
    season: Optional[str],
    eligible_games: int,
    collected_games: int,
    total_event_rows: int,
) -> dict:
    """Build one row for the play-by-play coverage report.

    Args:
        season: Season label this row describes, or ``None`` for "all".
        eligible_games: Number of eligible (completed) games in games.csv.
        collected_games: Number of those eligible games now present in the
            play-by-play output.
        total_event_rows: Number of event rows in the output for those games.

    Returns:
        A dict with exactly :data:`COVERAGE_REPORT_COLUMNS` keys.
    """
    remaining = max(eligible_games - collected_games, 0)
    coverage_pct = round(collected_games / eligible_games * 100, 2) if eligible_games else 0.0
    return {
        "season": season if season is not None else "all",
        "eligible_games": eligible_games,
        "collected_games": collected_games,
        "remaining_games": remaining,
        "coverage_pct": coverage_pct,
        "total_event_rows": total_event_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def merge_failure_records(
    existing_df: Optional[pd.DataFrame],
    new_failures: Sequence[dict],
    succeeded_game_ids: Sequence[str],
) -> pd.DataFrame:
    """Merge new failures into an existing failure report without duplicating.

    A game appears at most once (keyed by ``game_id``).  Games that succeeded
    this run are removed (they are no longer failing), and any game in
    ``new_failures`` replaces its previous entry so the latest error wins.

    Args:
        existing_df: The previously-saved failures, or ``None`` if none exist.
        new_failures: Failure dicts from this run (see :func:`build_failure_record`).
        succeeded_game_ids: Game ids that were collected successfully this run.

    Returns:
        A de-duplicated failures DataFrame with :data:`FAILURE_REPORT_COLUMNS`.
    """
    new_df = pd.DataFrame(list(new_failures), columns=FAILURE_REPORT_COLUMNS)
    new_df["game_id"] = new_df["game_id"].astype(str)

    if existing_df is None or existing_df.empty:
        combined = new_df
    else:
        existing = existing_df.copy()
        existing["game_id"] = existing["game_id"].astype(str)
        drop_ids = set(map(str, succeeded_game_ids)) | set(new_df["game_id"])
        existing = existing[~existing["game_id"].isin(drop_ids)]
        combined = pd.concat([existing, new_df], ignore_index=True)

    combined = combined.drop_duplicates(subset=["game_id"], keep="last").reset_index(drop=True)
    return combined


def _flush_batch(frames: List[pd.DataFrame], output_path: Path) -> Tuple[int, pd.DataFrame]:
    """Validate a batch of collected games and append valid rows to the output.

    Args:
        frames: Normalized per-game DataFrames collected since the last flush.
        output_path: Master play-by-play CSV to append/update.

    Returns:
        A ``(valid_row_count, invalid_df)`` tuple.
    """
    if not frames:
        return 0, pd.DataFrame(columns=REQUIRED_PBP_COLUMNS)

    new_df = pd.concat(frames, ignore_index=True)
    valid, invalid = validate_play_by_play_dataframe(new_df)
    if not valid.empty:
        append_or_update_csv(valid, output_path, key_columns=["game_id", "event_num"])
    return len(valid), invalid


# ---------------------------------------------------------------------------
# NBA API access — lazily imported so helpers stay testable offline
# ---------------------------------------------------------------------------

def fetch_play_by_play(game_id: str) -> pd.DataFrame:
    """Fetch raw play-by-play rows for one game from ``PlayByPlayV3``.

    The import of ``nba_api`` is deferred to here so all the pure helpers in
    this module can be imported and unit-tested without a network connection.

    # FUTURE FALLBACK: if PlayByPlayV3 is ever deprecated, add a try/except
    # around the V3 call and fall back to the next available endpoint here.

    Args:
        game_id: String game ID with leading zeros (e.g. ``"0022400001"``).

    Returns:
        Raw DataFrame from ``PlayByPlayV3``.  Returns an empty DataFrame on
        any failure so the caller can handle it gracefully.

    Raises:
        ImportError: If ``nba_api`` is not installed (propagated to the
            caller so it can log a clear message).
    """
    from nba_api.stats.endpoints import playbyplayv3  # lazy import

    endpoint = playbyplayv3.PlayByPlayV3(game_id=game_id)
    raw = endpoint.get_data_frames()[0]
    return raw


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _write_coverage_report(
    season: Optional[str],
    eligible: pd.DataFrame,
    output_path: Path,
    coverage_report_path: Optional[Path] = None,
) -> dict:
    """Compute and persist the coverage report for the selected season.

    Reads the current output to count how many eligible games (and event rows)
    are now present, then upserts a per-season row into the coverage report.

    Returns the coverage record that was written.
    """
    eligible_ids = set(eligible["game_id"])
    collected_after = get_already_collected_game_ids(output_path)
    collected_eligible = eligible_ids & collected_after

    total_event_rows = 0
    if output_path.exists():
        out = pd.read_csv(output_path, usecols=["game_id"], dtype={"game_id": str})
        total_event_rows = int(out["game_id"].isin(eligible_ids).sum())

    record = build_coverage_record(
        season=season,
        eligible_games=len(eligible),
        collected_games=len(collected_eligible),
        total_event_rows=total_event_rows,
    )
    coverage_df = pd.DataFrame([record], columns=COVERAGE_REPORT_COLUMNS)
    report_path = coverage_report_path or config.PBP_COVERAGE_REPORT_PATH
    append_or_update_csv(coverage_df, report_path, key_columns=["season"])
    return record


def collect_play_by_play(
    season: Optional[str] = None,
    game_id: Optional[str] = None,
    limit: Optional[int] = DEFAULT_LIMIT,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    coverage_report_path: Optional[Path] = None,
) -> int:
    """Collect raw play-by-play data and append it to the master output file.

    The collector is idempotent: games already present in the output are skipped,
    so re-running never duplicates event rows.  For full-season collection,
    progress is saved to disk after every ``batch_size`` games so an API failure
    partway through does not discard already-collected games.

    Args:
        season: If given, restrict to games from this season (e.g.
            ``"2024-25"``).
        game_id: If given, collect only this single game.
        limit: Maximum number of new games to collect this run.  Defaults to
            :data:`DEFAULT_LIMIT` (10).  Pass ``0`` or ``None`` for **no cap**
            (collect every remaining eligible game).
        batch_size: Save progress to disk after this many collected games.
            Defaults to :data:`DEFAULT_BATCH_SIZE` (50).
        output_path: Where to write the output.  Defaults to
            :data:`config.RAW_PLAY_BY_PLAY_PATH`.
        games_path: Games schedule CSV for eligible games.  Defaults to
            :data:`config.RAW_GAMES_PATH`.
        coverage_report_path: Where to write the coverage report.  Defaults to
            :data:`config.PBP_COVERAGE_REPORT_PATH`.

    Returns:
        Process exit code: ``0`` on success, ``1`` if nothing was collected
        and the output file does not yet exist.
    """
    ensure_directories()

    if output_path is None:
        output_path = config.RAW_PLAY_BY_PLAY_PATH
    if games_path is None:
        games_path = config.RAW_GAMES_PATH
    if batch_size is None or batch_size <= 0:
        batch_size = DEFAULT_BATCH_SIZE

    rate_limit = get_rate_limit_seconds()
    unlimited = not limit or limit <= 0

    # --- load eligible games + current coverage -----------------------------
    eligible = load_eligible_games(games_path, season=season)
    already_collected = get_already_collected_game_ids(output_path)
    already_eligible_count = int(eligible["game_id"].isin(already_collected).sum())
    remaining_count = len(eligible) - already_eligible_count

    to_collect = filter_games_for_collection(
        eligible,
        already_collected=already_collected,
        game_id=game_id,
        limit=limit,
    )

    # --- progress plan ------------------------------------------------------
    print("Play-by-play collection plan:")
    print(f"  Season filter:               {season or 'all'}")
    print(f"  Eligible games:              {len(eligible)}")
    print(f"  Already collected:           {already_eligible_count}")
    print(f"  Remaining to collect:        {remaining_count}")
    print(f"  Requested limit:             {'unlimited' if unlimited else limit}")
    print(f"  Games selected this run:     {len(to_collect)}")
    print(f"  Batch save size:             {batch_size}")

    if to_collect.empty:
        if remaining_count <= 0 and len(eligible) > 0:
            print("\n  All eligible games are already collected. Nothing to do.")
        else:
            print("\n  Nothing new to collect.")
        record = _write_coverage_report(
            season, eligible, output_path, coverage_report_path=coverage_report_path
        )
        print(f"  Coverage: {record['collected_games']}/{record['eligible_games']} "
              f"games ({record['coverage_pct']}%), {record['total_event_rows']} event rows.")
        report_path = coverage_report_path or config.PBP_COVERAGE_REPORT_PATH
        print(f"  Coverage report:             {report_path}")
        return 0

    # --- per-game collection loop (with periodic batch saves) ---------------
    batch_frames: List[pd.DataFrame] = []
    invalid_frames: List[pd.DataFrame] = []
    failure_records: List[dict] = []
    succeeded_ids: List[str] = []
    success_count = 0
    fail_count = 0
    total_selected = len(to_collect)

    for position, (_, game_row) in enumerate(to_collect.iterrows(), start=1):
        gid = str(game_row["game_id"])
        print(f"  [{position}/{total_selected}] {gid} "
              f"({game_row['home_team']} vs {game_row['away_team']})...", end=" ")

        try:
            raw = fetch_play_by_play(gid)
        except Exception as exc:  # noqa: BLE001 - log and continue on any failure
            msg = str(exc)
            print(f"[failed] {msg[:80]}")
            log_api_attempt(game_id=gid, endpoint=ENDPOINT_NAME, status="failed", error_message=msg)
            failure_records.append(build_failure_record(game_row, ENDPOINT_NAME, "api_error", msg))
            fail_count += 1
            safe_sleep(rate_limit)
            continue

        if raw.empty:
            msg = "API returned zero rows."
            print("[empty]")
            log_api_attempt(game_id=gid, endpoint=ENDPOINT_NAME, status="failed", error_message=msg)
            failure_records.append(build_failure_record(game_row, ENDPOINT_NAME, "empty_response", msg))
            fail_count += 1
            safe_sleep(rate_limit)
            continue

        normalized = normalize_play_by_play_dataframe(raw, game_row)
        batch_frames.append(normalized)
        succeeded_ids.append(gid)
        log_api_attempt(game_id=gid, endpoint=ENDPOINT_NAME, status="success")
        success_count += 1
        print(f"[ok] {len(normalized)} events")
        safe_sleep(rate_limit)

        # Flush to disk after every full batch to protect progress.
        if len(batch_frames) >= batch_size:
            saved, invalid = _flush_batch(batch_frames, output_path)
            if not invalid.empty:
                invalid_frames.append(invalid)
            print(f"  ...batch saved ({saved} rows) -> {output_path}")
            batch_frames = []

    # Flush any remaining games from the final partial batch.
    if batch_frames:
        saved, invalid = _flush_batch(batch_frames, output_path)
        if not invalid.empty:
            invalid_frames.append(invalid)

    # --- invalid rows report ------------------------------------------------
    invalid_count = 0
    if invalid_frames:
        invalid_all = pd.concat(invalid_frames, ignore_index=True)
        invalid_count = len(invalid_all)
        if invalid_count > 0:
            save_csv(invalid_all, config.INVALID_PBP_REPORT_PATH)
            print(f"  [warning] {invalid_count} invalid row(s) -> {config.INVALID_PBP_REPORT_PATH}")

    # --- failures report (append/update, never duplicate a game) ------------
    if failure_records or config.PBP_FAILURES_REPORT_PATH.exists():
        existing_failures = None
        if config.PBP_FAILURES_REPORT_PATH.exists():
            existing_failures = pd.read_csv(
                config.PBP_FAILURES_REPORT_PATH, dtype={"game_id": str}
            )
        merged_failures = merge_failure_records(existing_failures, failure_records, succeeded_ids)
        if merged_failures.empty:
            # Everything previously failing now succeeds; remove the stale file.
            config.PBP_FAILURES_REPORT_PATH.unlink(missing_ok=True)
        else:
            save_csv(merged_failures, config.PBP_FAILURES_REPORT_PATH)

    # --- guard: nothing collected and no existing output --------------------
    if success_count == 0 and not output_path.exists():
        print("  [error] No valid play-by-play rows collected and no existing output file.")
        return 1

    # --- coverage + final summary -------------------------------------------
    final_df = pd.read_csv(output_path, dtype={"game_id": str})
    record = _write_coverage_report(
        season, eligible, output_path, coverage_report_path=coverage_report_path
    )

    print("\nDone.")
    print(f"  Eligible games:              {len(eligible)}")
    print(f"  Already collected (skipped): {already_eligible_count}")
    print(f"  Games selected this run:     {total_selected}")
    print(f"  Successful API pulls:        {success_count}")
    print(f"  Failed / empty pulls:        {fail_count}")
    print(f"  Invalid rows:                {invalid_count}")
    print(f"  Unique games now in output:  {final_df['game_id'].nunique()}")
    print(f"  Total rows now in output:    {len(final_df)}")
    print(f"  Coverage ({record['season']}):  "
          f"{record['collected_games']}/{record['eligible_games']} "
          f"({record['coverage_pct']}%)")
    print(f"  Output path:                 {output_path}")
    report_path = coverage_report_path or config.PBP_COVERAGE_REPORT_PATH
    print(f"  Coverage report:             {report_path}")
    if fail_count > 0:
        print(f"  Failure report:              {config.PBP_FAILURES_REPORT_PATH}")
    else:
        print("  Failure report:              none (all selected games succeeded)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run play-by-play collection."""
    parser = argparse.ArgumentParser(
        description="Collect raw NBA play-by-play data into data/raw/play_by_play.csv."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Maximum number of new games to collect (default: {DEFAULT_LIMIT}). "
             "Pass 0 for NO limit, i.e. collect every remaining eligible game "
             "(use with care — a full season is 1 200+ games).",
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Only collect games from this season, e.g. 2024-25.",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        dest="game_id",
        help="Collect only this single game ID, e.g. 0022400001.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        dest="batch_size",
        help=f"Save progress to disk after this many collected games "
             f"(default: {DEFAULT_BATCH_SIZE}).",
    )
    args = parser.parse_args(argv)

    # --limit 0 (or negative) means "no cap": collect all remaining games.
    limit = args.limit if args.limit > 0 else None

    print("Collecting NBA play-by-play data...")
    return collect_play_by_play(
        season=args.season,
        game_id=args.game_id,
        limit=limit,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    sys.exit(main())
