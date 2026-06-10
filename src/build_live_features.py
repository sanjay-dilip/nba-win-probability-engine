"""Build model-ready live win-probability features from raw play-by-play.

This is the live feature-engineering layer for Build 6.  It reads raw
play-by-play events from ``data/raw/play_by_play.csv`` (joined to schedule
metadata from ``data/raw/games.csv``) and produces **one row per event** that
describes the game state *after* that event.  It also derives a game-level
results file from the play-by-play final scores.

Outputs:
* ``data/processed/live_features.csv`` — one row per play-by-play event.
* ``data/processed/game_results.csv``  — one row per game (final score, winner).

LEAKAGE RULES:
* The final outcome (``home_team_won``) is a *training label*, not a feature.
  It is derived from play-by-play final scores (allowed — the target may use the
  final result) and repeated on every event row, but downstream modeling must
  NOT feed it in as an input feature.
* Every per-event feature describes the state *as of that event* only.  Running
  scores are forward-filled from earlier events, never from later ones, so no
  future information leaks into an earlier event's row.

This build does NOT: train models, write prediction scripts, create model
files, rebuild or patch the pre-game feature builder, or call nba_api.

Run directly:
    python src/build_live_features.py
    python src/build_live_features.py --season 2024-25
    python src/build_live_features.py --game-id 0022400001
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Allow running this file directly (``python src/build_live_features.py``) by
# making the project root importable — same pattern as the other builders.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import validate_required_columns  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A regulation NBA period is 12 minutes; an overtime period is 5 minutes.
SECONDS_PER_REGULATION_PERIOD = 720
SECONDS_PER_OVERTIME_PERIOD = 300
REGULATION_PERIODS = 4

# Columns the final live_features.csv must always contain, in output order.
REQUIRED_LIVE_FEATURE_COLUMNS = [
    "game_id",
    "event_num",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "period",
    "pctimestring",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "home_score",
    "away_score",
    "score_margin_home",
    "abs_score_margin",
    "event_msg_type",
    "event_msg_action_type",
    "event_type_label",
    "is_scoring_event",
    "is_turnover",
    "is_foul",
    "is_timeout",
    "is_rebound",
    "is_free_throw",
    "is_field_goal_attempt",
    "home_team_won",
]

# Columns that must never be null in a structurally valid live-feature row.
# (seconds_remaining_game is intentionally excluded — it is null only for the
# small number of rows with an unparseable clock, which is recorded as a build
# issue rather than discarding the row.)
NON_NULL_LIVE_COLUMNS = [
    "game_id",
    "event_num",
    "period",
    "home_score",
    "away_score",
    "home_team_won",
]

# Columns the final game_results.csv must always contain, in output order.
REQUIRED_GAME_RESULTS_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "winner",
    "home_team_won",
    "source",
]

NON_NULL_RESULTS_COLUMNS = [
    "game_id",
    "home_score",
    "away_score",
    "winner",
    "home_team_won",
]

# Columns for the build-issues report.
ISSUE_REPORT_COLUMNS = ["game_id", "issue_type", "message", "row_count"]

RESULT_SOURCE = "play_by_play"

# Maps the classic numeric EVENTMSGTYPE codes to readable labels.
EVENT_TYPE_NUMERIC_MAP = {
    1: "made_shot",
    2: "missed_shot",
    3: "free_throw",
    4: "rebound",
    5: "turnover",
    6: "foul",
    7: "violation",
    8: "substitution",
    9: "timeout",
    10: "jump_ball",
    11: "ejection",
    12: "start_period",
    13: "end_period",
}

# Maps the PlayByPlayV3 string action types (lower-cased) to the same labels.
# The collected play_by_play.csv stores these strings in event_msg_type.
EVENT_TYPE_STRING_MAP = {
    "made shot": "made_shot",
    "missed shot": "missed_shot",
    "free throw": "free_throw",
    "rebound": "rebound",
    "turnover": "turnover",
    "foul": "foul",
    "violation": "violation",
    "substitution": "substitution",
    "timeout": "timeout",
    "jump ball": "jump_ball",
    "ejection": "ejection",
    "instant replay": "instant_replay",
    # "period" is refined into start_period / end_period using the action type.
    "period": "period",
}

# Pre-compiled patterns for clock parsing.
_ISO_DURATION_RE = re.compile(r"^PT(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$")
_CLOCK_MMSS_RE = re.compile(r"^(\d+):(\d{1,2}(?:\.\d+)?)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_play_by_play_for_features(
    pbp_path: str | Path,
    season: Optional[str] = None,
    game_id: Optional[str] = None,
) -> pd.DataFrame:
    """Load raw play-by-play events, keeping ``game_id`` as a string.

    Args:
        pbp_path: Path to ``data/raw/play_by_play.csv``.
        season: If given, restrict to this season label.
        game_id: If given, restrict to this single game.

    Returns:
        The loaded (optionally filtered) DataFrame.

    Raises:
        FileNotFoundError: If ``pbp_path`` does not exist.
    """
    pbp_path = Path(pbp_path)
    if not pbp_path.exists():
        raise FileNotFoundError(
            f"Play-by-play file not found: {pbp_path}. "
            "Run 'python run_pipeline.py --mode collect_play_by_play' first."
        )

    df = pd.read_csv(pbp_path, dtype={"game_id": str})
    df["event_num"] = pd.to_numeric(df["event_num"], errors="coerce")

    if season is not None and "season" in df.columns:
        df = df[df["season"] == season].copy()
    if game_id is not None:
        df = df[df["game_id"] == str(game_id)].copy()

    return df.reset_index(drop=True)


def load_games_metadata(games_path: str | Path) -> pd.DataFrame:
    """Load schedule metadata, keeping id columns as strings.

    Args:
        games_path: Path to ``data/raw/games.csv``.

    Returns:
        The games DataFrame.

    Raises:
        FileNotFoundError: If ``games_path`` does not exist.
    """
    games_path = Path(games_path)
    if not games_path.exists():
        raise FileNotFoundError(
            f"Games schedule not found: {games_path}. "
            "Run 'python run_pipeline.py --mode collect_games' first."
        )
    return pd.read_csv(
        games_path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )


# ---------------------------------------------------------------------------
# Small, testable helpers
# ---------------------------------------------------------------------------

def parse_pctimestring_to_seconds(value: object) -> float:
    """Parse a play clock string into seconds remaining in the period.

    Supports:
    * ISO 8601 durations from PlayByPlayV3, e.g. ``"PT10M50.00S"`` or
      ``"PT45.00S"``.
    * Traditional ``MM:SS`` strings, e.g. ``"10:50"`` (with optional decimals).

    Args:
        value: The raw clock value (string, number, or NaN/None).

    Returns:
        Seconds remaining in the period as a float, or ``NaN`` if the value is
        missing or cannot be parsed.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return np.nan

    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return np.nan

    iso_match = _ISO_DURATION_RE.match(text)
    if iso_match:
        minutes = float(iso_match.group(1)) if iso_match.group(1) else 0.0
        seconds = float(iso_match.group(2)) if iso_match.group(2) else 0.0
        return minutes * 60.0 + seconds

    mmss_match = _CLOCK_MMSS_RE.match(text)
    if mmss_match:
        return float(mmss_match.group(1)) * 60.0 + float(mmss_match.group(2))

    return np.nan


def seconds_remaining_in_game(period: object, seconds_remaining_period: float) -> float:
    """Convert period + seconds-left-in-period into seconds left in the game.

    For regulation periods 1-4::

        seconds_remaining_game = (4 - period) * 720 + seconds_remaining_period

    so it counts down toward 0 at the end of the 4th quarter.

    For overtime periods (> 4) there is no fixed "remaining" time, so we express
    them as *negative* seconds past the end of regulation::

        seconds_remaining_game = -1 * ((period - 5) * 300 + (300 - seconds_remaining_period))

    This is a deliberately simple MVP convention: regulation is positive and
    counts down to 0; overtime is negative and grows more negative as OT
    progresses.  (We are intentionally not modelling OT more precisely yet.)

    Args:
        period: The period number (1-based).
        seconds_remaining_period: Seconds left in that period.

    Returns:
        Seconds remaining in the game (negative during overtime), or ``NaN`` if
        either input is missing.
    """
    if pd.isna(period) or pd.isna(seconds_remaining_period):
        return np.nan

    period = int(period)
    if period <= REGULATION_PERIODS:
        return (REGULATION_PERIODS - period) * SECONDS_PER_REGULATION_PERIOD + seconds_remaining_period

    overtime_elapsed = (period - 5) * SECONDS_PER_OVERTIME_PERIOD + (
        SECONDS_PER_OVERTIME_PERIOD - seconds_remaining_period
    )
    return -1.0 * overtime_elapsed


def map_event_type_label(event_msg_type: object, action_type: object = None) -> str:
    """Map a raw event type to a readable label.

    Handles both the classic numeric ``EVENTMSGTYPE`` codes (1-13) and the
    PlayByPlayV3 string action types (e.g. ``"Made Shot"``).  Unknown or missing
    values map to ``"unknown"``.

    Args:
        event_msg_type: Numeric code or string action type.
        action_type: Optional sub-type used to refine ``"period"`` into
            ``start_period`` / ``end_period``.

    Returns:
        A snake_case label string.
    """
    if event_msg_type is None or (not isinstance(event_msg_type, str) and pd.isna(event_msg_type)):
        return "unknown"

    # Numeric code path (also handles numeric-looking strings like "4").
    numeric_code: Optional[int] = None
    if isinstance(event_msg_type, (int, np.integer)):
        numeric_code = int(event_msg_type)
    elif isinstance(event_msg_type, float) and float(event_msg_type).is_integer():
        numeric_code = int(event_msg_type)
    elif isinstance(event_msg_type, str) and event_msg_type.strip().isdigit():
        numeric_code = int(event_msg_type.strip())

    if numeric_code is not None:
        return EVENT_TYPE_NUMERIC_MAP.get(numeric_code, "unknown")

    # String action-type path.
    label = EVENT_TYPE_STRING_MAP.get(str(event_msg_type).strip().lower(), "unknown")
    if label == "period":
        sub = "" if action_type is None or pd.isna(action_type) else str(action_type).strip().lower()
        if sub == "start":
            return "start_period"
        if sub == "end":
            return "end_period"
        return "start_period"  # default a bare period marker to start_period
    return label


def compute_event_indicator_flags(label: str) -> dict:
    """Return the boolean event-indicator flags for a given event label.

    ``is_scoring_event`` is NOT included here — it depends on the running score
    delta and is computed in :func:`forward_fill_scores`.

    Args:
        label: An ``event_type_label`` produced by :func:`map_event_type_label`.

    Returns:
        A dict of boolean flags.
    """
    return {
        "is_turnover": label == "turnover",
        "is_foul": label == "foul",
        "is_timeout": label == "timeout",
        "is_rebound": label == "rebound",
        "is_free_throw": label == "free_throw",
        "is_field_goal_attempt": label in {"made_shot", "missed_shot"},
    }


def forward_fill_scores(game_events: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """Forward-fill running scores within a single game's events.

    The events are sorted by ``event_num``.  ``scoreHome`` / ``scoreAway`` are
    converted to numbers, forward-filled, and any leading gaps (before the first
    recorded score) are filled with 0.  Derived score columns are added.

    A running NBA score is monotonically non-decreasing, but some PlayByPlayV3
    feeds append out-of-order administrative events (e.g. ``instant_replay``
    reviews) that carry a *higher* ``event_num`` yet a *stale, lower* score.
    Left unchecked, such a trailing row would "rewind" the running score and, in
    a handful of games, corrupt the derived final score / ``home_team_won``
    target.  We therefore apply a cumulative maximum (in ``event_num`` order) so
    the running score can never decrease.

    Args:
        game_events: Raw events for ONE game.

    Returns:
        A ``(processed_df, has_any_score)`` tuple.  ``has_any_score`` is True
        when the game had at least one row with both raw scores present.
    """
    g = game_events.sort_values("event_num").copy()

    raw_home = pd.to_numeric(g.get("scoreHome"), errors="coerce")
    raw_away = pd.to_numeric(g.get("scoreAway"), errors="coerce")
    has_any_score = bool(raw_home.notna().any() and raw_away.notna().any())

    home_score = raw_home.ffill().fillna(0).cummax()
    away_score = raw_away.ffill().fillna(0).cummax()

    g["home_score"] = home_score.to_numpy()
    g["away_score"] = away_score.to_numpy()
    g["score_margin_home"] = g["home_score"] - g["away_score"]
    g["abs_score_margin"] = g["score_margin_home"].abs()

    # A scoring event is one where the combined score increased vs the prior
    # event (detectable from the forward-filled running score).
    total = g["home_score"] + g["away_score"]
    g["is_scoring_event"] = total.diff().fillna(0) > 0

    return g.reset_index(drop=True), has_any_score


def extract_game_result(
    game_events_with_scores: pd.DataFrame,
    has_any_score: bool,
    game_meta: dict,
) -> Optional[dict]:
    """Derive the final game result from a game's forward-filled scores.

    Args:
        game_events_with_scores: Output of :func:`forward_fill_scores`.
        has_any_score: Whether the game had any usable raw scores.
        game_meta: Dict of game-level metadata (team names/ids, season, date).

    Returns:
        A result dict with :data:`REQUIRED_GAME_RESULTS_COLUMNS` keys, or
        ``None`` when no usable scores exist (so nothing is fabricated).
    """
    if not has_any_score or game_events_with_scores.empty:
        return None

    # The final score is the highest running score reached by each team.  Using
    # the maximum (rather than the last row by event_num) is robust to
    # out-of-order trailing events such as instant-replay reviews, which can
    # carry a stale, lower score on a higher event_num.
    home_score = float(game_events_with_scores["home_score"].max())
    away_score = float(game_events_with_scores["away_score"].max())
    home_team_won = 1 if home_score > away_score else 0
    winner = game_meta["home_team"] if home_team_won == 1 else game_meta["away_team"]

    return {
        "game_id": str(game_meta["game_id"]),
        "season": game_meta["season"],
        "game_date": game_meta["game_date"],
        "home_team": game_meta["home_team"],
        "away_team": game_meta["away_team"],
        "home_team_id": game_meta["home_team_id"],
        "away_team_id": game_meta["away_team_id"],
        "home_score": home_score,
        "away_score": away_score,
        "winner": winner,
        "home_team_won": home_team_won,
        "source": RESULT_SOURCE,
    }


def dedupe_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate ``game_id`` + ``event_num`` rows, keeping the last.

    Args:
        df: Raw play-by-play events.

    Returns:
        A ``(deduped_df, removed_count)`` tuple.
    """
    before = len(df)
    deduped = (
        df.sort_values(["game_id", "event_num"])
        .drop_duplicates(subset=["game_id", "event_num"], keep="last")
        .reset_index(drop=True)
    )
    return deduped, before - len(deduped)


def build_issue_record(
    game_id: str,
    issue_type: str,
    message: str,
    row_count: int,
) -> dict:
    """Build one row for the live-feature build-issues report.

    Args:
        game_id: The affected game's id (may be empty for dataset-level issues).
        issue_type: Short label (e.g. ``"missing_scores"``).
        message: Human-readable explanation.
        row_count: Number of rows affected.

    Returns:
        A dict with exactly :data:`ISSUE_REPORT_COLUMNS` keys.
    """
    return {
        "game_id": str(game_id),
        "issue_type": issue_type,
        "message": message,
        "row_count": int(row_count),
    }


# ---------------------------------------------------------------------------
# Core build
# ---------------------------------------------------------------------------

def _game_meta_from_games(game_id: str, games_lookup: dict, fallback_row: pd.Series) -> Tuple[dict, bool]:
    """Resolve game-level metadata, preferring games.csv over play-by-play.

    Returns ``(meta, missing_from_games)``.  When the game is absent from
    games.csv, team ids are ``NaN`` and the play-by-play row supplies the names.
    """
    if game_id in games_lookup:
        row = games_lookup[game_id]
        meta = {
            "game_id": game_id,
            "season": row.get("season"),
            "game_date": row.get("game_date"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "home_team_id": row.get("home_team_id"),
            "away_team_id": row.get("away_team_id"),
        }
        return meta, False

    meta = {
        "game_id": game_id,
        "season": fallback_row.get("season"),
        "game_date": fallback_row.get("game_date"),
        "home_team": fallback_row.get("home_team"),
        "away_team": fallback_row.get("away_team"),
        "home_team_id": np.nan,
        "away_team_id": np.nan,
    }
    return meta, True


def build_live_feature_rows(
    pbp_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[dict], int]:
    """Build the live-feature table, the game-results table, and build issues.

    Args:
        pbp_df: Raw play-by-play events.
        games_df: Schedule metadata (for team ids and missing-game detection).

    Returns:
        A ``(features_df, results_df, issues, duplicates_removed)`` tuple.
    """
    issues: List[dict] = []

    deduped, dup_removed = dedupe_events(pbp_df)
    if dup_removed > 0:
        issues.append(
            build_issue_record(
                "", "duplicate_events",
                "Duplicate game_id + event_num rows removed (kept last).",
                dup_removed,
            )
        )

    games_lookup = {str(r["game_id"]): r for _, r in games_df.iterrows()}

    processed_frames: List[pd.DataFrame] = []
    result_records: List[dict] = []

    for game_id, group in deduped.groupby("game_id", sort=True):
        game_id = str(game_id)
        meta, missing = _game_meta_from_games(game_id, games_lookup, group.iloc[0])
        if missing:
            issues.append(
                build_issue_record(
                    game_id, "missing_from_games",
                    "game_id present in play_by_play.csv but not in games.csv; "
                    "team IDs unavailable.",
                    len(group),
                )
            )

        scored, has_any_score = forward_fill_scores(group)

        # Stamp authoritative game-level metadata onto every event row.
        for column in ["season", "game_date", "home_team", "away_team",
                       "home_team_id", "away_team_id"]:
            scored[column] = meta[column]

        result = extract_game_result(scored, has_any_score, meta)
        if result is None:
            issues.append(
                build_issue_record(
                    game_id, "missing_scores",
                    "No usable scores in play-by-play; final result and "
                    "home_team_won could not be determined.",
                    len(group),
                )
            )
            scored["home_team_won"] = np.nan
        else:
            result_records.append(result)
            scored["home_team_won"] = result["home_team_won"]

        processed_frames.append(scored)

    if processed_frames:
        features = pd.concat(processed_frames, ignore_index=True)
    else:
        features = pd.DataFrame(columns=REQUIRED_LIVE_FEATURE_COLUMNS)

    if not features.empty:
        features = _add_time_and_label_features(features, issues)
        features["game_id"] = features["game_id"].astype(str)
        features = features[REQUIRED_LIVE_FEATURE_COLUMNS].reset_index(drop=True)

    if result_records:
        results = pd.DataFrame(result_records, columns=REQUIRED_GAME_RESULTS_COLUMNS)
        results["game_id"] = results["game_id"].astype(str)
    else:
        results = pd.DataFrame(columns=REQUIRED_GAME_RESULTS_COLUMNS)

    return features, results, issues, dup_removed


def _add_time_and_label_features(features: pd.DataFrame, issues: List[dict]) -> pd.DataFrame:
    """Add clock, period-time, event-label and indicator-flag columns."""
    features["seconds_remaining_period"] = features["pctimestring"].apply(
        parse_pctimestring_to_seconds
    )

    # Record games where a non-empty clock value failed to parse.
    raw_clock = features["pctimestring"]
    bad_clock = raw_clock.notna() & (raw_clock.astype(str).str.strip() != "") & \
        features["seconds_remaining_period"].isna()
    if bad_clock.any():
        for game_id, count in features.loc[bad_clock, "game_id"].value_counts().items():
            issues.append(
                build_issue_record(
                    str(game_id), "invalid_pctimestring",
                    "One or more pctimestring values could not be parsed.",
                    int(count),
                )
            )

    features["seconds_remaining_game"] = features.apply(
        lambda r: seconds_remaining_in_game(r["period"], r["seconds_remaining_period"]),
        axis=1,
    )

    features["event_type_label"] = features.apply(
        lambda r: map_event_type_label(r["event_msg_type"], r.get("event_msg_action_type")),
        axis=1,
    )

    label = features["event_type_label"]
    features["is_turnover"] = label == "turnover"
    features["is_foul"] = label == "foul"
    features["is_timeout"] = label == "timeout"
    features["is_rebound"] = label == "rebound"
    features["is_free_throw"] = label == "free_throw"
    features["is_field_goal_attempt"] = label.isin(["made_shot", "missed_shot"])

    return features


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_live_features_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split live features into structurally valid and invalid rows.

    Args:
        df: The live-feature DataFrame.

    Returns:
        A ``(valid_df, invalid_df)`` tuple.

    Raises:
        ValueError: If a required column is missing entirely.
    """
    validate_required_columns(df, REQUIRED_LIVE_FEATURE_COLUMNS, "live_features")
    if df.empty:
        return df.copy(), df.copy()
    null_mask = df[NON_NULL_LIVE_COLUMNS].isnull().any(axis=1)
    return df[~null_mask].copy(), df[null_mask].copy()


def validate_game_results_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split game results into structurally valid and invalid rows.

    Args:
        df: The game-results DataFrame.

    Returns:
        A ``(valid_df, invalid_df)`` tuple.

    Raises:
        ValueError: If a required column is missing entirely.
    """
    validate_required_columns(df, REQUIRED_GAME_RESULTS_COLUMNS, "game_results")
    if df.empty:
        return df.copy(), df.copy()
    null_mask = df[NON_NULL_RESULTS_COLUMNS].isnull().any(axis=1)
    return df[~null_mask].copy(), df[null_mask].copy()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_live_features(
    season: Optional[str] = None,
    game_id: Optional[str] = None,
    pbp_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
) -> int:
    """Build live features + game results and write them to CSV.

    Args:
        season: If given, restrict to this season label.
        game_id: If given, restrict to this single game.
        pbp_path: Source play-by-play. Defaults to :data:`config.RAW_PLAY_BY_PLAY_PATH`.
        games_path: Source schedule. Defaults to :data:`config.RAW_GAMES_PATH`.
        output_path: Live-feature output. Defaults to :data:`config.LIVE_FEATURES_PATH`.
        results_path: Game-results output. Defaults to :data:`config.GAME_RESULTS_PATH`.

    Returns:
        Process exit code: ``0`` on success, ``1`` if no valid live features
        could be produced.
    """
    ensure_directories()

    if pbp_path is None:
        pbp_path = config.RAW_PLAY_BY_PLAY_PATH
    if games_path is None:
        games_path = config.RAW_GAMES_PATH
    if output_path is None:
        output_path = config.LIVE_FEATURES_PATH
    if results_path is None:
        results_path = config.GAME_RESULTS_PATH

    print(f"  Input play-by-play path:     {pbp_path}")
    print(f"  Input games path:            {games_path}")

    pbp = load_play_by_play_for_features(pbp_path, season=season, game_id=game_id)
    games = load_games_metadata(games_path)
    print(f"  Raw event rows loaded:       {len(pbp)}")
    print(f"  Games in play-by-play:       {pbp['game_id'].nunique()}")

    features, results, issues, dup_removed = build_live_feature_rows(pbp, games)
    print(f"  Duplicate event rows removed:{dup_removed:>5}")

    valid_features, invalid_features = validate_live_features_dataframe(features)
    valid_results, invalid_results = validate_game_results_dataframe(results)

    invalid_feature_count = len(invalid_features)
    invalid_result_count = len(invalid_results)

    if invalid_feature_count > 0:
        save_csv(invalid_features, config.INVALID_LIVE_FEATURES_REPORT_PATH)
        print(f"  [warning] {invalid_feature_count} invalid live row(s) -> "
              f"{config.INVALID_LIVE_FEATURES_REPORT_PATH}")
    if invalid_result_count > 0:
        save_csv(invalid_results, config.INVALID_GAME_RESULTS_REPORT_PATH)
        print(f"  [warning] {invalid_result_count} invalid result row(s) -> "
              f"{config.INVALID_GAME_RESULTS_REPORT_PATH}")

    if issues:
        issues_df = pd.DataFrame(issues, columns=ISSUE_REPORT_COLUMNS)
        save_csv(issues_df, config.LIVE_FEATURE_BUILD_ISSUES_PATH)

    if valid_features.empty:
        print("  [error] No valid live feature rows produced. Nothing saved.")
        return 1

    save_csv(valid_features, output_path)
    save_csv(valid_results, results_path)

    target_count = int(valid_features["home_team_won"].notna().sum())

    print("\nDone.")
    print(f"  Live feature rows created:   {len(valid_features)}")
    print(f"  Game result rows created:    {len(valid_results)}")
    print(f"  Rows with target set:        {target_count}")
    print(f"  Invalid live feature rows:   {invalid_feature_count}")
    print(f"  Invalid game result rows:    {invalid_result_count}")
    print(f"  Build issues:                {len(issues)}")
    print(f"  Live features output:        {output_path}")
    print(f"  Game results output:         {results_path}")
    if invalid_feature_count > 0:
        print(f"  Invalid live rows report:    {config.INVALID_LIVE_FEATURES_REPORT_PATH}")
    if invalid_result_count > 0:
        print(f"  Invalid results report:      {config.INVALID_GAME_RESULTS_REPORT_PATH}")
    if issues:
        print(f"  Build issues report:         {config.LIVE_FEATURE_BUILD_ISSUES_PATH}")
    else:
        print("  Build issues report:         none")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run live feature building."""
    parser = argparse.ArgumentParser(
        description="Build model-ready live features into "
        "data/processed/live_features.csv (and game_results.csv)."
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Only build features for this season, e.g. 2024-25.",
    )
    parser.add_argument(
        "--game-id",
        default=None,
        dest="game_id",
        help="Only build features for this single game id, e.g. 0022400001.",
    )
    args = parser.parse_args(argv)

    print("Building live features...")
    return build_live_features(season=args.season, game_id=args.game_id)


if __name__ == "__main__":
    sys.exit(main())
