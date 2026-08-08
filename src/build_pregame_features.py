"""Build a leakage-safe pre-game feature dataset into ``data/processed/pregame_features.csv``.

This is the pre-game feature-engineering layer.  It reads the master
schedule from ``data/raw/games.csv`` and produces **one row per completed
game**, where every feature is computed using only games that happened
*strictly before* that game's date.  No information from the game itself (or
from later games) ever enters its own feature row.

LEAKAGE RULES (the whole point of this module):
* For each game, a team's rolling stats are built only from that team's
  *earlier* games (strictly earlier ``game_date``).  The current game is never
  included in its own features.
* Same-date games are treated as unavailable to each other (we use a strict
  ``<`` comparison on the date), since we have no intra-day ordering.
* ``data/raw/team_stats.csv`` holds **full-season aggregates** and is therefore
  NOT used here: those numbers already "know" how the rest of the season went
  and would leak the future into historical pre-game rows.

SOURCE OF FINAL SCORES (the target):
    ``games.csv`` is schedule metadata only — it carries NO score columns.  Final
    scores come from ``data/processed/game_results.csv`` (one row per game,
    derived from play-by-play in Build 6).  When that file is present, its
    ``home_score`` / ``away_score`` are merged onto the schedule by ``game_id``
    so that:
      * Date-based features (``games_played_before``, ``rest_days``) ARE built.
      * Outcome/score features (wins, losses, win_pct, points averages, recent
        win pct) ARE computed from each team's *strictly earlier* games only.
      * The target ``home_team_won`` is populated for every game with a result.
    This merge is leakage-safe: a final score is only ever used to label its own
    game (the target) and to feed *later* games' rolling history — never the
    current game's own features (the strict ``<`` date guard still applies).

    If ``game_results.csv`` is absent, the build degrades gracefully: date-based
    features are still produced, outcome/score features fall back to neutral
    priors / NaN, and the target is left null with a ``missing_target`` issue.

    ``data/raw/team_stats.csv`` (full-season aggregates) is still NOT used here,
    as it would leak end-of-season information into historical pre-game rows.

This build does NOT: build live features, train models, write prediction
scripts, create model files, or call nba_api.

Run directly:
    python src/build_pregame_features.py
    python src/build_pregame_features.py --recent-window 5
    python src/build_pregame_features.py --season 2024-25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Allow running this file directly (``python src/build_pregame_features.py``) by
# making the project root importable — same pattern as the collectors.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import validate_required_columns  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns the final pregame_features.csv must always contain, in output order.
REQUIRED_FEATURE_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "game_type",
    "home_games_played_before",
    "away_games_played_before",
    "home_wins_before",
    "away_wins_before",
    "home_losses_before",
    "away_losses_before",
    "home_win_pct_before",
    "away_win_pct_before",
    "win_pct_diff_before",
    "home_points_for_avg_before",
    "away_points_for_avg_before",
    "home_points_allowed_avg_before",
    "away_points_allowed_avg_before",
    "points_for_avg_diff_before",
    "points_allowed_avg_diff_before",
    "home_recent_win_pct_before",
    "away_recent_win_pct_before",
    "recent_win_pct_diff_before",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    "home_team_won",
]

# Columns that must never be null in a structurally valid feature row.  The
# target (home_team_won) is intentionally NOT here: when scores are unavailable
# the target is null, but the feature row itself is still structurally valid and
# the missing target is captured in the build-issues report instead.
NON_NULL_FEATURE_COLUMNS = ["game_id", "game_date", "home_team", "away_team"]

# Columns for the build-issues report.
ISSUE_REPORT_COLUMNS = ["game_id", "issue_type", "message"]

# Candidate (home, away) score column-name pairs to look for in games.csv.
SCORE_COLUMN_CANDIDATES = [
    ("home_score", "away_score"),
    ("home_points", "away_points"),
    ("home_pts", "away_pts"),
    ("pts_home", "pts_away"),
    ("HOME_PTS", "AWAY_PTS"),
]

# ---------------------------------------------------------------------------
# Documented MVP priors / fallbacks for missing history.
# ---------------------------------------------------------------------------
# A team's very first game of the season has no prior history, so win_pct and
# recent win pct fall back to a neutral 0.5 prior (no information = coin flip).
# Rest days are unknown for a first game and fall back to a typical value of 3.
NEUTRAL_WIN_PCT = 0.5
DEFAULT_REST_DAYS = 3.0
DEFAULT_RECENT_WINDOW = 5


# ---------------------------------------------------------------------------
# Loading / sorting
# ---------------------------------------------------------------------------

def sort_games(df: pd.DataFrame) -> pd.DataFrame:
    """Sort games by ``season``, then ``game_date``, then ``game_id``.

    A stable, deterministic order is required so rolling history is built in
    true chronological order.  ``game_id`` is the final tie-breaker.

    Args:
        df: A games DataFrame.

    Returns:
        A new, sorted DataFrame with a reset index.
    """
    return df.sort_values(["season", "game_date", "game_id"]).reset_index(drop=True)


def load_games_for_features(
    games_path: str | Path,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """Load completed games for feature building, keeping IDs as strings.

    Applies baseline filters: ``status == "final"`` and non-null core fields.
    The result is sorted chronologically (see :func:`sort_games`).

    Args:
        games_path: Path to the master schedule CSV.
        season: If given, restrict to this season label (e.g. ``"2024-25"``).

    Returns:
        A filtered, sorted DataFrame with ``game_id``/team-id columns as strings.

    Raises:
        FileNotFoundError: If ``games_path`` does not exist.
    """
    games_path = Path(games_path)
    if not games_path.exists():
        raise FileNotFoundError(
            f"Games schedule not found: {games_path}. "
            "Run 'python run_pipeline.py --mode collect_games' first."
        )

    df = pd.read_csv(
        games_path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )

    if "status" in df.columns:
        df = df[df["status"] == "final"].copy()
    for col in ["game_id", "game_date", "home_team", "away_team"]:
        if col in df.columns:
            df = df[df[col].notna()].copy()

    if season is not None and "season" in df.columns:
        df = df[df["season"] == season].copy()

    return sort_games(df)


# ---------------------------------------------------------------------------
# Game-results enrichment (Build 5 target source)
# ---------------------------------------------------------------------------

# Columns pulled from game_results.csv onto the schedule.  ``home_score`` /
# ``away_score`` are the first SCORE_COLUMN_CANDIDATES pair, so merging them in
# automatically activates the score-aware feature/target code path below.
GAME_RESULTS_MERGE_COLUMNS = ["game_id", "home_score", "away_score"]


def load_game_results(results_path: str | Path) -> Optional[pd.DataFrame]:
    """Load ``game_results.csv`` if it exists, keeping ``game_id`` as a string.

    Args:
        results_path: Path to ``data/processed/game_results.csv``.

    Returns:
        The results DataFrame, or ``None`` when the file is absent (the build
        then degrades to the no-score path instead of failing).
    """
    results_path = Path(results_path)
    if not results_path.exists():
        return None
    return pd.read_csv(results_path, dtype={"game_id": str})


def merge_game_results(
    games_df: pd.DataFrame,
    results_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Merge final scores from game results onto the schedule by ``game_id``.

    Adds ``home_score`` / ``away_score`` columns (left join), which lets the
    existing score-aware code path compute outcome features and the
    ``home_team_won`` target.  Games with no matching result keep null scores
    (and are reported as ``missing_target`` downstream).

    Args:
        games_df: The schedule DataFrame (``game_id`` as string).
        results_df: Output of :func:`load_game_results`, or ``None``.

    Returns:
        A games DataFrame with score columns merged in.  When ``results_df`` is
        ``None`` or empty, the input is returned unchanged.
    """
    if results_df is None or results_df.empty:
        return games_df

    available = [c for c in GAME_RESULTS_MERGE_COLUMNS if c in results_df.columns]
    if "game_id" not in available or len(available) < 3:
        # Without both score columns there is nothing useful to merge.
        return games_df

    res = results_df[available].copy()
    res["game_id"] = res["game_id"].astype(str)
    # If a result somehow appears twice, keep the last (incoming) row.
    res = res.drop_duplicates(subset=["game_id"], keep="last")

    merged = games_df.copy()
    merged["game_id"] = merged["game_id"].astype(str)
    # Drop any pre-existing score columns so the merge is the single source.
    merged = merged.drop(columns=["home_score", "away_score"], errors="ignore")
    return merged.merge(res, on="game_id", how="left")


# ---------------------------------------------------------------------------
# Score detection + target
# ---------------------------------------------------------------------------

def detect_score_columns(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """Return the (home_score, away_score) column names present in ``df``.

    Looks through :data:`SCORE_COLUMN_CANDIDATES` and returns the first pair
    where *both* columns exist.

    Args:
        df: A games DataFrame.

    Returns:
        A ``(home_col, away_col)`` tuple, or ``None`` if no score pair is found.
    """
    for home_col, away_col in SCORE_COLUMN_CANDIDATES:
        if home_col in df.columns and away_col in df.columns:
            return home_col, away_col
    return None


def calculate_home_team_won(
    row: pd.Series,
    score_columns: Optional[Tuple[str, str]],
) -> Optional[int]:
    """Compute the ``home_team_won`` target for one game from its scores.

    Args:
        row: A single game row.
        score_columns: The ``(home_col, away_col)`` score column names, or
            ``None`` if the schedule has no scores.

    Returns:
        ``1`` if the home team won, ``0`` if the away team won, or ``None`` when
        scores are unavailable or non-numeric (the target cannot be fabricated).
    """
    if score_columns is None:
        return None

    home_col, away_col = score_columns
    home_score = pd.to_numeric(row.get(home_col), errors="coerce")
    away_score = pd.to_numeric(row.get(away_col), errors="coerce")
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score == away_score:
        return None  # ties are not a valid binary target; treat as unknown
    return 1 if home_score > away_score else 0


# ---------------------------------------------------------------------------
# Team-game history (long format: one row per team per game)
# ---------------------------------------------------------------------------

def build_team_game_history(
    games_df: pd.DataFrame,
    score_columns: Optional[Tuple[str, str]],
) -> pd.DataFrame:
    """Expand games into a per-team history table used for rolling lookups.

    Produces two rows per game (home perspective and away perspective).  Each
    row records whether that team won and its points for/against — values that
    are only populated when ``score_columns`` is provided.  When scores are
    absent, ``won``/``points_for``/``points_allowed`` are ``NaN`` (unknown).

    Args:
        games_df: A games DataFrame (does not need to be pre-sorted).
        score_columns: The ``(home_col, away_col)`` score columns, or ``None``.

    Returns:
        A DataFrame with columns ``team_id`` (str), ``game_id`` (str),
        ``season``, ``game_date``, ``game_dt`` (datetime), ``is_home``,
        ``won``, ``points_for``, ``points_allowed``.
    """
    records: List[dict] = []
    for _, g in games_df.iterrows():
        home_won = calculate_home_team_won(g, score_columns)

        if score_columns is not None:
            home_col, away_col = score_columns
            home_pts = pd.to_numeric(g.get(home_col), errors="coerce")
            away_pts = pd.to_numeric(g.get(away_col), errors="coerce")
        else:
            home_pts = np.nan
            away_pts = np.nan

        game_dt = pd.to_datetime(g["game_date"], errors="coerce")

        records.append({
            "team_id": str(g["home_team_id"]),
            "game_id": str(g["game_id"]),
            "season": g["season"],
            "game_date": g["game_date"],
            "game_dt": game_dt,
            "is_home": True,
            "won": np.nan if home_won is None else float(home_won),
            "points_for": home_pts,
            "points_allowed": away_pts,
        })
        records.append({
            "team_id": str(g["away_team_id"]),
            "game_id": str(g["game_id"]),
            "season": g["season"],
            "game_date": g["game_date"],
            "game_dt": game_dt,
            "is_home": False,
            "won": np.nan if home_won is None else float(1 - home_won),
            "points_for": away_pts,
            "points_allowed": home_pts,
        })

    columns = [
        "team_id", "game_id", "season", "game_date", "game_dt",
        "is_home", "won", "points_for", "points_allowed",
    ]
    return pd.DataFrame(records, columns=columns)


def get_team_history_before_date(
    team_id: str,
    game_date: object,
    history_df: pd.DataFrame,
    season: Optional[str] = None,
) -> pd.DataFrame:
    """Return a team's history strictly before ``game_date``, sorted ascending.

    The strict ``<`` comparison is the core leakage guard: the current game and
    any same-date games are excluded. When ``season`` is given, only games from
    that season are included so rolling stats reset at season boundaries.

    Args:
        team_id: The team identifier (compared as a string).
        game_date: The current game's date (string or datetime).
        history_df: Output of :func:`build_team_game_history`.
        season: If given, restrict history to this season label only.

    Returns:
        The matching rows, sorted by ``game_dt`` then ``game_id`` (oldest first).
    """
    if history_df.empty or "team_id" not in history_df.columns:
        return history_df.copy()

    current_dt = pd.to_datetime(game_date, errors="coerce")
    mask = (history_df["team_id"] == str(team_id)) & (history_df["game_dt"] < current_dt)
    if season is not None and "season" in history_df.columns:
        mask &= history_df["season"] == season
    return history_df[mask].sort_values(["game_dt", "game_id"]).reset_index(drop=True)


def calculate_team_pregame_stats(
    team_id: str,
    game_date: object,
    history_df: pd.DataFrame,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    season: Optional[str] = None,
) -> dict:
    """Compute one team's pre-game rolling stats as of ``game_date``.

    Uses only games strictly before ``game_date``.  Applies documented neutral
    priors when there is no history, and leaves outcome/point stats as ``NaN``
    when the underlying scores are unknown (so nothing is fabricated).

    Args:
        team_id: The team identifier.
        game_date: The current game's date.
        history_df: Output of :func:`build_team_game_history`.
        recent_window: Number of most-recent prior games for the recent-form
            metric (default :data:`DEFAULT_RECENT_WINDOW`).
        season: If given, only prior games from this season count (season reset).

    Returns:
        A dict with ``games_played_before``, ``wins_before``, ``losses_before``,
        ``win_pct_before``, ``points_for_avg_before``,
        ``points_allowed_avg_before``, ``recent_win_pct_before`` and
        ``rest_days`` (NaN for a first game; filled later).
    """
    prior = get_team_history_before_date(
        team_id, game_date, history_df, season=season
    )
    games_played = len(prior)

    stats = {
        "games_played_before": games_played,
        "wins_before": 0,
        "losses_before": 0,
        "win_pct_before": NEUTRAL_WIN_PCT,
        "points_for_avg_before": np.nan,
        "points_allowed_avg_before": np.nan,
        "recent_win_pct_before": NEUTRAL_WIN_PCT,
        "rest_days": np.nan,
    }

    if games_played == 0:
        # First game of the season: neutral priors above are the answer.
        return stats

    won_values = prior["won"].dropna()
    points_for = prior["points_for"].dropna()
    points_allowed = prior["points_allowed"].dropna()

    if len(won_values) > 0:
        wins = int((won_values == 1).sum())
        losses = int((won_values == 0).sum())
        decided = wins + losses
        stats["wins_before"] = wins
        stats["losses_before"] = losses
        stats["win_pct_before"] = wins / decided if decided > 0 else NEUTRAL_WIN_PCT
    else:
        # Prior games exist but their outcomes are unknown (no scores). We must
        # not invent wins/losses; mark them unknown and keep a neutral win_pct.
        stats["wins_before"] = np.nan
        stats["losses_before"] = np.nan
        stats["win_pct_before"] = NEUTRAL_WIN_PCT

    if len(points_for) > 0:
        stats["points_for_avg_before"] = float(points_for.mean())
    if len(points_allowed) > 0:
        stats["points_allowed_avg_before"] = float(points_allowed.mean())

    recent_won = prior.tail(recent_window)["won"].dropna()
    if len(recent_won) > 0:
        stats["recent_win_pct_before"] = float(recent_won.mean())

    # Rest days: gap between this game and the team's most recent prior game.
    last_dt = prior["game_dt"].max()
    current_dt = pd.to_datetime(game_date, errors="coerce")
    if pd.notna(last_dt) and pd.notna(current_dt):
        stats["rest_days"] = float((current_dt - last_dt).days)

    return stats


def build_issue_record(game_id: str, issue_type: str, message: str) -> dict:
    """Build one row for the pre-game feature build-issues report.

    Args:
        game_id: The affected game's id.
        issue_type: Short label (e.g. ``"missing_target"``).
        message: Human-readable explanation.

    Returns:
        A dict with exactly :data:`ISSUE_REPORT_COLUMNS` keys.
    """
    return {"game_id": str(game_id), "issue_type": issue_type, "message": message}


def build_pregame_feature_rows(
    games_df: pd.DataFrame,
    recent_window: int = DEFAULT_RECENT_WINDOW,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Build the full pre-game feature table plus a list of build issues.

    Args:
        games_df: Completed games (will be sorted chronologically internally).
        recent_window: Recent-form window size.

    Returns:
        A ``(features_df, issues)`` tuple.  ``features_df`` has exactly
        :data:`REQUIRED_FEATURE_COLUMNS`; ``issues`` is a list of dicts built by
        :func:`build_issue_record`.
    """
    games_df = sort_games(games_df)
    score_columns = detect_score_columns(games_df)
    history = build_team_game_history(games_df, score_columns)

    rows: List[dict] = []
    issues: List[dict] = []

    for _, g in games_df.iterrows():
        game_id = str(g["game_id"])
        game_date = g["game_date"]

        home = calculate_team_pregame_stats(
            str(g["home_team_id"]), game_date, history, recent_window, season=g["season"]
        )
        away = calculate_team_pregame_stats(
            str(g["away_team_id"]), game_date, history, recent_window, season=g["season"]
        )

        target = calculate_home_team_won(g, score_columns)
        if target is None:
            issues.append(
                build_issue_record(
                    game_id,
                    "missing_target",
                    "home_team_won could not be computed (no usable score columns "
                    "in games.csv). Row kept for non-target features only.",
                )
            )

        row = {
            "game_id": game_id,
            "season": g["season"],
            "game_date": game_date,
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_team_id": str(g["home_team_id"]),
            "away_team_id": str(g["away_team_id"]),
            "game_type": g.get("game_type"),
            "home_games_played_before": home["games_played_before"],
            "away_games_played_before": away["games_played_before"],
            "home_wins_before": home["wins_before"],
            "away_wins_before": away["wins_before"],
            "home_losses_before": home["losses_before"],
            "away_losses_before": away["losses_before"],
            "home_win_pct_before": home["win_pct_before"],
            "away_win_pct_before": away["win_pct_before"],
            "home_points_for_avg_before": home["points_for_avg_before"],
            "away_points_for_avg_before": away["points_for_avg_before"],
            "home_points_allowed_avg_before": home["points_allowed_avg_before"],
            "away_points_allowed_avg_before": away["points_allowed_avg_before"],
            "home_recent_win_pct_before": home["recent_win_pct_before"],
            "away_recent_win_pct_before": away["recent_win_pct_before"],
            "home_rest_days": home["rest_days"],
            "away_rest_days": away["rest_days"],
            "home_team_won": np.nan if target is None else target,
        }
        rows.append(row)

    features = pd.DataFrame(rows)
    if features.empty:
        return pd.DataFrame(columns=REQUIRED_FEATURE_COLUMNS), issues

    # --- documented fallbacks for first-game / unknown values ---------------
    # Rest days are unknown for a team's first game; fill with a typical value.
    features["home_rest_days"] = features["home_rest_days"].fillna(DEFAULT_REST_DAYS)
    features["away_rest_days"] = features["away_rest_days"].fillna(DEFAULT_REST_DAYS)
    # Win-pct style features fall back to the neutral prior if still missing.
    for col in [
        "home_win_pct_before", "away_win_pct_before",
        "home_recent_win_pct_before", "away_recent_win_pct_before",
    ]:
        features[col] = features[col].fillna(NEUTRAL_WIN_PCT)
    # Note: point averages are intentionally left NaN when scores are unknown.

    # --- difference features (home perspective) -----------------------------
    features["win_pct_diff_before"] = (
        features["home_win_pct_before"] - features["away_win_pct_before"]
    )
    features["points_for_avg_diff_before"] = (
        features["home_points_for_avg_before"] - features["away_points_for_avg_before"]
    )
    features["points_allowed_avg_diff_before"] = (
        features["home_points_allowed_avg_before"] - features["away_points_allowed_avg_before"]
    )
    features["recent_win_pct_diff_before"] = (
        features["home_recent_win_pct_before"] - features["away_recent_win_pct_before"]
    )
    features["rest_days_diff"] = features["home_rest_days"] - features["away_rest_days"]

    features = features[REQUIRED_FEATURE_COLUMNS]
    return features.reset_index(drop=True), issues


def validate_pregame_features_dataframe(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the feature DataFrame into structurally valid and invalid rows.

    Confirms the required columns exist, then flags rows with a null in any
    :data:`NON_NULL_FEATURE_COLUMNS` field.  A null target does NOT make a row
    invalid (that is tracked separately as a build issue).

    Args:
        df: The feature DataFrame to validate.

    Returns:
        A ``(valid_df, invalid_df)`` tuple.

    Raises:
        ValueError: If a required column is missing entirely.
    """
    validate_required_columns(df, REQUIRED_FEATURE_COLUMNS, "pregame_features")

    if df.empty:
        return df.copy(), df.copy()

    null_mask = df[NON_NULL_FEATURE_COLUMNS].isnull().any(axis=1)
    valid_df = df[~null_mask].copy()
    invalid_df = df[null_mask].copy()
    return valid_df, invalid_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_pregame_features(
    season: Optional[str] = None,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
) -> int:
    """Build the pre-game feature dataset and write it to CSV.

    Args:
        season: If given, restrict to this season label.
        recent_window: Recent-form window size (default 5).
        games_path: Source schedule. Defaults to :data:`config.RAW_GAMES_PATH`.
        output_path: Destination. Defaults to :data:`config.PREGAME_FEATURES_PATH`.
        results_path: Final-score source merged in for the target. Defaults to
            :data:`config.GAME_RESULTS_PATH`.

    Returns:
        Process exit code: ``0`` on success, ``1`` if no valid feature rows
        could be produced.
    """
    ensure_directories()

    if games_path is None:
        games_path = config.RAW_GAMES_PATH
    if output_path is None:
        output_path = config.PREGAME_FEATURES_PATH
    if results_path is None:
        results_path = config.GAME_RESULTS_PATH

    print(f"  Input games path:            {games_path}")
    print(f"  Game results path:           {results_path}")

    games = load_games_for_features(games_path, season=season)
    print(f"  Games loaded:                {len(games)}")

    results_df = load_game_results(results_path)
    if results_df is None:
        print("  [warning] game_results.csv not found. Run "
              "'build_live_features' first to enable the home_team_won target.")
    else:
        games = merge_game_results(games, results_df)
        matched = int(games["home_score"].notna().sum()) if "home_score" in games else 0
        print(f"  Game results merged:         {matched}/{len(games)} games")

    score_columns = detect_score_columns(games)
    if score_columns is None:
        print("  [warning] No score columns available. Score-based features and "
              "the home_team_won target cannot be computed.")
    else:
        print(f"  Score columns detected:      {score_columns[0]} / {score_columns[1]}")

    eligible = len(games)
    print(f"  Games eligible for features: {eligible}")

    features, issues = build_pregame_feature_rows(games, recent_window=recent_window)

    valid, invalid = validate_pregame_features_dataframe(features)
    invalid_count = len(invalid)

    if invalid_count > 0:
        save_csv(invalid, config.INVALID_PREGAME_FEATURES_REPORT_PATH)
        print(f"  [warning] {invalid_count} invalid row(s) -> "
              f"{config.INVALID_PREGAME_FEATURES_REPORT_PATH}")

    if issues:
        issues_df = pd.DataFrame(issues, columns=ISSUE_REPORT_COLUMNS)
        save_csv(issues_df, config.PREGAME_FEATURE_BUILD_ISSUES_PATH)

    if valid.empty:
        print("  [error] No valid feature rows produced. Nothing saved.")
        return 1

    save_csv(valid, output_path)

    target_count = int(valid["home_team_won"].notna().sum())

    print("\nDone.")
    print(f"  Feature rows created:        {len(valid)}")
    print(f"  Rows with target set:        {target_count}")
    print(f"  Invalid rows:                {invalid_count}")
    print(f"  Build issues:                {len(issues)}")
    print(f"  Output path:                 {output_path}")
    if invalid_count > 0:
        print(f"  Invalid rows report:         {config.INVALID_PREGAME_FEATURES_REPORT_PATH}")
    if issues:
        print(f"  Build issues report:         {config.PREGAME_FEATURE_BUILD_ISSUES_PATH}")
    else:
        print("  Build issues report:         none")
    if target_count == 0:
        print("\n  NOTE: home_team_won is empty for all rows. Ensure "
              "game_results.csv exists\n        (run 'build_live_features') "
              "before training a model.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run pre-game feature building."""
    parser = argparse.ArgumentParser(
        description="Build leakage-safe pre-game features into "
        "data/processed/pregame_features.csv."
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=DEFAULT_RECENT_WINDOW,
        help=f"Number of previous games for recent-form features "
        f"(default: {DEFAULT_RECENT_WINDOW}).",
    )
    parser.add_argument(
        "--season",
        default=None,
        help="Only build features for this season, e.g. 2024-25.",
    )
    args = parser.parse_args(argv)

    print("Building pre-game features...")
    return build_pregame_features(
        season=args.season,
        recent_window=args.recent_window,
    )


if __name__ == "__main__":
    sys.exit(main())
