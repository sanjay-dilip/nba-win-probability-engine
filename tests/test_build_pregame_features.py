"""Tests for the pre-game feature builder helpers.

All tests exercise pure helper logic on small, hand-built DataFrames and never
call the NBA API.  ``build_pregame_features.py`` does not import ``nba_api`` at
all, so importing it here is safe and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.build_pregame_features import (  # noqa: E402
    DEFAULT_REST_DAYS,
    ISSUE_REPORT_COLUMNS,
    NEUTRAL_WIN_PCT,
    REQUIRED_FEATURE_COLUMNS,
    build_issue_record,
    build_pregame_feature_rows,
    build_pregame_features,
    build_team_game_history,
    calculate_home_team_won,
    calculate_team_pregame_stats,
    detect_score_columns,
    get_team_history_before_date,
    load_game_results,
    load_games_for_features,
    merge_game_results,
    sort_games,
    validate_pregame_features_dataframe,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _games_with_scores() -> pd.DataFrame:
    """Three games between team 100 and team 200, with final scores."""
    return pd.DataFrame(
        [
            # G1: team 100 (home) beats team 200, 110-100.
            {"game_id": "0022400001", "season": "2024-25", "game_date": "2024-11-01",
             "home_team": "Team A", "away_team": "Team B",
             "home_team_id": "100", "away_team_id": "200",
             "status": "final", "game_type": "regular_season",
             "home_score": 110, "away_score": 100},
            # G2: team 200 (home) beats team 100, 120-90.
            {"game_id": "0022400002", "season": "2024-25", "game_date": "2024-11-04",
             "home_team": "Team B", "away_team": "Team A",
             "home_team_id": "200", "away_team_id": "100",
             "status": "final", "game_type": "regular_season",
             "home_score": 120, "away_score": 90},
            # G3: team 100 (home) vs team 200 — the game we inspect.
            {"game_id": "0022400003", "season": "2024-25", "game_date": "2024-11-07",
             "home_team": "Team A", "away_team": "Team B",
             "home_team_id": "100", "away_team_id": "200",
             "status": "final", "game_type": "regular_season",
             "home_score": 101, "away_score": 99},
        ]
    )


def _games_no_scores() -> pd.DataFrame:
    """Same three games but without any score columns (current games.csv shape)."""
    return _games_with_scores().drop(columns=["home_score", "away_score"])


def _game_results() -> pd.DataFrame:
    """A game_results.csv-shaped frame matching the three _games_no_scores()."""
    return pd.DataFrame(
        [
            {"game_id": "0022400001", "home_score": 110, "away_score": 100,
             "home_team_won": 1, "source": "play_by_play"},
            {"game_id": "0022400002", "home_score": 120, "away_score": 90,
             "home_team_won": 1, "source": "play_by_play"},
            {"game_id": "0022400003", "home_score": 101, "away_score": 99,
             "home_team_won": 1, "source": "play_by_play"},
        ]
    )


def _history(rows: list[dict]) -> pd.DataFrame:
    """Build a team-game history DataFrame matching build_team_game_history()."""
    recs = []
    for r in rows:
        recs.append({
            "team_id": str(r["team_id"]),
            "game_id": str(r["game_id"]),
            "season": r.get("season", "2024-25"),
            "game_date": r["date"],
            "game_dt": pd.to_datetime(r["date"]),
            "is_home": r.get("is_home", True),
            "won": r["won"],
            "points_for": r.get("pf", np.nan),
            "points_allowed": r.get("pa", np.nan),
        })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# load_games_for_features / sort_games
# ---------------------------------------------------------------------------

def test_load_preserves_game_id_as_string(tmp_path):
    path = tmp_path / "games.csv"
    _games_no_scores().to_csv(path, index=False)
    df = load_games_for_features(path)
    assert pd.api.types.is_string_dtype(df["game_id"])
    assert df["game_id"].iloc[0] == "0022400001"
    assert pd.api.types.is_string_dtype(df["home_team_id"])


def test_load_sorts_by_season_date_id(tmp_path):
    path = tmp_path / "games.csv"
    shuffled = _games_no_scores().iloc[[2, 0, 1]]
    shuffled.to_csv(path, index=False)
    df = load_games_for_features(path)
    assert df["game_id"].tolist() == ["0022400001", "0022400002", "0022400003"]


def test_sort_games_orders_by_keys():
    df = _games_no_scores().iloc[[2, 1, 0]].reset_index(drop=True)
    ordered = sort_games(df)
    assert ordered["game_date"].tolist() == ["2024-11-01", "2024-11-04", "2024-11-07"]


# ---------------------------------------------------------------------------
# detect_score_columns / calculate_home_team_won
# ---------------------------------------------------------------------------

def test_detect_score_columns_found():
    assert detect_score_columns(_games_with_scores()) == ("home_score", "away_score")


def test_detect_score_columns_absent():
    assert detect_score_columns(_games_no_scores()) is None


def test_calculate_home_team_won_home_win():
    row = pd.Series({"home_score": 110, "away_score": 100})
    assert calculate_home_team_won(row, ("home_score", "away_score")) == 1


def test_calculate_home_team_won_away_win():
    row = pd.Series({"home_score": 90, "away_score": 120})
    assert calculate_home_team_won(row, ("home_score", "away_score")) == 0


def test_calculate_home_team_won_no_scores_returns_none():
    row = pd.Series({"home_team": "A"})
    assert calculate_home_team_won(row, None) is None


# ---------------------------------------------------------------------------
# game_results enrichment (load_game_results / merge_game_results)
# ---------------------------------------------------------------------------

def test_load_game_results_missing_returns_none(tmp_path):
    assert load_game_results(tmp_path / "nope.csv") is None


def test_load_game_results_preserves_string_id(tmp_path):
    path = tmp_path / "game_results.csv"
    _game_results().to_csv(path, index=False)
    df = load_game_results(path)
    assert pd.api.types.is_string_dtype(df["game_id"])
    assert df["game_id"].iloc[0] == "0022400001"


def test_merge_game_results_adds_score_columns():
    merged = merge_game_results(_games_no_scores(), _game_results())
    assert {"home_score", "away_score"}.issubset(merged.columns)
    # Scores land on the correct game_id (order-independent).
    by_id = merged.set_index("game_id")
    assert by_id.loc["0022400001", "home_score"] == 110
    assert by_id.loc["0022400002", "away_score"] == 90
    # The merge activates the score-aware path.
    assert detect_score_columns(merged) == ("home_score", "away_score")


def test_merge_game_results_none_returns_unchanged():
    games = _games_no_scores()
    merged = merge_game_results(games, None)
    assert detect_score_columns(merged) is None
    assert len(merged) == len(games)


def test_merge_game_results_unmatched_game_gets_null_score():
    results = _game_results().iloc[:2]  # drop result for game 0022400003
    merged = merge_game_results(_games_no_scores(), results)
    by_id = merged.set_index("game_id")
    assert pd.isna(by_id.loc["0022400003", "home_score"])


# ---------------------------------------------------------------------------
# get_team_history_before_date — strict date guard
# ---------------------------------------------------------------------------

def test_history_before_date_is_strict():
    hist = _history([
        {"team_id": "100", "game_id": "g1", "date": "2024-11-01", "won": 1.0},
        {"team_id": "100", "game_id": "g2", "date": "2024-11-03", "won": 0.0},
    ])
    # Querying "before 2024-11-03" must exclude the same-date game g2.
    prior = get_team_history_before_date("100", "2024-11-03", hist)
    assert prior["game_id"].tolist() == ["g1"]


def test_history_before_date_filters_team():
    hist = _history([
        {"team_id": "100", "game_id": "g1", "date": "2024-11-01", "won": 1.0},
        {"team_id": "200", "game_id": "g1", "date": "2024-11-01", "won": 0.0},
    ])
    prior = get_team_history_before_date("100", "2024-11-10", hist)
    assert len(prior) == 1
    assert prior["team_id"].iloc[0] == "100"


# ---------------------------------------------------------------------------
# calculate_team_pregame_stats
# ---------------------------------------------------------------------------

def test_first_game_neutral_priors():
    empty_hist = _history([])
    stats = calculate_team_pregame_stats("100", "2024-11-01", empty_hist)
    assert stats["games_played_before"] == 0
    assert stats["wins_before"] == 0
    assert stats["losses_before"] == 0
    assert stats["win_pct_before"] == NEUTRAL_WIN_PCT
    assert stats["recent_win_pct_before"] == NEUTRAL_WIN_PCT
    assert np.isnan(stats["rest_days"])
    assert np.isnan(stats["points_for_avg_before"])


def test_rolling_uses_only_prior_games():
    hist = _history([
        {"team_id": "100", "game_id": "g1", "date": "2024-11-01", "won": 1.0, "pf": 100, "pa": 90},
        {"team_id": "100", "game_id": "g2", "date": "2024-11-02", "won": 1.0, "pf": 110, "pa": 95},
        {"team_id": "100", "game_id": "g3", "date": "2024-11-03", "won": 0.0, "pf": 80, "pa": 88},
    ])
    # As of 2024-11-03, only g1 and g2 count (g3 is same-day / current).
    stats = calculate_team_pregame_stats("100", "2024-11-03", hist)
    assert stats["games_played_before"] == 2
    assert stats["wins_before"] == 2
    assert stats["losses_before"] == 0
    assert stats["win_pct_before"] == 1.0
    assert stats["points_for_avg_before"] == 105.0   # (100 + 110) / 2
    assert stats["points_allowed_avg_before"] == 92.5  # (90 + 95) / 2


def test_recent_win_pct_uses_last_n_games():
    rows = [
        {"team_id": "100", "game_id": f"g{i}", "date": f"2024-11-0{i}", "won": w}
        for i, w in enumerate([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], start=1)
    ]
    hist = _history(rows)
    stats = calculate_team_pregame_stats("100", "2024-11-10", hist, recent_window=5)
    # Overall win pct across 6 games is 3/6 = 0.5.
    assert stats["win_pct_before"] == 0.5
    # Recent (last 5) outcomes are [1, 1, 0, 0, 0] -> 2/5 = 0.4.
    assert stats["recent_win_pct_before"] == pytest.approx(0.4)


def test_rest_days_calculated():
    hist = _history([
        {"team_id": "100", "game_id": "g1", "date": "2024-11-01", "won": 1.0},
        {"team_id": "100", "game_id": "g2", "date": "2024-11-04", "won": 0.0},
    ])
    stats = calculate_team_pregame_stats("100", "2024-11-07", hist)
    # Most recent prior game was 2024-11-04 -> 3 days of rest.
    assert stats["rest_days"] == 3.0


def test_rolling_features_reset_across_seasons():
    """Prior-season games must not count toward the next season's rolling stats."""
    hist = _history([
        {
            "team_id": "100",
            "game_id": "g1",
            "season": "2023-24",
            "date": "2024-04-10",
            "won": 1.0,
            "pf": 120,
            "pa": 100,
        },
        {
            "team_id": "100",
            "game_id": "g2",
            "season": "2024-25",
            "date": "2024-10-25",
            "won": 0.0,
            "pf": 90,
            "pa": 100,
        },
    ])
    stats = calculate_team_pregame_stats(
        "100", "2024-11-01", hist, season="2024-25"
    )
    assert stats["games_played_before"] == 1
    assert stats["wins_before"] == 0
    assert stats["losses_before"] == 1
    assert stats["rest_days"] == 7.0


def test_rest_days_do_not_carry_from_previous_season():
    hist = _history([
        {
            "team_id": "100",
            "game_id": "g1",
            "season": "2023-24",
            "date": "2024-04-15",
            "won": 1.0,
        },
    ])
    stats = calculate_team_pregame_stats(
        "100", "2024-10-22", hist, season="2024-25"
    )
    assert stats["games_played_before"] == 0
    assert np.isnan(stats["rest_days"])


def test_unknown_outcomes_keep_neutral_win_pct():
    # Prior games exist but outcomes are unknown (no scores): wins/losses are
    # NaN (not fabricated) and win_pct stays at the neutral prior.
    hist = _history([
        {"team_id": "100", "game_id": "g1", "date": "2024-11-01", "won": np.nan},
        {"team_id": "100", "game_id": "g2", "date": "2024-11-02", "won": np.nan},
    ])
    stats = calculate_team_pregame_stats("100", "2024-11-05", hist)
    assert stats["games_played_before"] == 2
    assert np.isnan(stats["wins_before"])
    assert np.isnan(stats["losses_before"])
    assert stats["win_pct_before"] == NEUTRAL_WIN_PCT


# ---------------------------------------------------------------------------
# build_team_game_history
# ---------------------------------------------------------------------------

def test_build_history_two_rows_per_game_with_scores():
    hist = build_team_game_history(_games_with_scores(), ("home_score", "away_score"))
    assert len(hist) == 6  # 3 games x 2 perspectives
    g1 = hist[hist["game_id"] == "0022400001"]
    home_row = g1[g1["is_home"]].iloc[0]
    away_row = g1[~g1["is_home"]].iloc[0]
    assert home_row["won"] == 1.0
    assert away_row["won"] == 0.0
    assert home_row["points_for"] == 110
    assert away_row["points_for"] == 100


def test_build_history_won_is_nan_without_scores():
    hist = build_team_game_history(_games_no_scores(), None)
    assert hist["won"].isna().all()
    assert hist["points_for"].isna().all()


# ---------------------------------------------------------------------------
# build_pregame_feature_rows — full build, scores present
# ---------------------------------------------------------------------------

def test_required_output_columns_exist():
    features, _issues = build_pregame_feature_rows(_games_with_scores())
    assert list(features.columns) == REQUIRED_FEATURE_COLUMNS


def test_game_id_preserved_as_string_in_features():
    features, _ = build_pregame_feature_rows(_games_with_scores())
    assert features["game_id"].iloc[0] == "0022400001"
    assert all(isinstance(v, str) for v in features["game_id"])


def test_target_when_scores_exist():
    features, issues = build_pregame_feature_rows(_games_with_scores())
    by_id = features.set_index("game_id")
    assert by_id.loc["0022400001", "home_team_won"] == 1  # A beat B at home
    assert by_id.loc["0022400002", "home_team_won"] == 1  # B beat A at home
    assert by_id.loc["0022400003", "home_team_won"] == 1  # A 101-99 at home
    assert issues == []  # every game got a target


def test_difference_features_calculated():
    features, _ = build_pregame_feature_rows(_games_with_scores())
    g3 = features.set_index("game_id").loc["0022400003"]
    # Home=A prior pf avg=(110+90)/2=100, away=B prior pf avg=(100+120)/2=110.
    assert g3["home_points_for_avg_before"] == pytest.approx(100.0)
    assert g3["away_points_for_avg_before"] == pytest.approx(110.0)
    assert g3["points_for_avg_diff_before"] == pytest.approx(-10.0)
    assert g3["points_allowed_avg_diff_before"] == pytest.approx(10.0)
    assert g3["win_pct_diff_before"] == pytest.approx(0.0)  # both 0.5
    assert g3["rest_days_diff"] == pytest.approx(0.0)       # both rested 3 days


def test_first_game_features_in_full_build():
    features, _ = build_pregame_feature_rows(_games_with_scores())
    g1 = features.set_index("game_id").loc["0022400001"]
    assert g1["home_games_played_before"] == 0
    assert g1["away_games_played_before"] == 0
    assert g1["home_win_pct_before"] == NEUTRAL_WIN_PCT
    # Rest days for a first game fall back to the documented default.
    assert g1["home_rest_days"] == DEFAULT_REST_DAYS
    assert g1["away_rest_days"] == DEFAULT_REST_DAYS


# ---------------------------------------------------------------------------
# build_pregame_feature_rows — missing scores path
# ---------------------------------------------------------------------------

def test_missing_scores_produces_issue_not_fabricated_target():
    features, issues = build_pregame_feature_rows(_games_no_scores())
    # Target must be null for every row (never fabricated)...
    assert features["home_team_won"].isna().all()
    # ...and every game must be recorded as a build issue.
    assert len(issues) == len(features)
    assert all(rec["issue_type"] == "missing_target" for rec in issues)
    # Non-score features are still produced.
    g3 = features.set_index("game_id").loc["0022400003"]
    assert g3["home_games_played_before"] == 2
    assert g3["home_rest_days"] == 3.0


# ---------------------------------------------------------------------------
# validate_pregame_features_dataframe
# ---------------------------------------------------------------------------

def test_validate_splits_valid_and_invalid():
    features, _ = build_pregame_feature_rows(_games_with_scores())
    bad = features.iloc[0].copy()
    bad["game_id"] = "0022400099"
    bad["home_team"] = None
    combined = pd.concat([features, bad.to_frame().T], ignore_index=True)
    valid, invalid = validate_pregame_features_dataframe(combined)
    assert len(valid) == len(features)
    assert len(invalid) == 1
    assert invalid.iloc[0]["game_id"] == "0022400099"


def test_validate_null_target_is_not_invalid():
    # A null home_team_won (missing scores) does not make a row structurally
    # invalid — those rows still belong in the feature file.
    features, _ = build_pregame_feature_rows(_games_no_scores())
    valid, invalid = validate_pregame_features_dataframe(features)
    assert len(valid) == len(features)
    assert invalid.empty


def test_validate_raises_on_missing_column():
    df = pd.DataFrame({"game_id": ["x"]})
    with pytest.raises(ValueError):
        validate_pregame_features_dataframe(df)


# ---------------------------------------------------------------------------
# build_issue_record
# ---------------------------------------------------------------------------

def test_build_issue_record_columns():
    rec = build_issue_record("0022400001", "missing_target", "no scores")
    assert list(rec.keys()) == ISSUE_REPORT_COLUMNS
    assert rec["game_id"] == "0022400001"
    assert rec["issue_type"] == "missing_target"


# ---------------------------------------------------------------------------
# build_pregame_features — end-to-end orchestration with game_results.csv
# ---------------------------------------------------------------------------

def _redirect_report_paths(monkeypatch, tmp_path):
    """Point the build's report outputs at tmp_path so tests never touch real files."""
    import src.build_pregame_features as mod
    monkeypatch.setattr(
        mod.config, "PREGAME_FEATURE_BUILD_ISSUES_PATH",
        tmp_path / "issues.csv",
    )
    monkeypatch.setattr(
        mod.config, "INVALID_PREGAME_FEATURES_REPORT_PATH",
        tmp_path / "invalid.csv",
    )


def test_build_with_results_populates_target(tmp_path, monkeypatch):
    _redirect_report_paths(monkeypatch, tmp_path)
    games_path = tmp_path / "games.csv"
    results_path = tmp_path / "game_results.csv"
    output_path = tmp_path / "pregame_features.csv"
    _games_no_scores().to_csv(games_path, index=False)
    _game_results().to_csv(results_path, index=False)

    rc = build_pregame_features(
        games_path=games_path,
        output_path=output_path,
        results_path=results_path,
    )
    assert rc == 0

    out = pd.read_csv(output_path, dtype={"game_id": str})
    assert list(out.columns) == REQUIRED_FEATURE_COLUMNS
    # Every game now has a real, leakage-safe target from game_results.
    assert out["home_team_won"].notna().all()
    assert out.set_index("game_id").loc["0022400001", "home_team_won"] == 1


def test_build_without_results_leaves_target_null(tmp_path, monkeypatch):
    _redirect_report_paths(monkeypatch, tmp_path)
    games_path = tmp_path / "games.csv"
    output_path = tmp_path / "pregame_features.csv"
    missing_results = tmp_path / "absent_game_results.csv"
    _games_no_scores().to_csv(games_path, index=False)

    rc = build_pregame_features(
        games_path=games_path,
        output_path=output_path,
        results_path=missing_results,
    )
    assert rc == 0

    out = pd.read_csv(output_path, dtype={"game_id": str})
    assert out["home_team_won"].isna().all()
