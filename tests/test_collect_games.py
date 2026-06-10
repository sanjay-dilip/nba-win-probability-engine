"""Tests for the game schedule collection helpers.

These tests exercise pure helper logic only. They never call the NBA API:
``collect_games.py`` imports ``nba_api`` lazily inside its fetch function, so
importing the module and its helpers here is safe and offline.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is importable when running pytest from anywhere.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.collect_games import (  # noqa: E402
    REQUIRED_GAME_COLUMNS,
    SKIPPED_REPORT_COLUMNS,
    build_game_rows_from_team_rows,
    parse_matchup_location,
    parse_seasons,
    season_type_to_game_type,
    summarize_raw_counts,
    validate_games_dataframe,
)
from src.utils import append_or_update_csv  # noqa: E402


def _team_row(game_id, team_id, team_name, matchup, game_date="2024-11-12"):
    """Build a single raw team-level row like LeagueGameFinder returns."""
    return {
        "GAME_ID": game_id,
        "TEAM_ID": team_id,
        "TEAM_NAME": team_name,
        "GAME_DATE": game_date,
        "MATCHUP": matchup,
    }


def _complete_game_rows() -> pd.DataFrame:
    """Two team rows representing a single complete game (Boston home)."""
    return pd.DataFrame(
        [
            _team_row("0022400001", 1610612738, "Boston Celtics", "BOS vs. MIA"),
            _team_row("0022400001", 1610612748, "Miami Heat", "MIA @ BOS"),
        ]
    )


def _build(df):
    """Convenience wrapper that returns (games_df, skipped_df)."""
    return build_game_rows_from_team_rows(df, season="2024-25", game_type="regular_season")


# ---------------------------------------------------------------------------
# parse_matchup_location
# ---------------------------------------------------------------------------
def test_parse_matchup_location_home():
    assert parse_matchup_location("BOS vs. MIA") == "home"


def test_parse_matchup_location_away():
    assert parse_matchup_location("MIA @ BOS") == "away"


def test_parse_matchup_location_invalid_raises():
    with pytest.raises(ValueError):
        parse_matchup_location("BOS - MIA")


def test_parse_matchup_location_non_string_raises():
    with pytest.raises(ValueError):
        parse_matchup_location(None)


# ---------------------------------------------------------------------------
# season_type_to_game_type
# ---------------------------------------------------------------------------
def test_season_type_to_game_type_mapping():
    assert season_type_to_game_type("Regular Season") == "regular_season"
    assert season_type_to_game_type("Playoffs") == "playoffs"
    # Unknown types fall back to regular_season.
    assert season_type_to_game_type("Preseason") == "regular_season"


# ---------------------------------------------------------------------------
# build_game_rows_from_team_rows — happy path
# ---------------------------------------------------------------------------
def test_complete_two_row_game_creates_one_final_row():
    games, skipped = _build(_complete_game_rows())

    assert len(games) == 1
    assert skipped.empty
    row = games.iloc[0]
    assert row["game_id"] == "0022400001"
    assert row["home_team"] == "Boston Celtics"
    assert row["away_team"] == "Miami Heat"
    assert row["home_team_id"] == 1610612738
    assert row["away_team_id"] == 1610612748
    assert row["season"] == "2024-25"
    assert row["game_type"] == "regular_season"
    assert row["status"] == "final"


def test_build_game_rows_has_required_columns():
    games, skipped = _build(_complete_game_rows())
    assert list(games.columns) == REQUIRED_GAME_COLUMNS
    assert list(skipped.columns) == SKIPPED_REPORT_COLUMNS


# ---------------------------------------------------------------------------
# build_game_rows_from_team_rows — skip diagnostics
# ---------------------------------------------------------------------------
def test_only_home_row_is_captured_as_skipped():
    df = pd.DataFrame([_team_row("G1", 1, "Home Team", "HOM vs. AWY")])
    games, skipped = _build(df)

    assert games.empty
    assert len(skipped) == 1
    assert skipped.iloc[0]["reason_skipped"] == "missing_away_row"
    assert skipped.iloc[0]["home_row_count"] == 1
    assert skipped.iloc[0]["away_row_count"] == 0


def test_only_away_row_is_captured_as_skipped():
    df = pd.DataFrame([_team_row("G1", 2, "Away Team", "AWY @ HOM")])
    games, skipped = _build(df)

    assert games.empty
    assert len(skipped) == 1
    assert skipped.iloc[0]["reason_skipped"] == "missing_home_row"


def test_invalid_matchup_format_is_captured_as_skipped():
    df = pd.DataFrame(
        [
            _team_row("G1", 1, "Home Team", "HOM - AWY"),  # no vs. / @
            _team_row("G1", 2, "Away Team", "AWY @ HOM"),
        ]
    )
    games, skipped = _build(df)

    assert games.empty
    assert len(skipped) == 1
    assert "invalid_matchup_format" in skipped.iloc[0]["reason_skipped"]


def test_duplicate_home_rows_is_captured_as_skipped():
    df = pd.DataFrame(
        [
            _team_row("G1", 1, "Home A", "HMA vs. AWY"),
            _team_row("G1", 3, "Home B", "HMB vs. AWY"),  # second home row
            _team_row("G1", 2, "Away Team", "AWY @ HOM"),
        ]
    )
    games, skipped = _build(df)

    assert games.empty
    assert len(skipped) == 1
    assert "multiple_home_rows" in skipped.iloc[0]["reason_skipped"]
    assert skipped.iloc[0]["home_row_count"] == 2
    assert skipped.iloc[0]["away_row_count"] == 1


def test_skipped_report_has_required_columns():
    df = pd.DataFrame([_team_row("G1", 1, "Home Team", "HOM vs. AWY")])
    _games, skipped = _build(df)
    assert list(skipped.columns) == SKIPPED_REPORT_COLUMNS
    # All diagnostic fields should be populated for the skipped group.
    for column in SKIPPED_REPORT_COLUMNS:
        assert pd.notna(skipped.iloc[0][column])


def test_valid_rows_returned_even_when_some_groups_skipped():
    # One complete game + one incomplete (home only) game in the same batch.
    df = pd.concat(
        [
            _complete_game_rows(),
            pd.DataFrame([_team_row("0022400099", 9, "Lonely Team", "LON vs. XXX")]),
        ],
        ignore_index=True,
    )
    games, skipped = _build(df)

    assert len(games) == 1
    assert games.iloc[0]["game_id"] == "0022400001"
    assert len(skipped) == 1
    assert skipped.iloc[0]["game_id"] == "0022400099"


# ---------------------------------------------------------------------------
# summarize_raw_counts
# ---------------------------------------------------------------------------
def test_summarize_raw_counts():
    df = _complete_game_rows()
    summary = summarize_raw_counts(df)
    assert summary["raw_row_count"] == 2
    assert summary["unique_game_id_count"] == 1
    assert summary["expected_game_count_from_raw_pairs"] == 1


# ---------------------------------------------------------------------------
# validate_games_dataframe
# ---------------------------------------------------------------------------
def test_validate_games_dataframe_splits_valid_and_invalid():
    games, _skipped = _build(_complete_game_rows())

    # Add an invalid row missing the away team.
    bad_row = games.iloc[0].copy()
    bad_row["game_id"] = "0022400002"
    bad_row["away_team"] = None
    games_with_bad = pd.concat([games, bad_row.to_frame().T], ignore_index=True)

    valid, invalid = validate_games_dataframe(games_with_bad)
    assert len(valid) == 1
    assert len(invalid) == 1
    assert invalid.iloc[0]["game_id"] == "0022400002"


def test_validate_games_dataframe_missing_column_raises():
    df = pd.DataFrame({"game_id": ["x"]})
    with pytest.raises(ValueError):
        validate_games_dataframe(df)


# ---------------------------------------------------------------------------
# parse_seasons
# ---------------------------------------------------------------------------
def test_parse_seasons_from_comma_string():
    assert parse_seasons("2021-22, 2022-23 ,2023-24") == [
        "2021-22",
        "2022-23",
        "2023-24",
    ]


def test_parse_seasons_from_list():
    assert parse_seasons(["2023-24", " 2024-25 "]) == ["2023-24", "2024-25"]


# ---------------------------------------------------------------------------
# Duplicate handling (via the shared append_or_update_csv utility)
# ---------------------------------------------------------------------------
def test_duplicate_game_ids_keep_latest(tmp_path):
    out_path = tmp_path / "games.csv"

    first, _ = _build(_complete_game_rows())
    append_or_update_csv(first, out_path, key_columns=["game_id"])

    # Re-collect the same game but with a corrected away team name.
    updated_rows = _complete_game_rows()
    updated_rows.loc[updated_rows["TEAM_NAME"] == "Miami Heat", "TEAM_NAME"] = "Miami HEAT"
    updated, _ = _build(updated_rows)
    final_df = append_or_update_csv(updated, out_path, key_columns=["game_id"])

    # No duplicate game_id, and the latest version wins.
    assert final_df["game_id"].is_unique
    assert len(final_df) == 1
    assert final_df.iloc[0]["away_team"] == "Miami HEAT"
