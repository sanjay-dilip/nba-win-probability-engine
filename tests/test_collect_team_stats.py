"""Tests for the team-stats collection helpers.

All tests exercise pure helper logic and never call the NBA API.
``collect_team_stats.py`` imports ``nba_api`` lazily inside
``fetch_team_stats_for_season()``, so importing the module here is safe and
offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.collect_team_stats import (  # noqa: E402
    DEFAULT_SEASONS,
    DEDUP_KEY_COLUMNS,
    FAILURE_REPORT_COLUMNS,
    REQUIRED_TEAM_STATS_COLUMNS,
    build_failure_record,
    get_default_seasons,
    normalize_season_type,
    normalize_team_stats_dataframe,
    parse_seasons,
    validate_team_stats_dataframe,
)
from src.utils import append_or_update_csv  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / factories
# ---------------------------------------------------------------------------

def _raw_team_stats_rows(include_advanced: bool = False) -> pd.DataFrame:
    """Return a tiny mock LeagueDashTeamStats DataFrame (uppercase columns)."""
    rows = [
        {
            "TEAM_ID": 1610612738,
            "TEAM_NAME": "Boston Celtics",
            "GP": 82,
            "W": 60,
            "L": 22,
            "W_PCT": 0.732,
            "MIN": 48.3,
            "PTS": 120.6,
            "FGM": 42.1,
            "FGA": 88.5,
            "FG_PCT": 0.476,
            "FG3M": 16.5,
            "FG3A": 42.3,
            "FG3_PCT": 0.390,
            "FTM": 15.2,
            "FTA": 19.1,
            "FT_PCT": 0.795,
            "OREB": 10.1,
            "DREB": 34.2,
            "REB": 44.3,
            "AST": 26.8,
            "TOV": 12.4,
            "STL": 7.1,
            "BLK": 5.2,
            "PF": 18.3,
            "PLUS_MINUS": 9.4,
        },
        {
            "TEAM_ID": 1610612744,
            "TEAM_NAME": "Golden State Warriors",
            "GP": 82,
            "W": 46,
            "L": 36,
            "W_PCT": 0.561,
            "MIN": 48.2,
            "PTS": 115.5,
            "FGM": 43.0,
            "FGA": 89.1,
            "FG_PCT": 0.482,
            "FG3M": 15.0,
            "FG3A": 39.0,
            "FG3_PCT": 0.385,
            "FTM": 14.0,
            "FTA": 17.5,
            "FT_PCT": 0.800,
            "OREB": 9.5,
            "DREB": 33.0,
            "REB": 42.5,
            "AST": 29.5,
            "TOV": 14.0,
            "STL": 7.5,
            "BLK": 4.5,
            "PF": 19.0,
            "PLUS_MINUS": 3.1,
        },
    ]
    df = pd.DataFrame(rows)
    if include_advanced:
        df["OFF_RATING"] = [120.1, 117.5]
        df["DEF_RATING"] = [110.6, 114.2]
        df["NET_RATING"] = [9.5, 3.3]
        df["PACE"] = [98.2, 100.1]
        df["TS_PCT"] = [0.601, 0.588]
    return df


# ---------------------------------------------------------------------------
# parse_seasons
# ---------------------------------------------------------------------------

def test_parse_seasons_from_comma_string():
    assert parse_seasons("2022-23, 2023-24 ,2024-25") == [
        "2022-23",
        "2023-24",
        "2024-25",
    ]


def test_parse_seasons_from_list():
    assert parse_seasons(["2023-24", " 2024-25 "]) == ["2023-24", "2024-25"]


def test_get_default_seasons_uses_env(monkeypatch):
    monkeypatch.setenv("NBA_SEASONS", "2021-22,2022-23")
    assert get_default_seasons() == ["2021-22", "2022-23"]


def test_get_default_seasons_falls_back(monkeypatch):
    monkeypatch.delenv("NBA_SEASONS", raising=False)
    assert get_default_seasons() == list(DEFAULT_SEASONS)


# ---------------------------------------------------------------------------
# normalize_season_type
# ---------------------------------------------------------------------------

def test_normalize_season_type_regular():
    assert normalize_season_type("Regular Season") == "Regular Season"
    assert normalize_season_type("regular_season") == "Regular Season"
    assert normalize_season_type("  REGULAR  ") == "Regular Season"


def test_normalize_season_type_playoffs():
    assert normalize_season_type("Playoffs") == "Playoffs"
    assert normalize_season_type("playoff") == "Playoffs"


def test_normalize_season_type_invalid_raises():
    with pytest.raises(ValueError):
        normalize_season_type("Preseason")


def test_normalize_season_type_non_string_raises():
    with pytest.raises(ValueError):
        normalize_season_type(None)


# ---------------------------------------------------------------------------
# normalize_team_stats_dataframe
# ---------------------------------------------------------------------------

def test_normalize_has_required_columns():
    out = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    for col in REQUIRED_TEAM_STATS_COLUMNS:
        assert col in out.columns, f"Missing required column: {col}"


def test_normalize_stamps_season_metadata():
    out = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    assert (out["season"] == "2024-25").all()
    assert (out["season_type"] == "Regular Season").all()


def test_normalize_renames_main_columns():
    out = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    first = out.iloc[0]
    assert first["team_name"] == "Boston Celtics"
    assert first["games_played"] == 82
    assert first["wins"] == 60
    assert first["losses"] == 22
    assert first["points"] == 120.6
    assert first["plus_minus"] == 9.4


def test_normalize_team_id_is_string():
    out = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    assert pd.api.types.is_string_dtype(out["team_id"])
    assert out["team_id"].iloc[0] == "1610612738"


def test_normalize_includes_advanced_when_present():
    out = normalize_team_stats_dataframe(
        _raw_team_stats_rows(include_advanced=True), "2024-25", "Regular Season"
    )
    for col in ["off_rating", "def_rating", "net_rating", "pace", "ts_pct"]:
        assert col in out.columns
    assert out["off_rating"].iloc[0] == 120.1


def test_normalize_omits_advanced_when_absent():
    out = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    for col in ["off_rating", "def_rating", "net_rating", "pace", "ts_pct"]:
        assert col not in out.columns


def test_normalize_missing_source_column_raises():
    raw = _raw_team_stats_rows().drop(columns=["GP"])
    with pytest.raises(ValueError):
        normalize_team_stats_dataframe(raw, "2024-25", "Regular Season")


# ---------------------------------------------------------------------------
# validate_team_stats_dataframe
# ---------------------------------------------------------------------------

def test_validate_splits_valid_and_invalid():
    df = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    # Inject an invalid row (null games_played).
    bad = df.iloc[0].copy()
    bad["team_id"] = "9999999999"
    bad["games_played"] = None
    combined = pd.concat([df, bad.to_frame().T], ignore_index=True)
    valid, invalid = validate_team_stats_dataframe(combined)
    assert len(valid) == len(df)
    assert len(invalid) == 1
    assert invalid.iloc[0]["team_id"] == "9999999999"


def test_validate_raises_on_missing_column():
    df = pd.DataFrame({"season": ["2024-25"], "season_type": ["Regular Season"]})
    with pytest.raises(ValueError):
        validate_team_stats_dataframe(df)


def test_validate_empty_dataframe_returns_empty_pair():
    empty = pd.DataFrame(columns=REQUIRED_TEAM_STATS_COLUMNS)
    valid, invalid = validate_team_stats_dataframe(empty)
    assert valid.empty
    assert invalid.empty


# ---------------------------------------------------------------------------
# build_failure_record
# ---------------------------------------------------------------------------

def test_build_failure_record_has_required_columns():
    record = build_failure_record(
        "2024-25", "Regular Season", "LeagueDashTeamStats", "empty_response", "Zero rows."
    )
    assert list(record.keys()) == FAILURE_REPORT_COLUMNS
    assert record["season"] == "2024-25"
    assert record["season_type"] == "Regular Season"
    assert record["reason_failed"] == "empty_response"
    assert record["endpoint"] == "LeagueDashTeamStats"


# ---------------------------------------------------------------------------
# De-duplication by season + season_type + team_id (via append_or_update_csv)
# ---------------------------------------------------------------------------

def test_dedup_by_season_type_team_id(tmp_path):
    out_path = tmp_path / "team_stats.csv"
    df1 = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    append_or_update_csv(df1, out_path, key_columns=DEDUP_KEY_COLUMNS)

    # Re-collect the same season with one win total corrected.
    raw2 = _raw_team_stats_rows()
    raw2.loc[raw2["TEAM_NAME"] == "Boston Celtics", "W"] = 61
    df2 = normalize_team_stats_dataframe(raw2, "2024-25", "Regular Season")
    final = append_or_update_csv(df2, out_path, key_columns=DEDUP_KEY_COLUMNS)

    # No duplicate rows, and the latest version wins.
    assert len(final) == len(df1)
    celtics = final[final["team_id"] == "1610612738"].iloc[0]
    assert int(celtics["wins"]) == 61


def test_dedup_keeps_separate_season_types(tmp_path):
    out_path = tmp_path / "team_stats.csv"
    reg = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Regular Season")
    append_or_update_csv(reg, out_path, key_columns=DEDUP_KEY_COLUMNS)

    playoffs = normalize_team_stats_dataframe(_raw_team_stats_rows(), "2024-25", "Playoffs")
    final = append_or_update_csv(playoffs, out_path, key_columns=DEDUP_KEY_COLUMNS)

    # Same teams + season but a different season_type are kept as distinct rows.
    assert len(final) == len(reg) + len(playoffs)
