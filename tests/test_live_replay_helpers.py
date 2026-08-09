"""Tests for Live Replay dashboard helpers.

Pure-function tests only — Streamlit is never launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    REQUIRED_LIVE_PREDICTION_COLUMNS,
    add_replay_columns,
    build_game_label,
    build_games_catalog,
    compute_game_elapsed_seconds,
    filter_games_catalog,
    format_game_clock,
    missing_required_columns,
    top_momentum_swings,
)


def _predictions_sample() -> pd.DataFrame:
    rows = []
    for game_id, home, away, date in [
        ("0022400001", "Boston Celtics", "Atlanta Hawks", "2024-11-12"),
        ("0022400002", "Detroit Pistons", "Miami Heat", "2024-11-12"),
        ("0022400003", "Boston Celtics", "Charlotte Hornets", "2024-11-15"),
    ]:
        for event_num, secs, prob in [(1, 2880.0, 0.55), (2, 2800.0, 0.60), (3, 100.0, 0.80)]:
            rows.append({
                "game_id": game_id,
                "event_num": event_num,
                "season": "2024-25",
                "game_date": date,
                "home_team": home,
                "away_team": away,
                "period": 1,
                "pctimestring": "PT06M00.00S",
                "seconds_remaining_period": 360.0,
                "seconds_remaining_game": secs,
                "home_score": 0.0,
                "away_score": 0.0,
                "score_margin_home": 0.0,
                "abs_score_margin": 0.0,
                "event_type_label": "made_shot",
                "home_win_probability": prob,
                "away_win_probability": 1.0 - prob,
                "predicted_winner": home,
                "predicted_label": 1,
                "actual_home_team_won": 1,
                "prediction_correct": True,
            })
    return pd.DataFrame(rows)


def test_build_game_label():
    row = pd.Series({
        "game_date": "2024-11-12",
        "home_team": "Boston Celtics",
        "away_team": "Atlanta Hawks",
    })
    assert build_game_label(row) == "2024-11-12 | Boston Celtics vs Atlanta Hawks"


def test_missing_required_columns_detects_gaps():
    df = _predictions_sample().drop(columns=["event_num"])
    missing = missing_required_columns(df)
    assert "event_num" in missing


def test_filter_by_season_and_team():
    catalog = build_games_catalog(_predictions_sample())
    by_team = filter_games_catalog(catalog, team="Boston Celtics")
    assert len(by_team) == 2
    assert all(
        (by_team["home_team"] == "Boston Celtics") | (by_team["away_team"] == "Boston Celtics")
    )


def test_filter_all_teams_returns_all():
    catalog = build_games_catalog(_predictions_sample())
    assert len(filter_games_catalog(catalog, team="All teams")) == len(catalog)


def test_compute_game_elapsed_seconds_increases_over_time():
    secs = pd.Series([2880.0, 2000.0, 100.0])
    elapsed = compute_game_elapsed_seconds(secs)
    assert elapsed.tolist() == pytest.approx([0.0, 880.0, 2780.0])


def test_probability_changes_computed():
    game = _predictions_sample()[_predictions_sample()["game_id"] == "0022400001"]
    enriched = add_replay_columns(game)
    assert enriched["probability_change"].iloc[0] == 0.0
    assert enriched["probability_change"].iloc[1] == pytest.approx(0.05)


def test_top_momentum_swings_sorted_by_abs_change():
    game = _predictions_sample()[_predictions_sample()["game_id"] == "0022400001"]
    swings = top_momentum_swings(game, n=2)
    assert len(swings) == 2
    # Largest swing is 0.80 - 0.60 = 0.20 on the last event.
    assert swings.iloc[0]["home_win_probability"] == pytest.approx(0.80)


def test_validate_required_columns_complete_sample():
    df = _predictions_sample()
    assert missing_required_columns(df) == []


def test_format_game_clock_on_sample_iso_duration():
    assert format_game_clock("PT06M00.00S") == "6:00"
