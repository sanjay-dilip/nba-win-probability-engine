"""Tests for Pre-game Predictor dashboard helpers.

Pure-function tests only — Streamlit is never launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    REQUIRED_PREGAME_PREDICTION_COLUMNS,
    build_game_label,
    build_games_catalog,
    build_pregame_feature_sections,
    build_team_comparison_frame,
    compute_pregame_filtered_summary,
    filter_games_catalog,
    missing_required_columns,
)


def _predictions_sample() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "game_id": "0022400001",
            "season": "2024-25",
            "game_date": "2024-11-12",
            "home_team": "Boston Celtics",
            "away_team": "Atlanta Hawks",
            "home_team_id": "100",
            "away_team_id": "200",
            "game_type": "regular_season",
            "home_win_probability": 0.80,
            "away_win_probability": 0.20,
            "predicted_winner": "Boston Celtics",
            "predicted_label": 1,
            "actual_home_team_won": 1,
            "prediction_correct": True,
        },
        {
            "game_id": "0022400002",
            "season": "2024-25",
            "game_date": "2024-11-12",
            "home_team": "Detroit Pistons",
            "away_team": "Miami Heat",
            "home_team_id": "101",
            "away_team_id": "201",
            "game_type": "regular_season",
            "home_win_probability": 0.51,
            "away_win_probability": 0.49,
            "predicted_winner": "Detroit Pistons",
            "predicted_label": 1,
            "actual_home_team_won": 0,
            "prediction_correct": False,
        },
        {
            "game_id": "0022400003",
            "season": "2024-25",
            "game_date": "2024-11-15",
            "home_team": "Boston Celtics",
            "away_team": "Charlotte Hornets",
            "home_team_id": "100",
            "away_team_id": "202",
            "game_type": "regular_season",
            "home_win_probability": 0.30,
            "away_win_probability": 0.70,
            "predicted_winner": "Charlotte Hornets",
            "predicted_label": 0,
            "actual_home_team_won": 1,
            "prediction_correct": False,
        },
    ])


def _features_row() -> pd.Series:
    return pd.Series({
        "game_id": "0022400001",
        "home_team": "Boston Celtics",
        "away_team": "Atlanta Hawks",
        "home_games_played_before": 10,
        "away_games_played_before": 10,
        "home_win_pct_before": 0.7,
        "away_win_pct_before": 0.4,
        "win_pct_diff_before": 0.3,
        "home_points_for_avg_before": 115.0,
        "away_points_for_avg_before": 108.0,
        "points_for_avg_diff_before": 7.0,
        "home_points_allowed_avg_before": 105.0,
        "away_points_allowed_avg_before": 112.0,
        "points_allowed_avg_diff_before": -7.0,
        "home_recent_win_pct_before": 0.8,
        "away_recent_win_pct_before": 0.4,
        "recent_win_pct_diff_before": 0.4,
        "home_rest_days": 2.0,
        "away_rest_days": 1.0,
        "rest_days_diff": 1.0,
    })


def test_pregame_required_columns_complete():
    assert missing_required_columns(_predictions_sample(), REQUIRED_PREGAME_PREDICTION_COLUMNS) == []


def test_pregame_required_columns_detects_missing():
    df = _predictions_sample().drop(columns=["predicted_winner"])
    assert "predicted_winner" in missing_required_columns(df, REQUIRED_PREGAME_PREDICTION_COLUMNS)


def test_filter_by_team_pregame():
    catalog = build_games_catalog(_predictions_sample())
    filtered = filter_games_catalog(catalog, team="Boston Celtics")
    assert len(filtered) == 2


def test_build_pregame_feature_sections():
    sections = build_pregame_feature_sections(_features_row())
    assert "Season record before game" in sections
    assert sections["Season record before game"]["Home win %"] == pytest.approx(0.7)


def test_build_team_comparison_frame_has_both_teams():
    comp = build_team_comparison_frame(_features_row())
    assert set(comp["team"]) == {"Boston Celtics", "Atlanta Hawks"}
    assert "Win %" in comp["metric"].values


def test_filtered_summary_metrics():
    summary = compute_pregame_filtered_summary(_predictions_sample())
    assert summary["game_count"] == 3
    assert summary["prediction_accuracy"] == pytest.approx(1 / 3)
    assert summary["most_confident_home_prob"] == pytest.approx(0.80)
    assert summary["closest_home_prob"] == pytest.approx(0.51)


def test_most_and_closest_game_labels():
    summary = compute_pregame_filtered_summary(_predictions_sample())
    assert "Boston Celtics vs Atlanta Hawks" in summary["most_confident_game"]
    assert "Detroit Pistons vs Miami Heat" in summary["closest_game"]
