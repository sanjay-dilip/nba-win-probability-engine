"""Tests for 2025-26 NBA Finals showcase dashboard helpers (Build 20.8)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    build_finals_overview_display,
    build_finals_win_probability_chart,
    build_projected_path_display_table,
    finals_game_can_show_replay,
    find_next_upcoming_finals_prediction,
    format_finals_game_selector_label,
    format_finals_matchup,
    format_game_status_for_finals,
    format_projection_type,
)


def _sample_summary_row(**overrides) -> pd.Series:
    base = {
        "season": "2025-26",
        "game_id": "0042500401",
        "finals_game_number": 1,
        "game_date": "2026-06-03",
        "home_team": "San Antonio Spurs",
        "away_team": "New York Knicks",
        "game_status": "completed",
        "prediction_available": True,
        "replay_available": True,
        "final_result_available": True,
        "predicted_winner_start": "San Antonio Spurs",
        "predicted_winner_final": "New York Knicks",
        "final_winner": "New York Knicks",
        "start_prediction_correct_if_final": False,
        "final_prediction_correct_if_final": True,
        "notes": "",
    }
    base.update(overrides)
    return pd.Series(base)


def test_format_finals_matchup():
    row = _sample_summary_row()
    assert format_finals_matchup(row) == "New York Knicks @ San Antonio Spurs"


def test_format_finals_game_selector_label_if_necessary():
    row = _sample_summary_row(
        finals_game_number=6,
        game_id="",
        home_team="",
        away_team="",
        game_status="not_available_yet",
    )
    label = format_finals_game_selector_label(row)
    assert "Game 6" in label
    assert "if necessary" in label


def test_build_finals_overview_display_seven_rows():
    rows = [_sample_summary_row(finals_game_number=n) for n in range(1, 8)]
    rows[3] = _sample_summary_row(
        finals_game_number=4,
        game_id="",
        home_team="",
        away_team="",
        game_status="not_available_yet",
        prediction_available=False,
        replay_available=False,
        predicted_winner_start=None,
        predicted_winner_final=None,
        final_winner=None,
    )
    df = pd.DataFrame(rows)
    overview = build_finals_overview_display(df)
    assert len(overview) == 7
    assert list(overview["Game"].astype(int)) == list(range(1, 8))


def test_finals_game_can_show_replay():
    assert finals_game_can_show_replay(_sample_summary_row()) is True
    assert finals_game_can_show_replay(_sample_summary_row(replay_available=False)) is False
    assert finals_game_can_show_replay(_sample_summary_row(game_id="")) is False


def test_build_finals_win_probability_chart():
    preds = pd.DataFrame(
        {
            "game_id": ["0042500401"] * 3,
            "event_num": [1, 2, 3],
            "home_team": ["San Antonio Spurs"] * 3,
            "away_team": ["New York Knicks"] * 3,
            "period": [1, 1, 1],
            "pctimestring": ["12:00", "11:00", "10:00"],
            "seconds_remaining_game": [2880, 2820, 2760],
            "home_score": [0, 2, 4],
            "away_score": [0, 0, 2],
            "home_win_probability": [0.55, 0.56, 0.57],
            "event_msg_type": [12, 1, 1],
        }
    )
    fig = build_finals_win_probability_chart(preds, "San Antonio Spurs")
    assert fig.data[0].name == "San Antonio Spurs win probability"


def test_find_next_upcoming_finals_prediction():
    df = pd.DataFrame(
        [
            {
                "finals_game_number": 1,
                "game_id": "0042500401",
                "final_result_available": True,
                "pregame_prediction_available": True,
            },
            {
                "finals_game_number": 4,
                "game_id": "",
                "final_result_available": False,
                "pregame_prediction_available": False,
            },
            {
                "finals_game_number": 5,
                "game_id": "0042500405",
                "final_result_available": False,
                "pregame_prediction_available": True,
                "predicted_winner_pregame": "San Antonio Spurs",
            },
        ]
    )
    nxt = find_next_upcoming_finals_prediction(df)
    assert nxt is not None
    assert int(nxt["finals_game_number"]) == 5


def test_find_next_upcoming_finals_prediction_chooses_game_four():
    df = pd.DataFrame(
        [
            {
                "finals_game_number": 1,
                "game_id": "0042500401",
                "final_result_available": True,
                "pregame_prediction_available": True,
            },
            {
                "finals_game_number": 4,
                "game_id": "0042500404",
                "final_result_available": False,
                "pregame_prediction_available": True,
                "predicted_winner_pregame": "New York Knicks",
            },
        ]
    )
    nxt = find_next_upcoming_finals_prediction(df)
    assert nxt is not None
    assert int(nxt["finals_game_number"]) == 4


def test_summarize_projected_series_state_actual_vs_projected():
    from dashboard_utils import summarize_projected_series_state

    df = pd.DataFrame(
        [
            {
                "finals_game_number": 1,
                "series_team_a": "New York Knicks",
                "series_team_b": "San Antonio Spurs",
                "projection_type": "actual_official",
                "actual_winner": "New York Knicks",
                "team_a_wins_after": 1,
                "team_b_wins_after": 0,
                "series_over_after_game": False,
            },
            {
                "finals_game_number": 4,
                "series_team_a": "New York Knicks",
                "series_team_b": "San Antonio Spurs",
                "projection_type": "model_projected",
                "projected_winner_used": "New York Knicks",
                "team_a_wins_after": 3,
                "team_b_wins_after": 1,
                "series_over_after_game": False,
            },
        ]
    )
    state = summarize_projected_series_state(df)
    assert "New York Knicks 1" in state["actual_score_text"]
    assert "3" in state["projected_score_text"]


def test_format_projection_type_labels():
    assert format_projection_type("model_projected") == "Model projection"
    assert format_projection_type("actual_manual_override") == "Actual (manual correction)"
    assert format_projection_type("conditional_model_projected") == "Conditional model projection"


def test_format_game_status_for_finals_if_necessary_scheduled():
    assert format_game_status_for_finals("if_necessary_scheduled") == "If necessary, scheduled"
    assert format_game_status_for_finals("scheduled") == "Scheduled"


def test_format_finals_game_selector_label_if_necessary_scheduled():
    row = _sample_summary_row(
        finals_game_number=6,
        game_id="0042500406",
        home_team="New York Knicks",
        away_team="San Antonio Spurs",
        game_status="if_necessary_scheduled",
    )
    label = format_finals_game_selector_label(row)
    assert "If necessary, scheduled" in label
    assert "if necessary" in label


def test_build_projected_path_display_table_conditional_label():
    df = pd.DataFrame(
        [
            {
                "finals_game_number": 6,
                "home_team": "New York Knicks",
                "away_team": "San Antonio Spurs",
                "projection_type": "conditional_model_projected",
                "actual_winner": None,
                "predicted_winner_pregame": "San Antonio Spurs",
                "projected_winner_used": "San Antonio Spurs",
                "team_a_wins_before": 3,
                "team_b_wins_before": 2,
                "team_a_wins_after": 3,
                "team_b_wins_after": 3,
                "game_needed_under_projection": True,
                "notes": "",
            }
        ]
    )
    table = build_projected_path_display_table(df)
    assert table.iloc[0]["Path type"] == "Conditional model projection"
