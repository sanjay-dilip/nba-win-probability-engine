"""Tests for playoff / NBA Finals case-study helpers (Build 20.6).

No nba_api calls, no model training, no live data collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.playoff_case_study import (  # noqa: E402
    CASE_STUDY_COLUMNS,
    FINALS_LIVE_EXPORT_COLUMNS,
    FINALS_SCHEDULE_OVERRIDE_COLUMNS,
    PROJECTED_SERIES_COLUMNS,
    UPCOMING_PREDICTIONS_COLUMNS,
    add_finals_game_numbers,
    add_playoff_round_labels,
    apply_manual_overrides_to_finals_results,
    build_finals_live_predictions_export,
    run_export_finals_live_predictions_for_deploy,
    build_finals_pregame_features,
    build_finals_projected_series_report,
    build_finals_upcoming_predictions_report,
    build_merged_finals_games_df,
    build_nba_finals_case_study_summary,
    extract_playoff_round_from_game_id,
    filter_playoff_games,
    get_actual_finals_results,
    identify_nba_finals_games,
    is_nba_finals_game_id,
    load_finals_schedule_overrides,
    merge_finals_metadata_with_overrides,
    normalize_finals_schedule_overrides,
    normalize_season_type,
    project_finals_series_path,
    summarize_playoff_coverage,
    validate_finals_schedule_overrides,
)


def _sample_playoff_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "0042200401",
                "season": "2022-23",
                "season_type": "Playoffs",
                "game_date": "2023-06-01",
                "home_team": "DEN",
                "away_team": "MIA",
                "status": "final",
            },
            {
                "game_id": "0042200402",
                "season": "2022-23",
                "season_type": "Playoffs",
                "game_date": "2023-06-04",
                "home_team": "DEN",
                "away_team": "MIA",
                "status": "final",
            },
            {
                "game_id": "0042200403",
                "season": "2022-23",
                "season_type": "Playoffs",
                "game_date": "2023-06-07",
                "home_team": "MIA",
                "away_team": "DEN",
                "status": "final",
            },
            {
                "game_id": "0042200404",
                "season": "2022-23",
                "season_type": "Playoffs",
                "game_date": "2023-06-09",
                "home_team": "MIA",
                "away_team": "DEN",
                "status": "final",
            },
            {
                "game_id": "0042200311",
                "season": "2022-23",
                "season_type": "Playoffs",
                "game_date": "2023-05-18",
                "home_team": "BOS",
                "away_team": "PHI",
                "status": "final",
            },
        ]
    )


def test_normalize_season_type():
    assert normalize_season_type("Playoffs") == "Playoffs"
    assert normalize_season_type("playoff") == "Playoffs"
    assert normalize_season_type("Regular Season") == "Regular Season"
    assert normalize_season_type(None) == "Regular Season"


def test_filter_playoff_games():
    games = _sample_playoff_games()
    regular = pd.concat(
        [
            games,
            pd.DataFrame(
                [
                    {
                        "game_id": "0022200001",
                        "season": "2022-23",
                        "season_type": "Regular Season",
                        "game_date": "2022-10-18",
                        "home_team": "BOS",
                        "away_team": "PHI",
                        "status": "final",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    playoffs = filter_playoff_games(regular)
    assert len(playoffs) == 5
    assert all(playoffs["season_type"] == "Playoffs")


def test_identify_nba_finals_games():
    finals = identify_nba_finals_games(_sample_playoff_games())
    assert not finals.empty
    assert set(finals["home_team"].tolist() + finals["away_team"].tolist()) == {"DEN", "MIA"}
    assert len(finals) == 4
    assert all(finals["game_id"].map(is_nba_finals_game_id))


def test_extract_playoff_round_from_game_id():
    assert extract_playoff_round_from_game_id("0042500311") == "003"
    assert extract_playoff_round_from_game_id("0042500401") == "004"
    assert is_nba_finals_game_id("0042500403")
    assert not is_nba_finals_game_id("0042500317")
    assert extract_playoff_round_from_game_id("invalid") is None


def test_identify_nba_finals_games_2025_26_prefers_round_four():
    """Conference Finals (round 003) must not be chosen over NBA Finals (round 004)."""
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500311",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-05-18",
                "home_team": "Oklahoma City Thunder",
                "away_team": "San Antonio Spurs",
                "status": "final",
            },
            {
                "game_id": "0042500312",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-05-20",
                "home_team": "Oklahoma City Thunder",
                "away_team": "San Antonio Spurs",
                "status": "final",
            },
            {
                "game_id": "0042500313",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-05-22",
                "home_team": "San Antonio Spurs",
                "away_team": "Oklahoma City Thunder",
                "status": "final",
            },
            {
                "game_id": "0042500314",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-05-24",
                "home_team": "San Antonio Spurs",
                "away_team": "Oklahoma City Thunder",
                "status": "final",
            },
            {
                "game_id": "0042500315",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-05-26",
                "home_team": "Oklahoma City Thunder",
                "away_team": "San Antonio Spurs",
                "status": "final",
            },
            {
                "game_id": "0042500316",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-05-28",
                "home_team": "San Antonio Spurs",
                "away_team": "Oklahoma City Thunder",
                "status": "final",
            },
            {
                "game_id": "0042500317",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-05-30",
                "home_team": "Oklahoma City Thunder",
                "away_team": "San Antonio Spurs",
                "status": "final",
            },
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
            {
                "game_id": "0042500402",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-05",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
            {
                "game_id": "0042500403",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-08",
                "home_team": "New York Knicks",
                "away_team": "San Antonio Spurs",
                "status": "final",
            },
        ]
    )
    finals = identify_nba_finals_games(games)
    assert len(finals) == 3
    teams = set(finals["home_team"].tolist() + finals["away_team"].tolist())
    assert teams == {"San Antonio Spurs", "New York Knicks"}
    assert "Oklahoma City Thunder" not in teams
    assert list(finals.sort_values("game_date")["game_id"]) == [
        "0042500401",
        "0042500402",
        "0042500403",
    ]


def test_add_finals_game_numbers():
    labeled = add_finals_game_numbers(_sample_playoff_games())
    finals = labeled.loc[labeled["playoff_round"] == "NBA Finals"].sort_values("game_date")
    assert list(finals["finals_game_number"].astype(int)) == [1, 2, 3, 4]


def test_summarize_playoff_coverage():
    games = _sample_playoff_games()
    pbp = pd.DataFrame({"game_id": ["0042200401", "0042200402"]})
    summary = summarize_playoff_coverage(games, pbp)
    assert summary.iloc[0]["playoff_games"] == 5
    assert summary.iloc[0]["games_with_pbp"] == 2
    assert summary.iloc[0]["finals_games_detected"] == 4


def test_build_nba_finals_case_study_summary_always_seven_rows(tmp_path):
    games_2025 = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
            {
                "game_id": "0042500402",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-05",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
            {
                "game_id": "0042500403",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-08",
                "home_team": "New York Knicks",
                "away_team": "San Antonio Spurs",
                "status": "final",
            },
            {
                "game_id": "0042500404",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-10",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "scheduled",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games_2025.to_csv(games_path, index=False)

    summary = build_nba_finals_case_study_summary(
        focus_season="2025-26",
        games_path=games_path,
        predictions_path=tmp_path / "missing_preds.csv",
        pbp_path=tmp_path / "missing_pbp.csv",
        overrides_path=tmp_path / "no_overrides.csv",
    )
    assert list(summary.columns) == CASE_STUDY_COLUMNS
    assert len(summary) == 7
    assert list(summary["finals_game_number"].astype(int)) == list(range(1, 8))

    game4 = summary.loc[summary["finals_game_number"] == 4].iloc[0]
    assert game4["game_id"] == "0042500404"
    assert game4["game_status"] == "scheduled"
    assert game4["replay_available"] == False

    game5 = summary.loc[summary["finals_game_number"] == 5].iloc[0]
    assert game5["game_id"] == ""
    assert game5["game_status"] == "not_available_yet"
    assert "not yet collected" in game5["notes"]

    game6 = summary.loc[summary["finals_game_number"] == 6].iloc[0]
    assert game6["game_status"] == "not_available_yet"
    assert "If necessary" in game6["notes"] or "awaiting" in game6["notes"]

    game7 = summary.loc[summary["finals_game_number"] == 7].iloc[0]
    assert game7["game_status"] == "not_available_yet"
    assert "If necessary" in game7["notes"] or "awaiting" in game7["notes"]


def test_build_nba_finals_case_study_prediction_correctness_separate(tmp_path):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)

    preds = pd.DataFrame(
        {
            "game_id": ["0042500401", "0042500401"],
            "event_num": [1, 100],
            "home_win_probability": [0.55, 0.45],
            "predicted_winner": ["San Antonio Spurs", "New York Knicks"],
        }
    )
    preds_path = tmp_path / "playoff_live_predictions.csv"
    preds.to_csv(preds_path, index=False)

    results = pd.DataFrame(
        {"game_id": ["0042500401"], "winner": ["New York Knicks"]}
    )
    results_path = tmp_path / "playoff_game_results.csv"
    results.to_csv(results_path, index=False)

    pbp = pd.DataFrame({"game_id": ["0042500401"], "event_num": [1]})
    pbp_path = tmp_path / "playoff_play_by_play.csv"
    pbp.to_csv(pbp_path, index=False)

    summary = build_nba_finals_case_study_summary(
        focus_season="2025-26",
        games_path=games_path,
        predictions_path=preds_path,
        results_path=results_path,
        pbp_path=pbp_path,
        overrides_path=tmp_path / "no_overrides.csv",
    )
    game1 = summary.loc[summary["finals_game_number"] == 1].iloc[0]
    assert game1["predicted_winner_start"] == "San Antonio Spurs"
    assert game1["predicted_winner_final"] == "New York Knicks"
    assert game1["final_winner"] == "New York Knicks"
    assert game1["start_prediction_correct_if_final"] == False
    assert game1["final_prediction_correct_if_final"] == True
    assert game1["prediction_available"] == True
    assert game1["replay_available"] == True


def test_build_nba_finals_case_study_replay_false_without_pbp(tmp_path):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
            {
                "game_id": "0042500402",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-05",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "scheduled",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)

    summary = build_nba_finals_case_study_summary(
        focus_season="2025-26",
        games_path=games_path,
        predictions_path=tmp_path / "missing_preds.csv",
        pbp_path=tmp_path / "missing_pbp.csv",
        overrides_path=tmp_path / "no_overrides.csv",
    )
    game2 = summary.loc[summary["finals_game_number"] == 2].iloc[0]
    assert game2["game_id"] == "0042500402"
    assert game2["game_status"] == "scheduled"
    assert game2["replay_available"] == False
    assert game2["prediction_available"] == False


def test_build_finals_live_predictions_export_filters_to_finals_games(tmp_path):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
            {
                "game_id": "0042500402",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-05",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            },
            {
                "game_id": "0022500001",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-04-20",
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "status": "final",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)

    predictions = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "event_num": 1,
                "season": "2025-26",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "period": 1,
                "pctimestring": "12:00",
                "seconds_remaining_period": 720,
                "seconds_remaining_game": 2880,
                "home_score": 0,
                "away_score": 0,
                "score_margin_home": 0,
                "abs_score_margin": 0,
                "event_type_label": "Jump Ball",
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "predicted_winner": "San Antonio Spurs",
                "predicted_label": 1,
                "actual_home_team_won": 1,
                "prediction_correct": True,
            },
            {
                "game_id": "0042500402",
                "event_num": 1,
                "season": "2025-26",
                "game_date": "2026-06-05",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "period": 1,
                "pctimestring": "12:00",
                "seconds_remaining_period": 720,
                "seconds_remaining_game": 2880,
                "home_score": 0,
                "away_score": 0,
                "score_margin_home": 0,
                "abs_score_margin": 0,
                "event_type_label": "Jump Ball",
                "home_win_probability": 0.52,
                "away_win_probability": 0.48,
                "predicted_winner": "San Antonio Spurs",
                "predicted_label": 1,
                "actual_home_team_won": 0,
                "prediction_correct": False,
            },
            {
                "game_id": "0022500001",
                "event_num": 1,
                "season": "2025-26",
                "game_date": "2026-04-20",
                "home_team": "Boston Celtics",
                "away_team": "Miami Heat",
                "period": 1,
                "pctimestring": "12:00",
                "seconds_remaining_period": 720,
                "seconds_remaining_game": 2880,
                "home_score": 0,
                "away_score": 0,
                "score_margin_home": 0,
                "abs_score_margin": 0,
                "event_type_label": "Jump Ball",
                "home_win_probability": 0.60,
                "away_win_probability": 0.40,
                "predicted_winner": "Boston Celtics",
                "predicted_label": 1,
                "actual_home_team_won": 1,
                "prediction_correct": True,
            },
        ]
    )
    predictions_path = tmp_path / "playoff_live_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    export_df = build_finals_live_predictions_export(
        focus_season="2025-26",
        games_path=games_path,
        predictions_path=predictions_path,
    )

    assert list(export_df.columns) == FINALS_LIVE_EXPORT_COLUMNS
    assert set(export_df["game_id"]) == {"0042500401", "0042500402"}
    assert list(export_df.loc[export_df["game_id"] == "0042500401", "finals_game_number"]) == [1]
    assert list(export_df.loc[export_df["game_id"] == "0042500402", "finals_game_number"]) == [2]


def test_build_finals_live_predictions_export_missing_files_returns_empty(tmp_path):
    export_df = build_finals_live_predictions_export(
        focus_season="2025-26",
        games_path=tmp_path / "missing_games.csv",
        predictions_path=tmp_path / "missing_preds.csv",
    )
    assert list(export_df.columns) == FINALS_LIVE_EXPORT_COLUMNS
    assert export_df.empty


def test_run_export_finals_live_predictions_writes_file_and_reports_counts(tmp_path, monkeypatch, capsys):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
            }
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)

    predictions = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "event_num": 1,
                "season": "2025-26",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "period": 1,
                "pctimestring": "12:00",
                "seconds_remaining_period": 720,
                "seconds_remaining_game": 2880,
                "home_score": 0,
                "away_score": 0,
                "score_margin_home": 0,
                "abs_score_margin": 0,
                "event_type_label": "Jump Ball",
                "home_win_probability": 0.55,
                "away_win_probability": 0.45,
                "predicted_winner": "San Antonio Spurs",
                "predicted_label": 1,
                "actual_home_team_won": 1,
                "prediction_correct": True,
            }
        ]
    )
    predictions_path = tmp_path / "playoff_live_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    deploy_path = tmp_path / "deploy" / "finals_live_predictions.csv"
    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", games_path)
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", predictions_path)
    monkeypatch.setattr(config, "FINALS_LIVE_PREDICTIONS_DEPLOY_PATH", deploy_path)

    rc = run_export_finals_live_predictions_for_deploy()

    assert rc == 0
    assert deploy_path.exists()
    written = pd.read_csv(deploy_path, dtype={"game_id": str})
    assert set(written["game_id"]) == {"0042500401"}
    out = capsys.readouterr().out
    assert "Rows:  1" in out
    assert "Games: 1" in out


def test_run_export_finals_live_predictions_fails_loud_when_empty(tmp_path, monkeypatch, capsys):
    deploy_path = tmp_path / "deploy" / "finals_live_predictions.csv"
    deploy_path.parent.mkdir(parents=True)
    deploy_path.write_text("season,game_id\n2025-26,0042500401\n")

    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", tmp_path / "missing_games.csv")
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", tmp_path / "missing_preds.csv")
    monkeypatch.setattr(config, "FINALS_LIVE_PREDICTIONS_DEPLOY_PATH", deploy_path)

    rc = run_export_finals_live_predictions_for_deploy()

    assert rc == 1
    out = capsys.readouterr().out
    assert "ERROR" in out
    # existing deploy file must not be clobbered with an empty export
    assert deploy_path.read_text() == "season,game_id\n2025-26,0042500401\n"


def test_playoff_paths_separate_from_regular_season():
    assert config.PLAYOFF_GAMES_PATH != config.RAW_GAMES_PATH
    assert config.PLAYOFF_PLAY_BY_PLAY_PATH != config.RAW_PLAY_BY_PLAY_PATH
    assert config.PLAYOFF_LIVE_FEATURES_PATH != config.LIVE_FEATURES_PATH
    assert config.PLAYOFF_LIVE_PREDICTIONS_PATH != config.LIVE_PREDICTIONS_PATH
    assert config.FINALS_PREGAME_FEATURES_PATH != config.PREGAME_FEATURES_PATH
    assert config.FINALS_PREGAME_PREDICTIONS_PATH != config.PREGAME_PREDICTIONS_PATH
    assert config.FINALS_SCHEDULE_OVERRIDES_PATH != config.PLAYOFF_GAMES_PATH
    assert "manual" in str(config.FINALS_SCHEDULE_OVERRIDES_PATH)
    assert "playoffs" in str(config.PLAYOFF_PROCESSED_DIR)


def test_build_finals_upcoming_predictions_report_seven_rows(tmp_path, monkeypatch):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)
    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", games_path)
    monkeypatch.setattr(config, "PLAYOFF_PLAY_BY_PLAY_PATH", tmp_path / "missing_pbp.csv")
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", tmp_path / "missing_preds.csv")
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", tmp_path / "missing_results.csv")
    monkeypatch.setattr(config, "FINALS_PREGAME_PREDICTIONS_PATH", tmp_path / "missing_pregame.csv")
    monkeypatch.setattr(config, "FINALS_SCHEDULE_OVERRIDES_PATH", tmp_path / "no_overrides.csv")

    report = build_finals_upcoming_predictions_report(focus_season="2025-26")
    assert list(report.columns) == UPCOMING_PREDICTIONS_COLUMNS
    assert len(report) == 7
    game1 = report.loc[report["finals_game_number"] == 1].iloc[0]
    assert game1["pregame_prediction_available"] == False
    game5 = report.loc[report["finals_game_number"] == 5].iloc[0]
    assert game5["pregame_prediction_available"] == False
    assert "not yet collected" in str(game5["notes"]).lower() or game5["notes"]


def test_build_finals_pregame_features_uses_history_not_target(tmp_path, monkeypatch):
    """Feature rows are built from pre-game history; target label optional."""
    playoff_games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    regular_games = pd.DataFrame(
        [
            {
                "game_id": "0022500001",
                "season": "2025-26",
                "game_date": "2025-10-20",
                "home_team": "San Antonio Spurs",
                "away_team": "Team A",
                "home_team_id": "1610612759",
                "away_team_id": "1610612750",
                "status": "final",
                "game_type": "regular_season",
            },
        ]
    )
    regular_results = pd.DataFrame(
        {"game_id": ["0022500001"], "home_score": [110], "away_score": [100], "winner": ["San Antonio Spurs"]}
    )
    playoff_results = pd.DataFrame(
        {"game_id": ["0042500401"], "home_score": [105], "away_score": [102], "winner": ["San Antonio Spurs"]}
    )

    pg_path = tmp_path / "playoff_games.csv"
    rg_path = tmp_path / "regular_games.csv"
    rr_path = tmp_path / "regular_results.csv"
    pr_path = tmp_path / "playoff_results.csv"
    out_path = tmp_path / "finals_features.csv"
    playoff_games.to_csv(pg_path, index=False)
    regular_games.to_csv(rg_path, index=False)
    regular_results.to_csv(rr_path, index=False)
    playoff_results.to_csv(pr_path, index=False)

    monkeypatch.setattr(config, "RAW_GAMES_PATH", rg_path)
    monkeypatch.setattr(config, "GAME_RESULTS_PATH", rr_path)
    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", pg_path)
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", pr_path)
    monkeypatch.setattr(config, "FINALS_PREGAME_FEATURES_PATH", out_path)
    monkeypatch.setattr(config, "FINALS_SCHEDULE_OVERRIDES_PATH", tmp_path / "no_overrides.csv")

    features = build_finals_pregame_features(games_path=pg_path, output_path=out_path)
    assert len(features) == 1
    assert features.iloc[0]["game_id"] == "0042500401"
    assert "home_win_pct_before" in features.columns
    assert features.iloc[0]["home_games_played_before"] >= 1


def test_upcoming_report_merges_pregame_prediction(tmp_path, monkeypatch):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
            {
                "game_id": "0042500402",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-10",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "scheduled",
                "game_type": "playoffs",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)
    pregame_preds = pd.DataFrame(
        {
            "game_id": ["0042500402"],
            "home_win_probability": [0.58],
            "away_win_probability": [0.42],
            "predicted_winner": ["San Antonio Spurs"],
            "predicted_label": [1],
            "actual_home_team_won": [None],
            "prediction_correct": [None],
        }
    )
    pregame_path = tmp_path / "finals_pregame_predictions.csv"
    pregame_preds.to_csv(pregame_path, index=False)

    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", games_path)
    monkeypatch.setattr(config, "PLAYOFF_PLAY_BY_PLAY_PATH", tmp_path / "missing_pbp.csv")
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", tmp_path / "missing_preds.csv")
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", tmp_path / "missing_results.csv")
    monkeypatch.setattr(config, "FINALS_PREGAME_PREDICTIONS_PATH", pregame_path)
    monkeypatch.setattr(config, "FINALS_SCHEDULE_OVERRIDES_PATH", tmp_path / "no_overrides.csv")

    report = build_finals_upcoming_predictions_report(
        focus_season="2025-26",
        pregame_predictions_path=pregame_path,
    )
    row = report.loc[report["finals_game_number"] == 2].iloc[0]
    assert row["pregame_prediction_available"] == True
    assert row["replay_available"] == False
    assert row["predicted_winner_pregame"] == "San Antonio Spurs"


def test_collect_playoff_games_dry_run_does_not_call_api(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_pipeline", ROOT_DIR / "run_pipeline.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    called = {"count": 0}

    def fake_collect(*args, **kwargs):
        called["count"] += 1
        return 0

    monkeypatch.setattr(module, "collect_playoff_games", fake_collect)
    rc = module.dispatch_mode("collect_playoff_games", dry_run=True)
    assert rc == 0
    assert called["count"] == 0


def _sample_override_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2025-26",
                "finals_game_number": 4,
                "game_id": "0042500404",
                "game_date": "2026-06-11",
                "home_team": "New York Knicks",
                "away_team": "San Antonio Spurs",
                "home_team_id": "1610612752",
                "away_team_id": "1610612759",
                "status": "scheduled",
                "season_type": "Playoffs",
                "source": "manual_schedule",
                "notes": "Manual schedule only",
            },
            {
                "season": "2025-26",
                "finals_game_number": 5,
                "game_id": "0042500405",
                "game_date": "2026-06-13",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "scheduled",
                "season_type": "Playoffs",
                "source": "manual_schedule",
                "notes": "Manual schedule only",
            },
            {
                "season": "2025-26",
                "finals_game_number": 6,
                "game_id": "",
                "game_date": "",
                "home_team": "",
                "away_team": "",
                "home_team_id": "",
                "away_team_id": "",
                "status": "if_necessary",
                "season_type": "Playoffs",
                "source": "placeholder",
                "notes": "If necessary",
            },
        ]
    )


def test_override_loader_preserves_game_id_as_string(tmp_path):
    overrides = _sample_override_rows()
    path = tmp_path / "overrides.csv"
    overrides.to_csv(path, index=False)
    loaded = load_finals_schedule_overrides(path)
    assert loaded.iloc[0]["game_id"] == "0042500404"


def test_override_validator_rejects_result_columns():
    df = _sample_override_rows()
    df["home_score"] = 100
    with pytest.raises(ValueError, match="result columns"):
        validate_finals_schedule_overrides(df)


def test_override_validator_rejects_duplicate_game_numbers():
    df = _sample_override_rows()
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_finals_schedule_overrides(dup)


def test_local_metadata_wins_over_override_for_completed_games():
    local = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
                "finals_game_number": 1,
                "playoff_round": "NBA Finals",
            }
        ]
    )
    overrides = normalize_finals_schedule_overrides(_sample_override_rows())
    overrides.iloc[0, overrides.columns.get_loc("finals_game_number")] = 1
    overrides.iloc[0, overrides.columns.get_loc("home_team")] = "Override Team"
    merged = merge_finals_metadata_with_overrides(local, overrides, "2025-26")
    assert merged[1]["home_team"] == "San Antonio Spurs"


def test_override_fills_games_four_and_five_when_local_missing():
    local = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "status": "final",
                "finals_game_number": 1,
                "playoff_round": "NBA Finals",
            }
        ]
    )
    overrides = normalize_finals_schedule_overrides(_sample_override_rows())
    merged = merge_finals_metadata_with_overrides(local, overrides, "2025-26")
    assert merged[4]["home_team"] == "New York Knicks"
    assert str(merged[5]["status"]) == "scheduled"


def test_pregame_features_for_scheduled_override_without_results(tmp_path, monkeypatch):
    playoff_games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    regular_games = pd.DataFrame(
        [
            {
                "game_id": "0022500001",
                "season": "2025-26",
                "game_date": "2025-10-20",
                "home_team": "San Antonio Spurs",
                "away_team": "Team A",
                "home_team_id": "1610612759",
                "away_team_id": "1610612750",
                "status": "final",
                "game_type": "regular_season",
            },
        ]
    )
    regular_results = pd.DataFrame(
        {"game_id": ["0022500001"], "home_score": [110], "away_score": [100], "winner": ["San Antonio Spurs"]}
    )
    playoff_results = pd.DataFrame(
        {"game_id": ["0042500401"], "home_score": [105], "away_score": [102], "winner": ["San Antonio Spurs"]}
    )
    overrides = _sample_override_rows()

    pg_path = tmp_path / "playoff_games.csv"
    rg_path = tmp_path / "regular_games.csv"
    rr_path = tmp_path / "regular_results.csv"
    pr_path = tmp_path / "playoff_results.csv"
    ov_path = tmp_path / "overrides.csv"
    out_path = tmp_path / "finals_features.csv"
    playoff_games.to_csv(pg_path, index=False)
    regular_games.to_csv(rg_path, index=False)
    regular_results.to_csv(rr_path, index=False)
    playoff_results.to_csv(pr_path, index=False)
    overrides.to_csv(ov_path, index=False)

    monkeypatch.setattr(config, "RAW_GAMES_PATH", rg_path)
    monkeypatch.setattr(config, "GAME_RESULTS_PATH", rr_path)
    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", pg_path)
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", pr_path)
    monkeypatch.setattr(config, "FINALS_PREGAME_FEATURES_PATH", out_path)

    features = build_finals_pregame_features(
        games_path=pg_path,
        output_path=out_path,
        overrides_path=ov_path,
    )
    assert set(features["game_id"].astype(str)) >= {"0042500401", "0042500404", "0042500405"}
    scheduled = features.loc[features["game_id"] == "0042500404"].iloc[0]
    assert pd.isna(scheduled["home_team_won"])


def test_upcoming_report_game_four_pregame_when_override_and_prediction(tmp_path, monkeypatch):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)
    overrides = _sample_override_rows()
    ov_path = tmp_path / "overrides.csv"
    overrides.to_csv(ov_path, index=False)
    pregame_preds = pd.DataFrame(
        {
            "game_id": ["0042500404"],
            "home_win_probability": [0.61],
            "away_win_probability": [0.39],
            "predicted_winner": ["New York Knicks"],
            "predicted_label": [1],
            "actual_home_team_won": [None],
            "prediction_correct": [None],
        }
    )
    pregame_path = tmp_path / "finals_pregame_predictions.csv"
    pregame_preds.to_csv(pregame_path, index=False)

    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", games_path)
    monkeypatch.setattr(config, "PLAYOFF_PLAY_BY_PLAY_PATH", tmp_path / "missing_pbp.csv")
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", tmp_path / "missing_preds.csv")
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", tmp_path / "missing_results.csv")
    monkeypatch.setattr(config, "FINALS_PREGAME_PREDICTIONS_PATH", pregame_path)
    monkeypatch.setattr(config, "FINALS_SCHEDULE_OVERRIDES_PATH", ov_path)

    report = build_finals_upcoming_predictions_report(focus_season="2025-26")
    game4 = report.loc[report["finals_game_number"] == 4].iloc[0]
    assert game4["pregame_prediction_available"] == True
    assert game4["game_status"] == "scheduled"
    game6 = report.loc[report["finals_game_number"] == 6].iloc[0]
    assert game6["pregame_prediction_available"] == False


def _sample_upcoming_for_projection() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": "2025-26",
                "game_id": "0042500401",
                "finals_game_number": 1,
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "game_status": "completed",
                "pregame_prediction_available": True,
                "predicted_winner_pregame": "New York Knicks",
                "final_result_available": True,
                "final_winner": "New York Knicks",
            },
            {
                "season": "2025-26",
                "game_id": "0042500402",
                "finals_game_number": 2,
                "game_date": "2026-06-05",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "game_status": "completed",
                "pregame_prediction_available": True,
                "predicted_winner_pregame": "San Antonio Spurs",
                "final_result_available": True,
                "final_winner": "New York Knicks",
            },
            {
                "season": "2025-26",
                "game_id": "0042500403",
                "finals_game_number": 3,
                "game_date": "2026-06-08",
                "home_team": "New York Knicks",
                "away_team": "San Antonio Spurs",
                "game_status": "completed",
                "pregame_prediction_available": True,
                "predicted_winner_pregame": "San Antonio Spurs",
                "final_result_available": True,
                "final_winner": "San Antonio Spurs",
            },
            {
                "season": "2025-26",
                "game_id": "0042500404",
                "finals_game_number": 4,
                "game_date": "2026-06-11",
                "home_team": "New York Knicks",
                "away_team": "San Antonio Spurs",
                "game_status": "scheduled",
                "pregame_prediction_available": True,
                "predicted_winner_pregame": "New York Knicks",
                "final_result_available": False,
                "final_winner": None,
            },
            {
                "season": "2025-26",
                "game_id": "0042500405",
                "finals_game_number": 5,
                "game_date": "2026-06-13",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "game_status": "scheduled",
                "pregame_prediction_available": True,
                "predicted_winner_pregame": "San Antonio Spurs",
                "final_result_available": False,
                "final_winner": None,
            },
            {
                "season": "2025-26",
                "game_id": "",
                "finals_game_number": 6,
                "game_date": "",
                "home_team": "",
                "away_team": "",
                "game_status": "if_necessary",
                "pregame_prediction_available": False,
                "predicted_winner_pregame": None,
                "final_result_available": False,
                "final_winner": None,
            },
            {
                "season": "2025-26",
                "game_id": "",
                "finals_game_number": 7,
                "game_date": "",
                "home_team": "",
                "away_team": "",
                "game_status": "if_necessary",
                "pregame_prediction_available": False,
                "predicted_winner_pregame": None,
                "final_result_available": False,
                "final_winner": None,
            },
        ]
    )


def test_projected_path_uses_actual_results_first():
    upcoming = _sample_upcoming_for_projection()
    actual = get_actual_finals_results(upcoming)
    path = project_finals_series_path(upcoming, actual)
    g1 = path.loc[path["finals_game_number"] == 1].iloc[0]
    assert g1["projection_type"] == "actual_official"
    assert g1["projected_winner_used"] == "New York Knicks"


def test_manual_override_wins_over_official_result():
    upcoming = _sample_upcoming_for_projection()
    actual = get_actual_finals_results(upcoming)
    actual["0042500404"] = ("San Antonio Spurs", "manual_override")
    path = project_finals_series_path(upcoming, actual)
    g4 = path.loc[path["finals_game_number"] == 4].iloc[0]
    assert g4["projection_type"] == "actual_manual_override"
    assert g4["projected_winner_used"] == "San Antonio Spurs"


def test_model_projections_only_for_future_games():
    upcoming = _sample_upcoming_for_projection()
    actual = get_actual_finals_results(
        upcoming, playoff_results_path=Path("__no_such_results_file__.csv")
    )
    path = project_finals_series_path(upcoming, actual)
    g5 = path.loc[path["finals_game_number"] == 5].iloc[0]
    assert g5["projection_type"] == "model_projected"
    assert g5["projected_winner_used"] == "San Antonio Spurs"


def test_projected_path_marks_later_games_not_needed_after_clinch():
    upcoming = _sample_upcoming_for_projection().copy()
    upcoming.loc[upcoming["finals_game_number"] == 2, "final_winner"] = "New York Knicks"
    upcoming.loc[upcoming["finals_game_number"] == 3, "final_winner"] = "New York Knicks"
    actual = get_actual_finals_results(upcoming)
    path = project_finals_series_path(upcoming, actual)
    g5 = path.loc[path["finals_game_number"] == 5].iloc[0]
    assert g5["projection_type"] == "not_needed_under_projection"


def test_game_six_needed_but_prediction_unavailable():
    upcoming = _sample_upcoming_for_projection()
    actual = get_actual_finals_results(
        upcoming, playoff_results_path=Path("__no_such_results_file__.csv")
    )
    path = project_finals_series_path(upcoming, actual)
    g6 = path.loc[path["finals_game_number"] == 6].iloc[0]
    assert g6["game_needed_under_projection"] == True
    assert g6["projection_type"] == "needed_but_prediction_unavailable"


def test_game_seven_not_needed_when_not_three_three():
    upcoming = _sample_upcoming_for_projection()
    actual = get_actual_finals_results(
        upcoming, playoff_results_path=Path("__no_such_results_file__.csv")
    )
    path = project_finals_series_path(upcoming, actual)
    g7 = path.loc[path["finals_game_number"] == 7].iloc[0]
    assert g7["projection_type"] == "if_necessary_pending"
    assert "Game 7 depends on Game 6" in str(g7["notes"])


def test_projected_report_schema_stable(tmp_path):
    upcoming = _sample_upcoming_for_projection()
    up_path = tmp_path / "upcoming.csv"
    upcoming.to_csv(up_path, index=False)
    report = build_finals_projected_series_report(
        upcoming_path=up_path,
        manual_path=tmp_path / "missing_manual.csv",
        playoff_results_path=tmp_path / "missing_results.csv",
    )
    assert list(report.columns) == PROJECTED_SERIES_COLUMNS
    assert len(report) == 7


def test_manual_override_not_written_by_projected_path(tmp_path, monkeypatch):
    manual_path = tmp_path / "postgame_results.csv"
    upcoming = _sample_upcoming_for_projection()
    up_path = tmp_path / "upcoming.csv"
    upcoming.to_csv(up_path, index=False)
    build_finals_projected_series_report(
        upcoming_path=up_path,
        manual_path=manual_path,
        playoff_results_path=tmp_path / "missing_results.csv",
    )
    assert not manual_path.exists()


def _if_necessary_scheduled_override_rows() -> pd.DataFrame:
    rows = _sample_override_rows().to_dict("records")
    rows[2] = {
        "season": "2025-26",
        "finals_game_number": 6,
        "game_id": "0042500406",
        "game_date": "2026-06-16",
        "home_team": "New York Knicks",
        "away_team": "San Antonio Spurs",
        "home_team_id": "1610612752",
        "away_team_id": "1610612759",
        "status": "if_necessary_scheduled",
        "season_type": "Playoffs",
        "source": "manual_schedule",
        "notes": "Game 6 if necessary",
    }
    rows.append(
        {
            "season": "2025-26",
            "finals_game_number": 7,
            "game_id": "0042500407",
            "game_date": "2026-06-18",
            "home_team": "San Antonio Spurs",
            "away_team": "New York Knicks",
            "home_team_id": "1610612759",
            "away_team_id": "1610612752",
            "status": "if_necessary_scheduled",
            "season_type": "Playoffs",
            "source": "manual_schedule",
            "notes": "Game 7 if necessary",
        }
    )
    return pd.DataFrame(rows)


def test_if_necessary_scheduled_accepted_with_complete_metadata():
    df = _if_necessary_scheduled_override_rows()
    validate_finals_schedule_overrides(df)


def test_if_necessary_scheduled_rejected_if_required_metadata_missing():
    df = _if_necessary_scheduled_override_rows()
    df.loc[df["finals_game_number"] == 6, "game_date"] = ""
    with pytest.raises(ValueError, match="if_necessary_scheduled requires"):
        validate_finals_schedule_overrides(df)


def test_games_six_seven_metadata_in_finals_summary(tmp_path, monkeypatch):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)
    overrides = _if_necessary_scheduled_override_rows()
    ov_path = tmp_path / "overrides.csv"
    overrides.to_csv(ov_path, index=False)

    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", games_path)
    monkeypatch.setattr(config, "PLAYOFF_PLAY_BY_PLAY_PATH", tmp_path / "missing_pbp.csv")
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", tmp_path / "missing_preds.csv")
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", tmp_path / "missing_results.csv")
    monkeypatch.setattr(config, "FINALS_SCHEDULE_OVERRIDES_PATH", ov_path)

    summary = build_nba_finals_case_study_summary(focus_season="2025-26")
    assert len(summary) == 7
    g6 = summary.loc[summary["finals_game_number"] == 6].iloc[0]
    g7 = summary.loc[summary["finals_game_number"] == 7].iloc[0]
    assert g6["game_status"] == "if_necessary_scheduled"
    assert g6["home_team"] == "New York Knicks"
    assert g6["final_result_available"] == False
    assert g6["replay_available"] == False
    assert g7["game_status"] == "if_necessary_scheduled"
    assert g7["away_team"] == "New York Knicks"
    assert g7["final_winner"] is None or pd.isna(g7["final_winner"])


def test_pregame_features_for_if_necessary_scheduled_games(tmp_path, monkeypatch):
    playoff_games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    regular_games = pd.DataFrame(
        [
            {
                "game_id": "0022500001",
                "season": "2025-26",
                "game_date": "2025-10-20",
                "home_team": "San Antonio Spurs",
                "away_team": "Team A",
                "home_team_id": "1610612759",
                "away_team_id": "1610612750",
                "status": "final",
                "game_type": "regular_season",
            },
        ]
    )
    regular_results = pd.DataFrame(
        {"game_id": ["0022500001"], "home_score": [110], "away_score": [100], "winner": ["San Antonio Spurs"]}
    )
    playoff_results = pd.DataFrame(
        {"game_id": ["0042500401"], "home_score": [105], "away_score": [102], "winner": ["San Antonio Spurs"]}
    )
    overrides = _if_necessary_scheduled_override_rows()

    pg_path = tmp_path / "playoff_games.csv"
    rg_path = tmp_path / "regular_games.csv"
    rr_path = tmp_path / "regular_results.csv"
    pr_path = tmp_path / "playoff_results.csv"
    ov_path = tmp_path / "overrides.csv"
    out_path = tmp_path / "finals_features.csv"
    playoff_games.to_csv(pg_path, index=False)
    regular_games.to_csv(rg_path, index=False)
    regular_results.to_csv(rr_path, index=False)
    playoff_results.to_csv(pr_path, index=False)
    overrides.to_csv(ov_path, index=False)

    monkeypatch.setattr(config, "RAW_GAMES_PATH", rg_path)
    monkeypatch.setattr(config, "GAME_RESULTS_PATH", rr_path)
    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", pg_path)
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", pr_path)
    monkeypatch.setattr(config, "FINALS_PREGAME_FEATURES_PATH", out_path)

    features = build_finals_pregame_features(
        games_path=pg_path,
        output_path=out_path,
        overrides_path=ov_path,
    )
    assert "0042500406" in set(features["game_id"].astype(str))
    assert "0042500407" in set(features["game_id"].astype(str))


def test_upcoming_report_marks_g6_g7_pregame_when_scored(tmp_path, monkeypatch):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)
    overrides = _if_necessary_scheduled_override_rows()
    ov_path = tmp_path / "overrides.csv"
    overrides.to_csv(ov_path, index=False)
    pregame_preds = pd.DataFrame(
        {
            "game_id": ["0042500406", "0042500407"],
            "home_win_probability": [0.58, 0.55],
            "away_win_probability": [0.42, 0.45],
            "predicted_winner": ["New York Knicks", "San Antonio Spurs"],
            "predicted_label": [1, 1],
            "actual_home_team_won": [None, None],
            "prediction_correct": [None, None],
        }
    )
    pregame_path = tmp_path / "finals_pregame_predictions.csv"
    pregame_preds.to_csv(pregame_path, index=False)

    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", games_path)
    monkeypatch.setattr(config, "PLAYOFF_PLAY_BY_PLAY_PATH", tmp_path / "missing_pbp.csv")
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", tmp_path / "missing_preds.csv")
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", tmp_path / "missing_results.csv")
    monkeypatch.setattr(config, "FINALS_PREGAME_PREDICTIONS_PATH", pregame_path)
    monkeypatch.setattr(config, "FINALS_SCHEDULE_OVERRIDES_PATH", ov_path)

    report = build_finals_upcoming_predictions_report(focus_season="2025-26")
    g6 = report.loc[report["finals_game_number"] == 6].iloc[0]
    g7 = report.loc[report["finals_game_number"] == 7].iloc[0]
    assert g6["pregame_prediction_available"] == True
    assert g6["game_status"] == "if_necessary_scheduled"
    assert "Conditional pre-game prediction" in str(g6["notes"])
    assert g7["pregame_prediction_available"] == True


def _sample_upcoming_with_conditional_g6_g7() -> pd.DataFrame:
    rows = _sample_upcoming_for_projection().to_dict("records")
    for row in rows:
        if row["finals_game_number"] == 6:
            row.update(
                {
                    "game_id": "0042500406",
                    "game_date": "2026-06-16",
                    "home_team": "New York Knicks",
                    "away_team": "San Antonio Spurs",
                    "game_status": "if_necessary_scheduled",
                    "pregame_prediction_available": True,
                    "predicted_winner_pregame": "San Antonio Spurs",
                    "final_result_available": False,
                    "final_winner": None,
                }
            )
        elif row["finals_game_number"] == 7:
            row.update(
                {
                    "game_id": "0042500407",
                    "game_date": "2026-06-18",
                    "home_team": "San Antonio Spurs",
                    "away_team": "New York Knicks",
                    "game_status": "if_necessary_scheduled",
                    "pregame_prediction_available": True,
                    "predicted_winner_pregame": "San Antonio Spurs",
                    "final_result_available": False,
                    "final_winner": None,
                }
            )
    return pd.DataFrame(rows)


def test_game_six_used_when_projected_path_after_game_five_is_three_two():
    upcoming = _sample_upcoming_with_conditional_g6_g7()
    actual = get_actual_finals_results(
        upcoming, playoff_results_path=Path("__no_such_results_file__.csv")
    )
    path = project_finals_series_path(upcoming, actual)
    g6 = path.loc[path["finals_game_number"] == 6].iloc[0]
    assert g6["game_needed_under_projection"] == True
    assert g6["projection_type"] == "conditional_model_projected"
    assert g6["projected_winner_used"] == "San Antonio Spurs"
    assert g6["team_a_wins_after"] == 3
    assert g6["team_b_wins_after"] == 3


def test_game_seven_needed_only_if_path_after_game_six_is_three_three():
    upcoming = _sample_upcoming_with_conditional_g6_g7()
    actual = get_actual_finals_results(
        upcoming, playoff_results_path=Path("__no_such_results_file__.csv")
    )
    path = project_finals_series_path(upcoming, actual)
    g7 = path.loc[path["finals_game_number"] == 7].iloc[0]
    assert g7["game_needed_under_projection"] == True
    assert g7["projection_type"] == "conditional_model_projected"
    assert g7["projected_winner_used"] == "San Antonio Spurs"


def test_game_seven_not_marked_not_needed_before_game_six_processed():
    upcoming = _sample_upcoming_for_projection()
    actual = get_actual_finals_results(
        upcoming, playoff_results_path=Path("__no_such_results_file__.csv")
    )
    path = project_finals_series_path(upcoming, actual)
    g6 = path.loc[path["finals_game_number"] == 6].iloc[0]
    g7 = path.loc[path["finals_game_number"] == 7].iloc[0]
    assert g6["game_needed_under_projection"] == True
    assert g7["projection_type"] != "not_needed_under_projection"


def test_game_seven_not_needed_when_knicks_clinch_in_game_six():
    upcoming = _sample_upcoming_with_conditional_g6_g7()
    rows = upcoming.to_dict("records")
    for row in rows:
        if row["finals_game_number"] == 6:
            row["predicted_winner_pregame"] = "New York Knicks"
    upcoming = pd.DataFrame(rows)
    actual = get_actual_finals_results(upcoming)
    path = project_finals_series_path(upcoming, actual)
    g6 = path.loc[path["finals_game_number"] == 6].iloc[0]
    g7 = path.loc[path["finals_game_number"] == 7].iloc[0]
    assert g6["series_over_after_game"] == True
    assert g7["projection_type"] == "not_needed_under_projection"


def test_conditional_model_projected_used_for_g6_g7():
    upcoming = _sample_upcoming_with_conditional_g6_g7()
    actual = get_actual_finals_results(
        upcoming, playoff_results_path=Path("__no_such_results_file__.csv")
    )
    path = project_finals_series_path(upcoming, actual)
    g6 = path.loc[path["finals_game_number"] == 6].iloc[0]
    g7 = path.loc[path["finals_game_number"] == 7].iloc[0]
    assert g6["projection_type"] == "conditional_model_projected"
    assert g7["projection_type"] == "conditional_model_projected"


def test_g6_g7_no_actual_results_in_summary(tmp_path, monkeypatch):
    games = pd.DataFrame(
        [
            {
                "game_id": "0042500401",
                "season": "2025-26",
                "season_type": "Playoffs",
                "game_date": "2026-06-03",
                "home_team": "San Antonio Spurs",
                "away_team": "New York Knicks",
                "home_team_id": "1610612759",
                "away_team_id": "1610612752",
                "status": "final",
                "game_type": "playoffs",
            },
        ]
    )
    games_path = tmp_path / "playoff_games.csv"
    games.to_csv(games_path, index=False)
    overrides = _if_necessary_scheduled_override_rows()
    ov_path = tmp_path / "overrides.csv"
    overrides.to_csv(ov_path, index=False)

    monkeypatch.setattr(config, "PLAYOFF_GAMES_PATH", games_path)
    monkeypatch.setattr(config, "PLAYOFF_PLAY_BY_PLAY_PATH", tmp_path / "missing_pbp.csv")
    monkeypatch.setattr(config, "PLAYOFF_LIVE_PREDICTIONS_PATH", tmp_path / "missing_preds.csv")
    monkeypatch.setattr(config, "PLAYOFF_GAME_RESULTS_PATH", tmp_path / "missing_results.csv")
    monkeypatch.setattr(config, "FINALS_SCHEDULE_OVERRIDES_PATH", ov_path)

    summary = build_nba_finals_case_study_summary(focus_season="2025-26")
    for game_num in (6, 7):
        row = summary.loc[summary["finals_game_number"] == game_num].iloc[0]
        assert row["final_result_available"] == False
        assert row["final_winner"] is None or pd.isna(row["final_winner"])
