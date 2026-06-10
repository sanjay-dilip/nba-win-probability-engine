"""Tests for multi-season coverage checks (Build 18.5).

Uses temp CSVs only — no nba_api, no training, no live collection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.multiseason_coverage import (  # noqa: E402
    COVERAGE_COLUMNS,
    build_multiseason_coverage_report,
    count_rows_by_season,
    count_unique_games_by_season,
    get_play_by_play_coverage_by_season,
    should_allow_multiseason_training_from_coverage,
)


def _write_games(path: Path, seasons: dict[str, int]) -> None:
    rows = []
    for season, n in seasons.items():
        for i in range(n):
            rows.append({"game_id": f"{season.replace('-', '')}{i:04d}", "season": season})
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_pbp(path: Path, seasons: dict[str, int], events_per_game: int = 10) -> None:
    rows = []
    for season, n_games in seasons.items():
        for i in range(n_games):
            gid = f"{season.replace('-', '')}{i:04d}"
            for e in range(events_per_game):
                rows.append({"game_id": gid, "season": season, "event_num": e + 1})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_count_rows_by_season(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "season": ["2024-25", "2024-25", "2023-24"],
        "game_id": ["0022400001", "0022400002", "0022300001"],
    }).to_csv(path, index=False)
    assert count_rows_by_season(path) == {"2024-25": 2, "2023-24": 1}


def test_count_unique_games_preserves_string_game_id(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame({
        "season": ["2024-25", "2024-25"],
        "game_id": ["0022400001", "0022400001"],
    }).to_csv(path, index=False)
    df = pd.read_csv(path, dtype={"game_id": str})
    assert df["game_id"].iloc[0] == "0022400001"
    assert count_unique_games_by_season(path) == {"2024-25": 1}


def test_missing_games_season_causes_fail(tmp_path):
    games = tmp_path / "games.csv"
    _write_games(games, {"2024-25": 1200})
    pbp = tmp_path / "pbp.csv"
    _write_pbp(pbp, {"2024-25": 1200})

    report = build_multiseason_coverage_report(
        ["2022-23", "2024-25"],
        games_path=games,
        play_by_play_path=pbp,
        live_features_path=tmp_path / "missing_live.csv",
        game_results_path=tmp_path / "missing_gr.csv",
        pregame_features_path=tmp_path / "missing_pf.csv",
    )
    games_2223 = report[
        (report["section"] == "games") & (report["season"] == "2022-23")
    ].iloc[0]
    assert games_2223["status"] == "fail"


def test_pbp_coverage_below_threshold_fails(tmp_path):
    games = tmp_path / "games.csv"
    _write_games(games, {"2024-25": 100})
    pbp = tmp_path / "pbp.csv"
    _write_pbp(pbp, {"2024-25": 50})

    rows = get_play_by_play_coverage_by_season(games, pbp, ["2024-25"])
    assert rows[0]["status"] in ("fail", "warning")
    assert rows[0]["coverage_pct"] < 95.0


def test_pbp_coverage_above_threshold_passes(tmp_path):
    games = tmp_path / "games.csv"
    _write_games(games, {"2024-25": 100})
    pbp = tmp_path / "pbp.csv"
    _write_pbp(pbp, {"2024-25": 96})

    rows = get_play_by_play_coverage_by_season(games, pbp, ["2024-25"])
    assert rows[0]["status"] == "pass"
    assert rows[0]["coverage_pct"] >= 95.0


def test_feature_coverage_below_threshold_fails(tmp_path):
    games = tmp_path / "games.csv"
    _write_games(games, {"2024-25": 100})
    features = tmp_path / "features.csv"
    _write_games(features, {"2024-25": 50})

    report = build_multiseason_coverage_report(
        ["2024-25"],
        games_path=games,
        play_by_play_path=tmp_path / "pbp.csv",
        live_features_path=features,
        game_results_path=features,
        pregame_features_path=features,
    )
    live = report[(report["section"] == "live_features")].iloc[0]
    assert live["status"] in ("fail", "warning")


def test_coverage_report_has_required_columns(tmp_path):
    games = tmp_path / "games.csv"
    _write_games(games, {"2024-25": 1200})
    report = build_multiseason_coverage_report(
        ["2024-25"],
        games_path=games,
        play_by_play_path=tmp_path / "pbp.csv",
        live_features_path=tmp_path / "live.csv",
        game_results_path=tmp_path / "gr.csv",
        pregame_features_path=tmp_path / "pf.csv",
    )
    assert list(report.columns) == COVERAGE_COLUMNS


def test_missing_downstream_files_fail_with_recommended_action(tmp_path):
    games = tmp_path / "games.csv"
    _write_games(games, {"2024-25": 1200})
    report = build_multiseason_coverage_report(
        ["2024-25"],
        games_path=games,
        play_by_play_path=tmp_path / "missing_pbp.csv",
        live_features_path=tmp_path / "missing_live.csv",
        game_results_path=tmp_path / "missing_gr.csv",
        pregame_features_path=tmp_path / "missing_pf.csv",
    )
    pbp = report[(report["section"] == "play_by_play")].iloc[0]
    assert pbp["status"] == "fail"
    assert "collect_play_by_play" in pbp["recommended_action"].lower()


def test_should_allow_training_from_coverage_false_when_failures(tmp_path):
    games = tmp_path / "games.csv"
    _write_games(games, {"2024-25": 1200})
    report = build_multiseason_coverage_report(
        ["2024-25"],
        games_path=games,
        play_by_play_path=tmp_path / "missing.csv",
        live_features_path=tmp_path / "missing.csv",
        game_results_path=tmp_path / "missing.csv",
        pregame_features_path=tmp_path / "missing.csv",
        train_seasons=["2023-24"],
        test_season="2024-25",
    )
    assert not should_allow_multiseason_training_from_coverage(report)


def test_runbook_file_exists():
    runbook = ROOT_DIR / "MULTISEASON_RUNBOOK.md"
    assert runbook.exists()
    text = runbook.read_text(encoding="utf-8")
    assert "check_multiseason_coverage" in text
    assert "collect_play_by_play_multi_season" in text
    assert "prepare_multiseason_training_data" in text
