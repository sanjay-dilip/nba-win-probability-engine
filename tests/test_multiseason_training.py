"""Tests for multi-season training readiness.

Pure-function tests with fake data — no nba_api, no heavy model training.
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

from src import config  # noqa: E402
from src.multiseason_training import (  # noqa: E402
    build_multiseason_training_readiness,
    check_min_games_by_season,
    get_available_seasons,
    should_allow_training,
    split_by_train_test_seasons,
    validate_train_test_seasons,
)


def _pregame_row(season: str, game_id: str, target: float = 1.0) -> dict:
    return {
        "game_id": game_id,
        "season": season,
        "game_date": "2024-11-01",
        "home_team_won": target,
        "home_games_played_before": 5,
        "away_games_played_before": 5,
        "home_wins_before": 3,
        "away_wins_before": 2,
        "home_losses_before": 2,
        "away_losses_before": 3,
        "home_win_pct_before": 0.6,
        "away_win_pct_before": 0.4,
        "win_pct_diff_before": 0.2,
        "home_points_for_avg_before": 110.0,
        "away_points_for_avg_before": 108.0,
        "home_points_allowed_avg_before": 105.0,
        "away_points_allowed_avg_before": 107.0,
        "points_for_avg_diff_before": 2.0,
        "points_allowed_avg_diff_before": -2.0,
        "home_recent_win_pct_before": 0.6,
        "away_recent_win_pct_before": 0.4,
        "recent_win_pct_diff_before": 0.2,
        "home_rest_days": 2.0,
        "away_rest_days": 3.0,
        "rest_days_diff": -1.0,
        "game_type": "regular_season",
    }


def _live_rows(season: str, game_id: str, n_events: int = 120) -> list[dict]:
    rows = []
    for i in range(n_events):
        rows.append({
            "game_id": game_id,
            "season": season,
            "game_date": "2024-11-01",
            "home_team_won": 1,
            "seconds_remaining_game": float(2880 - i * 10),
            "home_score": float(i),
            "away_score": float(i - 1),
            "score_margin_home": 1.0,
            "abs_score_margin": 1.0,
            "period": 1,
            "seconds_remaining_period": 600.0,
            "event_type_label": "made_shot",
            "game_type": "regular_season",
            "is_scoring_event": True,
            "is_turnover": False,
            "is_foul": False,
            "is_timeout": False,
            "is_rebound": False,
            "is_free_throw": False,
            "is_field_goal_attempt": True,
        })
    return rows


def test_validate_train_test_seasons_accepts_valid_split():
    train, test = validate_train_test_seasons(["2022-23", "2023-24"], "2024-25")
    assert train == ["2022-23", "2023-24"]
    assert test == "2024-25"


def test_validate_train_test_seasons_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_train_test_seasons(["2022-23", "2024-25"], "2024-25")


def test_validate_train_test_seasons_rejects_empty_train():
    with pytest.raises(ValueError):
        validate_train_test_seasons([], "2024-25")


def test_split_by_train_test_seasons():
    df = pd.DataFrame([
        {"season": "2022-23", "game_id": "g1", "x": 1},
        {"season": "2023-24", "game_id": "g2", "x": 2},
        {"season": "2024-25", "game_id": "g3", "x": 3},
    ])
    train, test = split_by_train_test_seasons(df, ["2022-23", "2023-24"], "2024-25")
    assert train["season"].tolist() == ["2022-23", "2023-24"]
    assert test["season"].tolist() == ["2024-25"]


def test_readiness_fails_when_pregame_missing_train_season(tmp_path, monkeypatch):
    pregame = tmp_path / "pregame.csv"
    rows = [_pregame_row("2024-25", f"0022400{i:03d}") for i in range(150)]
    pd.DataFrame(rows).to_csv(pregame, index=False)
    live = tmp_path / "live.csv"
    live_rows = []
    for i in range(150):
        live_rows.extend(_live_rows("2024-25", f"0022400{i:03d}", n_events=120))
    pd.DataFrame(live_rows).to_csv(live, index=False)

    readiness = build_multiseason_training_readiness(
        ["2022-23", "2023-24"],
        "2024-25",
        pregame_path=pregame,
        live_path=live,
        min_games=100,
        min_live_rows=10_000,
    )
    assert not should_allow_training(readiness)
    missing = readiness[
        (readiness["status"] == "fail")
        & (readiness["check"].str.contains("2022-23"))
    ]
    assert not missing.empty


def test_readiness_fails_when_live_missing_test_season(tmp_path):
    pregame_rows = []
    for season in ["2022-23", "2023-24", "2024-25"]:
        for i in range(120):
            pregame_rows.append(_pregame_row(season, f"{season[:2]}{i:04d}"))
    pregame = tmp_path / "pregame.csv"
    pd.DataFrame(pregame_rows).to_csv(pregame, index=False)

    live_rows = []
    for season in ["2022-23", "2023-24"]:
        for i in range(120):
            live_rows.extend(_live_rows(season, f"{season[:2]}{i:04d}", n_events=120))
    live = tmp_path / "live.csv"
    pd.DataFrame(live_rows).to_csv(live, index=False)

    readiness = build_multiseason_training_readiness(
        ["2022-23", "2023-24"],
        "2024-25",
        pregame_path=pregame,
        live_path=live,
        min_games=100,
        min_live_rows=10_000,
    )
    assert not should_allow_training(readiness, sections=["live", "split"])
    fail_test = readiness[
        (readiness["section"] == "live")
        & (readiness["check"].str.contains("2024-25"))
        & (readiness["status"] == "fail")
    ]
    assert not fail_test.empty


def test_readiness_fails_on_null_targets(tmp_path):
    pregame_rows = [_pregame_row("2024-25", f"0022400{i:03d}") for i in range(120)]
    pregame_rows[0]["home_team_won"] = np.nan
    pregame = tmp_path / "pregame.csv"
    pd.DataFrame(pregame_rows).to_csv(pregame, index=False)

    live_rows = []
    for i in range(120):
        live_rows.extend(_live_rows("2024-25", f"0022400{i:03d}", n_events=120))
    live = tmp_path / "live.csv"
    pd.DataFrame(live_rows).to_csv(live, index=False)

    readiness = build_multiseason_training_readiness(
        ["2023-24"],
        "2024-25",
        pregame_path=pregame,
        live_path=live,
        min_games=100,
        min_live_rows=10_000,
    )
    null_checks = readiness[readiness["check"].str.contains("target")]
    assert (null_checks["status"] == "fail").any()


def test_readiness_passes_with_small_fake_data_and_lower_thresholds(tmp_path):
    pregame_rows = []
    for season in ["2022-23", "2023-24", "2024-25"]:
        for i in range(5):
            pregame_rows.append(_pregame_row(season, f"{season.replace('-', '')}{i}"))
    pregame = tmp_path / "pregame.csv"
    pd.DataFrame(pregame_rows).to_csv(pregame, index=False)

    live_rows = []
    for season in ["2022-23", "2023-24", "2024-25"]:
        for i in range(5):
            live_rows.extend(_live_rows(season, f"{season.replace('-', '')}{i}", n_events=50))
    live = tmp_path / "live.csv"
    pd.DataFrame(live_rows).to_csv(live, index=False)

    readiness = build_multiseason_training_readiness(
        ["2022-23", "2023-24"],
        "2024-25",
        pregame_path=pregame,
        live_path=live,
        min_games=3,
        min_live_rows=100,
    )
    assert should_allow_training(readiness)


def test_multiseason_artifact_paths_differ_from_single_season():
    assert config.PREGAME_MODEL_MULTISEASON_PATH != config.PREGAME_MODEL_PATH
    assert config.LIVE_MODEL_MULTISEASON_PATH != config.LIVE_MODEL_PATH
    assert "multiseason" in config.PREGAME_MODEL_MULTISEASON_PATH.name


def test_get_available_seasons():
    df = pd.DataFrame({"season": ["2024-25", "2023-24", "2024-25"]})
    assert get_available_seasons(df) == ["2023-24", "2024-25"]


def test_check_min_games_by_season():
    df = pd.DataFrame({
        "season": ["2024-25"] * 5,
        "game_id": [f"g{i}" for i in range(5)],
    })
    rows = check_min_games_by_season(
        df, "season", "game_id", ["2024-25"], min_games=3, dataset_name="test", section="t"
    )
    assert rows[0]["status"] == "pass"
