"""Tests for data freshness helpers.

Pure-function tests only — no Streamlit, nba_api, or pipeline mutations.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    format_status_label,
    get_asset_status,
    get_status_items,
    load_freshness_summary,
)
from src.data_freshness import (  # noqa: E402
    SUMMARY_COLUMNS,
    build_data_freshness_summary,
    count_statuses,
    count_csv_rows,
    file_status,
    get_top_issues,
    get_unique_count,
    read_latest_refresh_log,
    summarize_comparison_reports,
    summarize_manual_overrides,
    _apply_stale_rules,
)


def test_file_status_missing_required(tmp_path):
    assert file_status(tmp_path / "missing.csv", required=True) == "missing"


def test_file_status_missing_optional(tmp_path):
    assert file_status(tmp_path / "missing.csv", required=False) == "not_required"


def test_file_status_empty_csv_warning(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("game_id\n", encoding="utf-8")
    assert file_status(path, required=True) == "warning"


def test_file_status_nonempty_csv_ok(tmp_path):
    path = tmp_path / "games.csv"
    pd.DataFrame([{"game_id": "0022400001", "home_team": "A"}]).to_csv(path, index=False)
    assert file_status(path, required=True) == "ok"


def test_count_csv_rows(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame([{"game_id": "0022400001"}, {"game_id": "0022400002"}]).to_csv(path, index=False)
    assert count_csv_rows(path) == 2


def test_unique_game_count_preserves_string_ids(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame([{"game_id": "0022400001"}, {"game_id": "0022400001"}]).to_csv(path, index=False)
    assert get_unique_count(path, "game_id", dtype={"game_id": str}) == 1
    df = pd.read_csv(path, dtype={"game_id": str})
    assert df.loc[0, "game_id"] == "0022400001"


def test_stale_detection_when_dependency_is_newer(tmp_path):
    model = tmp_path / "pregame_model.pkl"
    predictions = tmp_path / "pregame_predictions.csv"
    model.write_bytes(b"model")
    pd.DataFrame([{"game_id": "0022400001"}]).to_csv(predictions, index=False)

    old_time = time.time() - 100
    os.utime(predictions, (old_time, old_time))
    os.utime(model, (time.time(), time.time()))

    summary = pd.DataFrame(
        [
            {
                "section": "models",
                "asset": "pregame_model",
                "path": str(model.name),
                "status": "ok",
                "row_count": "",
                "unique_games": "",
                "last_modified": "",
                "notes": "",
            },
            {
                "section": "predictions",
                "asset": "pregame_predictions",
                "path": str(predictions.name),
                "status": "ok",
                "row_count": 1,
                "unique_games": 1,
                "last_modified": "",
                "notes": "",
            },
        ]
    )

    # Patch asset_paths resolution by using filenames relative to tmp_path.
    import src.data_freshness as freshness_mod

    original_root = freshness_mod.config.ROOT_DIR
    freshness_mod.config.ROOT_DIR = tmp_path
    try:
        updated = _apply_stale_rules(summary)
    finally:
        freshness_mod.config.ROOT_DIR = original_root

    pred_status = updated.loc[updated["asset"] == "pregame_predictions", "status"].iloc[0]
    assert pred_status == "stale"


def test_freshness_summary_has_required_columns(tmp_path, monkeypatch):
    games = tmp_path / "data" / "raw" / "games.csv"
    games.parent.mkdir(parents=True)
    pd.DataFrame([{"game_id": "0022400001"}]).to_csv(games, index=False)

    import src.data_freshness as freshness_mod

    monkeypatch.setattr(freshness_mod.config, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(freshness_mod.config, "RAW_GAMES_PATH", games)
    monkeypatch.setattr(
        freshness_mod.config,
        "RAW_PLAY_BY_PLAY_PATH",
        tmp_path / "data/raw/play_by_play.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "RAW_TEAM_STATS_PATH",
        tmp_path / "data/raw/team_stats.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "PREGAME_FEATURES_PATH",
        tmp_path / "data/processed/pregame_features.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "LIVE_FEATURES_PATH",
        tmp_path / "data/processed/live_features.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "GAME_RESULTS_PATH",
        tmp_path / "data/processed/game_results.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "PREGAME_PREDICTIONS_PATH",
        tmp_path / "data/processed/pregame_predictions.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "LIVE_PREDICTIONS_PATH",
        tmp_path / "data/processed/live_predictions.csv",
    )
    monkeypatch.setattr(freshness_mod.config, "PREGAME_MODEL_PATH", tmp_path / "models/pregame_model.pkl")
    monkeypatch.setattr(freshness_mod.config, "LIVE_MODEL_PATH", tmp_path / "models/live_model.pkl")
    monkeypatch.setattr(
        freshness_mod.config,
        "PREGAME_FEATURE_COLUMNS_PATH",
        tmp_path / "models/pregame_feature_columns.pkl",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "LIVE_FEATURE_COLUMNS_PATH",
        tmp_path / "models/live_feature_columns.pkl",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "PREGAME_MODEL_METRICS_PATH",
        tmp_path / "outputs/reports/pregame_model_metrics.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "LIVE_MODEL_METRICS_PATH",
        tmp_path / "outputs/reports/live_model_metrics.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "EVALUATION_SUMMARY_PATH",
        tmp_path / "outputs/reports/evaluation_summary.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "PBP_COVERAGE_REPORT_PATH",
        tmp_path / "outputs/reports/play_by_play_coverage_report.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "POSTGAME_RESULTS_PATH",
        tmp_path / "data/manual/postgame_results.csv",
    )
    monkeypatch.setattr(
        freshness_mod.config,
        "DATA_REFRESH_LOG_PATH",
        tmp_path / "data/logs/data_refresh_log.csv",
    )

    summary = build_data_freshness_summary()
    assert list(summary.columns) == SUMMARY_COLUMNS
    manual = summary.loc[summary["asset"] == "postgame_results"].iloc[0]
    assert manual["status"] == "not_required"


def test_summarize_manual_overrides_optional_missing(tmp_path):
    note = summarize_manual_overrides(tmp_path / "missing.csv")
    assert "optional" in note


def test_read_latest_refresh_log_missing(tmp_path):
    assert read_latest_refresh_log(tmp_path / "missing.csv") is None


def test_read_latest_refresh_log_returns_latest_row(tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame(
        [
            {"timestamp": "t1", "game_id": "1", "endpoint": "A", "status": "success"},
            {"timestamp": "t2", "game_id": "2", "endpoint": "B", "status": "error"},
        ]
    ).to_csv(path, index=False)
    latest = read_latest_refresh_log(path)
    assert latest is not None
    assert latest["timestamp"] == "t2"
    assert latest["endpoint"] == "B"


def test_count_statuses_and_top_issues():
    summary = pd.DataFrame(
        [
            {"section": "a", "asset": "x", "path": "p", "status": "ok", "row_count": "", "unique_games": "", "last_modified": "", "notes": ""},
            {"section": "b", "asset": "y", "path": "p", "status": "missing", "row_count": "", "unique_games": "", "last_modified": "", "notes": ""},
            {"section": "c", "asset": "z", "path": "p", "status": "stale", "row_count": "", "unique_games": "", "last_modified": "", "notes": ""},
        ]
    )
    counts = count_statuses(summary)
    assert counts["ok"] == 1
    assert counts["missing"] == 1
    assert counts["stale"] == 1

    issues = get_top_issues(summary, limit=2)
    assert len(issues) == 2
    assert issues[0]["status"] == "missing"


def test_dashboard_freshness_helpers(tmp_path):
    report = tmp_path / "freshness.csv"
    pd.DataFrame(
        [
            {"section": "predictions", "asset": "pregame_predictions", "path": "p", "status": "ok", "row_count": 10, "unique_games": 10, "last_modified": "", "notes": ""},
            {"section": "reports", "asset": "evaluation_summary", "path": "e", "status": "stale", "row_count": "", "unique_games": "", "last_modified": "", "notes": ""},
        ]
    ).to_csv(report, index=False)

    loaded = load_freshness_summary(report)
    assert get_asset_status(loaded, "pregame_predictions") == "ok"
    assert get_asset_status(loaded, "evaluation_summary") == "stale"
    assert get_asset_status(loaded, "missing_asset") == "unknown"

    items = get_status_items(loaded, ["pregame_predictions", "evaluation_summary"])
    assert items[0] == ("pregame_predictions", "ok")
    assert "Ready" in format_status_label("ok")


def test_comparison_reports_are_optional(tmp_path, monkeypatch):
    from src import config

    comparison_paths = [
        "MODEL_COMPARISON_SUMMARY_PATH",
        "MODEL_COMPARISON_DETAIL_PATH",
        "PHASE_COMPARISON_SUMMARY_PATH",
        "CALIBRATION_COMPARISON_SUMMARY_PATH",
    ]
    for index, attr in enumerate(comparison_paths):
        monkeypatch.setattr(config, attr, tmp_path / f"missing_{index}.csv")

    df = summarize_comparison_reports()
    assert len(df) == 4
    assert (df["status"] == "not_required").all()


def test_playoff_case_study_assets_are_optional(tmp_path, monkeypatch):
    from src import config
    from src.data_freshness import summarize_playoff_case_study_files

    playoff_paths = [
        "PLAYOFF_GAMES_PATH",
        "PLAYOFF_PLAY_BY_PLAY_PATH",
        "PLAYOFF_LIVE_FEATURES_PATH",
        "PLAYOFF_LIVE_PREDICTIONS_PATH",
        "NBA_FINALS_CASE_STUDY_SUMMARY_PATH",
        "PLAYOFF_COVERAGE_REPORT_PATH",
        "FINALS_PREGAME_FEATURES_PATH",
        "FINALS_PREGAME_PREDICTIONS_PATH",
        "FINALS_UPCOMING_PREDICTIONS_REPORT_PATH",
        "FINALS_SCHEDULE_OVERRIDES_PATH",
        "FINALS_PROJECTED_SERIES_PATH",
    ]
    for index, attr in enumerate(playoff_paths):
        monkeypatch.setattr(config, attr, tmp_path / f"missing_{index}.csv")

    df = summarize_playoff_case_study_files()
    assert len(df) == 11
    assert (df["status"] == "not_required").all()

