"""Tests for project QA helpers (Build 16).

Pure-function tests only — no Streamlit, nba_api, pipeline steps, or training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.project_qa import (  # noqa: E402
    SUMMARY_COLUMNS,
    build_project_qa_summary,
    check_context_file,
    check_multiseason_coverage_support,
    check_optional_comparison_reports,
    check_optional_multiseason_artifacts,
    check_optional_playoff_case_study,
    check_pipeline_modes,
    check_readme_commands,
    check_required_paths,
    check_season_columns,
    compute_overall_qa_status,
    summarize_data_freshness,
)


def test_required_path_check_passes_when_file_exists(tmp_path):
    file_path = tmp_path / "games.csv"
    file_path.write_text("game_id\n0022400001\n", encoding="utf-8")
    df = check_required_paths({"raw_games": file_path})
    assert df.iloc[0]["status"] == "pass"


def test_required_path_check_fails_when_missing(tmp_path):
    df = check_required_paths({"raw_games": tmp_path / "missing.csv"})
    assert df.iloc[0]["status"] == "fail"


def test_required_path_check_warns_on_empty_csv(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("game_id\n", encoding="utf-8")
    df = check_required_paths({"raw_games": file_path})
    assert df.iloc[0]["status"] == "warning"


def test_qa_summary_has_required_columns():
    df = build_project_qa_summary()
    assert list(df.columns) == SUMMARY_COLUMNS
    assert not df.empty


def test_freshness_stale_only_is_warning(tmp_path):
    freshness = tmp_path / "freshness.csv"
    pd.DataFrame(
        [
            {"asset": "pregame_model_metrics", "status": "stale"},
            {"asset": "raw_games", "status": "ok"},
            {"asset": "postgame_results", "status": "not_required"},
        ]
    ).to_csv(freshness, index=False)

    df = summarize_data_freshness(freshness)
    present = df.loc[df["check"] == "required_assets_present"].iloc[0]
    assert present["status"] == "warning"


def test_freshness_missing_required_is_fail(tmp_path):
    freshness = tmp_path / "freshness.csv"
    pd.DataFrame(
        [
            {"asset": "pregame_predictions", "status": "missing"},
            {"asset": "raw_games", "status": "ok"},
        ]
    ).to_csv(freshness, index=False)

    df = summarize_data_freshness(freshness)
    present = df.loc[df["check"] == "required_assets_present"].iloc[0]
    assert present["status"] == "fail"


def test_readme_command_check_detects_missing_command(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n", encoding="utf-8")
    df = check_readme_commands(readme)
    row = df.loc[df["check"] == "required_commands_documented"].iloc[0]
    assert row["status"] == "fail"


def test_context_section_check_detects_missing_section(tmp_path):
    context = tmp_path / "CONTEXT.md"
    context.write_text("# Title\n\n## Project Overview\n", encoding="utf-8")
    df = check_context_file(context)
    row = df.loc[df["check"] == "required_sections_present"].iloc[0]
    assert row["status"] == "fail"


def test_pipeline_mode_check_confirms_expected_modes():
    df = check_pipeline_modes()
    modes_row = df.loc[df["check"] == "expected_modes_registered"].iloc[0]
    assert modes_row["status"] == "pass"
    no_all = df.loc[df["check"] == "no_generic_all_mode"].iloc[0]
    assert no_all["status"] == "pass"


def test_overall_qa_status_pass_warning_fail():
    assert compute_overall_qa_status(pd.DataFrame([{"status": "pass"}])) == "pass"
    assert compute_overall_qa_status(pd.DataFrame([{"status": "warning"}])) == "warning"
    assert compute_overall_qa_status(
        pd.DataFrame([{"status": "pass"}, {"status": "fail"}])
    ) == "fail"


def test_qa_dry_run_does_not_execute_callables():
    import importlib.util

    pipeline_path = ROOT_DIR / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline_qa_test", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec.loader.exec_module(module)

    calls: list[str] = []

    def fake_freshness() -> int:
        calls.append("freshness")
        return 0

    def fake_qa() -> int:
        calls.append("qa")
        return 0

    steps = [("check_data_freshness", fake_freshness), ("project_qa", fake_qa)]
    rc = module.run_grouped_mode("qa", steps, dry_run=True)
    assert rc == 0
    assert calls == []


def test_season_column_check_passes_when_present(tmp_path, monkeypatch):
    games = tmp_path / "games.csv"
    pd.DataFrame({"game_id": ["0022400001"], "season": ["2024-25"]}).to_csv(
        games, index=False
    )
    monkeypatch.setattr(
        "src.project_qa.config.RAW_GAMES_PATH",
        games,
    )
    df = check_season_columns()
    row = df.loc[df["check"] == "raw_games"].iloc[0]
    assert row["status"] == "pass"


def test_season_column_check_fails_when_missing(tmp_path, monkeypatch):
    games = tmp_path / "games.csv"
    pd.DataFrame({"game_id": ["0022400001"]}).to_csv(games, index=False)
    monkeypatch.setattr("src.project_qa.config.RAW_GAMES_PATH", games)
    df = check_season_columns()
    row = df.loc[df["check"] == "raw_games"].iloc[0]
    assert row["status"] == "fail"


def test_optional_multiseason_artifacts_not_applicable_when_missing(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(
        config,
        "PREGAME_MODEL_MULTISEASON_PATH",
        tmp_path / "missing_pregame_multiseason.pkl",
    )
    df = check_optional_multiseason_artifacts()
    row = df.loc[df["check"] == "multiseason_models_present"].iloc[0]
    assert row["status"] == "not_applicable"


def test_coverage_support_not_applicable_when_report_missing():
    df = check_multiseason_coverage_support()
    row = df.loc[df["check"] == "coverage_report_generated"].iloc[0]
    assert row["status"] in ("not_applicable", "pass")


def test_comparison_reports_not_applicable_when_missing(tmp_path, monkeypatch):
    from src import config

    monkeypatch.setattr(
        config,
        "MODEL_COMPARISON_SUMMARY_PATH",
        tmp_path / "missing_summary.csv",
    )
    df = check_optional_comparison_reports()
    module_row = df.loc[df["check"] == "compare_module_present"].iloc[0]
    report_row = df.loc[df["check"] == "comparison_reports_generated"].iloc[0]
    assert module_row["status"] == "pass"
    assert report_row["status"] == "not_applicable"


def test_comparison_reports_pass_when_present(tmp_path, monkeypatch):
    from src import config

    summary_path = tmp_path / "model_comparison_summary.csv"
    pd.DataFrame(
        [
            {
                "model": "pregame",
                "metric": "accuracy",
                "single_season_value": 0.7,
                "multiseason_value": 0.68,
                "difference": -0.02,
            }
        ]
    ).to_csv(summary_path, index=False)
    monkeypatch.setattr(config, "MODEL_COMPARISON_SUMMARY_PATH", summary_path)
    df = check_optional_comparison_reports()
    report_row = df.loc[df["check"] == "comparison_reports_generated"].iloc[0]
    assert report_row["status"] == "pass"


def test_playoff_case_study_not_applicable_when_outputs_missing(tmp_path, monkeypatch):
    from src import config

    playoff_paths = [
        "PLAYOFF_GAMES_PATH",
        "PLAYOFF_PLAY_BY_PLAY_PATH",
        "PLAYOFF_LIVE_FEATURES_PATH",
        "PLAYOFF_LIVE_PREDICTIONS_PATH",
        "NBA_FINALS_CASE_STUDY_SUMMARY_PATH",
        "FINALS_PREGAME_FEATURES_PATH",
        "FINALS_PREGAME_PREDICTIONS_PATH",
        "FINALS_UPCOMING_PREDICTIONS_REPORT_PATH",
        "FINALS_PROJECTED_SERIES_PATH",
    ]
    for index, attr in enumerate(playoff_paths):
        monkeypatch.setattr(config, attr, tmp_path / f"missing_{index}.csv")

    df = check_optional_playoff_case_study()
    module_row = df.loc[df["check"] == "playoff_module_present"].iloc[0]
    outputs_row = df.loc[df["check"] == "playoff_outputs_generated"].iloc[0]
    assert module_row["status"] == "pass"
    assert outputs_row["status"] == "not_applicable"


def test_github_finals_workflow_exists_and_skips_training():
    from src.project_qa import check_github_finals_workflow

    df = check_github_finals_workflow()
    row = df.iloc[0]
    assert row["check"] == "finals_refresh_workflow"
    assert row["status"] in {"pass", "not_applicable"}
    if row["status"] == "pass":
        text = Path(ROOT_DIR / ".github/workflows/finals_refresh.yml").read_text(encoding="utf-8")
        assert "train_pregame_model" not in text
        assert "train_live_model" not in text


def test_finals_deploy_export_not_applicable_when_missing(tmp_path, monkeypatch):
    from src import config
    from src.project_qa import check_finals_deploy_export

    monkeypatch.setattr(config, "FINALS_LIVE_PREDICTIONS_DEPLOY_PATH", tmp_path / "missing.csv")

    df = check_finals_deploy_export()
    row = df.iloc[0]
    assert row["check"] == "finals_live_predictions_export"
    assert row["status"] == "not_applicable"


def test_finals_deploy_export_fails_when_empty(tmp_path, monkeypatch):
    from src import config
    from src.project_qa import check_finals_deploy_export

    deploy_path = tmp_path / "finals_live_predictions.csv"
    deploy_path.write_text("season,game_id\n", encoding="utf-8")
    monkeypatch.setattr(config, "FINALS_LIVE_PREDICTIONS_DEPLOY_PATH", deploy_path)

    df = check_finals_deploy_export()
    row = df.iloc[0]
    assert row["status"] == "fail"


def test_finals_deploy_export_warns_on_missing_replay_available_game(tmp_path, monkeypatch):
    from src import config
    from src.project_qa import check_finals_deploy_export

    deploy_path = tmp_path / "finals_live_predictions.csv"
    pd.DataFrame(
        [{"season": "2025-26", "game_id": "0042500401", "event_num": 1}]
    ).to_csv(deploy_path, index=False)

    upcoming_path = tmp_path / "finals_upcoming_predictions.csv"
    pd.DataFrame(
        [
            {"game_id": "42500401", "replay_available": True},
            {"game_id": "42500402", "replay_available": True},
        ]
    ).to_csv(upcoming_path, index=False)

    monkeypatch.setattr(config, "FINALS_LIVE_PREDICTIONS_DEPLOY_PATH", deploy_path)
    monkeypatch.setattr(config, "FINALS_UPCOMING_PREDICTIONS_REPORT_PATH", upcoming_path)

    df = check_finals_deploy_export()
    row = df.iloc[0]
    assert row["status"] == "warning"
    assert "0042500402" in row["details"]


def test_finals_deploy_export_passes_when_covers_all_replay_available_games(tmp_path, monkeypatch):
    from src import config
    from src.project_qa import check_finals_deploy_export

    deploy_path = tmp_path / "finals_live_predictions.csv"
    pd.DataFrame(
        [
            {"season": "2025-26", "game_id": "0042500401", "event_num": 1},
            {"season": "2025-26", "game_id": "0042500402", "event_num": 1},
        ]
    ).to_csv(deploy_path, index=False)

    upcoming_path = tmp_path / "finals_upcoming_predictions.csv"
    pd.DataFrame(
        [
            # deliberately unpadded, matching the known formatting drift in
            # finals_upcoming_predictions.csv
            {"game_id": "42500401", "replay_available": True},
            {"game_id": "42500402", "replay_available": True},
        ]
    ).to_csv(upcoming_path, index=False)

    monkeypatch.setattr(config, "FINALS_LIVE_PREDICTIONS_DEPLOY_PATH", deploy_path)
    monkeypatch.setattr(config, "FINALS_UPCOMING_PREDICTIONS_REPORT_PATH", upcoming_path)

    df = check_finals_deploy_export()
    row = df.iloc[0]
    assert row["status"] == "pass"

