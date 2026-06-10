"""Tests for run_pipeline grouped modes (Build 15).

Orchestration tests only — no real pipeline steps, nba_api, or model training.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT_DIR / "run_pipeline.py"


def _load_pipeline_module():
    """Load run_pipeline.py as a module without executing main()."""
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    spec = importlib.util.spec_from_file_location("run_pipeline", PIPELINE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_pipeline"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def pipeline():
    return _load_pipeline_module()


def _step_names(steps: List) -> List[str]:
    return [name for name, _ in steps]


def test_build_features_step_order(pipeline):
    names = _step_names(pipeline.BUILD_FEATURES_STEPS)
    assert names == [
        "build_pregame_features",
        "build_live_features",
        "check_data_freshness",
    ]


def test_train_models_step_order(pipeline):
    names = _step_names(pipeline.TRAIN_MODELS_STEPS)
    assert names == [
        "train_pregame_model",
        "train_live_model",
        "check_data_freshness",
    ]


def test_score_outputs_step_order(pipeline):
    names = _step_names(pipeline.SCORE_OUTPUTS_STEPS)
    assert names == [
        "predict_all",
        "evaluate",
        "check_data_freshness",
    ]


def test_dashboard_ready_expands_submodes(pipeline):
    steps = []
    for sub_mode in pipeline.DASHBOARD_READY_SUBMODES:
        steps.extend(pipeline.get_grouped_mode_steps(sub_mode))
    names = _step_names(steps)
    assert names == [
        "build_pregame_features",
        "build_live_features",
        "check_data_freshness",
        "train_pregame_model",
        "train_live_model",
        "check_data_freshness",
        "predict_all",
        "evaluate",
        "check_data_freshness",
    ]


def test_refresh_full_season_play_by_play_steps_and_warning(pipeline, capsys):
    names = _step_names(pipeline.REFRESH_FULL_SEASON_PBP_STEPS)
    assert names == [
        "collect_play_by_play_full_season",
        "build_live_features",
        "predict_live",
        "evaluate",
        "check_data_freshness",
    ]
    assert "30-45" in pipeline.REFRESH_PBP_WARNING

    pipeline.run_refresh_full_season_play_by_play(dry_run=True)
    output = capsys.readouterr().out
    assert "refresh_full_season_play_by_play" in output
    assert "30-45" in output


def test_dry_run_does_not_execute_callables(pipeline):
    calls: list[str] = []

    def fake_step() -> int:
        calls.append("ran")
        return 0

    steps = [("fake_step", fake_step)]
    rc = pipeline.run_grouped_mode("test_mode", steps, dry_run=True)
    assert rc == 0
    assert calls == []


def test_grouped_mode_stops_after_failing_step(pipeline):
    calls: list[str] = []

    def ok_step() -> int:
        calls.append("ok")
        return 0

    def fail_step() -> int:
        calls.append("fail")
        return 1

    steps = [("ok_step", ok_step), ("fail_step", fail_step), ("never_step", ok_step)]
    rc = pipeline.run_grouped_mode("test_mode", steps, dry_run=False)
    assert rc == 1
    assert calls == ["ok", "fail"]


def test_existing_modes_still_registered(pipeline):
    modes = pipeline.get_available_modes()
    required = [
        "setup",
        "sample",
        "collect_games",
        "collect_play_by_play",
        "collect_play_by_play_full_season",
        "collect_team_stats",
        "build_pregame_features",
        "build_live_features",
        "train_pregame_model",
        "train_live_model",
        "predict_pregame",
        "predict_live",
        "predict_all",
        "evaluate",
        "check_data_freshness",
    ]
    for mode in required:
        assert mode in modes


def test_no_generic_all_mode(pipeline):
    modes = pipeline.get_available_modes()
    assert "all" not in modes


def test_list_modes_prints_grouped_modes(pipeline, capsys):
    pipeline.print_available_modes()
    output = capsys.readouterr().out
    assert "dashboard_ready" in output
    assert "build_features" in output
    assert " refresh_full_season_play_by_play" in output or "refresh_full_season_play_by_play" in output


def test_refresh_local_data_step_order(pipeline):
    names = _step_names(pipeline.REFRESH_LOCAL_DATA_STEPS)
    assert names == ["collect_games", "collect_team_stats"]


def test_collect_games_multi_season_step_order(pipeline, monkeypatch):
    calls: list[str] = []

    def fake_collect(seasons=None, **kwargs):
        calls.append(seasons[0])
        return 0

    monkeypatch.setattr(pipeline, "collect_games", fake_collect)
    pipeline.configure_seasons(seasons=["2022-23", "2023-24"])
    steps = pipeline.build_collect_games_multi_season_steps()
    assert _step_names(steps) == ["collect_games_2022-23", "collect_games_2023-24"]
    for _, fn in steps:
        fn()
    assert calls == ["2022-23", "2023-24"]


def test_collect_team_stats_multi_season_step_order(pipeline, monkeypatch):
    calls: list[str] = []

    def fake_collect(seasons=None, **kwargs):
        calls.append(seasons[0])
        return 0

    monkeypatch.setattr(pipeline, "collect_team_stats", fake_collect)
    pipeline.configure_seasons(seasons=["2022-23", "2024-25"])
    steps = pipeline.build_collect_team_stats_multi_season_steps()
    for _, fn in steps:
        fn()
    assert calls == ["2022-23", "2024-25"]


def test_collect_pbp_multi_season_dry_run_does_not_call_api(pipeline, monkeypatch, capsys):
    calls: list[str] = []

    def fake_pbp(**kwargs):
        calls.append(kwargs.get("season", ""))
        return 0

    monkeypatch.setattr(pipeline, "collect_play_by_play", fake_pbp)
    pipeline.configure_seasons(seasons=["2022-23", "2023-24"])
    rc = pipeline.run_collect_play_by_play_multi_season(dry_run=True)
    output = capsys.readouterr().out
    assert rc == 0
    assert calls == []
    assert "several hours" in output


def test_build_features_multi_season_has_no_api_steps(pipeline):
    names = _step_names(pipeline.BUILD_FEATURES_MULTI_SEASON_STEPS)
    assert names == [
        "build_pregame_features",
        "build_live_features",
        "check_data_freshness",
        "project_qa",
    ]
    assert "collect_games" not in names
    assert "collect_play_by_play" not in names


def test_default_season_is_2024_25(pipeline):
    pipeline.configure_seasons()
    assert pipeline.get_single_season() == "2024-25"
    assert pipeline.get_multi_seasons() == ["2022-23", "2023-24", "2024-25"]


def test_train_models_multiseason_step_order(pipeline):
    names = _step_names(pipeline.TRAIN_MODELS_MULTISEASON_STEPS)
    assert names == [
        "check_multiseason_training_readiness",
        "train_pregame_model_multiseason",
        "train_live_model_multiseason",
        "check_data_freshness",
        "project_qa",
    ]


def test_check_multiseason_training_readiness_mode_registered(pipeline):
    modes = pipeline.get_available_modes()
    assert "check_multiseason_training_readiness" in modes
    assert "train_models_multiseason" in modes


def test_train_models_multiseason_dry_run_does_not_train(pipeline, capsys):
    pipeline.configure_training_split(["2022-23", "2023-24"], "2024-25")
    rc = pipeline.run_train_models_multiseason(dry_run=True)
    output = capsys.readouterr().out
    assert rc == 0
    assert "train_models_multiseason" in output
    assert "check_multiseason_training_readiness" in output


def test_train_models_multiseason_stops_if_readiness_fails(pipeline, monkeypatch):
    pipeline.configure_training_split(["2022-23", "2023-24"], "2024-25")
    calls: list[str] = []

    def fail_readiness(*args, **kwargs) -> int:
        calls.append("readiness")
        return 1

    def should_not_run() -> int:
        calls.append("train")
        return 0

    monkeypatch.setattr(
        pipeline, "run_multiseason_training_readiness_check", fail_readiness
    )
    monkeypatch.setattr(
        pipeline, "run_train_pregame_model_multiseason", should_not_run
    )
    rc = pipeline.run_train_models_multiseason(dry_run=False)
    assert rc == 1
    assert calls == ["readiness"]


def test_configure_training_split_requires_both_args(pipeline):
    with pytest.raises(ValueError, match="train-seasons"):
        pipeline.configure_training_split(["2022-23"], None)
    with pytest.raises(ValueError, match="overlap"):
        pipeline.configure_training_split(["2024-25"], "2024-25")


def test_check_multiseason_coverage_mode_registered(pipeline):
    assert "check_multiseason_coverage" in pipeline.get_available_modes()
    assert "prepare_multiseason_training_data" in pipeline.get_available_modes()


def test_prepare_multiseason_training_data_step_order(pipeline):
    names = _step_names(pipeline.PREPARE_MULTISEASON_TRAINING_DATA_STEPS)
    assert names == [
        "build_live_features",
        "build_pregame_features",
        "check_multiseason_coverage",
        "check_multiseason_training_readiness",
    ]
    assert "collect_play_by_play" not in names
    assert "train_pregame_model" not in names


def test_compare_models_mode_registered(pipeline):
    assert "compare_models" in pipeline.get_available_modes()


def test_compare_models_dry_run(pipeline, capsys):
    rc = pipeline.run_compare_models(dry_run=True)
    output = capsys.readouterr().out
    assert rc == 0
    assert "compare_models" in output
    assert "DRY-RUN" in output
    assert "model_comparison_summary.csv" in output


def test_prepare_multiseason_training_data_dry_run(pipeline, capsys):
    pipeline.configure_seasons(seasons=["2022-23", "2023-24", "2024-25"])
    pipeline.configure_training_split(["2022-23", "2023-24"], "2024-25")
    rc = pipeline.run_prepare_multiseason_training_data(dry_run=True)
    output = capsys.readouterr().out
    assert rc == 0
    assert "prepare_multiseason_training_data" in output
    assert "Local post-collection prep" in output
    assert "check_multiseason_coverage" in output


def test_playoff_modes_registered(pipeline):
    modes = pipeline.get_available_modes()
    for mode in [
        "collect_playoff_games",
        "collect_playoff_play_by_play",
        "check_playoff_coverage",
        "build_playoff_live_features",
        "predict_playoff_live",
        "build_finals_case_study",
        "build_finals_pregame_predictions",
        "build_finals_projected_series_path",
        "run_playoff_case_study_pipeline",
    ]:
        assert mode in modes


def test_playoff_case_study_pipeline_step_order(pipeline):
    names = _step_names(pipeline.PLAYOFF_CASE_STUDY_STEPS)
    assert names == [
        "check_playoff_coverage",
        "build_playoff_live_features",
        "predict_playoff_live",
        "build_finals_case_study",
        "build_finals_pregame_predictions",
        "build_finals_projected_series_path",
        "qa",
    ]
    assert "collect_playoff_games" not in names
    assert "collect_playoff_play_by_play" not in names


def test_playoff_case_study_pipeline_dry_run(pipeline, capsys):
    pipeline.configure_seasons(seasons=["2022-23", "2023-24", "2024-25", "2025-26"])
    rc = pipeline.run_playoff_case_study_pipeline(dry_run=True)
    output = capsys.readouterr().out
    assert rc == 0
    assert "run_playoff_case_study_pipeline" in output
    assert "check_playoff_coverage" in output
    assert "build_finals_case_study" in output

