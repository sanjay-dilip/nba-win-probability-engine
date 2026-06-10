"""Central configuration for the NBA Win Probability Engine.

All project paths are defined here using ``pathlib`` so the rest of the
codebase never hard-codes file locations. Importing this module is cheap: it
only defines paths and (optionally) loads environment variables from a local
``.env`` file. It does not create folders or read data files at import time.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Root directory
# ---------------------------------------------------------------------------
# config.py lives in <root>/src/config.py, so the project root is two parents up.
ROOT_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Load environment variables from .env (if python-dotenv is installed)
# ---------------------------------------------------------------------------
# This lets settings like USE_SAMPLE_DATA, NBA_SEASONS and RATE_LIMIT_SECONDS
# come from a local .env file. If python-dotenv is not installed, or no .env
# file exists, we fall back to real environment variables and sensible defaults.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env")
except ImportError:
    # python-dotenv is optional; the project still works using OS env vars.
    pass

# ---------------------------------------------------------------------------
# Top-level directories
# ---------------------------------------------------------------------------
APP_DIR = ROOT_DIR / "app"
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
DOCS_DIR = ROOT_DIR / "docs"
ASSETS_DIR = ROOT_DIR / "assets"

# ---------------------------------------------------------------------------
# Data sub-directories
# ---------------------------------------------------------------------------
SAMPLE_DATA_DIR = DATA_DIR / "sample"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MANUAL_DATA_DIR = DATA_DIR / "manual"
LOGS_DIR = DATA_DIR / "logs"

# ---------------------------------------------------------------------------
# Output sub-directories
# ---------------------------------------------------------------------------
CHARTS_DIR = OUTPUTS_DIR / "charts"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# ---------------------------------------------------------------------------
# Sample data file paths (committed to the repo for the portfolio demo)
# ---------------------------------------------------------------------------
SAMPLE_GAMES_PATH = SAMPLE_DATA_DIR / "sample_games.csv"
SAMPLE_PREGAME_FEATURES_PATH = SAMPLE_DATA_DIR / "sample_pregame_features.csv"
SAMPLE_LIVE_FEATURES_PATH = SAMPLE_DATA_DIR / "sample_live_features.csv"
SAMPLE_PREDICTIONS_PATH = SAMPLE_DATA_DIR / "sample_predictions.csv"

# ---------------------------------------------------------------------------
# Raw data file paths (collected straight from the NBA API)
# ---------------------------------------------------------------------------
RAW_GAMES_PATH = RAW_DATA_DIR / "games.csv"
INVALID_GAMES_REPORT_PATH = REPORTS_DIR / "invalid_games_rows.csv"
SKIPPED_GAMES_REPORT_PATH = REPORTS_DIR / "skipped_games_report.csv"

# Play-by-play raw collection paths (Build 3)
RAW_PLAY_BY_PLAY_PATH = RAW_DATA_DIR / "play_by_play.csv"
INVALID_PBP_REPORT_PATH = REPORTS_DIR / "invalid_play_by_play_rows.csv"
PBP_FAILURES_REPORT_PATH = REPORTS_DIR / "play_by_play_collection_failures.csv"
# Coverage report written after every play-by-play collection run.
PBP_COVERAGE_REPORT_PATH = REPORTS_DIR / "play_by_play_coverage_report.csv"

# Team-stats raw collection paths (Build 4). Build 4 stores raw, season-level
# team statistics under data/raw/ (the data/processed/team_stats.csv path below
# is reserved for a later, cleaned/feature-ready version).
RAW_TEAM_STATS_PATH = RAW_DATA_DIR / "team_stats.csv"
INVALID_TEAM_STATS_REPORT_PATH = REPORTS_DIR / "invalid_team_stats_rows.csv"
TEAM_STATS_FAILURES_REPORT_PATH = REPORTS_DIR / "team_stats_collection_failures.csv"

# Pre-game feature build report paths (Build 5). The feature output itself uses
# the existing PREGAME_FEATURES_PATH (data/processed/pregame_features.csv) below.
INVALID_PREGAME_FEATURES_REPORT_PATH = REPORTS_DIR / "invalid_pregame_feature_rows.csv"
PREGAME_FEATURE_BUILD_ISSUES_PATH = REPORTS_DIR / "pregame_feature_build_issues.csv"

# Live feature build paths (Build 6). The live feature output itself uses the
# existing LIVE_FEATURES_PATH (data/processed/live_features.csv) below.
GAME_RESULTS_PATH = PROCESSED_DATA_DIR / "game_results.csv"
INVALID_LIVE_FEATURES_REPORT_PATH = REPORTS_DIR / "invalid_live_feature_rows.csv"
LIVE_FEATURE_BUILD_ISSUES_PATH = REPORTS_DIR / "live_feature_build_issues.csv"
INVALID_GAME_RESULTS_REPORT_PATH = REPORTS_DIR / "invalid_game_results_rows.csv"

# Pre-game model training artifacts (Build 7). The trained scikit-learn pipeline
# and the exact feature-column list are saved under models/; evaluation metrics
# and a calibration summary are written to outputs/reports/.
PREGAME_MODEL_PATH = MODELS_DIR / "pregame_model.pkl"
PREGAME_FEATURE_COLUMNS_PATH = MODELS_DIR / "pregame_feature_columns.pkl"
PREGAME_MODEL_METRICS_PATH = REPORTS_DIR / "pregame_model_metrics.csv"
PREGAME_MODEL_CALIBRATION_PATH = REPORTS_DIR / "pregame_model_calibration.csv"

# Live model training artifacts (Build 8). Same layout as the pre-game model:
# the fitted pipeline and feature-column dictionary live under models/; metrics,
# a calibration summary, and an optional by-phase breakdown go to outputs/reports/.
LIVE_MODEL_PATH = MODELS_DIR / "live_model.pkl"
LIVE_FEATURE_COLUMNS_PATH = MODELS_DIR / "live_feature_columns.pkl"
LIVE_MODEL_METRICS_PATH = REPORTS_DIR / "live_model_metrics.csv"
LIVE_MODEL_CALIBRATION_PATH = REPORTS_DIR / "live_model_calibration.csv"
LIVE_MODEL_PHASE_METRICS_PATH = REPORTS_DIR / "live_model_metrics_by_phase.csv"

# Multi-season model training artifacts (Build 18). Separate from single-season
# baseline artifacts so accidental retraining does not overwrite dashboard models.
PREGAME_MODEL_MULTISEASON_PATH = MODELS_DIR / "pregame_model_multiseason.pkl"
PREGAME_FEATURE_COLUMNS_MULTISEASON_PATH = (
    MODELS_DIR / "pregame_feature_columns_multiseason.pkl"
)
LIVE_MODEL_MULTISEASON_PATH = MODELS_DIR / "live_model_multiseason.pkl"
LIVE_FEATURE_COLUMNS_MULTISEASON_PATH = (
    MODELS_DIR / "live_feature_columns_multiseason.pkl"
)
PREGAME_MODEL_METRICS_MULTISEASON_PATH = (
    REPORTS_DIR / "pregame_model_metrics_multiseason.csv"
)
PREGAME_MODEL_CALIBRATION_MULTISEASON_PATH = (
    REPORTS_DIR / "pregame_model_calibration_multiseason.csv"
)
LIVE_MODEL_METRICS_MULTISEASON_PATH = (
    REPORTS_DIR / "live_model_metrics_multiseason.csv"
)
LIVE_MODEL_CALIBRATION_MULTISEASON_PATH = (
    REPORTS_DIR / "live_model_calibration_multiseason.csv"
)
LIVE_MODEL_PHASE_METRICS_MULTISEASON_PATH = (
    REPORTS_DIR / "live_model_metrics_by_phase_multiseason.csv"
)
MULTISEASON_TRAINING_READINESS_PATH = (
    REPORTS_DIR / "multiseason_training_readiness.csv"
)
MULTISEASON_COVERAGE_REPORT_PATH = (
    REPORTS_DIR / "multiseason_coverage_report.csv"
)

# Season defaults (Build 17). See src/season_config.py for validation helpers.
DEFAULT_SEASON = "2024-25"
MULTI_SEASON_TRAIN_START = "2022-23"
SUPPORTED_SEASONS = ["2022-23", "2023-24", "2024-25"]
FUTURE_TEST_SEASON = "2025-26"

# Evaluation layer outputs (Build 13). Summaries are written by src/evaluate.py
# from saved predictions and existing training metric reports — no retraining.
EVALUATION_SUMMARY_PATH = REPORTS_DIR / "evaluation_summary.csv"
PREGAME_PREDICTION_SUMMARY_PATH = REPORTS_DIR / "pregame_prediction_summary.csv"
LIVE_PREDICTION_SUMMARY_PATH = REPORTS_DIR / "live_prediction_summary.csv"
BIGGEST_MOMENTUM_SWINGS_PATH = REPORTS_DIR / "biggest_momentum_swings.csv"
EVALUATION_BY_TEAM_PATH = REPORTS_DIR / "evaluation_by_team.csv"

# Data freshness / system status (Build 14). Written by src/data_freshness.py.
DATA_FRESHNESS_REPORT_PATH = REPORTS_DIR / "data_freshness_summary.csv"

# Project QA (Build 16). Written by src/project_qa.py.
PROJECT_QA_REPORT_PATH = REPORTS_DIR / "project_qa_summary.csv"

# Model version comparison (Build 19). Written by src/compare_model_versions.py.
MODEL_COMPARISON_SUMMARY_PATH = REPORTS_DIR / "model_comparison_summary.csv"
MODEL_COMPARISON_DETAIL_PATH = REPORTS_DIR / "model_comparison_detail.csv"
PHASE_COMPARISON_SUMMARY_PATH = REPORTS_DIR / "live_phase_comparison_summary.csv"
CALIBRATION_COMPARISON_SUMMARY_PATH = REPORTS_DIR / "calibration_comparison_summary.csv"

# Playoff / NBA Finals case study (Build 20.6). Separate from regular-season paths.
PLAYOFF_RAW_DIR = DATA_DIR / "playoffs" / "raw"
PLAYOFF_PROCESSED_DIR = DATA_DIR / "playoffs" / "processed"
PLAYOFF_GAMES_PATH = PLAYOFF_RAW_DIR / "playoff_games.csv"
PLAYOFF_PLAY_BY_PLAY_PATH = PLAYOFF_RAW_DIR / "playoff_play_by_play.csv"
PLAYOFF_LIVE_FEATURES_PATH = PLAYOFF_PROCESSED_DIR / "playoff_live_features.csv"
PLAYOFF_GAME_RESULTS_PATH = PLAYOFF_PROCESSED_DIR / "playoff_game_results.csv"
PLAYOFF_LIVE_PREDICTIONS_PATH = PLAYOFF_PROCESSED_DIR / "playoff_live_predictions.csv"
PLAYOFF_COVERAGE_REPORT_PATH = REPORTS_DIR / "playoff_coverage_report.csv"
PLAYOFF_MODEL_PERFORMANCE_PATH = REPORTS_DIR / "playoff_model_performance.csv"
NBA_FINALS_CASE_STUDY_SUMMARY_PATH = REPORTS_DIR / "nba_finals_case_study_summary.csv"
FINALS_PREGAME_FEATURES_PATH = PLAYOFF_PROCESSED_DIR / "finals_pregame_features.csv"
FINALS_PREGAME_PREDICTIONS_PATH = PLAYOFF_PROCESSED_DIR / "finals_pregame_predictions.csv"
FINALS_UPCOMING_PREDICTIONS_REPORT_PATH = REPORTS_DIR / "finals_upcoming_predictions.csv"
FINALS_SCHEDULE_OVERRIDES_PATH = MANUAL_DATA_DIR / "finals_schedule_overrides.csv"
FINALS_PROJECTED_SERIES_PATH = REPORTS_DIR / "finals_projected_series_path.csv"
FINALS_AUTO_REFRESH_LOG_PATH = REPORTS_DIR / "finals_auto_refresh_log.csv"

# ---------------------------------------------------------------------------
# Full data file paths (populated in later builds; not created yet)
# ---------------------------------------------------------------------------
GAMES_PATH = PROCESSED_DATA_DIR / "games.csv"
PLAY_BY_PLAY_PATH = PROCESSED_DATA_DIR / "play_by_play.csv"
TEAM_STATS_PATH = PROCESSED_DATA_DIR / "team_stats.csv"
PREGAME_FEATURES_PATH = PROCESSED_DATA_DIR / "pregame_features.csv"
LIVE_FEATURES_PATH = PROCESSED_DATA_DIR / "live_features.csv"
# Prediction outputs (Build 9) — dashboard-ready CSVs alongside feature files.
PREGAME_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "pregame_predictions.csv"
LIVE_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "live_predictions.csv"
POSTGAME_RESULTS_PATH = MANUAL_DATA_DIR / "postgame_results.csv"
DATA_REFRESH_LOG_PATH = LOGS_DIR / "data_refresh_log.csv"

# ---------------------------------------------------------------------------
# Convenience collections used by ensure_directories()
# ---------------------------------------------------------------------------
ALL_DIRECTORIES = [
    APP_DIR,
    DATA_DIR,
    SAMPLE_DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MANUAL_DATA_DIR,
    LOGS_DIR,
    MODELS_DIR,
    MODELS_DIR / "sample",
    OUTPUTS_DIR,
    CHARTS_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    PLAYOFF_RAW_DIR,
    PLAYOFF_PROCESSED_DIR,
    DOCS_DIR,
    ASSETS_DIR,
]


def get_data_mode() -> str:
    """Return the active data mode based on the ``USE_SAMPLE_DATA`` env var.

    Returns:
        ``"sample"`` when ``USE_SAMPLE_DATA`` is truthy (the default), otherwise
        ``"full"``. Recognised truthy values are ``1``, ``true``, ``yes`` and
        ``on`` (case-insensitive).
    """
    raw_value = os.getenv("USE_SAMPLE_DATA", "true").strip().lower()
    use_sample = raw_value in {"1", "true", "yes", "on"}
    return "sample" if use_sample else "full"
