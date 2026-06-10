"""Command-line entry point for the NBA Win Probability Engine.

Individual modes run one pipeline step. Grouped modes orchestrate existing
steps in a safe order (Build 15). Use ``--dry-run`` to preview grouped modes.
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Optional, Tuple

from src import config
from src.collect_games import collect_games
from src.collect_play_by_play import collect_play_by_play
from src.collect_team_stats import collect_team_stats
from src.build_pregame_features import build_pregame_features
from src.build_live_features import build_live_features
from src.train_pregame_model import train_pregame_model
from src.train_live_model import train_live_model
from src.predict_pregame import predict_pregame
from src.predict_live import predict_live
from src.evaluate import run_evaluation
from src.data_freshness import run_data_freshness_check
from src.project_qa import run_project_qa
from src.multiseason_training import (
    build_multiseason_training_readiness,
    require_training_split_error_message,
    run_multiseason_training_readiness_check,
    save_multiseason_training_readiness,
    should_allow_training,
)
from src.multiseason_coverage import run_multiseason_coverage_check
from src.compare_model_versions import run_model_comparison
from src.playoff_case_study import (
    build_playoff_live_features_from_raw,
    collect_playoff_games,
    collect_playoff_play_by_play,
    predict_playoff_live_games,
    run_build_finals_case_study,
    run_build_finals_pregame_predictions,
    run_build_finals_projected_series_path,
    run_check_playoff_coverage,
)
from src.season_config import parse_pipeline_season_args
from src.utils import ensure_directories

Step = Tuple[str, Callable[[], int]]

# Active season selection set by main() from --season / --seasons.
_single_season: str = "2024-25"
_multi_seasons: List[str] = ["2022-23", "2023-24", "2024-25"]

# Active train/test split for multi-season training modes (Build 18).
_train_seasons: Optional[List[str]] = None
_test_season: Optional[str] = None

MULTISEASON_TRAINING_MODES = {
    "check_multiseason_training_readiness",
    "train_pregame_model_multiseason",
    "train_live_model_multiseason",
    "train_models_multiseason",
    "prepare_multiseason_training_data",
}

MODES_REQUIRING_TRAIN_SPLIT = MULTISEASON_TRAINING_MODES


def configure_seasons(
    season: Optional[str] = None,
    seasons: Optional[List[str]] = None,
) -> None:
    """Store pipeline season args for individual and grouped mode runners."""
    global _single_season, _multi_seasons
    single, multi = parse_pipeline_season_args(season, seasons)
    _single_season = single
    _multi_seasons = multi


def get_single_season() -> str:
    """Return the active single season (individual modes)."""
    return _single_season


def get_multi_seasons() -> List[str]:
    """Return the active season list (multi-season grouped modes)."""
    return list(_multi_seasons)


def configure_training_split(
    train_seasons: Optional[List[str]] = None,
    test_season: Optional[str] = None,
) -> None:
    """Store --train-seasons / --test-season for multi-season training modes."""
    global _train_seasons, _test_season
    if train_seasons is None and test_season is None:
        raise ValueError(require_training_split_error_message())
    if train_seasons is None or test_season is None:
        raise ValueError(require_training_split_error_message())
    from src.multiseason_training import validate_train_test_seasons

    validated_train, validated_test = validate_train_test_seasons(
        train_seasons, test_season
    )
    _train_seasons = validated_train
    _test_season = validated_test


def get_train_seasons() -> List[str]:
    """Return active train seasons (multi-season training modes)."""
    if _train_seasons is None:
        raise ValueError(require_training_split_error_message())
    return list(_train_seasons)


def get_test_season() -> str:
    """Return active test season (multi-season training modes)."""
    if _test_season is None:
        raise ValueError(require_training_split_error_message())
    return _test_season


REFRESH_PBP_WARNING = (
    "WARNING: refresh_full_season_play_by_play may take 30-45+ minutes "
    "and makes many NBA API calls."
)

MULTI_SEASON_PBP_WARNING = (
    "WARNING: API-heavy — collect_play_by_play_multi_season may take "
    "several hours across multiple seasons."
)

PLAYOFF_PBP_WARNING = (
    "WARNING: API-heavy — collect_playoff_play_by_play makes many NBA API calls. "
    "Playoff volumes are smaller than full regular seasons but collection is still "
    "explicit and user-triggered."
)


# ---------------------------------------------------------------------------
# Individual step runners (unchanged behavior)
# ---------------------------------------------------------------------------

def run_setup() -> int:
    """Create every project directory and report success."""
    print("Setting up project directories...")
    ensure_directories()
    for directory in config.ALL_DIRECTORIES:
        print(f"  [ok] {directory}")
    print("Setup complete. Project folder structure is ready.")
    return 0


def run_sample() -> int:
    """Confirm the sample data files exist and print their paths."""
    print("Checking sample data files...")
    sample_files = [
        config.SAMPLE_GAMES_PATH,
        config.SAMPLE_PREGAME_FEATURES_PATH,
        config.SAMPLE_LIVE_FEATURES_PATH,
        config.SAMPLE_PREDICTIONS_PATH,
    ]

    all_present = True
    for path in sample_files:
        if path.exists():
            print(f"  [found]   {path}")
        else:
            all_present = False
            print(f"  [missing] {path}")

    if all_present:
        print("All sample data files are present.")
        return 0

    print("Some sample data files are missing. Check the data/sample/ folder.")
    return 1


def run_collect_games() -> int:
    """Collect the NBA game schedule into data/raw/games.csv."""
    season = get_single_season()
    print(f"Collecting NBA game schedule for season {season}...")
    return collect_games(seasons=[season])


def run_collect_play_by_play() -> int:
    """Collect raw play-by-play for up to 10 new games into data/raw/play_by_play.csv."""
    season = get_single_season()
    print(f"Collecting NBA play-by-play data (limit: 10 games, season {season})...")
    return collect_play_by_play(season=season, limit=10)


def run_collect_play_by_play_full_season() -> int:
    """Collect ALL remaining regular-season play-by-play for one season (idempotent)."""
    season = get_single_season()
    print(f"Collecting full-season NBA play-by-play ({season}, no limit)...")
    print("This is idempotent — already-collected games are skipped. It may take a while.")
    return collect_play_by_play(season=season, limit=0)


def run_collect_team_stats() -> int:
    """Collect raw team season stats into data/raw/team_stats.csv."""
    season = get_single_season()
    print(f"Collecting NBA team stats for season {season}...")
    return collect_team_stats(seasons=[season])


def run_build_pregame_features() -> int:
    """Build leakage-safe pre-game features into data/processed/pregame_features.csv."""
    print("Building pre-game features (recent window: 5 games)...")
    return build_pregame_features(recent_window=5)


def run_build_live_features() -> int:
    """Build live features + game results from play-by-play into data/processed/."""
    print("Building live features and game results...")
    return build_live_features()


def run_train_pregame_model() -> int:
    """Train the baseline pre-game winner model from pregame_features.csv."""
    print("Training pre-game model (Logistic Regression baseline)...")
    return train_pregame_model(test_size=0.2)


def run_train_live_model() -> int:
    """Train the baseline live win-probability model from live_features.csv."""
    print("Training live model (Logistic Regression baseline)...")
    return train_live_model(test_size=0.2)


def run_check_multiseason_training_readiness() -> int:
    """Verify feature files contain requested train/test seasons."""
    print("Checking multi-season training readiness...")
    return run_multiseason_training_readiness_check(
        get_train_seasons(),
        get_test_season(),
        verbose=True,
    )


def run_check_multiseason_coverage() -> int:
    """Inspect local files and write multi-season coverage report."""
    print("Checking multi-season data coverage (local files only)...")
    train = _train_seasons
    test = _test_season
    return run_multiseason_coverage_check(
        get_multi_seasons(),
        train_seasons=train,
        test_season=test,
        verbose=True,
    )


def _pregame_readiness_ok() -> bool:
    readiness = build_multiseason_training_readiness(
        get_train_seasons(), get_test_season()
    )
    save_multiseason_training_readiness(readiness)
    return should_allow_training(readiness, sections=["pregame", "split"])


def _live_readiness_ok() -> bool:
    readiness = build_multiseason_training_readiness(
        get_train_seasons(), get_test_season()
    )
    save_multiseason_training_readiness(readiness)
    return should_allow_training(readiness, sections=["live", "split"])


def run_train_pregame_model_multiseason() -> int:
    """Train pre-game model on explicit train seasons; test on hold-out season."""
    print("Training multi-season pre-game model...")
    if not _pregame_readiness_ok():
        print("Pre-game readiness failed — training blocked.")
        return 1
    return train_pregame_model(
        train_seasons=get_train_seasons(),
        test_season=get_test_season(),
        model_path=config.PREGAME_MODEL_MULTISEASON_PATH,
        feature_columns_path=config.PREGAME_FEATURE_COLUMNS_MULTISEASON_PATH,
        metrics_path=config.PREGAME_MODEL_METRICS_MULTISEASON_PATH,
        calibration_path=config.PREGAME_MODEL_CALIBRATION_MULTISEASON_PATH,
    )


def run_train_live_model_multiseason() -> int:
    """Train live model on explicit train seasons; test on hold-out season."""
    print("Training multi-season live model...")
    if not _live_readiness_ok():
        print("Live readiness failed — training blocked.")
        return 1
    return train_live_model(
        train_seasons=get_train_seasons(),
        test_season=get_test_season(),
        model_path=config.LIVE_MODEL_MULTISEASON_PATH,
        feature_columns_path=config.LIVE_FEATURE_COLUMNS_MULTISEASON_PATH,
        metrics_path=config.LIVE_MODEL_METRICS_MULTISEASON_PATH,
        calibration_path=config.LIVE_MODEL_CALIBRATION_MULTISEASON_PATH,
        phase_metrics_path=config.LIVE_MODEL_PHASE_METRICS_MULTISEASON_PATH,
    )


def run_predict_pregame() -> int:
    """Generate pre-game predictions from saved model artifacts."""
    print("Generating pre-game predictions...")
    return predict_pregame()


def run_predict_live() -> int:
    """Generate live predictions from saved model artifacts."""
    print("Generating live predictions...")
    return predict_live()


def run_predict_all() -> int:
    """Run pre-game and live prediction in sequence."""
    print("Generating all predictions (pre-game, then live)...")
    rc = predict_pregame()
    if rc != 0:
        return rc
    return predict_live()


def run_evaluate() -> int:
    """Summarize saved predictions and write evaluation reports."""
    print("Running evaluation (reads predictions + training metrics)...")
    return run_evaluation(verbose=True)


def run_check_data_freshness() -> int:
    """Inspect project files and write the data freshness summary report."""
    print("Checking data freshness and system status...")
    return run_data_freshness_check(verbose=True)


def run_compare_models(dry_run: bool = False) -> int:
    """Compare single-season baseline vs multi-season holdout metric reports."""
    print("Comparing model versions (reads existing reports only)...")
    return run_model_comparison(verbose=True, dry_run=dry_run)


def run_collect_playoff_games_mode() -> int:
    """Collect playoff game metadata into data/playoffs/raw/playoff_games.csv."""
    seasons = get_multi_seasons()
    print(f"Collecting playoff games for seasons: {', '.join(seasons)}")
    return collect_playoff_games(seasons)


def run_collect_playoff_play_by_play_mode() -> int:
    """Collect playoff play-by-play into data/playoffs/raw/playoff_play_by_play.csv."""
    print(PLAYOFF_PBP_WARNING)
    seasons = get_multi_seasons()
    print(f"Collecting playoff play-by-play for seasons: {', '.join(seasons)}")
    return collect_playoff_play_by_play(seasons)


def run_check_playoff_coverage_mode() -> int:
    """Inspect local playoff files and write coverage report."""
    return run_check_playoff_coverage(get_multi_seasons())


def run_build_playoff_live_features_mode() -> int:
    """Build playoff live features from local playoff raw files."""
    print("Building playoff live features (separate from regular-season paths)...")
    return build_playoff_live_features_from_raw(seasons=get_multi_seasons())


def run_predict_playoff_live_mode() -> int:
    """Score playoff live features with the primary multi-season live model."""
    print("Generating playoff live predictions (primary multi-season model)...")
    return predict_playoff_live_games()


def run_build_finals_case_study_mode() -> int:
    """Write NBA Finals case-study summary report."""
    return run_build_finals_case_study()


def run_build_finals_pregame_predictions_mode() -> int:
    """Build Finals pre-game features and predictions for upcoming games."""
    return run_build_finals_pregame_predictions()


def run_build_finals_projected_series_path_mode() -> int:
    """Build projected Finals series path report."""
    return run_build_finals_projected_series_path()


# ---------------------------------------------------------------------------
# Grouped mode step definitions (Build 15)
# ---------------------------------------------------------------------------

BUILD_FEATURES_STEPS: List[Step] = [
    ("build_pregame_features", run_build_pregame_features),
    ("build_live_features", run_build_live_features),
    ("check_data_freshness", run_check_data_freshness),
]

TRAIN_MODELS_STEPS: List[Step] = [
    ("train_pregame_model", run_train_pregame_model),
    ("train_live_model", run_train_live_model),
    ("check_data_freshness", run_check_data_freshness),
]

SCORE_OUTPUTS_STEPS: List[Step] = [
    ("predict_all", run_predict_all),
    ("evaluate", run_evaluate),
    ("check_data_freshness", run_check_data_freshness),
]

REFRESH_LOCAL_DATA_STEPS: List[Step] = [
    ("collect_games", run_collect_games),
    ("collect_team_stats", run_collect_team_stats),
]

REFRESH_FULL_SEASON_PBP_STEPS: List[Step] = [
    ("collect_play_by_play_full_season", run_collect_play_by_play_full_season),
    ("build_live_features", run_build_live_features),
    ("predict_live", run_predict_live),
    ("evaluate", run_evaluate),
    ("check_data_freshness", run_check_data_freshness),
]

DASHBOARD_READY_SUBMODES = ["build_features", "train_models", "score_outputs"]

QA_STEPS: List[Step] = [
    ("check_data_freshness", run_check_data_freshness),
    ("project_qa", run_project_qa),
]

BUILD_FEATURES_MULTI_SEASON_STEPS: List[Step] = [
    ("build_pregame_features", run_build_pregame_features),
    ("build_live_features", run_build_live_features),
    ("check_data_freshness", run_check_data_freshness),
    ("project_qa", run_project_qa),
]

TRAIN_MODELS_MULTISEASON_STEPS: List[Step] = [
    ("check_multiseason_training_readiness", run_check_multiseason_training_readiness),
    ("train_pregame_model_multiseason", run_train_pregame_model_multiseason),
    ("train_live_model_multiseason", run_train_live_model_multiseason),
    ("check_data_freshness", run_check_data_freshness),
    ("project_qa", run_project_qa),
]

PREPARE_MULTISEASON_TRAINING_DATA_STEPS: List[Step] = [
    ("build_live_features", run_build_live_features),
    ("build_pregame_features", run_build_pregame_features),
    ("check_multiseason_coverage", run_check_multiseason_coverage),
    ("check_multiseason_training_readiness", run_check_multiseason_training_readiness),
]


def _collect_games_season(season: str) -> int:
    print(f"Collecting NBA game schedule for season {season}...")
    return collect_games(seasons=[season])


def _collect_team_stats_season(season: str) -> int:
    print(f"Collecting NBA team stats for season {season}...")
    return collect_team_stats(seasons=[season])


def _collect_pbp_full_season(season: str) -> int:
    print(f"Collecting full-season play-by-play for season {season}...")
    return collect_play_by_play(season=season, limit=0)


def build_multi_season_steps(
    prefix: str,
    seasons: List[str],
    runner: Callable[[str], int],
) -> List[Step]:
    """Build one pipeline step per season using ``runner(season)``."""
    return [
        (f"{prefix}_{season}", (lambda s=season: runner(s)))
        for season in seasons
    ]


def build_collect_games_multi_season_steps(seasons: Optional[List[str]] = None) -> List[Step]:
    return build_multi_season_steps(
        "collect_games",
        seasons or get_multi_seasons(),
        _collect_games_season,
    )


def build_collect_team_stats_multi_season_steps(
    seasons: Optional[List[str]] = None,
) -> List[Step]:
    return build_multi_season_steps(
        "collect_team_stats",
        seasons or get_multi_seasons(),
        _collect_team_stats_season,
    )


def build_collect_pbp_multi_season_steps(seasons: Optional[List[str]] = None) -> List[Step]:
    return build_multi_season_steps(
        "collect_play_by_play_full_season",
        seasons or get_multi_seasons(),
        _collect_pbp_full_season,
    )


def build_refresh_multi_season_metadata_steps(seasons: Optional[List[str]] = None) -> List[Step]:
    seasons = seasons or get_multi_seasons()
    steps: List[Step] = []
    steps.extend(build_collect_games_multi_season_steps(seasons))
    steps.extend(build_collect_team_stats_multi_season_steps(seasons))
    steps.extend([
        ("check_data_freshness", run_check_data_freshness),
        ("project_qa", run_project_qa),
    ])
    return steps


def get_grouped_mode_steps(mode_name: str) -> List[Step]:
    """Return the ordered steps for a grouped mode name."""
    grouped = {
        "build_features": BUILD_FEATURES_STEPS,
        "train_models": TRAIN_MODELS_STEPS,
        "score_outputs": SCORE_OUTPUTS_STEPS,
        "refresh_local_data": REFRESH_LOCAL_DATA_STEPS,
        "refresh_full_season_play_by_play": REFRESH_FULL_SEASON_PBP_STEPS,
    }
    if mode_name not in grouped:
        raise KeyError(f"Not a grouped mode: {mode_name}")
    return grouped[mode_name]


def get_available_modes() -> Dict[str, str]:
    """Return all pipeline mode names and short descriptions."""
    return {
        "setup": "Create project folder structure",
        "sample": "Verify committed sample CSV files exist",
        "collect_games": "Collect NBA game schedule into data/raw/games.csv",
        "collect_play_by_play": "Collect play-by-play for up to 10 games",
        "collect_play_by_play_full_season": "Collect all remaining season play-by-play (API-heavy)",
        "collect_team_stats": "Collect team season stats into data/raw/team_stats.csv",
        "build_pregame_features": "Build leakage-safe pre-game features",
        "build_live_features": "Build live features and game_results.csv",
        "train_pregame_model": "Train pre-game Logistic Regression baseline",
        "train_live_model": "Train live Logistic Regression baseline",
        "predict_pregame": "Score pre-game features with saved model",
        "predict_live": "Score live features with saved model",
        "predict_all": "Run pre-game and live prediction",
        "evaluate": "Write evaluation reports from saved predictions",
        "check_data_freshness": "Inspect files and write freshness summary",
        "build_features": "Grouped: rebuild pre-game + live features, then freshness check",
        "train_models": "Grouped: train both models, then freshness check",
        "score_outputs": "Grouped: predict_all + evaluate + freshness check",
        "dashboard_ready": "Grouped: build_features + train_models + score_outputs (no API)",
        "refresh_local_data": "Grouped: collect games + team stats (lightweight API)",
        "refresh_full_season_play_by_play": "Grouped: full PBP collect + live rebuild + score (API-heavy)",
        "qa": "Grouped: check_data_freshness + project QA report (no API, no training)",
        "collect_games_multi_season": "Grouped: collect_games for each season in --seasons (API-light)",
        "collect_team_stats_multi_season": "Grouped: collect_team_stats for each season (API-light)",
        "refresh_multi_season_metadata": "Grouped: multi-season games + team stats + freshness + qa (API-light)",
        "collect_play_by_play_multi_season": "Grouped: full PBP for each season in --seasons (API-heavy)",
        "build_features_multi_season": "Grouped: rebuild features + freshness + qa (local only)",
        "check_multiseason_training_readiness": "Check feature coverage for train/test seasons (no training)",
        "train_pregame_model_multiseason": "Train pre-game model on --train-seasons, test on --test-season",
        "train_live_model_multiseason": "Train live model on --train-seasons, test on --test-season",
        "train_models_multiseason": "Grouped: readiness + both multiseason models + freshness + qa",
        "check_multiseason_coverage": "Inspect local season coverage (no API, no training)",
        "prepare_multiseason_training_data": "Grouped: rebuild features + coverage + readiness (local only)",
        "compare_models": "Compare single-season vs multi-season metric reports (no training)",
        "collect_playoff_games": "Collect playoff game metadata (season_type=Playoffs, separate paths)",
        "collect_playoff_play_by_play": "Collect playoff play-by-play (API-heavy, separate paths)",
        "check_playoff_coverage": "Inspect local playoff files and write coverage report (no API)",
        "build_playoff_live_features": "Build playoff live features from local playoff raw files",
        "predict_playoff_live": "Score playoff live features with primary multi-season live model",
        "build_finals_case_study": "Write NBA Finals case-study summary report",
        "build_finals_pregame_predictions": "Build Finals pre-game predictions (primary multiseason model)",
        "build_finals_projected_series_path": "Build projected Finals series path report",
        "run_playoff_case_study_pipeline": "Grouped: coverage + features + predict + finals report + qa (no API collection)",
    }


def print_mode_plan(mode_name: str, steps: List[Step], dry_run: bool = False) -> None:
    """Print the ordered steps for a grouped mode."""
    flag = "DRY-RUN" if dry_run else "LIVE"
    print(f"\n=== Mode: {mode_name} ({flag}) ===")
    if mode_name == "refresh_full_season_play_by_play":
        print(REFRESH_PBP_WARNING)
    if mode_name == "collect_play_by_play_multi_season":
        print(MULTI_SEASON_PBP_WARNING)
    if mode_name == "refresh_multi_season_metadata":
        print("API-light metadata refresh — games + team stats only (no play-by-play).")
    if mode_name == "dashboard_ready":
        print("Uses existing local raw data only — does not call the NBA API.")
    if mode_name == "build_features_multi_season":
        print("Local feature rebuild only — no NBA API calls.")
    if mode_name == "train_models_multiseason":
        print("Multi-season training — saves separate multiseason artifacts.")
        print(f"  Train seasons: {', '.join(get_train_seasons())}")
        print(f"  Test season:   {get_test_season()}")
    if mode_name == "run_playoff_case_study_pipeline":
        print("Local playoff case-study pipeline — no API collection.")
        print(f"  Seasons: {', '.join(get_multi_seasons())}")
    if mode_name == "collect_playoff_play_by_play":
        print(PLAYOFF_PBP_WARNING)
    if mode_name == "prepare_multiseason_training_data":
        print("Local post-collection prep — no API calls, no model training.")
        print(f"  Coverage seasons: {', '.join(get_multi_seasons())}")
        print(f"  Train seasons: {', '.join(get_train_seasons())}")
        print(f"  Test season:   {get_test_season()}")
    if mode_name == "check_multiseason_coverage":
        print("Inspects local CSV files only — does not call the NBA API.")
    if mode_name in {
        "collect_games_multi_season",
        "collect_team_stats_multi_season",
        "refresh_multi_season_metadata",
        "collect_play_by_play_multi_season",
        "build_features_multi_season",
    }:
        print(f"Seasons: {', '.join(get_multi_seasons())}")
    for index, (step_name, _) in enumerate(steps, start=1):
        print(f"  Step {index}/{len(steps)}: {step_name}")
    print()


def run_step(step_name: str, callable_fn: Callable[[], int]) -> int:
    """Run one pipeline step with start/end logging."""
    print(f"\n--- Starting step: {step_name} ---")
    rc = callable_fn()
    if rc == 0:
        print(f"--- Finished step: {step_name} (ok) ---")
    else:
        print(f"--- Failed step: {step_name} (exit code {rc}) ---")
    return rc


def run_grouped_mode(mode_name: str, steps: List[Step], dry_run: bool = False) -> int:
    """Execute (or preview) a grouped pipeline mode."""
    print_mode_plan(mode_name, steps, dry_run=dry_run)
    if dry_run:
        print(f"[dry-run] Skipping execution for mode: {mode_name}")
        return 0

    print(f"Starting grouped mode: {mode_name}")
    for step_name, step_fn in steps:
        rc = run_step(step_name, step_fn)
        if rc != 0:
            print(f"\nGrouped mode '{mode_name}' stopped at failed step: {step_name}")
            return rc

    print(f"\nGrouped mode '{mode_name}' completed successfully.")
    return 0


def run_dashboard_ready(dry_run: bool = False) -> int:
    """Prepare local dashboard outputs without collecting new API data."""
    steps = []
    for sub_mode in DASHBOARD_READY_SUBMODES:
        steps.extend(get_grouped_mode_steps(sub_mode))

    print_mode_plan("dashboard_ready", steps, dry_run=dry_run)
    if dry_run:
        print("[dry-run] Skipping execution for mode: dashboard_ready")
        return 0

    print("Starting grouped mode: dashboard_ready")
    print("Uses existing local raw data only — does not call the NBA API.")
    for sub_mode in DASHBOARD_READY_SUBMODES:
        rc = run_grouped_mode(sub_mode, get_grouped_mode_steps(sub_mode), dry_run=False)
        if rc != 0:
            print(f"\nGrouped mode 'dashboard_ready' stopped during sub-mode: {sub_mode}")
            return rc

    print("\nGrouped mode 'dashboard_ready' completed successfully.")
    return 0


def run_refresh_full_season_play_by_play(dry_run: bool = False) -> int:
    """API-heavy grouped mode for full-season play-by-play refresh."""
    if not dry_run:
        print(REFRESH_PBP_WARNING)
    return run_grouped_mode(
        "refresh_full_season_play_by_play",
        REFRESH_FULL_SEASON_PBP_STEPS,
        dry_run=dry_run,
    )


def run_collect_games_multi_season(dry_run: bool = False) -> int:
    return run_grouped_mode(
        "collect_games_multi_season",
        build_collect_games_multi_season_steps(),
        dry_run=dry_run,
    )


def run_collect_team_stats_multi_season(dry_run: bool = False) -> int:
    return run_grouped_mode(
        "collect_team_stats_multi_season",
        build_collect_team_stats_multi_season_steps(),
        dry_run=dry_run,
    )


def run_refresh_multi_season_metadata(dry_run: bool = False) -> int:
    return run_grouped_mode(
        "refresh_multi_season_metadata",
        build_refresh_multi_season_metadata_steps(),
        dry_run=dry_run,
    )


def run_collect_play_by_play_multi_season(dry_run: bool = False) -> int:
    if not dry_run:
        print(MULTI_SEASON_PBP_WARNING)
    return run_grouped_mode(
        "collect_play_by_play_multi_season",
        build_collect_pbp_multi_season_steps(),
        dry_run=dry_run,
    )


def run_build_features_multi_season(dry_run: bool = False) -> int:
    return run_grouped_mode(
        "build_features_multi_season",
        BUILD_FEATURES_MULTI_SEASON_STEPS,
        dry_run=dry_run,
    )


def run_train_models_multiseason(dry_run: bool = False) -> int:
    return run_grouped_mode(
        "train_models_multiseason",
        TRAIN_MODELS_MULTISEASON_STEPS,
        dry_run=dry_run,
    )


def run_prepare_multiseason_training_data(dry_run: bool = False) -> int:
    return run_grouped_mode(
        "prepare_multiseason_training_data",
        PREPARE_MULTISEASON_TRAINING_DATA_STEPS,
        dry_run=dry_run,
    )


def run_qa(dry_run: bool = False) -> int:
    """Run freshness check then project QA; save project_qa_summary.csv."""
    return run_grouped_mode("qa", QA_STEPS, dry_run=dry_run)


PLAYOFF_CASE_STUDY_STEPS: List[Step] = [
    ("check_playoff_coverage", run_check_playoff_coverage_mode),
    ("build_playoff_live_features", run_build_playoff_live_features_mode),
    ("predict_playoff_live", run_predict_playoff_live_mode),
    ("build_finals_case_study", run_build_finals_case_study_mode),
    ("build_finals_pregame_predictions", run_build_finals_pregame_predictions_mode),
    ("build_finals_projected_series_path", run_build_finals_projected_series_path_mode),
    ("qa", run_qa),
]


def run_playoff_case_study_pipeline(dry_run: bool = False) -> int:
    """Run local playoff case-study steps (no API collection)."""
    return run_grouped_mode(
        "run_playoff_case_study_pipeline",
        PLAYOFF_CASE_STUDY_STEPS,
        dry_run=dry_run,
    )


def print_available_modes() -> None:
    """Print all registered modes for ``--list-modes``."""
    print("Available pipeline modes:\n")
    for mode, description in sorted(get_available_modes().items()):
        print(f"  {mode:<35} {description}")
    print("\nGrouped modes support --dry-run to preview steps without executing.")
    print("Example: python run_pipeline.py --mode dashboard_ready --dry-run")


def dispatch_mode(mode: str, dry_run: bool = False) -> int:
    """Route a mode name to its runner."""
    grouped_handlers = {
        "build_features": lambda: run_grouped_mode(
            "build_features", BUILD_FEATURES_STEPS, dry_run=dry_run
        ),
        "train_models": lambda: run_grouped_mode(
            "train_models", TRAIN_MODELS_STEPS, dry_run=dry_run
        ),
        "score_outputs": lambda: run_grouped_mode(
            "score_outputs", SCORE_OUTPUTS_STEPS, dry_run=dry_run
        ),
        "refresh_local_data": lambda: run_grouped_mode(
            "refresh_local_data", REFRESH_LOCAL_DATA_STEPS, dry_run=dry_run
        ),
        "refresh_full_season_play_by_play": lambda: run_refresh_full_season_play_by_play(
            dry_run=dry_run
        ),
        "dashboard_ready": lambda: run_dashboard_ready(dry_run=dry_run),
        "qa": lambda: run_qa(dry_run=dry_run),
        "collect_games_multi_season": lambda: run_collect_games_multi_season(dry_run=dry_run),
        "collect_team_stats_multi_season": lambda: run_collect_team_stats_multi_season(
            dry_run=dry_run
        ),
        "refresh_multi_season_metadata": lambda: run_refresh_multi_season_metadata(
            dry_run=dry_run
        ),
        "collect_play_by_play_multi_season": lambda: run_collect_play_by_play_multi_season(
            dry_run=dry_run
        ),
        "build_features_multi_season": lambda: run_build_features_multi_season(dry_run=dry_run),
        "train_models_multiseason": lambda: run_train_models_multiseason(dry_run=dry_run),
        "prepare_multiseason_training_data": lambda: run_prepare_multiseason_training_data(
            dry_run=dry_run
        ),
        "run_playoff_case_study_pipeline": lambda: run_playoff_case_study_pipeline(
            dry_run=dry_run
        ),
    }
    if mode in grouped_handlers:
        return grouped_handlers[mode]()

    individual_handlers: Dict[str, Callable[[], int]] = {
        "setup": run_setup,
        "sample": run_sample,
        "collect_games": run_collect_games,
        "collect_play_by_play": run_collect_play_by_play,
        "collect_play_by_play_full_season": run_collect_play_by_play_full_season,
        "collect_team_stats": run_collect_team_stats,
        "build_pregame_features": run_build_pregame_features,
        "build_live_features": run_build_live_features,
        "train_pregame_model": run_train_pregame_model,
        "train_live_model": run_train_live_model,
        "predict_pregame": run_predict_pregame,
        "predict_live": run_predict_live,
        "predict_all": run_predict_all,
        "evaluate": run_evaluate,
        "check_data_freshness": run_check_data_freshness,
        "check_multiseason_training_readiness": run_check_multiseason_training_readiness,
        "check_multiseason_coverage": run_check_multiseason_coverage,
        "train_pregame_model_multiseason": run_train_pregame_model_multiseason,
        "train_live_model_multiseason": run_train_live_model_multiseason,
        "compare_models": lambda: run_compare_models(dry_run=dry_run),
        "collect_playoff_games": run_collect_playoff_games_mode,
        "collect_playoff_play_by_play": run_collect_playoff_play_by_play_mode,
        "check_playoff_coverage": run_check_playoff_coverage_mode,
        "build_playoff_live_features": run_build_playoff_live_features_mode,
        "predict_playoff_live": run_predict_playoff_live_mode,
        "build_finals_case_study": run_build_finals_case_study_mode,
        "build_finals_pregame_predictions": run_build_finals_pregame_predictions_mode,
        "build_finals_projected_series_path": run_build_finals_projected_series_path_mode,
    }

    if mode not in individual_handlers:
        print(f"Unknown mode: {mode}")
        return 1

    if dry_run:
        print(f"\n=== Mode: {mode} (DRY-RUN) ===")
        if mode in {
            "collect_games",
            "collect_play_by_play",
            "collect_play_by_play_full_season",
            "collect_team_stats",
        }:
            print(f"  Season: {get_single_season()}")
        if mode in {
            "collect_playoff_games",
            "collect_playoff_play_by_play",
            "check_playoff_coverage",
            "build_playoff_live_features",
            "build_finals_pregame_predictions",
            "build_finals_projected_series_path",
            "run_playoff_case_study_pipeline",
        }:
            print(f"  Seasons: {', '.join(get_multi_seasons())}")
        if mode == "collect_playoff_play_by_play":
            print(PLAYOFF_PBP_WARNING)
        if mode == "check_multiseason_coverage":
            print(f"  Seasons: {', '.join(get_multi_seasons())}")
            print("  Would inspect local files and write multiseason_coverage_report.csv")
        if mode == "compare_models":
            return run_compare_models(dry_run=True)
        print(f"  Step 1/1: {mode}")
        print(f"\n[dry-run] Skipping execution for mode: {mode}")
        return 0

    return individual_handlers[mode]()


def main() -> int:
    """Parse arguments and dispatch to the requested mode."""
    modes = sorted(get_available_modes().keys())
    parser = argparse.ArgumentParser(description="NBA Win Probability Engine pipeline.")
    parser.add_argument(
        "--mode",
        choices=modes,
        help="Pipeline mode to run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview grouped mode steps without executing them.",
    )
    parser.add_argument(
        "--season",
        help="Single season for individual modes (default: 2024-25).",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        help="Multiple seasons for grouped multi-season modes "
        "(default: 2022-23 2023-24 2024-25).",
    )
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        help="Train seasons for multi-season training modes (required with --test-season).",
    )
    parser.add_argument(
        "--test-season",
        help="Hold-out test season for multi-season training modes (required with --train-seasons).",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="Print available modes and exit.",
    )
    args = parser.parse_args()

    if args.list_modes:
        print_available_modes()
        return 0

    if not args.mode:
        parser.error("--mode is required unless --list-modes is used")

    try:
        configure_seasons(args.season, args.seasons)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.mode in MODES_REQUIRING_TRAIN_SPLIT:
        try:
            configure_training_split(args.train_seasons, args.test_season)
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1

    return dispatch_mode(args.mode, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
