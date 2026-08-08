"""Project-level QA and stability checks.

Inspects required files, pipeline registration, README/CONTEXT consistency,
and data freshness status. Does **not** collect data, train models, call
``nba_api``, or run expensive pipeline steps.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

QaStatus = Literal["pass", "warning", "fail", "not_applicable"]

SUMMARY_COLUMNS = [
    "section",
    "check",
    "status",
    "details",
    "recommended_action",
]

EXPECTED_PIPELINE_MODES = [
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
    "build_features",
    "train_models",
    "score_outputs",
    "dashboard_ready",
    "refresh_local_data",
    "refresh_full_season_play_by_play",
    "qa",
    "collect_games_multi_season",
    "collect_team_stats_multi_season",
    "refresh_multi_season_metadata",
    "collect_play_by_play_multi_season",
    "build_features_multi_season",
    "check_multiseason_training_readiness",
    "train_pregame_model_multiseason",
    "train_live_model_multiseason",
    "train_models_multiseason",
    "check_multiseason_coverage",
    "prepare_multiseason_training_data",
    "compare_models",
    "collect_playoff_games",
    "collect_playoff_play_by_play",
    "check_playoff_coverage",
    "build_playoff_live_features",
    "predict_playoff_live",
    "build_finals_case_study",
    "build_finals_pregame_predictions",
    "build_finals_projected_series_path",
    "run_playoff_case_study_pipeline",
]

REQUIRED_README_COMMANDS = [
    "python run_pipeline.py --list-modes",
    "python run_pipeline.py --mode dashboard_ready --dry-run",
    "python run_pipeline.py --mode dashboard_ready",
    "python run_pipeline.py --mode check_data_freshness",
    "python run_pipeline.py --mode evaluate",
    "streamlit run app/Home.py",
    "pytest",
]

REQUIRED_CONTEXT_SECTIONS = [
    "Project Overview",
    "Permanent Constraints",
    "Completed Builds",
    "Current Pipeline Modes",
    "Current Data State",
    "Immediate Next Step",
    "Future Build Instruction",
    "Maintenance Note",
]

STATUS_SEVERITY = {"fail": 0, "warning": 1, "pass": 2, "not_applicable": 3}


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(config.ROOT_DIR))
    except ValueError:
        return str(path)


def _row(
    section: str,
    check: str,
    status: QaStatus,
    details: str = "",
    recommended_action: str = "",
) -> dict:
    return {
        "section": section,
        "check": check,
        "status": status,
        "details": details,
        "recommended_action": recommended_action,
    }


def _file_check(path: Path, label: str) -> dict:
    """Return one QA row for a required file on disk."""
    rel = _relative(path)
    if not path.exists():
        return _row(
            "files",
            label,
            "fail",
            f"Missing: {rel}",
            f"Create or restore {rel}",
        )
    if path.suffix.lower() == ".csv":
        try:
            df = pd.read_csv(path)
            if df.empty:
                return _row(
                    "files",
                    label,
                    "warning",
                    f"Empty CSV (no data rows): {rel}",
                    f"Regenerate {rel}",
                )
        except pd.errors.EmptyDataError:
            return _row(
                "files",
                label,
                "warning",
                f"Empty CSV: {rel}",
                f"Regenerate {rel}",
            )
    return _row("files", label, "pass", f"Found: {rel}")


def check_required_paths(paths: Dict[str, Path], section: str = "files") -> pd.DataFrame:
    """Check a mapping of label -> path and return QA rows."""
    rows = []
    for label, path in paths.items():
        row = _file_check(path, label)
        row["section"] = section
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def check_required_data_files() -> pd.DataFrame:
    """Verify core raw and processed CSV files exist."""
    paths = {
        "raw_games": config.RAW_GAMES_PATH,
        "raw_play_by_play": config.RAW_PLAY_BY_PLAY_PATH,
        "raw_team_stats": config.RAW_TEAM_STATS_PATH,
        "pregame_features": config.PREGAME_FEATURES_PATH,
        "live_features": config.LIVE_FEATURES_PATH,
        "game_results": config.GAME_RESULTS_PATH,
        "pregame_predictions": config.PREGAME_PREDICTIONS_PATH,
        "live_predictions": config.LIVE_PREDICTIONS_PATH,
    }
    return check_required_paths(paths, section="data_files")


def check_required_model_artifacts() -> pd.DataFrame:
    """Verify saved model pickle and feature-column files exist."""
    paths = {
        "pregame_model": config.PREGAME_MODEL_PATH,
        "pregame_feature_columns": config.PREGAME_FEATURE_COLUMNS_PATH,
        "live_model": config.LIVE_MODEL_PATH,
        "live_feature_columns": config.LIVE_FEATURE_COLUMNS_PATH,
    }
    return check_required_paths(paths, section="model_artifacts")


def check_optional_multiseason_artifacts() -> pd.DataFrame:
    """Optional QA for multi-season models — not_applicable until generated."""
    rows: List[dict] = []
    model_path = config.PREGAME_MODEL_MULTISEASON_PATH
    if not model_path.exists():
        rows.append(
            _row(
                "multiseason",
                "multiseason_models_present",
                "not_applicable",
                "Multi-season models not generated yet",
            )
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    rows.append(
        _row(
            "multiseason",
            "multiseason_models_present",
            "pass",
            "Multi-season model artifacts found on disk",
        )
    )
    paired = {
        "pregame_model_multiseason": (
            config.PREGAME_MODEL_MULTISEASON_PATH,
            config.PREGAME_MODEL_METRICS_MULTISEASON_PATH,
        ),
        "live_model_multiseason": (
            config.LIVE_MODEL_MULTISEASON_PATH,
            config.LIVE_MODEL_METRICS_MULTISEASON_PATH,
        ),
    }
    for label, (model, metrics) in paired.items():
        if metrics.exists():
            rows.append(
                _row(
                    "multiseason",
                    f"{label}_metrics_report",
                    "pass",
                    f"Paired metrics report exists: {metrics.name}",
                )
            )
        else:
            rows.append(
                _row(
                    "multiseason",
                    f"{label}_metrics_report",
                    "warning",
                    f"Model exists but metrics missing: {metrics.name}",
                    "Re-run multi-season training for this model",
                )
            )

    readiness = config.MULTISEASON_TRAINING_READINESS_PATH
    if readiness.exists():
        rows.append(
            _row(
                "multiseason",
                "readiness_report_present",
                "pass",
                f"Readiness report: {readiness.name}",
            )
        )
    else:
        rows.append(
            _row(
                "multiseason",
                "readiness_report_present",
                "not_applicable",
                "No readiness report on disk",
            )
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def check_multiseason_coverage_support() -> pd.DataFrame:
    """Verify coverage module, runbook, and optional coverage report."""
    rows: List[dict] = []
    coverage_module = config.ROOT_DIR / "src" / "multiseason_coverage.py"
    if coverage_module.exists():
        rows.append(
            _row("multiseason", "coverage_module_present", "pass", str(coverage_module.name))
        )
    else:
        rows.append(
            _row(
                "multiseason",
                "coverage_module_present",
                "fail",
                "src/multiseason_coverage.py missing",
                "Add multiseason coverage module",
            )
        )

    runbook = config.ROOT_DIR / "MULTISEASON_RUNBOOK.md"
    if runbook.exists():
        rows.append(_row("multiseason", "runbook_present", "pass", "MULTISEASON_RUNBOOK.md"))
    else:
        rows.append(
            _row(
                "multiseason",
                "runbook_present",
                "fail",
                "MULTISEASON_RUNBOOK.md missing",
                "Add multi-season runbook",
            )
        )

    report_path = config.MULTISEASON_COVERAGE_REPORT_PATH
    if not report_path.exists():
        rows.append(
            _row(
                "multiseason",
                "coverage_report_generated",
                "not_applicable",
                "Coverage report not generated yet",
            )
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    try:
        report = pd.read_csv(report_path)
    except pd.errors.EmptyDataError:
        rows.append(
            _row(
                "multiseason",
                "coverage_report_generated",
                "warning",
                "Coverage report is empty",
            )
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    rows.append(
        _row(
            "multiseason",
            "coverage_report_generated",
            "pass",
            f"{len(report)} rows in coverage report",
        )
    )
    fails = report[report["status"] == "fail"]
    if fails.empty:
        rows.append(
            _row(
                "multiseason",
                "coverage_report_status",
                "pass",
                "No fail rows in coverage report",
            )
        )
    else:
        seasons = sorted(fails["season"].unique().tolist())
        rows.append(
            _row(
                "multiseason",
                "coverage_report_status",
                "warning",
                f"Coverage report has {len(fails)} fail row(s) for: {', '.join(seasons)}",
                "Complete PBP collection and feature rebuild (see MULTISEASON_RUNBOOK.md)",
            )
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def check_optional_comparison_reports() -> pd.DataFrame:
    """Optional QA for model version comparison reports."""
    rows: List[dict] = []
    compare_module = config.ROOT_DIR / "src" / "compare_model_versions.py"
    if compare_module.exists():
        rows.append(
            _row(
                "comparison",
                "compare_module_present",
                "pass",
                "src/compare_model_versions.py",
            )
        )
    else:
        rows.append(
            _row(
                "comparison",
                "compare_module_present",
                "fail",
                "src/compare_model_versions.py missing",
                "Add compare_model_versions module",
            )
        )

    summary_path = config.MODEL_COMPARISON_SUMMARY_PATH
    if not summary_path.exists():
        rows.append(
            _row(
                "comparison",
                "comparison_reports_generated",
                "not_applicable",
                "Comparison reports not generated yet — run compare_models",
            )
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    try:
        summary = pd.read_csv(summary_path)
    except pd.errors.EmptyDataError:
        rows.append(
            _row(
                "comparison",
                "comparison_reports_generated",
                "warning",
                "model_comparison_summary.csv is empty",
                "Re-run: python run_pipeline.py --mode compare_models",
            )
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    rows.append(
        _row(
            "comparison",
            "comparison_reports_generated",
            "pass",
            f"{len(summary)} comparison rows in model_comparison_summary.csv",
        )
    )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def check_optional_playoff_case_study() -> pd.DataFrame:
    """Optional QA for playoff / NBA Finals case-study support."""
    rows: List[dict] = []
    module_path = config.ROOT_DIR / "src" / "playoff_case_study.py"
    if module_path.exists():
        rows.append(
            _row(
                "playoffs",
                "playoff_module_present",
                "pass",
                "src/playoff_case_study.py",
            )
        )
    else:
        rows.append(
            _row(
                "playoffs",
                "playoff_module_present",
                "fail",
                "src/playoff_case_study.py missing",
                "Add playoff case-study module",
            )
        )

    optional_paths = {
        "playoff_games": config.PLAYOFF_GAMES_PATH,
        "playoff_play_by_play": config.PLAYOFF_PLAY_BY_PLAY_PATH,
        "playoff_live_features": config.PLAYOFF_LIVE_FEATURES_PATH,
        "playoff_live_predictions": config.PLAYOFF_LIVE_PREDICTIONS_PATH,
        "nba_finals_case_study_summary": config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH,
        "finals_upcoming_predictions": config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH,
        "finals_projected_series_path": config.FINALS_PROJECTED_SERIES_PATH,
    }
    present = [label for label, path in optional_paths.items() if path.exists()]
    if not present:
        rows.append(
            _row(
                "playoffs",
                "playoff_outputs_generated",
                "not_applicable",
                "Playoff case-study files not generated yet — run collect_playoff_* then "
                "run_playoff_case_study_pipeline",
            )
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    rows.append(
        _row(
            "playoffs",
            "playoff_outputs_generated",
            "pass",
            f"{len(present)}/{len(optional_paths)} playoff assets present: "
            f"{', '.join(present)}",
        )
    )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def check_finals_schedule_overrides() -> pd.DataFrame:
    """Validate optional Finals schedule override file (no result columns)."""
    path = config.FINALS_SCHEDULE_OVERRIDES_PATH
    if not path.exists():
        return pd.DataFrame(
            [
                _row(
                    "manual",
                    "finals_schedule_overrides",
                    "not_applicable",
                    "Optional schedule override file not present",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    try:
        df = pd.read_csv(path, dtype={"game_id": str})
    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            [
                _row(
                    "manual",
                    "finals_schedule_overrides",
                    "warning",
                    "Override file exists but is empty",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    from src.playoff_case_study import (  # noqa: WPS433 — local import avoids cycle at load
        FORBIDDEN_OVERRIDE_COLUMNS,
        validate_finals_schedule_overrides,
    )

    forbidden = [c for c in df.columns if c in FORBIDDEN_OVERRIDE_COLUMNS]
    if forbidden:
        return pd.DataFrame(
            [
                _row(
                    "manual",
                    "finals_schedule_overrides",
                    "warning",
                    f"Forbidden result/leakage columns present: {forbidden}",
                    "Remove score/winner columns from finals_schedule_overrides.csv",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    try:
        validate_finals_schedule_overrides(df)
    except ValueError as exc:
        return pd.DataFrame(
            [
                _row(
                    "manual",
                    "finals_schedule_overrides",
                    "fail",
                    str(exc),
                    "Fix data/manual/finals_schedule_overrides.csv",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    return pd.DataFrame(
        [
            _row(
                "manual",
                "finals_schedule_overrides",
                "pass",
                f"{len(df)} override row(s) at {_relative(path)}",
            )
        ],
        columns=SUMMARY_COLUMNS,
    )


def check_required_reports() -> pd.DataFrame:
    """Verify key evaluation and freshness reports exist."""
    paths = {
        "pregame_model_metrics": config.PREGAME_MODEL_METRICS_PATH,
        "live_model_metrics": config.LIVE_MODEL_METRICS_PATH,
        "evaluation_summary": config.EVALUATION_SUMMARY_PATH,
        "data_freshness_summary": config.DATA_FRESHNESS_REPORT_PATH,
        "pbp_coverage_report": config.PBP_COVERAGE_REPORT_PATH,
    }
    return check_required_paths(paths, section="reports")


def check_dashboard_dependencies() -> pd.DataFrame:
    """Verify Streamlit app entry points exist."""
    paths = {
        "app_home": config.APP_DIR / "Home.py",
        "dashboard_utils": config.APP_DIR / "dashboard_utils.py",
        "page_pregame_predictor": config.APP_DIR / "pages" / "1_Pregame_Predictor.py",
        "page_live_replay": config.APP_DIR / "pages" / "2_Live_Replay.py",
        "page_postgame_override": config.APP_DIR / "pages" / "3_Postgame_Override.py",
        "page_model_performance": config.APP_DIR / "pages" / "4_Model_Performance.py",
        "page_nba_finals_case_study": config.APP_DIR / "pages" / "5_NBA_Finals_Case_Study.py",
        "page_2025_26_nba_finals": config.APP_DIR / "pages" / "6_2025_26_NBA_Finals.py",
    }
    return check_required_paths(paths, section="dashboard")


def check_season_columns() -> pd.DataFrame:
    """Verify key datasets include a ``season`` column when present on disk."""
    paths = {
        "raw_games": config.RAW_GAMES_PATH,
        "raw_play_by_play": config.RAW_PLAY_BY_PLAY_PATH,
        "pregame_features": config.PREGAME_FEATURES_PATH,
        "live_features": config.LIVE_FEATURES_PATH,
        "game_results": config.GAME_RESULTS_PATH,
    }
    rows: List[dict] = []
    for label, path in paths.items():
        rel = _relative(path)
        if not path.exists():
            rows.append(
                _row(
                    "season_columns",
                    label,
                    "not_applicable",
                    f"File missing: {rel}",
                )
            )
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns.tolist()
        except pd.errors.EmptyDataError:
            rows.append(
                _row(
                    "season_columns",
                    label,
                    "warning",
                    f"Empty file: {rel}",
                    "Regenerate with season column",
                )
            )
            continue
        if "season" in columns:
            rows.append(
                _row("season_columns", label, "pass", f"season column present in {rel}")
            )
        else:
            rows.append(
                _row(
                    "season_columns",
                    label,
                    "fail",
                    f"Missing season column in {rel}",
                    "Rebuild dataset with season-aware pipeline",
                )
            )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _load_pipeline_module():
    """Load run_pipeline.py for mode registration checks."""
    pipeline_path = config.ROOT_DIR / "run_pipeline.py"
    spec = importlib.util.spec_from_file_location("run_pipeline", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_pipeline_modes() -> pd.DataFrame:
    """Verify pipeline modes and CLI flags are registered as expected."""
    rows: List[dict] = []
    pipeline = _load_pipeline_module()
    available = pipeline.get_available_modes()

    missing_modes = [m for m in EXPECTED_PIPELINE_MODES if m not in available]
    if missing_modes:
        rows.append(
            _row(
                "pipeline",
                "expected_modes_registered",
                "fail",
                f"Missing modes: {missing_modes}",
                "Register missing modes in run_pipeline.py",
            )
        )
    else:
        rows.append(
            _row(
                "pipeline",
                "expected_modes_registered",
                "pass",
                f"All {len(EXPECTED_PIPELINE_MODES)} expected modes registered",
            )
        )

    if "all" in available:
        rows.append(
            _row(
                "pipeline",
                "no_generic_all_mode",
                "fail",
                "Generic 'all' mode is registered",
                "Remove the generic all mode",
            )
        )
    else:
        rows.append(
            _row(
                "pipeline",
                "no_generic_all_mode",
                "pass",
                "No generic all mode (by design)",
            )
        )

    parser_source = config.ROOT_DIR / "run_pipeline.py"
    source = parser_source.read_text(encoding="utf-8")
    if "--dry-run" in source:
        rows.append(_row("pipeline", "dry_run_flag", "pass", "--dry-run supported"))
    else:
        rows.append(
            _row(
                "pipeline",
                "dry_run_flag",
                "fail",
                "--dry-run not found in run_pipeline.py",
                "Add --dry-run support",
            )
        )

    if "--list-modes" in source:
        rows.append(_row("pipeline", "list_modes_flag", "pass", "--list-modes supported"))
    else:
        rows.append(
            _row(
                "pipeline",
                "list_modes_flag",
                "fail",
                "--list-modes not found in run_pipeline.py",
                "Add --list-modes support",
            )
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def check_readme_commands(readme_path: Optional[Path] = None) -> pd.DataFrame:
    """Verify README.md documents key commands."""
    path = readme_path or (config.ROOT_DIR / "README.md")
    rows: List[dict] = []

    if not path.exists():
        return pd.DataFrame(
            [_row("readme", "readme_exists", "fail", "README.md missing", "Restore README.md")],
            columns=SUMMARY_COLUMNS,
        )

    text = path.read_text(encoding="utf-8")
    missing = [cmd for cmd in REQUIRED_README_COMMANDS if cmd not in text]
    if missing:
        rows.append(
            _row(
                "readme",
                "required_commands_documented",
                "fail",
                f"Missing commands: {missing}",
                "Add missing commands to README.md",
            )
        )
    else:
        rows.append(
            _row(
                "readme",
                "required_commands_documented",
                "pass",
                f"All {len(REQUIRED_README_COMMANDS)} required commands documented",
            )
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def check_github_finals_workflow() -> pd.DataFrame:
    """Verify optional GitHub Actions Finals refresh workflow exists and is safe."""
    workflow_path = config.ROOT_DIR / ".github" / "workflows" / "finals_refresh.yml"
    if not workflow_path.exists():
        return pd.DataFrame(
            [
                _row(
                    "deploy",
                    "finals_refresh_workflow",
                    "not_applicable",
                    "Optional workflow not present",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    text = workflow_path.read_text(encoding="utf-8").lower()
    issues: List[str] = []
    if "train_pregame_model" in text or "train_live_model" in text:
        issues.append("contains training commands")
    if "workflow_dispatch" not in text:
        issues.append("missing workflow_dispatch trigger")

    if issues:
        return pd.DataFrame(
            [
                _row(
                    "deploy",
                    "finals_refresh_workflow",
                    "warning",
                    "; ".join(issues),
                    "Review .github/workflows/finals_refresh.yml",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    return pd.DataFrame(
        [
            _row(
                "deploy",
                "finals_refresh_workflow",
                "pass",
                "finals_refresh.yml present; no training commands detected",
            )
        ],
        columns=SUMMARY_COLUMNS,
    )


def check_finals_deploy_export() -> pd.DataFrame:
    """Verify the deploy-safe Finals live-predictions export is present, non-empty,
    and covers every Finals game marked replay-available."""
    deploy_path = config.FINALS_LIVE_PREDICTIONS_DEPLOY_PATH
    if not deploy_path.exists():
        return pd.DataFrame(
            [
                _row(
                    "deploy",
                    "finals_live_predictions_export",
                    "not_applicable",
                    "Deploy export not generated yet — run "
                    "export_finals_live_predictions_for_deploy",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    try:
        export_df = pd.read_csv(deploy_path, dtype={"game_id": str})
    except pd.errors.EmptyDataError:
        export_df = pd.DataFrame()

    if export_df.empty:
        return pd.DataFrame(
            [
                _row(
                    "deploy",
                    "finals_live_predictions_export",
                    "fail",
                    f"{_relative(deploy_path)} exists but has 0 rows",
                    "Re-run export_finals_live_predictions_for_deploy",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    # game_id is zero-padded to 10 characters in the deploy export; some sibling
    # report CSVs are known to drop the leading zeros, so normalize before
    # comparing rather than treating that formatting drift as missing games.
    exported_games = {gid.zfill(10) for gid in export_df["game_id"].astype(str).unique()}

    expected_games: set[str] = set()
    upcoming_path = config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH
    if upcoming_path.exists():
        upcoming = pd.read_csv(upcoming_path, dtype={"game_id": str})
        if "replay_available" in upcoming.columns:
            expected_games = {
                gid.zfill(10)
                for gid in upcoming.loc[
                    upcoming["replay_available"] == True, "game_id"  # noqa: E712
                ].astype(str)
            }

    missing_games = sorted(expected_games - exported_games)
    if missing_games:
        return pd.DataFrame(
            [
                _row(
                    "deploy",
                    "finals_live_predictions_export",
                    "warning",
                    f"{len(exported_games)} game(s) in export; missing replay-available "
                    f"game(s): {missing_games}",
                    "Re-run export_finals_live_predictions_for_deploy after collecting "
                    "the missing games' play-by-play",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    return pd.DataFrame(
        [
            _row(
                "deploy",
                "finals_live_predictions_export",
                "pass",
                f"{len(export_df)} rows across {len(exported_games)} game(s)",
            )
        ],
        columns=SUMMARY_COLUMNS,
    )


def check_context_file(context_path: Optional[Path] = None) -> pd.DataFrame:
    """Verify CONTEXT.md exists and contains required sections."""
    path = context_path or (config.ROOT_DIR / "CONTEXT.md")
    rows: List[dict] = []

    if not path.exists():
        return pd.DataFrame(
            [
                _row(
                    "context",
                    "context_exists",
                    "fail",
                    "CONTEXT.md missing",
                    "Create CONTEXT.md handoff file",
                )
            ],
            columns=SUMMARY_COLUMNS,
        )

    text = path.read_text(encoding="utf-8")
    rows.append(_row("context", "context_exists", "pass", f"Found: {_relative(path)}"))

    missing_sections = [s for s in REQUIRED_CONTEXT_SECTIONS if f"## {s}" not in text]
    if missing_sections:
        rows.append(
            _row(
                "context",
                "required_sections_present",
                "fail",
                f"Missing sections: {missing_sections}",
                "Add missing sections to CONTEXT.md",
            )
        )
    else:
        rows.append(
            _row(
                "context",
                "required_sections_present",
                "pass",
                f"All {len(REQUIRED_CONTEXT_SECTIONS)} required sections present",
            )
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def summarize_data_freshness(
    freshness_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Summarize data_freshness_summary.csv into QA pass/warning/fail rows."""
    path = freshness_path or config.DATA_FRESHNESS_REPORT_PATH
    rows: List[dict] = []

    if not path.exists():
        rows.append(
            _row(
                "freshness",
                "freshness_report_exists",
                "warning",
                "data_freshness_summary.csv not found",
                "Run: python run_pipeline.py --mode check_data_freshness",
            )
        )
        return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)

    df = pd.read_csv(path)
    counts = df["status"].value_counts().to_dict()
    details = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    rows.append(
        _row(
            "freshness",
            "freshness_status_counts",
            "pass",
            details,
        )
    )

    required_assets = df.loc[df["status"] != "not_required"]
    missing = required_assets.loc[required_assets["status"] == "missing"]
    stale = required_assets.loc[required_assets["status"] == "stale"]
    warnings = required_assets.loc[required_assets["status"] == "warning"]

    if not missing.empty:
        missing_list = ", ".join(missing["asset"].tolist())
        rows.append(
            _row(
                "freshness",
                "required_assets_present",
                "fail",
                f"Missing required assets: {missing_list}",
                "Restore missing files or re-run the pipeline step that creates them",
            )
        )
    elif not stale.empty and warnings.empty:
        stale_list = ", ".join(stale["asset"].tolist())
        rows.append(
            _row(
                "freshness",
                "required_assets_present",
                "warning",
                f"Stale assets (likely mtime ordering): {stale_list}",
                "Re-run evaluate/train if outputs are genuinely out of date; otherwise safe to ignore",
            )
        )
    elif not stale.empty or not warnings.empty:
        issues = []
        if not stale.empty:
            issues.append(f"stale: {', '.join(stale['asset'].tolist())}")
        if not warnings.empty:
            issues.append(f"warning: {', '.join(warnings['asset'].tolist())}")
        rows.append(
            _row(
                "freshness",
                "required_assets_present",
                "warning",
                "; ".join(issues),
                "Review freshness report and regenerate affected outputs if needed",
            )
        )
    else:
        rows.append(
            _row(
                "freshness",
                "required_assets_present",
                "pass",
                "No missing required freshness assets",
            )
        )

    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def build_project_qa_summary(
    readme_path: Optional[Path] = None,
    context_path: Optional[Path] = None,
    freshness_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Run all QA checks and return a combined summary DataFrame."""
    parts = [
        check_required_data_files(),
        check_required_model_artifacts(),
        check_required_reports(),
        check_dashboard_dependencies(),
        check_season_columns(),
        check_optional_multiseason_artifacts(),
        check_multiseason_coverage_support(),
        check_optional_comparison_reports(),
        check_optional_playoff_case_study(),
        check_finals_schedule_overrides(),
        check_github_finals_workflow(),
        check_finals_deploy_export(),
        check_pipeline_modes(),
        check_readme_commands(readme_path),
        check_context_file(context_path),
        summarize_data_freshness(freshness_path),
    ]
    return pd.concat(parts, ignore_index=True)


def save_project_qa_summary(
    summary_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the QA summary CSV to disk."""
    out_path = Path(output_path) if output_path is not None else config.PROJECT_QA_REPORT_PATH
    save_csv(summary_df, out_path)
    return out_path


def count_qa_statuses(summary_df: pd.DataFrame) -> Dict[str, int]:
    """Count checks by QA status."""
    counts = {s: 0 for s in STATUS_SEVERITY}
    if summary_df.empty:
        return counts
    for status, count in summary_df["status"].value_counts().items():
        counts[str(status)] = int(count)
    return counts


def compute_overall_qa_status(summary_df: pd.DataFrame) -> QaStatus:
    """Return overall pass / warning / fail from individual check rows."""
    if summary_df.empty:
        return "fail"
    statuses = set(summary_df["status"].tolist())
    if "fail" in statuses:
        return "fail"
    if "warning" in statuses:
        return "warning"
    return "pass"


def get_top_qa_issues(summary_df: pd.DataFrame, limit: int = 5) -> List[dict]:
    """Return the most important warning/fail checks for console output."""
    if summary_df.empty:
        return []
    issues = summary_df.loc[summary_df["status"].isin(["fail", "warning"])].copy()
    if issues.empty:
        return []
    issues["_priority"] = issues["status"].map(lambda s: STATUS_SEVERITY.get(s, 99))
    issues = issues.sort_values(["_priority", "section", "check"]).head(limit)
    return issues.drop(columns=["_priority"]).to_dict("records")


def run_project_qa(verbose: bool = True) -> int:
    """Run QA checks, save report, and print summary. Returns 0 pass/warn, 1 fail."""
    ensure_directories()
    summary_df = build_project_qa_summary()
    out_path = save_project_qa_summary(summary_df)
    counts = count_qa_statuses(summary_df)
    overall = compute_overall_qa_status(summary_df)

    if verbose:
        total = len(summary_df)
        print("Project QA complete.")
        print(f"  Checks run:   {total}")
        print(f"  pass:         {counts.get('pass', 0)}")
        print(f"  warning:      {counts.get('warning', 0)}")
        print(f"  fail:         {counts.get('fail', 0)}")
        print(f"  not_applicable: {counts.get('not_applicable', 0)}")
        print(f"  Output:       {out_path}")
        print(f"  Overall:      {overall}")

        issues = get_top_qa_issues(summary_df)
        if issues:
            print("\nTop issues:")
            for issue in issues:
                print(
                    f"  - [{issue['status']}] {issue['section']}/{issue['check']}: "
                    f"{issue['details']}"
                )
                if issue.get("recommended_action"):
                    print(f"      -> {issue['recommended_action']}")
        else:
            print("\nNo warnings or failures detected.")

    return 1 if overall == "fail" else 0


def main() -> int:
    """CLI entry point for ``python src/project_qa.py``."""
    return run_project_qa(verbose=True)


if __name__ == "__main__":
    sys.exit(main())
