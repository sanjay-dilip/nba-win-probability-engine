"""Data freshness and system status helpers.

Inspects project files to report whether data, models, predictions, reports,
and dashboard dependencies are present, missing, stale, or empty.

This module does **not** collect data, train models, or generate predictions.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.season_config import summarize_seasons_in_csv  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

Status = Literal["ok", "missing", "stale", "warning", "not_required", "unknown"]

SUMMARY_COLUMNS = [
    "section",
    "asset",
    "path",
    "status",
    "row_count",
    "unique_games",
    "last_modified",
    "notes",
]

STATUS_PRIORITY = {
    "missing": 0,
    "stale": 1,
    "warning": 2,
    "unknown": 3,
    "ok": 4,
    "not_required": 5,
}


def get_file_modified_time(path: Union[str, Path]) -> str:
    """Return ISO-8601 UTC modified time for ``path``, or empty if unavailable."""
    csv_path = Path(path)
    if not csv_path.exists():
        return ""
    mtime = csv_path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()


def count_csv_rows(
    path: Union[str, Path],
    dtype: Optional[dict] = None,
) -> Optional[int]:
    """Count rows in a CSV without loading the full file into memory."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, dtype=dtype or {})
    except pd.errors.EmptyDataError:
        return 0
    return len(df)


def get_unique_count(
    path: Union[str, Path],
    column: str,
    dtype: Optional[dict] = None,
) -> Optional[int]:
    """Return unique values in ``column``, preserving string dtypes like ``game_id``."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        series = pd.read_csv(csv_path, dtype=dtype or {}, usecols=[column])[column]
    except (pd.errors.EmptyDataError, ValueError):
        return None
    if series.empty:
        return 0
    return int(series.nunique())


def file_status(
    path: Union[str, Path],
    required: bool = True,
) -> Status:
    """Return a simple status for a file on disk."""
    csv_path = Path(path)
    if not csv_path.exists():
        return "missing" if required else "not_required"

    if csv_path.suffix.lower() == ".csv":
        rows = count_csv_rows(csv_path)
        if rows == 0:
            return "warning"
        return "ok"

    if csv_path.stat().st_size == 0:
        return "warning"
    return "ok"


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(config.ROOT_DIR))
    except ValueError:
        return str(path)


def _is_stale(output_path: Path, dependency_paths: List[Path]) -> bool:
    """True when any dependency is newer than the output file."""
    if not output_path.exists():
        return False
    output_mtime = output_path.stat().st_mtime
    for dep in dependency_paths:
        if dep.exists() and dep.stat().st_mtime > output_mtime:
            return True
    return False


def read_latest_refresh_log(log_path: Union[str, Path]) -> Optional[dict]:
    """Return the most recent row from the data refresh log, or ``None``."""
    path = Path(log_path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    latest = df.iloc[-1]
    return {
        "timestamp": str(latest.get("timestamp", "")),
        "game_id": str(latest.get("game_id", "")),
        "endpoint": str(latest.get("endpoint", "")),
        "status": str(latest.get("status", "")),
    }


def summarize_coverage_report(path: Union[str, Path]) -> str:
    """Build a short note from the play-by-play coverage report."""
    df = _load_csv_if_present(path)
    if df is None or df.empty:
        return ""
    row = df.iloc[-1]
    parts = []
    if "coverage_pct" in row and pd.notna(row["coverage_pct"]):
        parts.append(f"coverage {row['coverage_pct']}%")
    if "collected_games" in row and pd.notna(row["collected_games"]):
        parts.append(f"{int(row['collected_games'])} games collected")
    if "total_event_rows" in row and pd.notna(row["total_event_rows"]):
        parts.append(f"{int(row['total_event_rows'])} events")
    return "; ".join(parts)


def summarize_manual_overrides(path: Union[str, Path]) -> str:
    """Build a short note for the optional manual override file."""
    df = _load_csv_if_present(path)
    if df is None:
        return "optional — not recorded"
    if df.empty:
        return "optional — empty"
    return f"{len(df)} manual override(s)"


def _load_csv_if_present(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _inspect_csv_asset(
    section: str,
    asset: str,
    path: Path,
    required: bool,
    game_id_column: Optional[str] = "game_id",
    notes: str = "",
    dtype: Optional[dict] = None,
) -> dict:
    """Build one freshness summary row for a CSV asset."""
    status = file_status(path, required=required)
    row_count: Union[int, str] = ""
    unique_games: Union[int, str] = ""

    if path.exists() and path.suffix.lower() == ".csv":
        counted = count_csv_rows(path, dtype=dtype)
        if counted is not None:
            row_count = counted
        if game_id_column and counted and counted > 0:
            unique = get_unique_count(path, game_id_column, dtype=dtype)
            if unique is not None:
                unique_games = unique

    return {
        "section": section,
        "asset": asset,
        "path": _relative_path(path),
        "status": status,
        "row_count": row_count,
        "unique_games": unique_games,
        "last_modified": get_file_modified_time(path),
        "notes": notes,
    }


def _inspect_binary_asset(
    section: str,
    asset: str,
    path: Path,
    required: bool,
    notes: str = "",
) -> dict:
    """Build one freshness summary row for a non-CSV artifact."""
    return {
        "section": section,
        "asset": asset,
        "path": _relative_path(path),
        "status": file_status(path, required=required),
        "row_count": "",
        "unique_games": "",
        "last_modified": get_file_modified_time(path),
        "notes": notes,
    }


def summarize_model_artifacts() -> pd.DataFrame:
    """Inspect saved model pickle and feature-column files."""
    rows = [
        _inspect_binary_asset("models", "pregame_model", config.PREGAME_MODEL_PATH, required=True),
        _inspect_binary_asset("models", "live_model", config.LIVE_MODEL_PATH, required=True),
        _inspect_binary_asset(
            "models",
            "pregame_feature_columns",
            config.PREGAME_FEATURE_COLUMNS_PATH,
            required=True,
        ),
        _inspect_binary_asset(
            "models",
            "live_feature_columns",
            config.LIVE_FEATURE_COLUMNS_PATH,
            required=True,
        ),
    ]
    return pd.DataFrame(rows)


def summarize_prediction_files() -> pd.DataFrame:
    """Inspect prediction CSV outputs."""
    dtype = {"game_id": str}
    rows = [
        _inspect_csv_asset(
            "predictions",
            "pregame_predictions",
            config.PREGAME_PREDICTIONS_PATH,
            required=True,
            dtype=dtype,
        ),
        _inspect_csv_asset(
            "predictions",
            "live_predictions",
            config.LIVE_PREDICTIONS_PATH,
            required=True,
            dtype=dtype,
        ),
    ]
    return pd.DataFrame(rows)


def summarize_multiseason_model_artifacts() -> pd.DataFrame:
    """Inspect optional multi-season model artifacts (not required for dashboard)."""
    rows = [
        _inspect_binary_asset(
            "models",
            "pregame_model_multiseason",
            config.PREGAME_MODEL_MULTISEASON_PATH,
            required=False,
        ),
        _inspect_binary_asset(
            "models",
            "live_model_multiseason",
            config.LIVE_MODEL_MULTISEASON_PATH,
            required=False,
        ),
        _inspect_binary_asset(
            "models",
            "pregame_feature_columns_multiseason",
            config.PREGAME_FEATURE_COLUMNS_MULTISEASON_PATH,
            required=False,
        ),
        _inspect_binary_asset(
            "models",
            "live_feature_columns_multiseason",
            config.LIVE_FEATURE_COLUMNS_MULTISEASON_PATH,
            required=False,
        ),
    ]
    return pd.DataFrame(rows)


def summarize_multiseason_reports() -> pd.DataFrame:
    """Inspect optional multi-season training reports."""
    rows = [
        _inspect_csv_asset(
            "reports",
            "multiseason_training_readiness",
            config.MULTISEASON_TRAINING_READINESS_PATH,
            required=False,
            game_id_column=None,
        ),
        _inspect_csv_asset(
            "reports",
            "multiseason_coverage_report",
            config.MULTISEASON_COVERAGE_REPORT_PATH,
            required=False,
            game_id_column=None,
        ),
        _inspect_csv_asset(
            "reports",
            "pregame_model_metrics_multiseason",
            config.PREGAME_MODEL_METRICS_MULTISEASON_PATH,
            required=False,
            game_id_column=None,
        ),
        _inspect_csv_asset(
            "reports",
            "live_model_metrics_multiseason",
            config.LIVE_MODEL_METRICS_MULTISEASON_PATH,
            required=False,
            game_id_column=None,
        ),
    ]
    return pd.DataFrame(rows)


def summarize_comparison_reports() -> pd.DataFrame:
    """Inspect optional model version comparison reports."""
    comparison_assets = [
        ("model_comparison_summary", config.MODEL_COMPARISON_SUMMARY_PATH),
        ("model_comparison_detail", config.MODEL_COMPARISON_DETAIL_PATH),
        ("live_phase_comparison_summary", config.PHASE_COMPARISON_SUMMARY_PATH),
        ("calibration_comparison_summary", config.CALIBRATION_COMPARISON_SUMMARY_PATH),
    ]
    rows = [
        _inspect_csv_asset(
            "reports",
            asset,
            path,
            required=False,
            game_id_column=None,
            notes="optional — run compare_models to generate",
        )
        for asset, path in comparison_assets
    ]
    return pd.DataFrame(rows)


def summarize_report_files() -> pd.DataFrame:
    """Inspect key evaluation and coverage reports."""
    rows = [
        _inspect_csv_asset(
            "reports",
            "pregame_model_metrics",
            config.PREGAME_MODEL_METRICS_PATH,
            required=True,
            game_id_column=None,
        ),
        _inspect_csv_asset(
            "reports",
            "live_model_metrics",
            config.LIVE_MODEL_METRICS_PATH,
            required=True,
            game_id_column=None,
        ),
        _inspect_csv_asset(
            "reports",
            "evaluation_summary",
            config.EVALUATION_SUMMARY_PATH,
            required=True,
            game_id_column=None,
        ),
        _inspect_csv_asset(
            "reports",
            "pbp_coverage_report",
            config.PBP_COVERAGE_REPORT_PATH,
            required=True,
            game_id_column=None,
            notes=summarize_coverage_report(config.PBP_COVERAGE_REPORT_PATH),
        ),
    ]
    return pd.DataFrame(rows)


def _apply_stale_rules(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Mark outputs stale when a dependency file is newer."""
    out = summary_df.copy()
    asset_paths = {
        row["asset"]: config.ROOT_DIR / row["path"]
        for _, row in out.iterrows()
    }

    stale_rules = {
        "pregame_predictions": ["pregame_model"],
        "live_predictions": ["live_model"],
        "pregame_model_metrics": ["pregame_model"],
        "live_model_metrics": ["live_model"],
        "evaluation_summary": ["pregame_predictions", "live_predictions"],
    }

    for asset, deps in stale_rules.items():
        if asset not in asset_paths:
            continue
        target_idx = out.index[out["asset"] == asset]
        if target_idx.empty:
            continue
        idx = target_idx[0]
        if out.at[idx, "status"] not in {"ok", "warning"}:
            continue
        dep_paths = [asset_paths[d] for d in deps if d in asset_paths]
        if _is_stale(asset_paths[asset], dep_paths):
            out.at[idx, "status"] = "stale"
            out.at[idx, "notes"] = (
                str(out.at[idx, "notes"]) + "; dependency newer than file"
            ).strip("; ")

    return out


def summarize_playoff_case_study_files() -> pd.DataFrame:
    """Inspect optional playoff / NBA Finals case-study assets."""
    dtype = {"game_id": str}
    rows = [
        _inspect_csv_asset(
            "playoffs",
            "playoff_games",
            config.PLAYOFF_GAMES_PATH,
            required=False,
            dtype=dtype,
            notes="optional — run collect_playoff_games",
        ),
        _inspect_csv_asset(
            "playoffs",
            "playoff_play_by_play",
            config.PLAYOFF_PLAY_BY_PLAY_PATH,
            required=False,
            dtype=dtype,
            notes="optional — run collect_playoff_play_by_play",
        ),
        _inspect_csv_asset(
            "playoffs",
            "playoff_live_features",
            config.PLAYOFF_LIVE_FEATURES_PATH,
            required=False,
            dtype=dtype,
            notes="optional — run build_playoff_live_features",
        ),
        _inspect_csv_asset(
            "playoffs",
            "playoff_live_predictions",
            config.PLAYOFF_LIVE_PREDICTIONS_PATH,
            required=False,
            dtype=dtype,
            notes="optional — run predict_playoff_live",
        ),
        _inspect_csv_asset(
            "playoffs",
            "nba_finals_case_study_summary",
            config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH,
            required=False,
            game_id_column=None,
            notes="optional — run build_finals_case_study",
        ),
        _inspect_csv_asset(
            "playoffs",
            "playoff_coverage_report",
            config.PLAYOFF_COVERAGE_REPORT_PATH,
            required=False,
            game_id_column=None,
            notes="optional — run check_playoff_coverage",
        ),
        _inspect_csv_asset(
            "playoffs",
            "finals_pregame_features",
            config.FINALS_PREGAME_FEATURES_PATH,
            required=False,
            dtype=dtype,
            notes="optional — run build_finals_pregame_predictions",
        ),
        _inspect_csv_asset(
            "playoffs",
            "finals_pregame_predictions",
            config.FINALS_PREGAME_PREDICTIONS_PATH,
            required=False,
            dtype=dtype,
            notes="optional — run build_finals_pregame_predictions",
        ),
        _inspect_csv_asset(
            "playoffs",
            "finals_upcoming_predictions",
            config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH,
            required=False,
            game_id_column=None,
            notes="optional — run build_finals_pregame_predictions",
        ),
        _inspect_csv_asset(
            "manual",
            "finals_schedule_overrides",
            config.FINALS_SCHEDULE_OVERRIDES_PATH,
            required=False,
            game_id_column="game_id",
            dtype=dtype,
            notes=summarize_manual_overrides(config.FINALS_SCHEDULE_OVERRIDES_PATH),
        ),
        _inspect_csv_asset(
            "playoffs",
            "finals_projected_series_path",
            config.FINALS_PROJECTED_SERIES_PATH,
            required=False,
            game_id_column=None,
            notes="optional — run build_finals_projected_series_path",
        ),
    ]
    return pd.DataFrame(rows)


def build_data_freshness_summary() -> pd.DataFrame:
    """Inspect all tracked assets and return a structured summary DataFrame."""
    dtype = {"game_id": str}
    latest_log = read_latest_refresh_log(config.DATA_REFRESH_LOG_PATH)
    log_note = ""
    if latest_log:
        log_note = (
            f"latest log {latest_log['timestamp']} "
            f"({latest_log['endpoint']}, {latest_log['status']})"
        )

    rows: List[dict] = []

    raw_assets = [
        ("raw_games", config.RAW_GAMES_PATH),
        ("raw_play_by_play", config.RAW_PLAY_BY_PLAY_PATH),
        ("raw_team_stats", config.RAW_TEAM_STATS_PATH),
    ]
    for asset, path in raw_assets:
        note = summarize_seasons_in_csv(path)
        rows.append(
            _inspect_csv_asset(
                "raw_data", asset, path, required=True, dtype=dtype, notes=note
            )
        )

    processed_assets = [
        ("pregame_features", config.PREGAME_FEATURES_PATH),
        ("live_features", config.LIVE_FEATURES_PATH),
        ("game_results", config.GAME_RESULTS_PATH),
    ]
    for asset, path in processed_assets:
        rows.append(
            _inspect_csv_asset("processed_data", asset, path, required=True, dtype=dtype)
        )

    rows.extend(summarize_prediction_files().to_dict("records"))
    rows.extend(summarize_model_artifacts().to_dict("records"))
    rows.extend(summarize_multiseason_model_artifacts().to_dict("records"))
    rows.extend(summarize_report_files().to_dict("records"))
    rows.extend(summarize_multiseason_reports().to_dict("records"))
    rows.extend(summarize_comparison_reports().to_dict("records"))
    rows.extend(summarize_playoff_case_study_files().to_dict("records"))

    manual_status = file_status(config.POSTGAME_RESULTS_PATH, required=False)
    rows.append(
        {
            "section": "manual",
            "asset": "postgame_results",
            "path": _relative_path(config.POSTGAME_RESULTS_PATH),
            "status": manual_status,
            "row_count": count_csv_rows(config.POSTGAME_RESULTS_PATH, dtype=dtype) or "",
            "unique_games": get_unique_count(
                config.POSTGAME_RESULTS_PATH, "game_id", dtype=dtype
            )
            if config.POSTGAME_RESULTS_PATH.exists()
            else "",
            "last_modified": get_file_modified_time(config.POSTGAME_RESULTS_PATH),
            "notes": summarize_manual_overrides(config.POSTGAME_RESULTS_PATH),
        }
    )

    log_status = file_status(config.DATA_REFRESH_LOG_PATH, required=False)
    rows.append(
        {
            "section": "logs",
            "asset": "data_refresh_log",
            "path": _relative_path(config.DATA_REFRESH_LOG_PATH),
            "status": log_status,
            "row_count": count_csv_rows(config.DATA_REFRESH_LOG_PATH) or "",
            "unique_games": "",
            "last_modified": get_file_modified_time(config.DATA_REFRESH_LOG_PATH),
            "notes": log_note,
        }
    )

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return _apply_stale_rules(summary_df)


def save_data_freshness_summary(
    summary_df: pd.DataFrame,
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write the freshness summary CSV to disk."""
    out_path = Path(path) if path is not None else config.DATA_FRESHNESS_REPORT_PATH
    save_csv(summary_df, out_path)
    return out_path


def count_statuses(summary_df: pd.DataFrame) -> Dict[str, int]:
    """Count assets by status value."""
    counts = {status: 0 for status in STATUS_PRIORITY}
    if summary_df.empty:
        return counts
    for status, count in summary_df["status"].value_counts().items():
        counts[str(status)] = int(count)
    return counts


def get_top_issues(summary_df: pd.DataFrame, limit: int = 5) -> List[dict]:
    """Return the most important non-ok assets for console display."""
    if summary_df.empty:
        return []

    issues = summary_df.loc[summary_df["status"].isin(["missing", "stale", "warning"])].copy()
    if issues.empty:
        return []

    issues["_priority"] = issues["status"].map(STATUS_PRIORITY)
    issues = issues.sort_values(["_priority", "section", "asset"]).head(limit)
    return issues.drop(columns=["_priority"]).to_dict("records")


def compute_overall_status(summary_df: pd.DataFrame) -> Status:
    """Return a single overall status for the project."""
    if summary_df.empty:
        return "unknown"

    statuses = summary_df["status"].tolist()
    if "missing" in statuses:
        return "missing"
    if "stale" in statuses:
        return "stale"
    if "warning" in statuses:
        return "warning"
    if all(s in {"ok", "not_required"} for s in statuses):
        return "ok"
    return "unknown"


def run_data_freshness_check(verbose: bool = True) -> int:
    """Build, save, and optionally print the data freshness summary."""
    ensure_directories()
    summary_df = build_data_freshness_summary()
    out_path = save_data_freshness_summary(summary_df)
    counts = count_statuses(summary_df)

    if verbose:
        total = len(summary_df)
        print("Data freshness check complete.")
        print(f"  Assets checked: {total}")
        print(f"  ok:          {counts.get('ok', 0)}")
        print(f"  warning:     {counts.get('warning', 0)}")
        print(f"  missing:     {counts.get('missing', 0)}")
        print(f"  stale:       {counts.get('stale', 0)}")
        print(f"  not_required:{counts.get('not_required', 0)}")
        print(f"  Output:      {out_path}")
        print(f"  Overall:     {compute_overall_status(summary_df)}")

        issues = get_top_issues(summary_df)
        if issues:
            print("\nTop issues:")
            for issue in issues:
                print(
                    f"  - [{issue['status']}] {issue['section']}/{issue['asset']} "
                    f"({issue['path']})"
                )
                if issue.get("notes"):
                    print(f"      {issue['notes']}")
        else:
            print("\nNo issues detected.")

    return 0


def main() -> int:
    """CLI entry point for ``python src/data_freshness.py``."""
    return run_data_freshness_check(verbose=True)


if __name__ == "__main__":
    sys.exit(main())
