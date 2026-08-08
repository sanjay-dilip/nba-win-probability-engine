"""Multi-season data coverage checks.

Inspects local CSV files and summarizes season-level coverage. Does not call
``nba_api``, build features, or train models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

CoverageStatus = Literal["pass", "warning", "fail", "not_applicable"]

COVERAGE_COLUMNS = [
    "section",
    "season",
    "expected_games",
    "available_games",
    "available_rows",
    "coverage_pct",
    "status",
    "notes",
    "recommended_action",
]

DEFAULT_GAMES_MIN = 1000
COVERAGE_THRESHOLD = 0.95

CRITICAL_SECTIONS = [
    "games",
    "play_by_play",
    "live_features",
    "game_results",
    "pregame_features",
]

_GAME_ID_DTYPE = {"game_id": str, "home_team_id": str, "away_team_id": str}


def _coverage_row(
    section: str,
    season: str,
    expected_games: int,
    available_games: int,
    available_rows: int,
    status: CoverageStatus,
    notes: str = "",
    recommended_action: str = "",
) -> dict:
    if expected_games > 0:
        coverage_pct = round(100.0 * available_games / expected_games, 2)
    elif available_games > 0:
        coverage_pct = 100.0
    else:
        coverage_pct = 0.0
    return {
        "section": section,
        "season": season,
        "expected_games": expected_games,
        "available_games": available_games,
        "available_rows": available_rows,
        "coverage_pct": coverage_pct,
        "status": status,
        "notes": notes,
        "recommended_action": recommended_action,
    }


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, dtype=_GAME_ID_DTYPE)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def count_rows_by_season(
    path: Path,
    season_col: str = "season",
) -> Dict[str, int]:
    """Count rows per season in a CSV file."""
    df = _load_csv(path)
    if df is None or df.empty or season_col not in df.columns:
        return {}
    counts = df.groupby(season_col, dropna=False).size()
    return {str(k): int(v) for k, v in counts.items()}


def count_unique_games_by_season(
    path: Path,
    season_col: str = "season",
    game_id_col: str = "game_id",
) -> Dict[str, int]:
    """Count unique ``game_id`` values per season."""
    df = _load_csv(path)
    if df is None or df.empty:
        return {}
    if season_col not in df.columns or game_id_col not in df.columns:
        return {}
    grouped = df.groupby(season_col, dropna=False)[game_id_col].nunique()
    return {str(k): int(v) for k, v in grouped.items()}


def get_expected_games_by_season(
    games_path: Path,
    seasons: Sequence[str],
) -> Dict[str, int]:
    """Return expected regular-season game counts from ``games.csv``."""
    counts = count_unique_games_by_season(games_path)
    return {season: counts.get(season, 0) for season in seasons}


def _status_for_games_section(available_games: int) -> CoverageStatus:
    if available_games >= DEFAULT_GAMES_MIN:
        return "pass"
    if available_games > 0:
        return "warning"
    return "fail"


def _status_for_coverage_ratio(
    available_games: int,
    expected_games: int,
) -> CoverageStatus:
    if expected_games <= 0:
        return "fail"
    ratio = available_games / expected_games
    if ratio >= COVERAGE_THRESHOLD:
        return "pass"
    if available_games > 0:
        return "warning"
    return "fail"


def _games_section_rows(
    games_path: Path,
    seasons: Sequence[str],
) -> List[dict]:
    rows: List[dict] = []
    if not games_path.exists():
        for season in seasons:
            rows.append(
                _coverage_row(
                    "games",
                    season,
                    0,
                    0,
                    0,
                    "fail",
                    f"File missing: {games_path.name}",
                    "Run refresh_multi_season_metadata or collect_games_multi_season",
                )
            )
        return rows

    row_counts = count_rows_by_season(games_path)
    game_counts = count_unique_games_by_season(games_path)

    for season in seasons:
        available = game_counts.get(season, 0)
        available_rows = row_counts.get(season, 0)
        status = _status_for_games_section(available)
        notes = f"{available} unique games in games.csv"
        action = ""
        if status == "fail":
            action = "Collect game schedule for this season"
        elif status == "warning":
            action = f"Expected at least {DEFAULT_GAMES_MIN} games; verify schedule collection"
        rows.append(
            _coverage_row(
                "games",
                season,
                available,
                available,
                available_rows,
                status,
                notes,
                action,
            )
        )
    return rows


def get_play_by_play_coverage_by_season(
    games_path: Path,
    play_by_play_path: Path,
    seasons: Sequence[str],
    expected_by_season: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Build play-by-play coverage rows for each season."""
    expected = expected_by_season or get_expected_games_by_season(games_path, seasons)
    rows: List[dict] = []

    if not play_by_play_path.exists():
        for season in seasons:
            exp = expected.get(season, 0)
            rows.append(
                _coverage_row(
                    "play_by_play",
                    season,
                    exp,
                    0,
                    0,
                    "fail",
                    "play_by_play.csv missing",
                    "Run collect_play_by_play_multi_season for this season (API-heavy)",
                )
            )
        return rows

    game_counts = count_unique_games_by_season(play_by_play_path)
    row_counts = count_rows_by_season(play_by_play_path)

    for season in seasons:
        exp = expected.get(season, 0)
        available = game_counts.get(season, 0)
        available_rows = row_counts.get(season, 0)
        if exp <= 0:
            rows.append(
                _coverage_row(
                    "play_by_play",
                    season,
                    0,
                    available,
                    available_rows,
                    "fail",
                    "No expected games — games.csv lacks this season",
                    "Collect game schedule before play-by-play",
                )
            )
            continue
        status = _status_for_coverage_ratio(available, exp)
        notes = f"{available}/{exp} games ({available_rows} event rows)"
        action = ""
        if status != "pass":
            action = (
                "Run collect_play_by_play_multi_season for this season (API-heavy); "
                "rerun if interrupted"
            )
        rows.append(
            _coverage_row(
                "play_by_play",
                season,
                exp,
                available,
                available_rows,
                status,
                notes,
                action,
            )
        )
    return rows


def get_feature_coverage_by_season(
    feature_path: Path,
    seasons: Sequence[str],
    dataset_name: str,
    expected_by_season: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Build feature-file coverage rows (live or pregame)."""
    section = dataset_name
    expected = expected_by_season or get_expected_games_by_season(
        config.RAW_GAMES_PATH, seasons
    )
    rows: List[dict] = []

    if not feature_path.exists():
        for season in seasons:
            exp = expected.get(season, 0)
            rows.append(
                _coverage_row(
                    section,
                    season,
                    exp,
                    0,
                    0,
                    "fail",
                    f"{feature_path.name} missing",
                    "Run build_features_multi_season or prepare_multiseason_training_data",
                )
            )
        return rows

    game_counts = count_unique_games_by_season(feature_path)
    row_counts = count_rows_by_season(feature_path)

    for season in seasons:
        exp = expected.get(season, 0)
        available = game_counts.get(season, 0)
        available_rows = row_counts.get(season, 0)
        if exp <= 0:
            rows.append(
                _coverage_row(
                    section,
                    season,
                    0,
                    available,
                    available_rows,
                    "fail",
                    "No expected games — games.csv lacks this season",
                    "Collect game schedule before rebuilding features",
                )
            )
            continue
        status = _status_for_coverage_ratio(available, exp)
        notes = f"{available}/{exp} games ({available_rows} rows)"
        action = ""
        if status != "pass":
            action = (
                "Collect play-by-play for this season, then run "
                "prepare_multiseason_training_data"
            )
        rows.append(
            _coverage_row(
                section,
                season,
                exp,
                available,
                available_rows,
                status,
                notes,
                action,
            )
        )
    return rows


def get_game_results_coverage_by_season(
    game_results_path: Path,
    seasons: Sequence[str],
    expected_by_season: Optional[Dict[str, int]] = None,
) -> List[dict]:
    """Build game_results.csv coverage rows."""
    return get_feature_coverage_by_season(
        game_results_path,
        seasons,
        "game_results",
        expected_by_season=expected_by_season,
    )


def _readiness_section_rows(
    report_rows: List[dict],
    train_seasons: Optional[Sequence[str]],
    test_season: Optional[str],
) -> List[dict]:
    if not train_seasons or not test_season:
        return [
            _coverage_row(
                "readiness",
                "all",
                0,
                0,
                0,
                "not_applicable",
                "Provide --train-seasons and --test-season for readiness summary",
                "Use check_multiseason_training_readiness with explicit split",
            )
        ]

    needed = list(train_seasons) + [test_season]
    df = pd.DataFrame(report_rows)
    critical = df[df["section"].isin(["play_by_play", "live_features", "pregame_features"])]

    all_pass = True
    details: List[str] = []
    for season in needed:
        season_rows = critical[critical["season"] == season]
        if season_rows.empty or (season_rows["status"] == "fail").any():
            all_pass = False
            details.append(f"{season}: incomplete")

    if all_pass:
        status: CoverageStatus = "pass"
        notes = f"Coverage sufficient for train {list(train_seasons)} / test {test_season}"
        action = "Run check_multiseason_training_readiness, then train_models_multiseason"
    else:
        status = "fail"
        notes = "; ".join(details) if details else "Coverage incomplete for train/test seasons"
        action = "Complete play-by-play collection and rebuild features before training"

    return [
        _coverage_row(
            "readiness",
            f"train={','.join(train_seasons)};test={test_season}",
            len(needed),
            0,
            0,
            status,
            notes,
            action,
        )
    ]


def build_multiseason_coverage_report(
    seasons: Sequence[str],
    train_seasons: Optional[Sequence[str]] = None,
    test_season: Optional[str] = None,
    games_path: Optional[Path] = None,
    play_by_play_path: Optional[Path] = None,
    live_features_path: Optional[Path] = None,
    game_results_path: Optional[Path] = None,
    pregame_features_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build the full multi-season coverage report."""
    games_path = games_path or config.RAW_GAMES_PATH
    play_by_play_path = play_by_play_path or config.RAW_PLAY_BY_PLAY_PATH
    live_features_path = live_features_path or config.LIVE_FEATURES_PATH
    game_results_path = game_results_path or config.GAME_RESULTS_PATH
    pregame_features_path = pregame_features_path or config.PREGAME_FEATURES_PATH

    rows: List[dict] = []
    rows.extend(_games_section_rows(games_path, seasons))
    expected = get_expected_games_by_season(games_path, seasons)
    rows.extend(
        get_play_by_play_coverage_by_season(
            games_path, play_by_play_path, seasons, expected
        )
    )
    rows.extend(
        get_feature_coverage_by_season(
            live_features_path, seasons, "live_features", expected
        )
    )
    rows.extend(get_game_results_coverage_by_season(game_results_path, seasons, expected))
    rows.extend(
        get_feature_coverage_by_season(
            pregame_features_path, seasons, "pregame_features", expected
        )
    )
    rows.extend(_readiness_section_rows(rows, train_seasons, test_season))
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def save_multiseason_coverage_report(
    report_df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """Write the coverage report CSV."""
    path = path or config.MULTISEASON_COVERAGE_REPORT_PATH
    ensure_directories()
    save_csv(report_df, path)
    return path


def get_missing_seasons_from_report(report_df: pd.DataFrame) -> List[str]:
    """Return seasons with ``fail`` status in critical sections."""
    if report_df.empty:
        return []
    critical = report_df[
        report_df["section"].isin(CRITICAL_SECTIONS) & (report_df["status"] == "fail")
    ]
    return sorted(critical["season"].unique().tolist())


def should_allow_multiseason_training_from_coverage(
    report_df: pd.DataFrame,
) -> bool:
    """Return True when no critical section has ``fail`` for readiness check."""
    if report_df.empty:
        return False
    readiness = report_df[report_df["section"] == "readiness"]
    if not readiness.empty and readiness.iloc[0]["status"] == "pass":
        return True
    if readiness.empty or readiness.iloc[0]["status"] == "not_applicable":
        critical = report_df[
            report_df["section"].isin(CRITICAL_SECTIONS) & (report_df["status"] == "fail")
        ]
        return critical.empty
    return False


def print_coverage_summary(report_df: pd.DataFrame) -> None:
    """Print per-season and per-section summary to stdout."""
    print("\nMulti-season coverage summary:")
    for section in CRITICAL_SECTIONS + ["readiness"]:
        section_df = report_df[report_df["section"] == section]
        if section_df.empty:
            continue
        print(f"\n  [{section}]")
        for _, row in section_df.iterrows():
            pct = row["coverage_pct"]
            pct_str = f"{pct:.1f}%" if pd.notna(pct) else "n/a"
            print(
                f"    {row['season']}: {row['status']} — "
                f"{row['available_games']}/{row['expected_games']} games "
                f"({pct_str})"
            )


def print_recommended_next_commands(
    report_df: pd.DataFrame,
    seasons: Sequence[str],
) -> None:
    """Print suggested pipeline commands based on coverage gaps."""
    missing = get_missing_seasons_from_report(report_df)
    pbp_fail = report_df[
        (report_df["section"] == "play_by_play") & (report_df["status"] != "pass")
    ]
    feature_fail = report_df[
        report_df["section"].isin(["live_features", "pregame_features", "game_results"])
        & (report_df["status"] != "pass")
    ]

    print("\nRecommended next steps:")
    if missing:
        print(f"  Missing/incomplete seasons: {', '.join(missing)}")
    if not pbp_fail.empty:
        need_pbp = sorted(pbp_fail["season"].unique().tolist())
        seasons_arg = " ".join(need_pbp)
        print(
            f"  1. Collect play-by-play (API-heavy):\n"
            f"     python run_pipeline.py --mode collect_play_by_play_multi_season "
            f"--seasons {seasons_arg} --dry-run"
        )
    if not pbp_fail.empty or not feature_fail.empty:
        seasons_arg = " ".join(seasons)
        print(
            f"  2. Rebuild features locally:\n"
            f"     python run_pipeline.py --mode prepare_multiseason_training_data "
            f"--seasons {seasons_arg} --train-seasons 2022-23 2023-24 --test-season 2024-25"
        )
    print(
        "  3. Confirm readiness:\n"
        "     python run_pipeline.py --mode check_multiseason_training_readiness "
        "--train-seasons 2022-23 2023-24 --test-season 2024-25"
    )
    print("  See MULTISEASON_RUNBOOK.md for the full safe sequence.")


def run_multiseason_coverage_check(
    seasons: Sequence[str],
    train_seasons: Optional[Sequence[str]] = None,
    test_season: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Build, save, and print coverage report. Exit 1 on critical failures."""
    report = build_multiseason_coverage_report(
        seasons,
        train_seasons=train_seasons,
        test_season=test_season,
    )
    out_path = save_multiseason_coverage_report(report)
    if verbose:
        print(f"Coverage report: {out_path}")
        counts = report["status"].value_counts().to_dict()
        for status in ["pass", "warning", "fail", "not_applicable"]:
            if status in counts:
                print(f"  {status}: {counts[status]}")
        print_coverage_summary(report)
        missing = get_missing_seasons_from_report(report)
        if missing:
            print(f"\nMissing/incomplete seasons: {', '.join(missing)}")
        print_recommended_next_commands(report, seasons)

    critical_fails = report[
        report["section"].isin(CRITICAL_SECTIONS) & (report["status"] == "fail")
    ]
    if not critical_fails.empty:
        if verbose:
            print("\nCoverage: FAIL — critical gaps remain.")
        return 1
    if verbose:
        print("\nCoverage: PASS — all critical sections meet thresholds.")
    return 0
