"""Multi-season model training readiness checks.

Validates that processed feature files contain the requested train/test seasons
before any multi-season retraining runs. Does not call ``nba_api`` or train models.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.season_config import validate_season_format, validate_season_list  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

ReadinessStatus = Literal["pass", "warning", "fail", "not_applicable"]

READINESS_COLUMNS = [
    "section",
    "check",
    "status",
    "details",
    "recommended_action",
]

TARGET_COLUMN = "home_team_won"
DEFAULT_MIN_GAMES = 100
DEFAULT_MIN_LIVE_ROWS = 10_000

TRAINING_SPLIT_EXAMPLE = (
    "python run_pipeline.py --mode check_multiseason_training_readiness "
    "--train-seasons 2022-23 2023-24 --test-season 2024-25"
)


def _row(
    section: str,
    check: str,
    status: ReadinessStatus,
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


def get_available_seasons(df: pd.DataFrame, season_col: str = "season") -> List[str]:
    """Return sorted unique season labels present in ``df``."""
    if df.empty or season_col not in df.columns:
        return []
    return sorted(df[season_col].dropna().astype(str).unique().tolist())


def validate_train_test_seasons(
    train_seasons: Sequence[str],
    test_season: str,
) -> Tuple[List[str], str]:
    """Validate train/test season lists and ensure they do not overlap.

    Returns:
        ``(validated_train_seasons, validated_test_season)``

    Raises:
        ValueError: If formats are invalid, lists are empty, or seasons overlap.
    """
    validated_train = validate_season_list(train_seasons)
    validated_test = validate_season_format(test_season)

    overlap = set(validated_train) & {validated_test}
    if overlap:
        raise ValueError(
            f"Train and test seasons must not overlap. Overlap: {sorted(overlap)}"
        )

    return validated_train, validated_test


def check_required_columns(
    df: Optional[pd.DataFrame],
    required_columns: Sequence[str],
    dataset_name: str,
    section: str,
) -> List[dict]:
    """Verify required columns exist in a feature DataFrame."""
    rows: List[dict] = []
    if df is None:
        for col in required_columns:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_column_{col}",
                    "fail",
                    f"{dataset_name} not loaded",
                    f"Build {dataset_name} before multi-season training",
                )
            )
        return rows

    for col in required_columns:
        if col in df.columns:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_column_{col}",
                    "pass",
                    f"{col} present in {dataset_name}",
                )
            )
        else:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_column_{col}",
                    "fail",
                    f"Missing column '{col}' in {dataset_name}",
                    f"Rebuild {dataset_name} with season column",
                )
            )
    return rows


def check_min_rows_by_season(
    df: pd.DataFrame,
    season_col: str,
    seasons: Sequence[str],
    min_rows: int,
    dataset_name: str,
    section: str,
) -> List[dict]:
    """Fail when a season has fewer than ``min_rows`` event/game rows."""
    rows: List[dict] = []
    if df is None or df.empty:
        return rows

    for season in seasons:
        count = int((df[season_col] == season).sum())
        if count >= min_rows:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_min_rows_{season}",
                    "pass",
                    f"{season}: {count} rows (min {min_rows})",
                )
            )
        else:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_min_rows_{season}",
                    "fail",
                    f"{season}: {count} rows, need at least {min_rows}",
                    "Collect play-by-play and rebuild live features for this season",
                )
            )
    return rows


def check_min_games_by_season(
    df: pd.DataFrame,
    season_col: str,
    game_id_col: str,
    seasons: Sequence[str],
    min_games: int,
    dataset_name: str,
    section: str,
) -> List[dict]:
    """Fail when a season has fewer than ``min_games`` unique games."""
    rows: List[dict] = []
    if df is None or df.empty:
        return rows

    for season in seasons:
        subset = df[df[season_col] == season]
        count = int(subset[game_id_col].nunique()) if not subset.empty else 0
        if count >= min_games:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_min_games_{season}",
                    "pass",
                    f"{season}: {count} games (min {min_games})",
                )
            )
        else:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_min_games_{season}",
                    "fail",
                    f"{season}: {count} games, need at least {min_games}",
                    "Collect schedule/PBP and rebuild features for this season",
                )
            )
    return rows


def _check_season_presence(
    available: Sequence[str],
    requested: Sequence[str],
    dataset_name: str,
    section: str,
) -> List[dict]:
    rows: List[dict] = []
    for season in requested:
        if season in available:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_season_{season}",
                    "pass",
                    f"{season} present in {dataset_name}",
                )
            )
        else:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_season_{season}",
                    "fail",
                    f"{season} missing from {dataset_name} (have: {', '.join(available) or 'none'})",
                    "Run build_features_multi_season after collecting raw data",
                )
            )
    return rows


def _check_target_not_null(
    df: pd.DataFrame,
    seasons: Sequence[str],
    season_col: str,
    dataset_name: str,
    section: str,
    label: str,
) -> List[dict]:
    rows: List[dict] = []
    if df is None or df.empty or TARGET_COLUMN not in df.columns:
        return rows

    for season in seasons:
        subset = df[df[season_col] == season]
        if subset.empty:
            continue
        null_count = int(subset[TARGET_COLUMN].isna().sum())
        if null_count == 0:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_target_{label}_{season}",
                    "pass",
                    f"{season}: all {label} rows have {TARGET_COLUMN}",
                )
            )
        else:
            rows.append(
                _row(
                    section,
                    f"{dataset_name}_target_{label}_{season}",
                    "fail",
                    f"{season}: {null_count} {label} rows missing {TARGET_COLUMN}",
                    f"Rebuild {dataset_name} for this season",
                )
            )
    return rows


def split_by_train_test_seasons(
    df: pd.DataFrame,
    train_seasons: Sequence[str],
    test_season: str,
    season_col: str = "season",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train and test subsets by season label."""
    train_mask = df[season_col].isin(list(train_seasons))
    test_mask = df[season_col] == test_season
    return (
        df[train_mask].copy().reset_index(drop=True),
        df[test_mask].copy().reset_index(drop=True),
    )


def should_allow_training(
    readiness_df: pd.DataFrame,
    sections: Optional[Sequence[str]] = None,
) -> bool:
    """Return True when no ``fail`` statuses exist (optionally filtered by section)."""
    df = readiness_df
    if sections is not None:
        df = readiness_df[readiness_df["section"].isin(sections)]
    if df.empty:
        return False
    return not (df["status"] == "fail").any()


def _load_features_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(
        path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )


def build_multiseason_training_readiness(
    train_seasons: Sequence[str],
    test_season: str,
    pregame_path: Optional[Path] = None,
    live_path: Optional[Path] = None,
    min_games: int = DEFAULT_MIN_GAMES,
    min_live_rows: int = DEFAULT_MIN_LIVE_ROWS,
) -> pd.DataFrame:
    """Build the full multi-season training readiness report."""
    validated_train, validated_test = validate_train_test_seasons(
        train_seasons, test_season
    )
    pregame_path = pregame_path or config.PREGAME_FEATURES_PATH
    live_path = live_path or config.LIVE_FEATURES_PATH

    rows: List[dict] = []

    # Shared split validation
    rows.append(
        _row(
            "split",
            "train_test_no_overlap",
            "pass",
            f"Train {validated_train}, test {validated_test}",
        )
    )

    pregame_df = _load_features_if_exists(pregame_path)
    live_df = _load_features_if_exists(live_path)

    if pregame_df is None:
        rows.append(
            _row(
                "pregame",
                "pregame_features_exists",
                "fail",
                f"File missing: {pregame_path.name}",
                "Run build_pregame_features or build_features_multi_season",
            )
        )
    else:
        rows.append(
            _row(
                "pregame",
                "pregame_features_exists",
                "pass",
                f"{len(pregame_df)} rows loaded",
            )
        )

    if live_df is None:
        rows.append(
            _row(
                "live",
                "live_features_exists",
                "fail",
                f"File missing: {live_path.name}",
                "Run build_live_features or build_features_multi_season",
            )
        )
    else:
        rows.append(
            _row(
                "live",
                "live_features_exists",
                "pass",
                f"{len(live_df)} rows loaded",
            )
        )

    required = ["season", "game_id", TARGET_COLUMN]
    rows.extend(check_required_columns(pregame_df, required, "pregame_features", "pregame"))
    rows.extend(check_required_columns(live_df, required, "live_features", "live"))

    if pregame_df is not None:
        available = get_available_seasons(pregame_df)
        rows.extend(
            _check_season_presence(
                available, validated_train, "pregame_features", "pregame"
            )
        )
        rows.extend(
            _check_season_presence(
                available, [validated_test], "pregame_features", "pregame"
            )
        )
        rows.extend(
            check_min_games_by_season(
                pregame_df,
                "season",
                "game_id",
                validated_train + [validated_test],
                min_games,
                "pregame_features",
                "pregame",
            )
        )
        rows.extend(
            _check_target_not_null(
                pregame_df,
                validated_train,
                "season",
                "pregame_features",
                "pregame",
                "train",
            )
        )
        rows.extend(
            _check_target_not_null(
                pregame_df,
                [validated_test],
                "season",
                "pregame_features",
                "pregame",
                "test",
            )
        )

    if live_df is not None:
        available = get_available_seasons(live_df)
        rows.extend(
            _check_season_presence(available, validated_train, "live_features", "live")
        )
        rows.extend(
            _check_season_presence(
                available, [validated_test], "live_features", "live"
            )
        )
        rows.extend(
            check_min_games_by_season(
                live_df,
                "season",
                "game_id",
                validated_train + [validated_test],
                min_games,
                "live_features",
                "live",
            )
        )
        rows.extend(
            check_min_rows_by_season(
                live_df,
                "season",
                validated_train + [validated_test],
                min_live_rows,
                "live_features",
                "live",
            )
        )
        rows.extend(
            _check_target_not_null(
                live_df,
                validated_train,
                "season",
                "live_features",
                "live",
                "train",
            )
        )
        rows.extend(
            _check_target_not_null(
                live_df,
                [validated_test],
                "season",
                "live_features",
                "live",
                "test",
            )
        )

    return pd.DataFrame(rows, columns=READINESS_COLUMNS)


def save_multiseason_training_readiness(
    readiness_df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """Write the readiness report CSV."""
    path = path or config.MULTISEASON_TRAINING_READINESS_PATH
    ensure_directories()
    save_csv(readiness_df, path)
    return path


def print_readiness_summary(readiness_df: pd.DataFrame) -> None:
    """Print pass/warning/fail counts and top failures."""
    counts = readiness_df["status"].value_counts().to_dict()
    print("\nMulti-season training readiness:")
    for status in ["pass", "warning", "fail", "not_applicable"]:
        if status in counts:
            print(f"  {status}: {counts[status]}")

    failures = readiness_df[readiness_df["status"] == "fail"]
    if not failures.empty:
        print("\nTop failures:")
        for _, row in failures.head(10).iterrows():
            print(f"  - [{row['section']}] {row['check']}: {row['details']}")
            if row["recommended_action"]:
                print(f"      -> {row['recommended_action']}")


def run_multiseason_training_readiness_check(
    train_seasons: Sequence[str],
    test_season: str,
    verbose: bool = True,
    min_games: int = DEFAULT_MIN_GAMES,
    min_live_rows: int = DEFAULT_MIN_LIVE_ROWS,
) -> int:
    """Build, save, and print the readiness report. Exit 1 if any fail status."""
    readiness = build_multiseason_training_readiness(
        train_seasons,
        test_season,
        min_games=min_games,
        min_live_rows=min_live_rows,
    )
    out_path = save_multiseason_training_readiness(readiness)
    if verbose:
        print(f"Readiness report: {out_path}")
        print_readiness_summary(readiness)

    if should_allow_training(readiness):
        if verbose:
            print("\nReadiness: PASS — training may proceed.")
        return 0

    if verbose:
        print("\nReadiness: FAIL — training blocked until issues are resolved.")
    return 1


def require_training_split_error_message() -> str:
    """Return a clear error when --train-seasons / --test-season are missing."""
    return (
        "Multi-season training modes require both --train-seasons and --test-season.\n"
        f"Example: {TRAINING_SPLIT_EXAMPLE}"
    )
