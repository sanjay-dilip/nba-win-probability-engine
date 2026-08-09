"""Foundation tests for the NBA Win Probability Engine.

These tests verify that the project layout, sample data, and core validation
helpers all behave as expected. They do not require any network access or
trained models.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is importable when running pytest from anywhere.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import validate_probability_column  # noqa: E402
from src.utils import compute_winner, ensure_directories, load_csv, save_csv  # noqa: E402

# Exact column order the manual post-game override CSV must use.
POSTGAME_RESULT_COLUMNS = [
    "game_id",
    "home_score",
    "away_score",
    "winner",
    "source",
    "confirmed_at",
    "notes",
]


def test_ensure_directories_creates_all_paths():
    """ensure_directories() should create every configured directory."""
    ensure_directories()
    for directory in config.ALL_DIRECTORIES:
        assert directory.exists(), f"Expected directory to exist: {directory}"
        assert directory.is_dir(), f"Expected a directory: {directory}"


def test_sample_files_exist():
    """All committed sample CSV files should be present."""
    sample_files = [
        config.SAMPLE_GAMES_PATH,
        config.SAMPLE_PREGAME_FEATURES_PATH,
        config.SAMPLE_LIVE_FEATURES_PATH,
        config.SAMPLE_PREDICTIONS_PATH,
    ]
    for path in sample_files:
        assert path.exists(), f"Missing sample file: {path}"


@pytest.mark.parametrize(
    "path, required_columns",
    [
        (
            config.SAMPLE_GAMES_PATH,
            [
                "game_id",
                "season",
                "game_date",
                "home_team",
                "away_team",
                "home_team_id",
                "away_team_id",
                "status",
                "game_type",
            ],
        ),
        (
            config.SAMPLE_PREGAME_FEATURES_PATH,
            [
                "game_id",
                "season",
                "game_date",
                "home_team",
                "away_team",
                "home_win_pct",
                "away_win_pct",
                "win_pct_diff",
                "home_net_rating",
                "away_net_rating",
                "net_rating_diff",
                "home_rest_days",
                "away_rest_days",
                "rest_days_diff",
                "home_recent_form",
                "away_recent_form",
                "recent_form_diff",
                "is_playoff",
                "home_team_won",
            ],
        ),
        (
            config.SAMPLE_LIVE_FEATURES_PATH,
            [
                "game_id",
                "event_num",
                "period",
                "seconds_remaining_period",
                "seconds_remaining_game",
                "home_score",
                "away_score",
                "score_margin_home",
                "abs_score_margin",
                "event_type",
                "home_team_won",
            ],
        ),
        (
            config.SAMPLE_PREDICTIONS_PATH,
            [
                "game_id",
                "event_num",
                "period",
                "seconds_remaining_game",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "home_win_probability",
                "away_win_probability",
                "predicted_winner",
            ],
        ),
    ],
)
def test_sample_csvs_have_required_columns(path, required_columns):
    """Each sample CSV should contain its required columns (via load_csv)."""
    df = load_csv(path, required_columns=required_columns)
    assert not df.empty, f"Sample file should not be empty: {path}"


def test_probability_column_is_valid():
    """home_win_probability in sample_predictions.csv must be valid [0, 1]."""
    df = load_csv(config.SAMPLE_PREDICTIONS_PATH)
    # Should not raise.
    validate_probability_column(df, "home_win_probability")


@pytest.mark.parametrize(
    "home_score, away_score, expected",
    [
        (110, 102, "home"),
        (98, 105, "away"),
        (100, 100, "tie"),
    ],
)
def test_compute_winner(home_score, away_score, expected):
    """compute_winner should map scores to home / away / tie correctly."""
    assert compute_winner(home_score, away_score) == expected


def test_manual_override_output_format(tmp_path):
    """A manual override record must use the expected column layout.

    Uses build_manual_result_record from src.manual_override.
    """
    from src.manual_override import build_manual_result_record  # noqa: E402

    record = build_manual_result_record(
        game_id="SAMPLE_GAME_001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=114,
        away_score=103,
        source="manual",
    )

    out_path = tmp_path / "postgame_results.csv"
    save_csv(pd.DataFrame([record]), out_path)

    written = pd.read_csv(out_path)
    assert list(written.columns) == POSTGAME_RESULT_COLUMNS
    assert written.loc[0, "winner"] == "Boston Celtics"
    assert written.loc[0, "source"] == "manual"
