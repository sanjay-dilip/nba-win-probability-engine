"""Tests for the live model training helpers (Build 8).

All tests exercise small, pure helpers on hand-built DataFrames and never call
the NBA API.  The one end-to-end test trains a tiny Logistic Regression on a few
games (fast) only to confirm the saved artifacts have the right shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.train_live_model import (  # noqa: E402
    CALIBRATION_COLUMNS,
    FORBIDDEN_COLUMNS,
    PHASE_METRICS_COLUMNS,
    TARGET_COLUMN,
    build_calibration_table,
    build_live_logistic_regression_pipeline,
    build_phase_metrics_table,
    chronological_game_train_test_split,
    evaluate_classifier,
    get_live_feature_columns,
    prepare_live_training_data,
    train_live_model,
    validate_game_split,
    validate_no_live_leakage_columns,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _live_df(n_games: int = 6, events_per_game: int = 8, string_event_types: bool = True):
    """A small live-features-shaped frame spanning several games.

    Each game's target alternates deterministically so any chronological game
    slice contains both classes (keeps helper tests stable).
    """
    rows = []
    for gi in range(n_games):
        game_id = f"00224000{gi:02d}"
        date = f"2024-11-{gi + 1:02d}"
        won = gi % 2  # alternate home win/loss by game
        for ei in range(events_per_game):
            secs = 2880 - ei * (2880 // events_per_game)  # counts down across the game
            rows.append({
                "game_id": game_id,
                "event_num": ei + 1,
                "season": "2024-25",
                "game_date": date,
                "home_team": "Team A",
                "away_team": "Team B",
                "home_team_id": "100",
                "away_team_id": "200",
                "period": min(4, ei // 2 + 1),
                "pctimestring": "PT06M00.00S",
                "seconds_remaining_period": 360.0,
                "seconds_remaining_game": float(secs),
                "home_score": float(ei * 3),
                "away_score": float(ei * 2),
                "score_margin_home": float(ei),
                "abs_score_margin": float(ei),
                "event_msg_type": "Made Shot" if string_event_types else (ei % 13) + 1,
                "event_msg_action_type": "Jump Shot" if string_event_types else 0,
                "event_type_label": "made_shot",
                "is_scoring_event": ei % 2 == 0,
                "is_turnover": False,
                "is_foul": False,
                "is_timeout": False,
                "is_rebound": False,
                "is_free_throw": False,
                "is_field_goal_attempt": True,
                "game_type": "regular_season",
                "home_team_won": won,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_live_feature_columns
# ---------------------------------------------------------------------------

def test_feature_columns_exclude_forbidden_leakage_columns():
    feats = get_live_feature_columns(_live_df())
    for forbidden in FORBIDDEN_COLUMNS:
        assert forbidden not in feats["all_features"]
    # Running score IS allowed live state (not a final-outcome column).
    assert "home_score" in feats["all_features"]
    assert "score_margin_home" in feats["all_features"]


def test_feature_columns_string_event_type_routed_to_categorical():
    feats = get_live_feature_columns(_live_df(string_event_types=True))
    assert "event_msg_type" in feats["categorical_features"]
    assert "event_msg_action_type" in feats["categorical_features"]
    assert "event_msg_type" not in feats["numeric_features"]


def test_feature_columns_numeric_event_type_routed_to_numeric():
    feats = get_live_feature_columns(_live_df(string_event_types=False))
    assert "event_msg_type" in feats["numeric_features"]
    assert "event_msg_action_type" in feats["numeric_features"]
    assert "event_msg_type" not in feats["categorical_features"]


def test_feature_columns_all_is_numeric_plus_categorical():
    feats = get_live_feature_columns(_live_df())
    assert feats["all_features"] == feats["numeric_features"] + feats["categorical_features"]


# ---------------------------------------------------------------------------
# validate_no_live_leakage_columns
# ---------------------------------------------------------------------------

def test_validate_raises_on_target_column():
    with pytest.raises(ValueError):
        validate_no_live_leakage_columns(["score_margin_home", "home_team_won"])


def test_validate_raises_on_winner_and_final_scores():
    with pytest.raises(ValueError):
        validate_no_live_leakage_columns(["period", "winner"])
    with pytest.raises(ValueError):
        validate_no_live_leakage_columns(["period", "final_home_score"])
    with pytest.raises(ValueError):
        validate_no_live_leakage_columns(["final_away_score"])


def test_validate_passes_on_clean_feature_list():
    validate_no_live_leakage_columns(get_live_feature_columns(_live_df())["all_features"])


# ---------------------------------------------------------------------------
# prepare_live_training_data
# ---------------------------------------------------------------------------

def test_missing_target_rows_dropped():
    df = _live_df(n_games=3, events_per_game=4)
    df.loc[0, TARGET_COLUMN] = np.nan
    df.loc[5, TARGET_COLUMN] = np.nan
    clean = prepare_live_training_data(df)
    assert len(clean) == len(df) - 2
    assert clean[TARGET_COLUMN].notna().all()
    assert pd.api.types.is_integer_dtype(clean[TARGET_COLUMN])


def test_prepare_raises_without_target_column():
    df = _live_df().drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError):
        prepare_live_training_data(df)


# ---------------------------------------------------------------------------
# chronological_game_train_test_split + validate_game_split
# ---------------------------------------------------------------------------

def test_split_keeps_games_whole_and_separated():
    df = _live_df(n_games=10, events_per_game=5)
    train_df, test_df = chronological_game_train_test_split(df, test_size=0.2)
    # No game_id in both sides.
    assert set(train_df["game_id"]).isdisjoint(set(test_df["game_id"]))
    # 10 games, 20% -> 2 test games, 8 train games.
    assert train_df["game_id"].nunique() == 8
    assert test_df["game_id"].nunique() == 2
    # Every event row of each game stays together (counts are multiples of 5).
    assert len(test_df) == 2 * 5


def test_split_preserves_chronological_game_order():
    # Shuffle rows first; the split must still order games by date, not shuffle.
    df = _live_df(n_games=6, events_per_game=4).sample(frac=1.0, random_state=1)
    train_df, test_df = chronological_game_train_test_split(df, test_size=0.5)
    assert train_df["game_date"].max() <= test_df["game_date"].min()


def test_validate_game_split_raises_on_overlap():
    df = _live_df(n_games=4, events_per_game=3)
    # Force an overlap by reusing the same frame on both sides.
    with pytest.raises(ValueError):
        validate_game_split(df, df)


def test_validate_game_split_passes_when_disjoint():
    df = _live_df(n_games=6, events_per_game=3)
    train_df, test_df = chronological_game_train_test_split(df, test_size=0.34)
    validate_game_split(train_df, test_df)  # must not raise


# ---------------------------------------------------------------------------
# evaluate_classifier
# ---------------------------------------------------------------------------

def test_metrics_normal_binary_case():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 0, 0]
    y_prob = [0.9, 0.2, 0.4, 0.1]
    m = evaluate_classifier(y_true, y_pred, y_prob)
    assert m["accuracy"] == pytest.approx(0.75)
    assert 0.0 <= m["roc_auc"] <= 1.0
    assert m["true_positive"] == 1
    assert m["true_negative"] == 2
    assert m["false_negative"] == 1
    assert m["false_positive"] == 0


def test_metrics_one_class_roc_auc_is_nan():
    with pytest.warns(UserWarning):
        m = evaluate_classifier([1, 1, 1], [1, 0, 1], [0.8, 0.3, 0.6])
    assert np.isnan(m["roc_auc"])
    assert not np.isnan(m["accuracy"])


# ---------------------------------------------------------------------------
# build_calibration_table
# ---------------------------------------------------------------------------

def test_calibration_table_columns_and_content():
    y_true = [1, 0, 1, 0, 1, 1, 0, 0]
    y_prob = [0.95, 0.05, 0.85, 0.15, 0.75, 0.65, 0.35, 0.25]
    table = build_calibration_table(y_true, y_prob, n_bins=10)
    assert list(table.columns) == CALIBRATION_COLUMNS
    assert table["row_count"].sum() == len(y_true)


# ---------------------------------------------------------------------------
# build_phase_metrics_table
# ---------------------------------------------------------------------------

def test_phase_metrics_columns_and_phases():
    secs = [2500, 2200, 1500, 800, 600, 100]  # early, early, mid, mid, late, late
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 1, 1, 0]
    y_prob = [0.8, 0.2, 0.7, 0.6, 0.55, 0.3]
    table = build_phase_metrics_table(secs, y_true, y_pred, y_prob)
    assert list(table.columns) == PHASE_METRICS_COLUMNS
    assert set(table["phase"]).issubset({"early_game", "mid_game", "late_game", "unknown"})
    assert table["row_count"].sum() == len(y_true)


# ---------------------------------------------------------------------------
# pipeline construction
# ---------------------------------------------------------------------------

def test_pipeline_builds_with_numeric_and_categorical():
    pipe = build_live_logistic_regression_pipeline(
        ["score_margin_home", "seconds_remaining_game"], ["event_type_label"]
    )
    assert pipe.named_steps["classifier"] is not None
    assert pipe.named_steps["preprocessor"] is not None


# ---------------------------------------------------------------------------
# end-to-end (tiny) — saved feature columns artifact is a dict
# ---------------------------------------------------------------------------

def test_saved_feature_columns_is_a_dict(tmp_path):
    features_path = tmp_path / "live_features.csv"
    _live_df(n_games=10, events_per_game=6).to_csv(features_path, index=False)

    model_path = tmp_path / "live_model.pkl"
    feature_columns_path = tmp_path / "live_feature_columns.pkl"
    metrics_path = tmp_path / "metrics.csv"
    calibration_path = tmp_path / "calibration.csv"
    phase_metrics_path = tmp_path / "phase.csv"

    rc = train_live_model(
        test_size=0.2,
        input_path=features_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        metrics_path=metrics_path,
        calibration_path=calibration_path,
        phase_metrics_path=phase_metrics_path,
    )
    assert rc == 0

    saved = joblib.load(feature_columns_path)
    assert isinstance(saved, dict)
    assert set(saved.keys()) == {"numeric_features", "categorical_features", "all_features"}
    assert TARGET_COLUMN not in saved["all_features"]
    assert model_path.exists()

    metrics = pd.read_csv(metrics_path)
    assert metrics.loc[0, "model_type"] == "logistic_regression"
    assert metrics.loc[0, "train_games"] == 8
    assert metrics.loc[0, "test_games"] == 2


def test_train_live_model_season_split_saves_to_custom_paths(tmp_path):
    rows = []
    for season, n_games in [("2022-23", 4), ("2024-25", 4)]:
        for gi in range(n_games):
            game_id = f"{season.replace('-', '')}{gi:02d}"
            for ei in range(8):
                rows.append({
                    "game_id": game_id,
                    "event_num": ei + 1,
                    "season": season,
                    "game_date": "2024-11-01",
                    "period": 1,
                    "seconds_remaining_period": 600.0,
                    "seconds_remaining_game": float(2880 - ei * 100),
                    "home_score": float(ei),
                    "away_score": float(ei - 1),
                    "score_margin_home": 1.0,
                    "abs_score_margin": 1.0,
                    "event_msg_type": "Made Shot",
                    "event_msg_action_type": "Jump Shot",
                    "event_type_label": "made_shot",
                    "is_scoring_event": True,
                    "is_turnover": False,
                    "is_foul": False,
                    "is_timeout": False,
                    "is_rebound": False,
                    "is_free_throw": False,
                    "is_field_goal_attempt": True,
                    "game_type": "regular_season",
                    "home_team_won": gi % 2,
                })
    features_path = tmp_path / "live_features.csv"
    pd.DataFrame(rows).to_csv(features_path, index=False)

    model_path = tmp_path / "live_model_multiseason.pkl"
    rc = train_live_model(
        train_seasons=["2022-23"],
        test_season="2024-25",
        input_path=features_path,
        model_path=model_path,
        feature_columns_path=tmp_path / "live_feature_columns_multiseason.pkl",
        metrics_path=tmp_path / "live_metrics_multiseason.csv",
        calibration_path=tmp_path / "live_calibration_multiseason.csv",
        phase_metrics_path=tmp_path / "live_phase_multiseason.csv",
    )
    assert rc == 0
    assert model_path.exists()
    assert "multiseason" in model_path.name
