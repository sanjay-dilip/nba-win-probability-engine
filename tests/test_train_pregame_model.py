"""Tests for the pre-game model training helpers.

All tests exercise small, pure helpers on hand-built DataFrames and never call
the NBA API.  The one end-to-end test trains a tiny Logistic Regression on a few
rows (fast) only to confirm the saved artifacts have the right shape.
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

from src.train_pregame_model import (  # noqa: E402
    ALLOWED_NUMERIC_FEATURES,
    CALIBRATION_COLUMNS,
    FORBIDDEN_COLUMNS,
    TARGET_COLUMN,
    build_calibration_table,
    build_logistic_regression_pipeline,
    chronological_train_test_split,
    evaluate_classifier,
    get_pregame_feature_columns,
    prepare_training_data,
    split_feature_types,
    train_pregame_model,
    validate_no_leakage_columns,
)
from src.multiseason_training import split_by_train_test_seasons  # noqa: E402


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _feature_df(n: int = 10) -> pd.DataFrame:
    """A small pregame-features-shaped frame with a usable target.

    The target alternates deterministically (``i % 2``) so any chronological
    hold-out slice contains both classes — keeping helper tests stable.
    """
    rows = []
    for i in range(n):
        rows.append({
            "game_id": f"00224000{i:02d}",
            "season": "2024-25",
            "game_date": f"2024-11-{(i % 28) + 1:02d}",
            "home_team": "Team A",
            "away_team": "Team B",
            "home_team_id": "100",
            "away_team_id": "200",
            "game_type": "regular_season",
            "home_games_played_before": i,
            "away_games_played_before": i,
            "home_wins_before": i // 2,
            "away_wins_before": i // 3,
            "home_losses_before": i - i // 2,
            "away_losses_before": i - i // 3,
            "home_win_pct_before": 0.5 + (i % 5) * 0.05,
            "away_win_pct_before": 0.5 - (i % 4) * 0.05,
            "win_pct_diff_before": (i % 5) * 0.05,
            "home_points_for_avg_before": 110.0 + i,
            "away_points_for_avg_before": 108.0 + i,
            "home_points_allowed_avg_before": 105.0 + i,
            "away_points_allowed_avg_before": 107.0 + i,
            "points_for_avg_diff_before": 2.0,
            "points_allowed_avg_diff_before": -2.0,
            "home_recent_win_pct_before": 0.6,
            "away_recent_win_pct_before": 0.4,
            "recent_win_pct_diff_before": 0.2,
            "home_rest_days": 2.0,
            "away_rest_days": 3.0,
            "rest_days_diff": -1.0,
            "home_team_won": i % 2,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# get_pregame_feature_columns
# ---------------------------------------------------------------------------

def test_feature_columns_exclude_forbidden_leakage_columns():
    df = _feature_df()
    cols = get_pregame_feature_columns(df)
    # No forbidden / id / outcome columns may appear in the feature list.
    for forbidden in FORBIDDEN_COLUMNS:
        assert forbidden not in cols
    # And the curated numeric features that exist are included.
    assert "home_win_pct_before" in cols
    assert "game_type" in cols


def test_feature_columns_only_returns_existing():
    df = _feature_df().drop(columns=["home_rest_days", "game_type"])
    cols = get_pregame_feature_columns(df)
    assert "home_rest_days" not in cols
    assert "game_type" not in cols


def test_split_feature_types_separates_numeric_and_categorical():
    df = _feature_df()
    numeric, categorical = split_feature_types(get_pregame_feature_columns(df))
    assert "game_type" in categorical
    assert "game_type" not in numeric
    assert set(numeric).issubset(set(ALLOWED_NUMERIC_FEATURES))


# ---------------------------------------------------------------------------
# validate_no_leakage_columns
# ---------------------------------------------------------------------------

def test_validate_raises_on_target_column():
    with pytest.raises(ValueError):
        validate_no_leakage_columns(["home_win_pct_before", "home_team_won"])


def test_validate_raises_on_final_scores():
    with pytest.raises(ValueError):
        validate_no_leakage_columns(["home_win_pct_before", "home_score"])
    with pytest.raises(ValueError):
        validate_no_leakage_columns(["away_score", "rest_days_diff"])


def test_validate_passes_on_clean_feature_list():
    # A clean list of allowed features must not raise.
    validate_no_leakage_columns(get_pregame_feature_columns(_feature_df()))


# ---------------------------------------------------------------------------
# prepare_training_data
# ---------------------------------------------------------------------------

def test_missing_target_rows_dropped():
    df = _feature_df(6)
    df.loc[0, TARGET_COLUMN] = np.nan
    df.loc[3, TARGET_COLUMN] = np.nan
    clean = prepare_training_data(df)
    assert len(clean) == 4
    assert clean[TARGET_COLUMN].notna().all()
    # Target is cast to int for supervised training.
    assert pd.api.types.is_integer_dtype(clean[TARGET_COLUMN])


def test_prepare_raises_without_target_column():
    df = _feature_df().drop(columns=[TARGET_COLUMN])
    with pytest.raises(ValueError):
        prepare_training_data(df)


# ---------------------------------------------------------------------------
# chronological_train_test_split
# ---------------------------------------------------------------------------

def test_split_preserves_date_order():
    df = _feature_df(10)
    train_df, test_df = chronological_train_test_split(df, test_size=0.2)
    # Every training date must be <= every test date (no future leaks into train).
    assert train_df["game_date"].max() <= test_df["game_date"].min()
    assert len(train_df) == 8
    assert len(test_df) == 2


def test_split_does_not_shuffle():
    # Feed deliberately unsorted dates; the split must sort ascending, not shuffle.
    df = _feature_df(6).sample(frac=1.0, random_state=42).reset_index(drop=True)
    train_df, test_df = chronological_train_test_split(df, test_size=0.5)
    combined = pd.concat([train_df, test_df], ignore_index=True)
    assert combined["game_date"].tolist() == sorted(combined["game_date"].tolist())


def test_split_rejects_bad_test_size():
    with pytest.raises(ValueError):
        chronological_train_test_split(_feature_df(), test_size=1.5)


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
    assert not np.isnan(m["log_loss"])
    assert not np.isnan(m["brier_score"])


def test_metrics_one_class_roc_auc_is_nan():
    # Only one class present in y_true -> ROC-AUC undefined -> NaN + warning.
    y_true = [1, 1, 1]
    y_pred = [1, 0, 1]
    y_prob = [0.8, 0.3, 0.6]
    with pytest.warns(UserWarning):
        m = evaluate_classifier(y_true, y_pred, y_prob)
    assert np.isnan(m["roc_auc"])
    # Other metrics remain computable.
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
    # Probabilities and rates are within [0, 1].
    assert table["avg_predicted_probability"].between(0, 1).all()
    assert table["actual_home_win_rate"].between(0, 1).all()


# ---------------------------------------------------------------------------
# pipeline construction
# ---------------------------------------------------------------------------

def test_pipeline_builds_with_numeric_and_categorical():
    pipe = build_logistic_regression_pipeline(
        ["home_win_pct_before", "rest_days_diff"], ["game_type"]
    )
    # Pipeline has a preprocessor + classifier and is unfitted but constructible.
    assert pipe.named_steps["classifier"] is not None
    assert pipe.named_steps["preprocessor"] is not None


# ---------------------------------------------------------------------------
# end-to-end (tiny) — saved feature columns are a list
# ---------------------------------------------------------------------------

def test_saved_feature_columns_is_a_list(tmp_path):
    features_path = tmp_path / "pregame_features.csv"
    _feature_df(20).to_csv(features_path, index=False)

    model_path = tmp_path / "pregame_model.pkl"
    feature_columns_path = tmp_path / "pregame_feature_columns.pkl"
    metrics_path = tmp_path / "metrics.csv"
    calibration_path = tmp_path / "calibration.csv"

    rc = train_pregame_model(
        test_size=0.2,
        input_path=features_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        metrics_path=metrics_path,
        calibration_path=calibration_path,
    )
    assert rc == 0

    saved_columns = joblib.load(feature_columns_path)
    assert isinstance(saved_columns, list)
    assert all(isinstance(c, str) for c in saved_columns)
    assert TARGET_COLUMN not in saved_columns
    assert model_path.exists()
    metrics = pd.read_csv(metrics_path)
    assert metrics.loc[0, "model_type"] == "logistic_regression"


def test_season_split_train_test(tmp_path):
    """Explicit season split puts only test-season rows in the hold-out set."""
    rows = []
    for season in ["2022-23", "2023-24", "2024-25"]:
        for i in range(12):
            row = _feature_df(1).iloc[0].to_dict()
            row["season"] = season
            row["game_id"] = f"{season.replace('-', '')}{i:03d}"
            row["game_date"] = f"2024-11-{(i % 28) + 1:02d}"
            row["home_team_won"] = i % 2
            rows.append(row)
    df = pd.DataFrame(rows)
    used = prepare_training_data(df)
    train_df, test_df = split_by_train_test_seasons(
        used, ["2022-23", "2023-24"], "2024-25"
    )
    assert set(train_df["season"].unique()) == {"2022-23", "2023-24"}
    assert set(test_df["season"].unique()) == {"2024-25"}


def test_train_pregame_model_season_split_saves_to_custom_paths(tmp_path):
    rows = []
    for season in ["2022-23", "2024-25"]:
        for i in range(15):
            row = _feature_df(1).iloc[0].to_dict()
            row["season"] = season
            row["game_id"] = f"{season.replace('-', '')}{i:03d}"
            row["home_team_won"] = i % 2
            rows.append(row)
    features_path = tmp_path / "pregame_features.csv"
    pd.DataFrame(rows).to_csv(features_path, index=False)

    model_path = tmp_path / "pregame_model_multiseason.pkl"
    feature_columns_path = tmp_path / "pregame_feature_columns_multiseason.pkl"
    metrics_path = tmp_path / "metrics_multiseason.csv"
    calibration_path = tmp_path / "calibration_multiseason.csv"

    rc = train_pregame_model(
        train_seasons=["2022-23"],
        test_season="2024-25",
        input_path=features_path,
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        metrics_path=metrics_path,
        calibration_path=calibration_path,
    )
    assert rc == 0
    assert model_path.exists()
    assert "multiseason" in model_path.name
