"""Tests for pre-game prediction helpers.

Uses tiny DataFrames and a mock estimator — never loads the real trained model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_validation import validate_prediction_probabilities  # noqa: E402
from src.predict_pregame import (  # noqa: E402
    PREGAME_OUTPUT_COLUMNS,
    TARGET_COLUMN,
    build_pregame_prediction_output,
    generate_pregame_predictions,
    validate_pregame_prediction_inputs,
)


class _MockModel:
    """Returns fixed home-win probabilities from predict_proba."""

    def __init__(self, probs: list[float]):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n = len(x)
        p = self._probs[:n] if len(self._probs) >= n else np.resize(self._probs, n)
        return np.column_stack([1.0 - p, p])


def _features_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": [f"002240000{i}" for i in range(n)],
        "season": ["2024-25"] * n,
        "game_date": [f"2024-11-{(i % 28) + 1:02d}" for i in range(n)],
        "home_team": ["Team A"] * n,
        "away_team": ["Team B"] * n,
        "home_team_id": ["100"] * n,
        "away_team_id": ["200"] * n,
        "game_type": ["regular_season"] * n,
        "home_win_pct_before": [0.6 - (i % 3) * 0.1 for i in range(n)],
        "away_win_pct_before": [0.4 + (i % 3) * 0.1 for i in range(n)],
        "win_pct_diff_before": [0.2 - (i % 3) * 0.1 for i in range(n)],
        TARGET_COLUMN: [1 if i % 2 == 0 else 0 for i in range(n)],
    })


FEATURE_COLS = ["home_win_pct_before", "away_win_pct_before", "win_pct_diff_before"]


def test_required_output_columns_exist():
    df = _features_df(2)
    out = build_pregame_prediction_output(df, [0.7, 0.3])
    assert list(out.columns) == PREGAME_OUTPUT_COLUMNS


def test_away_probability_is_complement():
    out = build_pregame_prediction_output(_features_df(1), [0.65])
    assert out.loc[0, "away_win_probability"] == pytest.approx(0.35)


def test_predicted_label_threshold():
    out = build_pregame_prediction_output(_features_df(2), [0.6, 0.4])
    assert out.loc[0, "predicted_label"] == 1
    assert out.loc[1, "predicted_label"] == 0


def test_predicted_winner():
    out = build_pregame_prediction_output(_features_df(2), [0.8, 0.2])
    assert out.loc[0, "predicted_winner"] == "Team A"
    assert out.loc[1, "predicted_winner"] == "Team B"


def test_prediction_correct_when_actual_labels_exist():
    out = build_pregame_prediction_output(_features_df(2), [0.8, 0.2])
    assert out.loc[0, "prediction_correct"] is True or out.loc[0, "prediction_correct"] == True  # noqa: E712
    assert out.loc[1, "prediction_correct"] is True or out.loc[1, "prediction_correct"] == True  # noqa: E712


def test_prediction_correct_nan_without_actual():
    df = _features_df(1).drop(columns=[TARGET_COLUMN])
    out = build_pregame_prediction_output(df, [0.5])
    assert pd.isna(out.loc[0, "prediction_correct"])
    assert pd.isna(out.loc[0, "actual_home_team_won"])


def test_missing_feature_column_raises():
    df = _features_df(1).drop(columns=["home_win_pct_before"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_pregame_prediction_inputs(df, FEATURE_COLS)


def test_forbidden_target_not_in_feature_list():
    with pytest.raises(ValueError, match="leakage"):
        validate_pregame_prediction_inputs(_features_df(1), FEATURE_COLS + [TARGET_COLUMN])


def test_generate_predictions_validates_probabilities():
    model = _MockModel([0.5, 0.9])
    probs = generate_pregame_predictions(model, _features_df(2), FEATURE_COLS)
    assert len(probs) == 2
    assert probs[1] == pytest.approx(0.9)


def test_probability_validation_rejects_null():
    with pytest.raises(ValueError, match="null"):
        validate_prediction_probabilities([0.5, np.nan])


def test_probability_validation_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        validate_prediction_probabilities([0.5, 1.5])
