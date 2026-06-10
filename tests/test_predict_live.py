"""Tests for live prediction helpers (Build 9).

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
from src.predict_live import (  # noqa: E402
    LIVE_OUTPUT_COLUMNS,
    TARGET_COLUMN,
    build_live_prediction_output,
    generate_live_predictions,
    normalize_live_feature_columns,
    validate_live_prediction_inputs,
)


class _MockModel:
    def __init__(self, probs: list[float]):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n = len(x)
        p = self._probs[:n] if len(self._probs) >= n else np.resize(self._probs, n)
        return np.column_stack([1.0 - p, p])


def _live_df(n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": ["0022400001"] * n,
        "event_num": list(range(1, n + 1)),
        "season": ["2024-25"] * n,
        "game_date": ["2024-11-01"] * n,
        "home_team": ["Team A"] * n,
        "away_team": ["Team B"] * n,
        "period": [min(4, (i // 2) + 1) for i in range(n)],
        "pctimestring": ["PT06M00.00S"] * n,
        "seconds_remaining_period": [360.0] * n,
        "seconds_remaining_game": [2880.0 - i * 100 for i in range(n)],
        "home_score": [float(i * 3) for i in range(n)],
        "away_score": [float(i * 2) for i in range(n)],
        "score_margin_home": [float(i) for i in range(n)],
        "abs_score_margin": [float(i) for i in range(n)],
        "event_type_label": ["made_shot"] * n,
        TARGET_COLUMN: [1] * n,
    })


FEATURE_DICT = {
    "numeric_features": ["seconds_remaining_game", "score_margin_home"],
    "categorical_features": ["event_type_label"],
    "all_features": ["seconds_remaining_game", "score_margin_home", "event_type_label"],
}


def test_normalize_feature_columns_dict():
    out = normalize_live_feature_columns(FEATURE_DICT)
    assert set(out.keys()) == {"numeric_features", "categorical_features", "all_features"}
    assert out["all_features"] == FEATURE_DICT["all_features"]


def test_required_output_columns_exist():
    out = build_live_prediction_output(_live_df(2), [0.7, 0.3])
    assert list(out.columns) == LIVE_OUTPUT_COLUMNS


def test_away_probability_is_complement():
    out = build_live_prediction_output(_live_df(1), [0.65])
    assert out.loc[0, "away_win_probability"] == pytest.approx(0.35)


def test_predicted_label_threshold():
    out = build_live_prediction_output(_live_df(2), [0.55, 0.45])
    assert out.loc[0, "predicted_label"] == 1
    assert out.loc[1, "predicted_label"] == 0


def test_predicted_winner():
    out = build_live_prediction_output(_live_df(1), [0.9])
    assert out.loc[0, "predicted_winner"] == "Team A"


def test_prediction_correct_when_actual_labels_exist():
    out = build_live_prediction_output(_live_df(1), [0.9])
    assert out.loc[0, "prediction_correct"] == True  # noqa: E712


def test_missing_feature_column_raises():
    df = _live_df(1).drop(columns=["event_type_label"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_live_prediction_inputs(df, FEATURE_DICT)


def test_forbidden_target_not_in_feature_list():
    bad = {**FEATURE_DICT, "all_features": FEATURE_DICT["all_features"] + [TARGET_COLUMN]}
    with pytest.raises(ValueError, match="leakage"):
        validate_live_prediction_inputs(_live_df(1), bad)


def test_generate_predictions_with_feature_dict():
    model = _MockModel([0.6, 0.4])
    probs = generate_live_predictions(model, _live_df(2), FEATURE_DICT)
    assert len(probs) == 2


def test_probability_validation_rejects_null():
    with pytest.raises(ValueError, match="null"):
        validate_prediction_probabilities([0.5, np.nan])


def test_probability_validation_rejects_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        validate_prediction_probabilities([-0.1])


def test_game_id_stays_string_in_output():
    out = build_live_prediction_output(_live_df(1), [0.5])
    assert out["game_id"].iloc[0] == "0022400001"
    assert isinstance(out["game_id"].iloc[0], str)
