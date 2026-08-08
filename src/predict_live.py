"""Generate live win-probability predictions.

Loads the saved live model and feature-column dictionary, scores every row in
``data/processed/live_features.csv``, and writes a dashboard-ready CSV to
``data/processed/live_predictions.csv``.

This module does NOT train models, call ``nba_api``, or use ``home_team_won`` as
a model input — that column appears in the output only as an optional actual
label for evaluation/reference.

Run directly:
    python src/predict_live.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import (  # noqa: E402
    validate_no_target_leakage,
    validate_prediction_probabilities,
    validate_required_columns,
)
from src.utils import ensure_directories, save_csv  # noqa: E402

TARGET_COLUMN = "home_team_won"

FORBIDDEN_INPUT_COLUMNS = [
    TARGET_COLUMN,
    "final_home_score",
    "final_away_score",
    "winner",
]

LIVE_OUTPUT_COLUMNS = [
    "game_id",
    "event_num",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "period",
    "pctimestring",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "home_score",
    "away_score",
    "score_margin_home",
    "abs_score_margin",
    "event_type_label",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "predicted_label",
    "actual_home_team_won",
    "prediction_correct",
]


def normalize_live_feature_columns(
    artifact: Union[List[str], dict],
) -> Dict[str, List[str]]:
    """Return feature-column dict from the saved artifact.

    ``src/train_live_model.py`` saves a dict with ``numeric_features``,
    ``categorical_features``, and ``all_features``.  A plain list is accepted
    as ``all_features`` only.
    """
    if isinstance(artifact, dict):
        if "all_features" not in artifact:
            raise ValueError(
                "live_feature_columns.pkl dict must contain 'all_features'."
            )
        return {
            "numeric_features": list(artifact.get("numeric_features", [])),
            "categorical_features": list(artifact.get("categorical_features", [])),
            "all_features": list(artifact["all_features"]),
        }
    cols = list(artifact)
    return {
        "numeric_features": cols,
        "categorical_features": [],
        "all_features": cols,
    }


def load_live_prediction_inputs(
    features_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, object, Dict[str, List[str]]]:
    """Load live features, the fitted model, and the feature-column dict.

    Returns:
        A ``(features_df, model, feature_column_dict)`` tuple.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    features_path = features_path or config.LIVE_FEATURES_PATH
    model_path = model_path or config.LIVE_MODEL_PATH
    feature_columns_path = feature_columns_path or config.LIVE_FEATURE_COLUMNS_PATH

    for path, label in [
        (features_path, "Live features"),
        (model_path, "Live model"),
        (feature_columns_path, "Live feature columns"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    df = pd.read_csv(
        features_path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )
    if "event_num" in df.columns:
        df["event_num"] = pd.to_numeric(df["event_num"], errors="coerce")

    model = joblib.load(model_path)
    feature_columns = normalize_live_feature_columns(
        joblib.load(feature_columns_path)
    )
    return df, model, feature_columns


def validate_live_prediction_inputs(
    df: pd.DataFrame,
    feature_columns: Union[Sequence[str], Dict[str, List[str]]],
) -> None:
    """Ensure all model feature columns exist and no forbidden columns are used."""
    if isinstance(feature_columns, dict):
        cols = feature_columns["all_features"]
    else:
        cols = list(feature_columns)

    validate_required_columns(df, cols, "live_features")
    validate_no_target_leakage(cols, FORBIDDEN_INPUT_COLUMNS)


def generate_live_predictions(
    model: object,
    df: pd.DataFrame,
    feature_columns: Union[Sequence[str], Dict[str, List[str]]],
) -> np.ndarray:
    """Return home-win probabilities for every live event row."""
    if isinstance(feature_columns, dict):
        cols = feature_columns["all_features"]
    else:
        cols = list(feature_columns)

    x = df[list(cols)]
    probabilities = model.predict_proba(x)[:, 1]
    validate_prediction_probabilities(probabilities)
    return np.asarray(probabilities, dtype=float)


def build_live_prediction_output(
    df: pd.DataFrame,
    probabilities: Sequence[float],
) -> pd.DataFrame:
    """Combine event metadata with model probabilities into the output schema."""
    home_prob = np.asarray(probabilities, dtype=float)
    away_prob = 1.0 - home_prob
    predicted_label = (home_prob >= 0.5).astype(int)

    def _col(name: str):
        return df[name] if name in df.columns else np.nan

    out = pd.DataFrame({
        "game_id": df["game_id"].astype(str),
        "event_num": _col("event_num"),
        "season": _col("season"),
        "game_date": _col("game_date"),
        "home_team": _col("home_team"),
        "away_team": _col("away_team"),
        "period": _col("period"),
        "pctimestring": _col("pctimestring"),
        "seconds_remaining_period": _col("seconds_remaining_period"),
        "seconds_remaining_game": _col("seconds_remaining_game"),
        "home_score": _col("home_score"),
        "away_score": _col("away_score"),
        "score_margin_home": _col("score_margin_home"),
        "abs_score_margin": _col("abs_score_margin"),
        "event_type_label": _col("event_type_label"),
        "home_win_probability": home_prob,
        "away_win_probability": away_prob,
        "predicted_label": predicted_label,
    })

    out["predicted_winner"] = np.where(
        out["predicted_label"] == 1,
        out["home_team"],
        out["away_team"],
    )

    if TARGET_COLUMN in df.columns:
        actual = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        out["actual_home_team_won"] = actual
        out["prediction_correct"] = np.where(
            actual.notna(),
            out["predicted_label"] == actual.astype("Int64"),
            np.nan,
        )
    else:
        out["actual_home_team_won"] = np.nan
        out["prediction_correct"] = np.nan

    return out[LIVE_OUTPUT_COLUMNS]


def save_live_predictions(
    predictions: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Write live predictions to CSV."""
    output_path = output_path or config.LIVE_PREDICTIONS_PATH
    return save_csv(predictions, output_path)


def compute_accuracy_if_labels_exist(predictions: pd.DataFrame) -> Optional[float]:
    """Return accuracy when ``actual_home_team_won`` is present, else ``None``."""
    if "actual_home_team_won" not in predictions.columns:
        return None
    actual = predictions["actual_home_team_won"]
    mask = actual.notna()
    if mask.sum() == 0:
        return None
    return float((predictions.loc[mask, "predicted_label"] == actual[mask].astype(int)).mean())


def predict_live(
    features_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    """Load artifacts, score live features, and save predictions.

    Returns:
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    ensure_directories()

    features_path = features_path or config.LIVE_FEATURES_PATH
    model_path = model_path or config.LIVE_MODEL_PATH
    feature_columns_path = feature_columns_path or config.LIVE_FEATURE_COLUMNS_PATH
    output_path = output_path or config.LIVE_PREDICTIONS_PATH

    print(f"  Input features path:         {features_path}")
    print(f"  Model path:                  {model_path}")
    print(f"  Feature columns path:        {feature_columns_path}")

    try:
        df, model, feature_columns = load_live_prediction_inputs(
            features_path=features_path,
            model_path=model_path,
            feature_columns_path=feature_columns_path,
        )
    except FileNotFoundError as exc:
        print(f"  [error] {exc}")
        return 1

    validate_live_prediction_inputs(df, feature_columns)

    if df.empty:
        print("  [error] No rows to score. Nothing saved.")
        return 1

    probabilities = generate_live_predictions(model, df, feature_columns)
    predictions = build_live_prediction_output(df, probabilities)
    save_live_predictions(predictions, output_path)

    accuracy = compute_accuracy_if_labels_exist(predictions)
    prob_min = float(predictions["home_win_probability"].min())
    prob_max = float(predictions["home_win_probability"].max())
    n_games = predictions["game_id"].nunique()

    print(f"  Rows scored:                 {len(predictions)}")
    print(f"  Unique games scored:       {n_games}")
    print(f"  Probability range:           {prob_min:.4f} – {prob_max:.4f}")
    print(f"  Output path:                 {output_path}")
    if accuracy is not None:
        print(f"  Prediction accuracy:         {accuracy:.4f} (actual labels present)")
    else:
        print("  Prediction accuracy:         n/a (no actual labels)")

    print("\nDone.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run live prediction."""
    parser = argparse.ArgumentParser(
        description="Generate live win-probability predictions."
    )
    parser.parse_args(argv)

    print("Generating live predictions...")
    return predict_live()


if __name__ == "__main__":
    sys.exit(main())
