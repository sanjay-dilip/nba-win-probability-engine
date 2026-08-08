"""Generate pre-game win-probability predictions.

Loads the saved pre-game model and feature-column artifact, scores every row in
``data/processed/pregame_features.csv``, and writes a dashboard-ready CSV to
``data/processed/pregame_predictions.csv``.

This module does NOT train models, call ``nba_api``, or use ``home_team_won`` as
a model input — that column appears in the output only as an optional actual
label for evaluation/reference.

Run directly:
    python src/predict_pregame.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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

# The target label — never a model input.
TARGET_COLUMN = "home_team_won"

FORBIDDEN_INPUT_COLUMNS = [TARGET_COLUMN, "home_score", "away_score", "winner"]

PREGAME_OUTPUT_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "game_type",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "predicted_label",
    "actual_home_team_won",
    "prediction_correct",
]


def normalize_pregame_feature_columns(
    artifact: Union[List[str], dict],
) -> List[str]:
    """Return a flat feature-column list from the saved artifact.

    ``src/train_pregame_model.py`` saves a plain ``list``; accept a dict with
    ``all_features`` too.
    """
    if isinstance(artifact, dict):
        if "all_features" in artifact:
            return list(artifact["all_features"])
        raise ValueError(
            "pregame_feature_columns.pkl dict must contain 'all_features'."
        )
    return list(artifact)


def load_pregame_prediction_inputs(
    features_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, object, List[str]]:
    """Load features, the fitted model, and the feature-column list.

    Returns:
        A ``(features_df, model, feature_columns)`` tuple.

    Raises:
        FileNotFoundError: If any required file is missing.
    """
    features_path = features_path or config.PREGAME_FEATURES_PATH
    model_path = model_path or config.PREGAME_MODEL_PATH
    feature_columns_path = feature_columns_path or config.PREGAME_FEATURE_COLUMNS_PATH

    for path, label in [
        (features_path, "Pre-game features"),
        (model_path, "Pre-game model"),
        (feature_columns_path, "Pre-game feature columns"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    df = pd.read_csv(
        features_path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )
    model = joblib.load(model_path)
    feature_columns = normalize_pregame_feature_columns(
        joblib.load(feature_columns_path)
    )
    return df, model, feature_columns


def validate_pregame_prediction_inputs(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> None:
    """Ensure all model feature columns exist and no forbidden columns are used.

    Raises:
        ValueError: If validation fails.
    """
    validate_required_columns(df, feature_columns, "pregame_features")
    validate_no_target_leakage(feature_columns, FORBIDDEN_INPUT_COLUMNS)


def generate_pregame_predictions(
    model: object,
    df: pd.DataFrame,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """Return home-win probabilities for every feature row.

    Args:
        model: A fitted scikit-learn pipeline with ``predict_proba``.
        df: Pre-game feature rows.
        feature_columns: Columns to pass to the model (no target leakage).

    Returns:
        A 1-D array of home-win probabilities (class 1).
    """
    x = df[list(feature_columns)]
    probabilities = model.predict_proba(x)[:, 1]
    validate_prediction_probabilities(probabilities)
    return np.asarray(probabilities, dtype=float)


def build_pregame_prediction_output(
    df: pd.DataFrame,
    probabilities: Sequence[float],
) -> pd.DataFrame:
    """Combine metadata with model probabilities into the output schema."""
    home_prob = np.asarray(probabilities, dtype=float)
    away_prob = 1.0 - home_prob
    predicted_label = (home_prob >= 0.5).astype(int)

    out = pd.DataFrame({
        "game_id": df["game_id"].astype(str),
        "season": df["season"] if "season" in df.columns else np.nan,
        "game_date": df["game_date"] if "game_date" in df.columns else np.nan,
        "home_team": df["home_team"] if "home_team" in df.columns else np.nan,
        "away_team": df["away_team"] if "away_team" in df.columns else np.nan,
        "home_team_id": df["home_team_id"].astype(str) if "home_team_id" in df.columns else np.nan,
        "away_team_id": df["away_team_id"].astype(str) if "away_team_id" in df.columns else np.nan,
        "game_type": df["game_type"] if "game_type" in df.columns else np.nan,
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

    return out[PREGAME_OUTPUT_COLUMNS]


def save_pregame_predictions(
    predictions: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> Path:
    """Write pre-game predictions to CSV."""
    output_path = output_path or config.PREGAME_PREDICTIONS_PATH
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


def predict_pregame(
    features_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    """Load artifacts, score pre-game features, and save predictions.

    Returns:
        Process exit code: ``0`` on success, ``1`` on failure.
    """
    ensure_directories()

    features_path = features_path or config.PREGAME_FEATURES_PATH
    model_path = model_path or config.PREGAME_MODEL_PATH
    feature_columns_path = feature_columns_path or config.PREGAME_FEATURE_COLUMNS_PATH
    output_path = output_path or config.PREGAME_PREDICTIONS_PATH

    print(f"  Input features path:         {features_path}")
    print(f"  Model path:                  {model_path}")
    print(f"  Feature columns path:        {feature_columns_path}")

    try:
        df, model, feature_columns = load_pregame_prediction_inputs(
            features_path=features_path,
            model_path=model_path,
            feature_columns_path=feature_columns_path,
        )
    except FileNotFoundError as exc:
        print(f"  [error] {exc}")
        return 1

    validate_pregame_prediction_inputs(df, feature_columns)

    if df.empty:
        print("  [error] No rows to score. Nothing saved.")
        return 1

    probabilities = generate_pregame_predictions(model, df, feature_columns)
    predictions = build_pregame_prediction_output(df, probabilities)
    save_pregame_predictions(predictions, output_path)

    accuracy = compute_accuracy_if_labels_exist(predictions)
    prob_min = float(predictions["home_win_probability"].min())
    prob_max = float(predictions["home_win_probability"].max())

    print(f"  Rows scored:                 {len(predictions)}")
    print(f"  Probability range:           {prob_min:.4f} – {prob_max:.4f}")
    print(f"  Output path:                 {output_path}")
    if accuracy is not None:
        print(f"  Prediction accuracy:         {accuracy:.4f} (actual labels present)")
    else:
        print("  Prediction accuracy:         n/a (no actual labels)")

    print("\nDone.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run pre-game prediction."""
    parser = argparse.ArgumentParser(
        description="Generate pre-game win-probability predictions."
    )
    parser.parse_args(argv)

    print("Generating pre-game predictions...")
    return predict_pregame()


if __name__ == "__main__":
    sys.exit(main())
