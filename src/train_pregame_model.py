"""Train a baseline pre-game winner-prediction model (Build 7).

This is the pre-game **model training** layer.  It reads the leakage-safe
pre-game feature table produced by Build 5
(``data/processed/pregame_features.csv``), trains a baseline Logistic
Regression classifier inside a scikit-learn ``Pipeline``, evaluates it on a
chronological hold-out split, and saves the fitted model plus training
artifacts.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:
* It does not train the live model, build prediction scripts, or touch the
  Streamlit dashboard.
* It never calls ``nba_api``.
* It never uses ``home_team_won`` as an input feature — that column is the
  target label only.  Final scores and any same-game outcome columns are
  likewise excluded from the feature set (see :data:`FORBIDDEN_COLUMNS`).

Leakage protection:
* Only the curated pre-game columns in :data:`ALLOWED_NUMERIC_FEATURES` and
  :data:`ALLOWED_CATEGORICAL_FEATURES` are ever fed to the model.
* :func:`validate_no_leakage_columns` hard-fails if a forbidden column sneaks
  into the feature list.
* The train/test split is **chronological** (no shuffling), so the model is
  always evaluated on games that happened *after* the ones it trained on.

Run directly:
    python src/train_pregame_model.py
    python src/train_pregame_model.py --test-size 0.2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Allow running this file directly (``python src/train_pregame_model.py``) by
# making the project root importable — same pattern as the builders.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.data_validation import validate_no_target_leakage  # noqa: E402
from src.multiseason_training import split_by_train_test_seasons  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The target label.  NEVER a model input feature.
TARGET_COLUMN = "home_team_won"

# Curated leakage-safe numeric feature columns (all computed from strictly
# earlier games by the Build 5 feature builder).
ALLOWED_NUMERIC_FEATURES = [
    "home_games_played_before",
    "away_games_played_before",
    "home_wins_before",
    "away_wins_before",
    "home_losses_before",
    "away_losses_before",
    "home_win_pct_before",
    "away_win_pct_before",
    "win_pct_diff_before",
    "home_points_for_avg_before",
    "away_points_for_avg_before",
    "home_points_allowed_avg_before",
    "away_points_allowed_avg_before",
    "points_for_avg_diff_before",
    "points_allowed_avg_diff_before",
    "home_recent_win_pct_before",
    "away_recent_win_pct_before",
    "recent_win_pct_diff_before",
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
]

# Curated leakage-safe categorical feature columns.
ALLOWED_CATEGORICAL_FEATURES = ["game_type"]

# Columns that must NEVER be used as model inputs.  These either identify the
# game (and would let the model "memorise" rows) or leak the final outcome.
FORBIDDEN_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "home_team_won",
    "home_score",
    "away_score",
    "winner",
]

MODEL_TYPE = "logistic_regression"

# Columns for the saved metrics CSV, in output order.
METRICS_COLUMNS = [
    "model_type",
    "n_total_rows",
    "n_used_rows",
    "n_dropped_missing_target",
    "n_train",
    "n_test",
    "n_features",
    "accuracy",
    "roc_auc",
    "log_loss",
    "brier_score",
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
]

CALIBRATION_COLUMNS = [
    "probability_bucket",
    "row_count",
    "avg_predicted_probability",
    "actual_home_win_rate",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_pregame_features(path: str | Path) -> pd.DataFrame:
    """Load the pre-game feature table, keeping ``game_id`` as a string.

    Args:
        path: Path to ``data/processed/pregame_features.csv``.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-game features not found: {path}. Run "
            "'python run_pipeline.py --mode build_pregame_features' first."
        )
    return pd.read_csv(
        path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )


# ---------------------------------------------------------------------------
# Feature selection + leakage guards
# ---------------------------------------------------------------------------

def get_pregame_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return the allowed pre-game feature columns that exist in ``df``.

    Only columns from :data:`ALLOWED_NUMERIC_FEATURES` and
    :data:`ALLOWED_CATEGORICAL_FEATURES` are considered, so forbidden /
    leaky columns can never be selected.  Numeric features come first, then
    categorical, preserving the curated order.

    Args:
        df: The pre-game feature DataFrame.

    Returns:
        The list of usable feature column names present in ``df``.
    """
    allowed = ALLOWED_NUMERIC_FEATURES + ALLOWED_CATEGORICAL_FEATURES
    return [col for col in allowed if col in df.columns]


def split_feature_types(
    feature_columns: Sequence[str],
) -> Tuple[List[str], List[str]]:
    """Split a feature list into (numeric, categorical) by the allowed sets.

    Args:
        feature_columns: Columns selected by :func:`get_pregame_feature_columns`.

    Returns:
        A ``(numeric_features, categorical_features)`` tuple.
    """
    numeric = [c for c in feature_columns if c in ALLOWED_NUMERIC_FEATURES]
    categorical = [c for c in feature_columns if c in ALLOWED_CATEGORICAL_FEATURES]
    return numeric, categorical


def validate_no_leakage_columns(feature_columns: Sequence[str]) -> None:
    """Raise if any forbidden (leaky) column is present in ``feature_columns``.

    Args:
        feature_columns: The proposed model-input columns.

    Raises:
        ValueError: If any column in :data:`FORBIDDEN_COLUMNS` is included
            (e.g. the target ``home_team_won`` or a final score).
    """
    validate_no_target_leakage(feature_columns, FORBIDDEN_COLUMNS)


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------

def prepare_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing target and cast the target to ``int``.

    Rows where ``home_team_won`` is null cannot be used for supervised
    training, so they are removed (never fabricated).

    Args:
        df: The pre-game feature DataFrame.

    Returns:
        A new DataFrame containing only rows with a usable target.

    Raises:
        ValueError: If the target column is missing entirely.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in features.")

    clean = df[df[TARGET_COLUMN].notna()].copy()
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(int)
    return clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Chronological split (no shuffle)
# ---------------------------------------------------------------------------

def chronological_train_test_split(
    df: pd.DataFrame,
    date_col: str = "game_date",
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train/test by date order, without shuffling.

    The data is sorted ascending by ``date_col`` (with ``game_id`` as a stable
    tie-breaker when present).  The earliest ``1 - test_size`` fraction becomes
    the training set and the most recent ``test_size`` fraction becomes the test
    set, so the model is always evaluated on *later* games.

    Args:
        df: The data to split (must contain ``date_col``).
        date_col: Name of the date column to sort on.
        test_size: Fraction of the most-recent rows to hold out for testing.

    Returns:
        A ``(train_df, test_df)`` tuple, both sorted ascending by date.

    Raises:
        ValueError: If ``date_col`` is missing or ``test_size`` is not in (0, 1).
    """
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found for chronological split.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    sort_cols = [date_col] + (["game_id"] if "game_id" in df.columns else [])
    ordered = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    n = len(ordered)
    n_test = int(round(n * test_size))
    # Keep at least one row on each side when there are >= 2 rows.
    n_test = max(1, min(n_test, n - 1)) if n >= 2 else 0
    split_idx = n - n_test

    train_df = ordered.iloc[:split_idx].reset_index(drop=True)
    test_df = ordered.iloc[split_idx:].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------

def build_logistic_regression_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    """Build the baseline Logistic Regression pipeline.

    Numeric features are median-imputed and standardised; categorical features
    are most-frequent-imputed and one-hot encoded (unknown categories at predict
    time are ignored).  The final estimator is a Logistic Regression.

    Args:
        numeric_features: Numeric input column names.
        categorical_features: Categorical input column names.

    Returns:
        An unfitted scikit-learn :class:`~sklearn.pipeline.Pipeline`.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, list(numeric_features)))
    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, list(categorical_features)))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> dict:
    """Compute classification metrics for the home-win target.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities of the home team winning.

    Returns:
        A dict with ``accuracy``, ``roc_auc``, ``log_loss``, ``brier_score`` and
        confusion-matrix counts (``true_positive``, ``true_negative``,
        ``false_positive``, ``false_negative``).  ``roc_auc`` is ``NaN`` (with a
        warning) when the test set contains only one class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)

    accuracy = float(accuracy_score(y_true, y_pred))

    # ROC-AUC is undefined when only one class is present in y_true.
    if len(np.unique(y_true)) < 2:
        warnings.warn(
            "ROC-AUC is undefined because the test set contains only one class; "
            "writing NaN.",
            stacklevel=2,
        )
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))

    # log_loss / brier are well-defined even with one class when labels are
    # pinned to [0, 1].
    ll = float(log_loss(y_true, y_prob, labels=[0, 1]))
    brier = float(brier_score_loss(y_true, y_prob))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "log_loss": ll,
        "brier_score": brier,
        "true_positive": int(tp),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
    }


def build_calibration_table(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    n_bins: int = 10,
) -> pd.DataFrame:
    """Summarise predicted probability vs actual home-win rate, in buckets.

    Predicted probabilities are grouped into ``n_bins`` equal-width buckets over
    ``[0, 1]``.  Empty buckets are omitted.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities of the home team winning.
        n_bins: Number of equal-width probability buckets.

    Returns:
        A DataFrame with :data:`CALIBRATION_COLUMNS`.
    """
    frame = pd.DataFrame({"y_true": np.asarray(y_true, dtype=float),
                          "y_prob": np.asarray(y_prob, dtype=float)})
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    labels = [f"{edges[i]:.1f}-{edges[i + 1]:.1f}" for i in range(n_bins)]
    frame["bucket"] = pd.cut(
        frame["y_prob"], bins=edges, labels=labels, include_lowest=True
    )

    rows: List[dict] = []
    for label in labels:
        bucket = frame[frame["bucket"] == label]
        if bucket.empty:
            continue
        rows.append({
            "probability_bucket": label,
            "row_count": int(len(bucket)),
            "avg_predicted_probability": float(bucket["y_prob"].mean()),
            "actual_home_win_rate": float(bucket["y_true"].mean()),
        })

    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def train_pregame_model(
    test_size: float = 0.2,
    train_seasons: Optional[List[str]] = None,
    test_season: Optional[str] = None,
    input_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
) -> int:
    """Train, evaluate, and save the baseline pre-game model.

    When ``train_seasons`` and ``test_season`` are both provided, the hold-out
    split is by season label instead of chronological date order.

    Args:
        test_size: Fraction of most-recent games held out (single-season mode).
        train_seasons: Seasons used for training (multi-season mode).
        test_season: Season held out for testing (multi-season mode).
        input_path: Source features. Defaults to :data:`config.PREGAME_FEATURES_PATH`.
        model_path: Where to save the pipeline. Defaults to
            :data:`config.PREGAME_MODEL_PATH`.
        feature_columns_path: Where to save the feature-column list. Defaults to
            :data:`config.PREGAME_FEATURE_COLUMNS_PATH`.
        metrics_path: Where to save metrics. Defaults to
            :data:`config.PREGAME_MODEL_METRICS_PATH`.
        calibration_path: Where to save the calibration table. Defaults to
            :data:`config.PREGAME_MODEL_CALIBRATION_PATH`.

    Returns:
        Process exit code: ``0`` on success, ``1`` if training cannot proceed.
    """
    ensure_directories()

    input_path = input_path or config.PREGAME_FEATURES_PATH
    model_path = model_path or config.PREGAME_MODEL_PATH
    feature_columns_path = feature_columns_path or config.PREGAME_FEATURE_COLUMNS_PATH
    metrics_path = metrics_path or config.PREGAME_MODEL_METRICS_PATH
    calibration_path = calibration_path or config.PREGAME_MODEL_CALIBRATION_PATH

    print(f"  Input path:                  {input_path}")

    df = load_pregame_features(input_path)
    n_total = len(df)
    print(f"  Total rows loaded:           {n_total}")

    used = prepare_training_data(df)
    n_used = len(used)
    n_dropped = n_total - n_used
    print(f"  Rows used (target present):  {n_used}")
    print(f"  Rows dropped (missing target):{n_dropped:>4}")

    if n_used < 2:
        print("  [error] Not enough labelled rows to train a model. Nothing saved.")
        return 1

    if (train_seasons is None) ^ (test_season is None):
        print("  [error] Provide both train_seasons and test_season for season split.")
        return 1

    feature_columns = get_pregame_feature_columns(used)
    validate_no_leakage_columns(feature_columns)
    numeric_features, categorical_features = split_feature_types(feature_columns)
    print(f"  Feature columns used ({len(feature_columns)}):")
    print(f"    numeric ({len(numeric_features)}):     {numeric_features}")
    print(f"    categorical ({len(categorical_features)}): {categorical_features}")

    if train_seasons is not None and test_season is not None:
        train_df, test_df = split_by_train_test_seasons(used, train_seasons, test_season)
        print(f"  Season split — train: {train_seasons}, test: {test_season}")
    else:
        train_df, test_df = chronological_train_test_split(used, test_size=test_size)
    print(f"  Train rows:                  {len(train_df)}")
    print(f"  Test rows:                   {len(test_df)}")

    if test_df.empty or train_df.empty:
        print("  [error] Train or test split is empty. Nothing saved.")
        return 1

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(int)

    pipeline = build_logistic_regression_pipeline(numeric_features, categorical_features)
    pipeline.fit(x_train, y_train)
    print(f"  Model type:                  {MODEL_TYPE}")

    y_pred = pipeline.predict(x_test)
    y_prob = pipeline.predict_proba(x_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_prob)
    metrics_row = {
        "model_type": MODEL_TYPE,
        "n_total_rows": n_total,
        "n_used_rows": n_used,
        "n_dropped_missing_target": n_dropped,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": len(feature_columns),
        **metrics,
    }

    print("\n  Metrics (test set):")
    print(f"    accuracy:    {metrics['accuracy']:.4f}")
    print(f"    roc_auc:     {metrics['roc_auc']:.4f}")
    print(f"    log_loss:    {metrics['log_loss']:.4f}")
    print(f"    brier_score: {metrics['brier_score']:.4f}")
    print(f"    confusion:   TP={metrics['true_positive']} TN={metrics['true_negative']} "
          f"FP={metrics['false_positive']} FN={metrics['false_negative']}")

    calibration = build_calibration_table(y_test, y_prob, n_bins=10)

    # Persist artifacts.
    metrics_df = pd.DataFrame([metrics_row], columns=METRICS_COLUMNS)
    save_csv(metrics_df, metrics_path)
    save_csv(calibration, calibration_path)
    joblib.dump(pipeline, model_path)
    joblib.dump(feature_columns, feature_columns_path)

    print("\nDone.")
    print(f"  Model artifact:              {model_path}")
    print(f"  Feature columns artifact:    {feature_columns_path}")
    print(f"  Metrics output:              {metrics_path}")
    print(f"  Calibration output:          {calibration_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run pre-game model training."""
    parser = argparse.ArgumentParser(
        description="Train a baseline pre-game winner-prediction model from "
        "data/processed/pregame_features.csv."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of most-recent games held out for testing (default 0.2).",
    )
    args = parser.parse_args(argv)

    print("Training pre-game model (Logistic Regression baseline)...")
    return train_pregame_model(test_size=args.test_size)


if __name__ == "__main__":
    sys.exit(main())
