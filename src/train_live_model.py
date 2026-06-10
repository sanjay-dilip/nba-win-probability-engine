"""Train a baseline live win-probability model (Build 8).

This is the live **model training** layer.  It reads the per-event live-feature
table produced by Build 6 (``data/processed/live_features.csv``), trains a
baseline Logistic Regression classifier inside a scikit-learn ``Pipeline``,
evaluates it on a **game-level chronological** hold-out split, and saves the
fitted model plus training artifacts.

WHAT THIS MODULE DELIBERATELY DOES NOT DO:
* It does not retrain the pre-game model, build prediction scripts, write
  prediction output files, or touch the Streamlit dashboard.
* It never calls ``nba_api``.
* It never uses ``home_team_won`` as an input feature — that column is the
  target label only.  Final-outcome columns (``winner`` etc.) are excluded too
  (see :data:`FORBIDDEN_COLUMNS`).

Leakage protection:
* Only live game-state columns that are known *as of each event* are fed to the
  model (running score, margin, clock, event type/flags).  The running
  ``home_score`` / ``away_score`` are legitimate live state, not final scores.
* :func:`validate_no_live_leakage_columns` hard-fails if a forbidden column
  sneaks into the feature list.
* The split is **by game** and **chronological**: every event row of a game
  goes to the same side, and no ``game_id`` appears in both train and test, so
  the model never sees any event of a game it is evaluated on.

Run directly:
    python src/train_live_model.py
    python src/train_live_model.py --test-size 0.2
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

# Allow running this file directly by making the project root importable.
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

# Candidate numeric live features (known as of each event).  Two of these
# (event_msg_type / event_msg_action_type) may arrive as strings depending on
# the source; when they are not numeric they are routed to the categorical set.
ALLOWED_NUMERIC_CANDIDATES = [
    "period",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "home_score",
    "away_score",
    "score_margin_home",
    "abs_score_margin",
    "event_msg_type",
    "event_msg_action_type",
    "is_scoring_event",
    "is_turnover",
    "is_foul",
    "is_timeout",
    "is_rebound",
    "is_free_throw",
    "is_field_goal_attempt",
]

# Columns that are always treated as categorical when present.
ALLOWED_CATEGORICAL_BASE = ["event_type_label", "game_type"]

# Candidates whose type is decided at runtime (numeric -> numeric set, otherwise
# categorical set).
TYPE_AMBIGUOUS_COLUMNS = {"event_msg_type", "event_msg_action_type"}

# Columns that must NEVER be used as model inputs (ids, metadata, and any final
# outcome).  Note: running home_score/away_score are allowed live state and are
# intentionally NOT here.
FORBIDDEN_COLUMNS = [
    "game_id",
    "event_num",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "home_team_won",
    "final_home_score",
    "final_away_score",
    "winner",
]

MODEL_TYPE = "logistic_regression"

# Game-phase boundaries (seconds remaining in regulation game).
EARLY_GAME_MIN_SECONDS = 2160   # > 36:00 remaining
LATE_GAME_MAX_SECONDS = 720     # <= 12:00 remaining

METRICS_COLUMNS = [
    "model_type",
    "n_total_rows",
    "n_used_rows",
    "n_dropped_missing_target",
    "n_unique_games",
    "train_games",
    "test_games",
    "train_rows",
    "test_rows",
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

PHASE_METRICS_COLUMNS = [
    "phase",
    "row_count",
    "accuracy",
    "log_loss",
    "brier_score",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_live_features(path: str | Path) -> pd.DataFrame:
    """Load the live-feature table, keeping ``game_id`` as a string.

    Args:
        path: Path to ``data/processed/live_features.csv``.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Live features not found: {path}. Run "
            "'python run_pipeline.py --mode build_live_features' first."
        )
    return pd.read_csv(
        path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )


# ---------------------------------------------------------------------------
# Feature selection + leakage guards
# ---------------------------------------------------------------------------

def get_live_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return the allowed live feature columns that exist in ``df``.

    Splits features into numeric and categorical.  The ambiguous columns
    ``event_msg_type`` / ``event_msg_action_type`` are placed in the numeric set
    only when their dtype is genuinely numeric; otherwise they are treated as
    categorical (handled safely either way).

    Args:
        df: The live-feature DataFrame.

    Returns:
        A dict with ``numeric_features``, ``categorical_features`` and
        ``all_features`` (numeric first, then categorical).
    """
    numeric: List[str] = []
    categorical: List[str] = []

    for col in ALLOWED_NUMERIC_CANDIDATES:
        if col not in df.columns:
            continue
        if col in TYPE_AMBIGUOUS_COLUMNS and not pd.api.types.is_numeric_dtype(df[col]):
            categorical.append(col)
        else:
            numeric.append(col)

    for col in ALLOWED_CATEGORICAL_BASE:
        if col in df.columns:
            categorical.append(col)

    return {
        "numeric_features": numeric,
        "categorical_features": categorical,
        "all_features": numeric + categorical,
    }


def validate_no_live_leakage_columns(feature_columns: Sequence[str]) -> None:
    """Raise if any forbidden (leaky) column is present in ``feature_columns``.

    Args:
        feature_columns: The proposed model-input columns.

    Raises:
        ValueError: If any column in :data:`FORBIDDEN_COLUMNS` is included
            (e.g. the target ``home_team_won`` or ``winner``).
    """
    validate_no_target_leakage(feature_columns, FORBIDDEN_COLUMNS)


# ---------------------------------------------------------------------------
# Target preparation
# ---------------------------------------------------------------------------

def prepare_live_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing target and cast the target to ``int``.

    Args:
        df: The live-feature DataFrame.

    Returns:
        A new DataFrame containing only rows with a usable target.

    Raises:
        ValueError: If the target column is missing entirely.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in live features.")

    clean = df[df[TARGET_COLUMN].notna()].copy()
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(int)
    return clean.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Game-level chronological split (no shuffle, no game in both sides)
# ---------------------------------------------------------------------------

def chronological_game_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_col: str = "game_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split event rows by *game*, chronologically, without shuffling.

    Unique games are ordered by ``date_col`` then ``game_id``; the earliest
    ``1 - test_size`` fraction of games form the training set and the most recent
    ``test_size`` fraction form the test set.  Every event row of a game is
    assigned to that game's side, so no ``game_id`` can appear in both sets.

    Args:
        df: Event-level live features (must contain ``game_id`` and ``date_col``).
        test_size: Fraction of the most-recent *games* held out for testing.
        date_col: Name of the date column used to order games.

    Returns:
        A ``(train_df, test_df)`` tuple of event rows.

    Raises:
        ValueError: If required columns are missing or ``test_size`` is invalid.
    """
    if "game_id" not in df.columns:
        raise ValueError("game_id column is required for a game-level split.")
    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found for chronological split.")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}.")

    # One row per game, ordered chronologically (game_id breaks ties stably).
    game_order = (
        df[["game_id", date_col]]
        .drop_duplicates(subset="game_id")
        .sort_values([date_col, "game_id"], kind="mergesort")
        .reset_index(drop=True)
    )

    n_games = len(game_order)
    n_test = int(round(n_games * test_size))
    n_test = max(1, min(n_test, n_games - 1)) if n_games >= 2 else 0
    split_idx = n_games - n_test

    train_games = set(game_order["game_id"].iloc[:split_idx])
    test_games = set(game_order["game_id"].iloc[split_idx:])

    train_df = df[df["game_id"].isin(train_games)].copy().reset_index(drop=True)
    test_df = df[df["game_id"].isin(test_games)].copy().reset_index(drop=True)
    return train_df, test_df


def validate_game_split(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Raise if any ``game_id`` appears in both the train and test sets.

    Args:
        train_df: Training event rows.
        test_df: Test event rows.

    Raises:
        ValueError: If the train/test game sets overlap.
    """
    overlap = set(train_df["game_id"]) & set(test_df["game_id"])
    if overlap:
        raise ValueError(
            f"Game-level split leak: {len(overlap)} game_id(s) appear in both "
            f"train and test (e.g. {sorted(overlap)[:5]})."
        )


# ---------------------------------------------------------------------------
# Model pipeline
# ---------------------------------------------------------------------------

def build_live_logistic_regression_pipeline(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Pipeline:
    """Build the baseline live Logistic Regression pipeline.

    Numeric features are median-imputed and standardised; categorical features
    are most-frequent-imputed and one-hot encoded (unknown categories ignored at
    predict time).  The final estimator is a Logistic Regression with a high
    iteration cap (the live dataset is large).

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
        confusion-matrix counts.  ``roc_auc`` is ``NaN`` (with a warning) when the
        evaluation set contains only one class.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)

    accuracy = float(accuracy_score(y_true, y_pred))

    if len(np.unique(y_true)) < 2:
        warnings.warn(
            "ROC-AUC is undefined because the evaluation set contains only one "
            "class; writing NaN.",
            stacklevel=2,
        )
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))

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

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities of the home team winning.
        n_bins: Number of equal-width probability buckets over ``[0, 1]``.

    Returns:
        A DataFrame with :data:`CALIBRATION_COLUMNS` (empty buckets omitted).
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


def _phase_label(seconds_remaining_game: float) -> str:
    """Bucket one ``seconds_remaining_game`` value into a game phase."""
    if pd.isna(seconds_remaining_game):
        return "unknown"
    if seconds_remaining_game > EARLY_GAME_MIN_SECONDS:
        return "early_game"
    if seconds_remaining_game <= LATE_GAME_MAX_SECONDS:
        return "late_game"
    return "mid_game"


def build_phase_metrics_table(
    seconds_remaining_game: Sequence[float],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Sequence[float],
) -> pd.DataFrame:
    """Compute accuracy / log loss / Brier per game phase on the test rows.

    Phases (by ``seconds_remaining_game``): ``early_game`` (> 2160),
    ``mid_game`` (720–2160), ``late_game`` (<= 720).  ``log_loss`` is ``NaN`` for
    a phase whose rows are all one class.

    Args:
        seconds_remaining_game: Per-row seconds remaining in the game.
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities of the home team winning.

    Returns:
        A DataFrame with :data:`PHASE_METRICS_COLUMNS`, one row per non-empty
        phase, ordered early -> mid -> late.
    """
    frame = pd.DataFrame({
        "secs": np.asarray(seconds_remaining_game, dtype=float),
        "y_true": np.asarray(y_true, dtype=int),
        "y_pred": np.asarray(y_pred, dtype=int),
        "y_prob": np.asarray(y_prob, dtype=float),
    })
    frame["phase"] = frame["secs"].apply(_phase_label)

    order = {"early_game": 0, "mid_game": 1, "late_game": 2, "unknown": 3}
    rows: List[dict] = []
    for phase, grp in sorted(frame.groupby("phase"), key=lambda kv: order.get(kv[0], 99)):
        if grp.empty:
            continue
        if grp["y_true"].nunique() < 2:
            ll = float("nan")
        else:
            ll = float(log_loss(grp["y_true"], grp["y_prob"], labels=[0, 1]))
        rows.append({
            "phase": phase,
            "row_count": int(len(grp)),
            "accuracy": float(accuracy_score(grp["y_true"], grp["y_pred"])),
            "log_loss": ll,
            "brier_score": float(brier_score_loss(grp["y_true"], grp["y_prob"])),
        })

    return pd.DataFrame(rows, columns=PHASE_METRICS_COLUMNS)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def train_live_model(
    test_size: float = 0.2,
    train_seasons: Optional[List[str]] = None,
    test_season: Optional[str] = None,
    input_path: Optional[Path] = None,
    model_path: Optional[Path] = None,
    feature_columns_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    calibration_path: Optional[Path] = None,
    phase_metrics_path: Optional[Path] = None,
) -> int:
    """Train, evaluate, and save the baseline live win-probability model.

    When ``train_seasons`` and ``test_season`` are both provided, the hold-out
    split is by season (all event rows of a game stay on one side).

    Args:
        test_size: Fraction of most-recent games held out (single-season mode).
        train_seasons: Seasons used for training (multi-season mode).
        test_season: Season held out for testing (multi-season mode).
        input_path: Source features. Defaults to :data:`config.LIVE_FEATURES_PATH`.
        model_path: Pipeline output. Defaults to :data:`config.LIVE_MODEL_PATH`.
        feature_columns_path: Feature-dict output. Defaults to
            :data:`config.LIVE_FEATURE_COLUMNS_PATH`.
        metrics_path: Metrics output. Defaults to :data:`config.LIVE_MODEL_METRICS_PATH`.
        calibration_path: Calibration output. Defaults to
            :data:`config.LIVE_MODEL_CALIBRATION_PATH`.
        phase_metrics_path: Phase-metrics output. Defaults to
            :data:`config.LIVE_MODEL_PHASE_METRICS_PATH`.

    Returns:
        Process exit code: ``0`` on success, ``1`` if training cannot proceed.
    """
    ensure_directories()

    input_path = input_path or config.LIVE_FEATURES_PATH
    model_path = model_path or config.LIVE_MODEL_PATH
    feature_columns_path = feature_columns_path or config.LIVE_FEATURE_COLUMNS_PATH
    metrics_path = metrics_path or config.LIVE_MODEL_METRICS_PATH
    calibration_path = calibration_path or config.LIVE_MODEL_CALIBRATION_PATH
    phase_metrics_path = phase_metrics_path or config.LIVE_MODEL_PHASE_METRICS_PATH

    print(f"  Input path:                  {input_path}")

    df = load_live_features(input_path)
    n_total = len(df)
    print(f"  Total rows loaded:           {n_total}")

    used = prepare_live_training_data(df)
    n_used = len(used)
    n_dropped = n_total - n_used
    n_games = used["game_id"].nunique()
    print(f"  Rows used (target present):  {n_used}")
    print(f"  Rows dropped (missing target):{n_dropped:>4}")
    print(f"  Unique games used:           {n_games}")

    if n_used < 2 or n_games < 2:
        print("  [error] Not enough labelled rows/games to train. Nothing saved.")
        return 1

    if (train_seasons is None) ^ (test_season is None):
        print("  [error] Provide both train_seasons and test_season for season split.")
        return 1

    feature_dict = get_live_feature_columns(used)
    numeric_features = feature_dict["numeric_features"]
    categorical_features = feature_dict["categorical_features"]
    all_features = feature_dict["all_features"]
    validate_no_live_leakage_columns(all_features)
    print(f"  Feature columns used ({len(all_features)}):")
    print(f"    numeric ({len(numeric_features)}):     {numeric_features}")
    print(f"    categorical ({len(categorical_features)}): {categorical_features}")

    if train_seasons is not None and test_season is not None:
        train_df, test_df = split_by_train_test_seasons(used, train_seasons, test_season)
        print(f"  Season split — train: {train_seasons}, test: {test_season}")
    else:
        train_df, test_df = chronological_game_train_test_split(used, test_size=test_size)
    validate_game_split(train_df, test_df)
    train_games = train_df["game_id"].nunique()
    test_games = test_df["game_id"].nunique()
    print(f"  Train games:                 {train_games}")
    print(f"  Test games:                  {test_games}")
    print(f"  Train rows:                  {len(train_df)}")
    print(f"  Test rows:                   {len(test_df)}")

    if train_df.empty or test_df.empty:
        print("  [error] Train or test split is empty. Nothing saved.")
        return 1

    x_train = train_df[all_features]
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df[all_features]
    y_test = test_df[TARGET_COLUMN].astype(int)

    pipeline = build_live_logistic_regression_pipeline(numeric_features, categorical_features)
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
        "n_unique_games": int(n_games),
        "train_games": int(train_games),
        "test_games": int(test_games),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "n_features": len(all_features),
        **metrics,
    }

    print("\n  Metrics (test event rows):")
    print(f"    accuracy:    {metrics['accuracy']:.4f}")
    print(f"    roc_auc:     {metrics['roc_auc']:.4f}")
    print(f"    log_loss:    {metrics['log_loss']:.4f}")
    print(f"    brier_score: {metrics['brier_score']:.4f}")
    print(f"    confusion:   TP={metrics['true_positive']} TN={metrics['true_negative']} "
          f"FP={metrics['false_positive']} FN={metrics['false_negative']}")

    calibration = build_calibration_table(y_test, y_prob, n_bins=10)
    phase_metrics = build_phase_metrics_table(
        test_df["seconds_remaining_game"], y_test, y_pred, y_prob
    )

    # Persist artifacts.
    metrics_df = pd.DataFrame([metrics_row], columns=METRICS_COLUMNS)
    save_csv(metrics_df, metrics_path)
    save_csv(calibration, calibration_path)
    save_csv(phase_metrics, phase_metrics_path)
    joblib.dump(pipeline, model_path)
    joblib.dump(feature_dict, feature_columns_path)

    print("\nDone.")
    print(f"  Model artifact:              {model_path}")
    print(f"  Feature columns artifact:    {feature_columns_path}")
    print(f"  Metrics output:              {metrics_path}")
    print(f"  Calibration output:          {calibration_path}")
    print(f"  Phase metrics output:        {phase_metrics_path}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse CLI arguments and run live model training."""
    parser = argparse.ArgumentParser(
        description="Train a baseline live win-probability model from "
        "data/processed/live_features.csv."
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of most-recent games held out for testing (default 0.2).",
    )
    args = parser.parse_args(argv)

    print("Training live model (Logistic Regression baseline)...")
    return train_live_model(test_size=args.test_size)


if __name__ == "__main__":
    sys.exit(main())
