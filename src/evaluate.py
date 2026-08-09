"""Project-level evaluation layer.

Reads saved prediction CSVs and existing model training metric reports, then
writes dashboard-ready evaluation summaries to ``outputs/reports/``.

This module does **not** train models, regenerate predictions, or call
``nba_api``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.manual_override import compare_manual_to_game_results, load_postgame_results  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

PROBABILITY_BUCKET_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

MOMENTUM_SWING_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "event_num",
    "period",
    "pctimestring",
    "home_score",
    "away_score",
    "event_type_label",
    "home_win_probability",
    "previous_home_win_probability",
    "probability_change",
    "absolute_probability_change",
]


def load_optional_csv(
    path: Union[str, Path],
    dtype: Optional[dict] = None,
) -> Optional[pd.DataFrame]:
    """Load a CSV when present; return ``None`` if the file is missing or empty."""
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, dtype=dtype or {})
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def _confidence_series(home_prob: pd.Series, away_prob: pd.Series) -> pd.Series:
    """Return max(home, away) win probability per row."""
    return pd.concat([home_prob, away_prob], axis=1).max(axis=1)


def _labeled_mask(df: pd.DataFrame) -> pd.Series:
    """Rows with a known actual outcome label."""
    if "actual_home_team_won" not in df.columns:
        return pd.Series(False, index=df.index)
    return df["actual_home_team_won"].notna()


def _accuracy_from_correct(df: pd.DataFrame, mask: Optional[pd.Series] = None) -> Optional[float]:
    """Share of rows where ``prediction_correct`` is truthy."""
    if "prediction_correct" not in df.columns:
        return None
    subset = df.loc[mask] if mask is not None else df
    valid = subset.loc[subset["prediction_correct"].notna()]
    if valid.empty:
        return None
    return float(valid["prediction_correct"].astype(bool).mean())


def _game_label(row: pd.Series) -> str:
    return f"{row['game_date']} | {row['home_team']} vs {row['away_team']}"


def calculate_probability_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket rows by ``home_win_probability`` and compute calibration stats."""
    if df.empty or "home_win_probability" not in df.columns:
        return pd.DataFrame(
            columns=[
                "probability_bucket",
                "row_count",
                "avg_predicted_home_probability",
                "actual_home_win_rate",
            ]
        )

    labeled = df.loc[_labeled_mask(df)].copy()
    if labeled.empty:
        return pd.DataFrame(
            columns=[
                "probability_bucket",
                "row_count",
                "avg_predicted_home_probability",
                "actual_home_win_rate",
            ]
        )

    probs = pd.to_numeric(labeled["home_win_probability"], errors="coerce")
    labeled = labeled.assign(_prob=probs).dropna(subset=["_prob"])

    rows: list[dict] = []
    for low, high in zip(PROBABILITY_BUCKET_EDGES[:-1], PROBABILITY_BUCKET_EDGES[1:]):
        if high < 1.0:
            bucket_mask = (labeled["_prob"] >= low) & (labeled["_prob"] < high)
            label = f"{low:.1f}-{high:.1f}"
        else:
            bucket_mask = (labeled["_prob"] >= low) & (labeled["_prob"] <= high)
            label = f"{low:.1f}-{high:.1f}"

        bucket = labeled.loc[bucket_mask]
        if bucket.empty:
            continue

        actual = bucket["actual_home_team_won"].astype(float)
        rows.append(
            {
                "probability_bucket": label,
                "row_count": len(bucket),
                "avg_predicted_home_probability": float(bucket["_prob"].mean()),
                "actual_home_win_rate": float(actual.mean()) if actual.notna().any() else None,
            }
        )

    return pd.DataFrame(rows)


def summarize_pregame_predictions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize full-dataset pre-game predictions.

    Returns:
        A tuple of (scalar summary DataFrame, probability-bucket DataFrame).
    """
    empty_summary = pd.DataFrame(columns=["section", "metric", "value"])
    empty_buckets = calculate_probability_buckets(df)

    if df is None or df.empty:
        return empty_summary, empty_buckets

    work = df.copy()
    work["game_id"] = work["game_id"].astype(str)
    labeled = _labeled_mask(work)
    confidence = _confidence_series(work["home_win_probability"], work["away_win_probability"])

    metrics: dict[str, Any] = {
        "total_predictions": len(work),
        "labeled_predictions": int(labeled.sum()),
        "avg_home_win_probability": float(work["home_win_probability"].mean()),
        "avg_away_win_probability": float(work["away_win_probability"].mean()),
        "avg_confidence": float(confidence.mean()),
    }

    if "predicted_label" in work.columns:
        metrics["home_pick_rate"] = float((work["predicted_label"] == 1).mean())
        metrics["away_pick_rate"] = float((work["predicted_label"] == 0).mean())

    accuracy = _accuracy_from_correct(work, labeled)
    if accuracy is not None:
        metrics["prediction_accuracy"] = accuracy
        correct = work.loc[labeled, "prediction_correct"].astype(bool)
        metrics["correct_predictions"] = int(correct.sum())
        metrics["incorrect_predictions"] = int((~correct).sum())

    if not work.empty:
        most_idx = confidence.idxmax()
        closest_idx = (work["home_win_probability"] - 0.5).abs().idxmin()
        metrics["most_confident_prediction"] = _game_label(work.loc[most_idx])
        metrics["most_confident_home_probability"] = float(work.loc[most_idx, "home_win_probability"])
        metrics["closest_prediction_to_50_50"] = _game_label(work.loc[closest_idx])
        metrics["closest_home_probability"] = float(work.loc[closest_idx, "home_win_probability"])

    summary_df = pd.DataFrame(
        [{"section": "summary", "metric": key, "value": value} for key, value in metrics.items()]
    )
    buckets_df = calculate_probability_buckets(work)
    return summary_df, buckets_df


def summarize_live_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize event-level live predictions (event-weighted metrics)."""
    empty = pd.DataFrame(columns=["section", "metric", "value"])
    if df is None or df.empty:
        return empty

    work = df.copy()
    work["game_id"] = work["game_id"].astype(str)
    labeled = _labeled_mask(work)
    confidence = _confidence_series(work["home_win_probability"], work["away_win_probability"])

    metrics = {
        "total_event_predictions": len(work),
        "unique_games": int(work["game_id"].nunique()),
        "avg_home_win_probability": float(work["home_win_probability"].mean()),
        "avg_away_win_probability": float(work["away_win_probability"].mean()),
        "avg_confidence": float(confidence.mean()),
    }

    if "predicted_label" in work.columns:
        metrics["home_pick_rate"] = float((work["predicted_label"] == 1).mean())
        metrics["away_pick_rate"] = float((work["predicted_label"] == 0).mean())

    accuracy = _accuracy_from_correct(work, labeled)
    if accuracy is not None:
        metrics["prediction_accuracy"] = accuracy

    rows = [
        {"section": "event_level", "metric": key, "value": value}
        for key, value in metrics.items()
    ]
    return pd.DataFrame(rows)


def extract_live_final_events(df: pd.DataFrame) -> pd.DataFrame:
    """Return one final-event row per game (max ``event_num``)."""
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    work["game_id"] = work["game_id"].astype(str)
    work["event_num"] = pd.to_numeric(work["event_num"], errors="coerce")
    work = work.dropna(subset=["event_num"])
    idx = work.groupby("game_id")["event_num"].idxmax()
    return work.loc[idx].reset_index(drop=True)


def summarize_live_final_events(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize live predictions using each game's final event only."""
    empty = pd.DataFrame(columns=["section", "metric", "value"])
    final_events = extract_live_final_events(df)
    if final_events.empty:
        return empty

    confidence = _confidence_series(
        final_events["home_win_probability"],
        final_events["away_win_probability"],
    )
    labeled = _labeled_mask(final_events)

    metrics = {
        "total_games": len(final_events),
        "avg_final_confidence": float(confidence.mean()),
    }

    if "predicted_label" in final_events.columns:
        metrics["number_home_final_predictions"] = int((final_events["predicted_label"] == 1).sum())
        metrics["number_away_final_predictions"] = int((final_events["predicted_label"] == 0).sum())

    accuracy = _accuracy_from_correct(final_events, labeled)
    if accuracy is not None:
        metrics["final_event_accuracy"] = accuracy

    rows = [
        {"section": "final_event", "metric": key, "value": value}
        for key, value in metrics.items()
    ]
    return pd.DataFrame(rows)


def calculate_biggest_momentum_swings(
    live_df: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    """Find the largest absolute home-win probability changes across all games."""
    empty = pd.DataFrame(columns=MOMENTUM_SWING_COLUMNS)
    if live_df is None or live_df.empty:
        return empty

    work = live_df.copy()
    work["game_id"] = work["game_id"].astype(str)
    work["event_num"] = pd.to_numeric(work["event_num"], errors="coerce")
    work = work.sort_values(["game_id", "event_num"])

    work["previous_home_win_probability"] = work.groupby("game_id")[
        "home_win_probability"
    ].shift(1)
    work["probability_change"] = work["home_win_probability"] - work["previous_home_win_probability"]
    work["absolute_probability_change"] = work["probability_change"].abs()

    # First event per game has no previous probability — exclude from ranking.
    ranked = work.loc[work["previous_home_win_probability"].notna()].copy()
    if ranked.empty:
        return empty

    top = ranked.nlargest(top_n, "absolute_probability_change")
    cols = [c for c in MOMENTUM_SWING_COLUMNS if c in top.columns]
    return top[cols].reset_index(drop=True)


def summarize_manual_overrides(
    manual_df: Optional[pd.DataFrame],
    game_results_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Summarize manual override records and mismatches vs official results."""
    rows = [{"metric": "manual_override_count", "value": 0}]

    if manual_df is None or manual_df.empty:
        return pd.DataFrame(rows)

    manual_df = manual_df.copy()
    manual_df["game_id"] = manual_df["game_id"].astype(str)
    rows = [{"metric": "manual_override_count", "value": len(manual_df)}]

    if "source" in manual_df.columns:
        for source, count in manual_df["source"].value_counts().items():
            rows.append({"metric": f"source_{source}", "value": int(count)})

    if game_results_df is not None and not game_results_df.empty:
        comparison = compare_manual_to_game_results(manual_df, game_results_df)
        mismatch_count = int((comparison["comparison_status"] == "mismatch").sum())
        match_count = int((comparison["comparison_status"] == "match").sum())
        rows.append({"metric": "mismatch_vs_game_results", "value": mismatch_count})
        rows.append({"metric": "match_vs_game_results", "value": match_count})

    return pd.DataFrame(rows)


def summarize_by_team(pregame_df: pd.DataFrame) -> pd.DataFrame:
    """Optional team-level pre-game prediction accuracy summary."""
    if pregame_df is None or pregame_df.empty:
        return pd.DataFrame(
            columns=["team", "games_as_home", "games_as_away", "total_games", "accuracy"]
        )

    work = pregame_df.copy()
    work["game_id"] = work["game_id"].astype(str)
    if "prediction_correct" not in work.columns:
        return pd.DataFrame(
            columns=["team", "games_as_home", "games_as_away", "total_games", "accuracy"]
        )

    labeled = work.loc[_labeled_mask(work) & work["prediction_correct"].notna()].copy()
    if labeled.empty:
        return pd.DataFrame(
            columns=["team", "games_as_home", "games_as_away", "total_games", "accuracy"]
        )

    teams = sorted(set(labeled["home_team"]).union(set(labeled["away_team"])))
    rows: list[dict] = []
    for team in teams:
        home_games = labeled.loc[labeled["home_team"] == team]
        away_games = labeled.loc[labeled["away_team"] == team]
        team_games = labeled.loc[
            (labeled["home_team"] == team) | (labeled["away_team"] == team)
        ]
        if team_games.empty:
            continue
        rows.append(
            {
                "team": team,
                "games_as_home": len(home_games),
                "games_as_away": len(away_games),
                "total_games": len(team_games),
                "accuracy": float(team_games["prediction_correct"].astype(bool).mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def load_metric_value(metrics_df: Optional[pd.DataFrame], metric_name: str) -> Optional[Any]:
    """Read a scalar metric from a one-row training metrics CSV."""
    if metrics_df is None or metrics_df.empty:
        return None
    if metric_name not in metrics_df.columns:
        return None
    value = metrics_df.iloc[0][metric_name]
    if pd.isna(value):
        return None
    return value


def build_evaluation_summary(
    pregame_metrics: Optional[pd.DataFrame],
    live_metrics: Optional[pd.DataFrame],
    live_phase_metrics: Optional[pd.DataFrame],
    pregame_summary: pd.DataFrame,
    live_event_summary: pd.DataFrame,
    live_final_summary: pd.DataFrame,
    manual_summary: pd.DataFrame,
    game_results_df: Optional[pd.DataFrame],
    pregame_predictions: Optional[pd.DataFrame],
    live_predictions: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build the compact key-value evaluation summary table."""
    rows: List[dict] = []

    def add(section: str, metric: str, value: Any, notes: str = "") -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "notes": notes,
            }
        )

    test_note = "Chronological hold-out test set from model training."
    ref_note = (
        "Full-dataset reference metric for dashboard use — "
        "not the same as chronological test-set performance."
    )

    for metric in ["accuracy", "roc_auc", "log_loss", "brier_score", "n_train", "n_test", "n_features"]:
        val = load_metric_value(pregame_metrics, metric)
        if val is not None:
            add("pregame_model_training", metric, val, test_note)

    for metric in ["accuracy", "roc_auc", "log_loss", "brier_score", "train_games", "test_games", "n_features"]:
        val = load_metric_value(live_metrics, metric)
        if val is not None:
            add("live_model_training", metric, val, test_note)

    if live_phase_metrics is not None and not live_phase_metrics.empty:
        for _, phase_row in live_phase_metrics.iterrows():
            phase = phase_row.get("phase", "unknown")
            for metric in ["accuracy", "log_loss", "brier_score", "row_count"]:
                if metric in phase_row and pd.notna(phase_row[metric]):
                    add(
                        "live_model_training",
                        f"{phase}_{metric}",
                        phase_row[metric],
                        test_note,
                    )

    if not pregame_summary.empty:
        for _, row in pregame_summary.iterrows():
            add("pregame_predictions_full_dataset", row["metric"], row["value"], ref_note)

    if not live_event_summary.empty:
        for _, row in live_event_summary.iterrows():
            add(
                "live_predictions_event_level",
                row["metric"],
                row["value"],
                "Event-weighted — rows within a game are correlated.",
            )

    if not live_final_summary.empty:
        for _, row in live_final_summary.iterrows():
            add("live_predictions_final_event", row["metric"], row["value"], ref_note)

    if game_results_df is not None and not game_results_df.empty:
        add("data_coverage", "game_results_count", len(game_results_df))
    if pregame_predictions is not None and not pregame_predictions.empty:
        add("data_coverage", "pregame_predictions_count", len(pregame_predictions))
    if live_predictions is not None and not live_predictions.empty:
        add("data_coverage", "live_event_predictions_count", len(live_predictions))
        add("data_coverage", "live_unique_games", int(live_predictions["game_id"].nunique()))

    if not manual_summary.empty:
        for _, row in manual_summary.iterrows():
            add("manual_overrides", row["metric"], row["value"])

    return pd.DataFrame(rows)


def run_evaluation(verbose: bool = True) -> int:
    """Run the full evaluation pipeline and write report CSVs.

    Returns:
        ``0`` on success, ``1`` when required prediction files are missing.
    """
    ensure_directories()

    pregame_predictions = load_optional_csv(
        config.PREGAME_PREDICTIONS_PATH,
        dtype={"game_id": str},
    )
    live_predictions = load_optional_csv(
        config.LIVE_PREDICTIONS_PATH,
        dtype={"game_id": str},
    )
    game_results = load_optional_csv(
        config.GAME_RESULTS_PATH,
        dtype={"game_id": str},
    )

    if pregame_predictions is None and live_predictions is None:
        if verbose:
            print("ERROR: No prediction files found. Run predict_all first.")
        return 1

    pregame_metrics = load_optional_csv(config.PREGAME_MODEL_METRICS_PATH)
    live_metrics = load_optional_csv(config.LIVE_MODEL_METRICS_PATH)
    live_phase_metrics = load_optional_csv(config.LIVE_MODEL_PHASE_METRICS_PATH)
    pregame_calibration = load_optional_csv(config.PREGAME_MODEL_CALIBRATION_PATH)
    live_calibration = load_optional_csv(config.LIVE_MODEL_CALIBRATION_PATH)

    manual_df = load_postgame_results(config.POSTGAME_RESULTS_PATH)
    if manual_df is not None and manual_df.empty:
        manual_df = None

    pregame_summary, pregame_buckets = summarize_pregame_predictions(
        pregame_predictions if pregame_predictions is not None else pd.DataFrame()
    )
    live_event_summary = summarize_live_predictions(
        live_predictions if live_predictions is not None else pd.DataFrame()
    )
    live_final_summary = summarize_live_final_events(
        live_predictions if live_predictions is not None else pd.DataFrame()
    )
    manual_summary = summarize_manual_overrides(manual_df, game_results)
    momentum_swings = calculate_biggest_momentum_swings(
        live_predictions if live_predictions is not None else pd.DataFrame()
    )
    by_team = summarize_by_team(
        pregame_predictions if pregame_predictions is not None else pd.DataFrame()
    )

    live_prediction_summary = pd.concat(
        [live_event_summary, live_final_summary],
        ignore_index=True,
    )

    evaluation_summary = build_evaluation_summary(
        pregame_metrics=pregame_metrics,
        live_metrics=live_metrics,
        live_phase_metrics=live_phase_metrics,
        pregame_summary=pregame_summary,
        live_event_summary=live_event_summary,
        live_final_summary=live_final_summary,
        manual_summary=manual_summary,
        game_results_df=game_results,
        pregame_predictions=pregame_predictions,
        live_predictions=live_predictions,
    )

    pregame_output = pregame_summary.copy()
    if not pregame_buckets.empty:
        bucket_part = pregame_buckets.copy()
        bucket_part["section"] = "probability_bucket"
        bucket_part["metric"] = bucket_part["probability_bucket"]
        bucket_part["value"] = bucket_part["row_count"]
        pregame_output = pd.concat([pregame_output, bucket_part], ignore_index=True, sort=False)

    save_csv(pregame_output, config.PREGAME_PREDICTION_SUMMARY_PATH)
    save_csv(live_prediction_summary, config.LIVE_PREDICTION_SUMMARY_PATH)
    save_csv(momentum_swings, config.BIGGEST_MOMENTUM_SWINGS_PATH)
    save_csv(evaluation_summary, config.EVALUATION_SUMMARY_PATH)
    if not by_team.empty:
        save_csv(by_team, config.EVALUATION_BY_TEAM_PATH)

    if verbose:
        _print_evaluation_summary(
            evaluation_summary,
            pregame_calibration is not None,
            live_calibration is not None,
        )

    return 0


def _print_evaluation_summary(
    evaluation_summary: pd.DataFrame,
    has_pregame_calibration: bool,
    has_live_calibration: bool,
) -> None:
    """Print a concise CLI summary after writing reports."""
    print("Evaluation complete. Reports written to outputs/reports/")
    print(f"  - {config.EVALUATION_SUMMARY_PATH.name}")
    print(f"  - {config.PREGAME_PREDICTION_SUMMARY_PATH.name}")
    print(f"  - {config.LIVE_PREDICTION_SUMMARY_PATH.name}")
    print(f"  - {config.BIGGEST_MOMENTUM_SWINGS_PATH.name}")

    if evaluation_summary.empty:
        return

    print("\nKey metrics:")
    highlights = [
        ("pregame_model_training", "accuracy"),
        ("live_model_training", "accuracy"),
        ("pregame_predictions_full_dataset", "prediction_accuracy"),
        ("live_predictions_event_level", "prediction_accuracy"),
        ("live_predictions_final_event", "final_event_accuracy"),
        ("manual_overrides", "manual_override_count"),
    ]
    for section, metric in highlights:
        match = evaluation_summary.loc[
            (evaluation_summary["section"] == section)
            & (evaluation_summary["metric"] == metric)
        ]
        if not match.empty:
            val = match.iloc[0]["value"]
            print(f"  {section}.{metric}: {val}")

    if has_pregame_calibration:
        print(f"  (existing) {config.PREGAME_MODEL_CALIBRATION_PATH.name}")
    if has_live_calibration:
        print(f"  (existing) {config.LIVE_MODEL_CALIBRATION_PATH.name}")

    if config.MODEL_COMPARISON_SUMMARY_PATH.exists():
        print(f"  (existing) {config.MODEL_COMPARISON_SUMMARY_PATH.name}")
        print("  Run compare_models to refresh model version comparison reports.")


def main() -> int:
    """CLI entry point for ``python src/evaluate.py``."""
    return run_evaluation(verbose=True)


if __name__ == "__main__":
    sys.exit(main())
