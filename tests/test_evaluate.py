"""Tests for the evaluation layer.

Pure-function tests only — no Streamlit, nba_api, or model training.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluate import (  # noqa: E402
    build_evaluation_summary,
    calculate_biggest_momentum_swings,
    calculate_probability_buckets,
    extract_live_final_events,
    load_metric_value,
    load_optional_csv,
    summarize_live_final_events,
    summarize_live_predictions,
    summarize_manual_overrides,
    summarize_pregame_predictions,
)


def _pregame_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "0022400001",
                "season": "2024-25",
                "game_date": "2024-11-12",
                "home_team": "Boston Celtics",
                "away_team": "Atlanta Hawks",
                "home_win_probability": 0.80,
                "away_win_probability": 0.20,
                "predicted_label": 1,
                "actual_home_team_won": 1,
                "prediction_correct": True,
            },
            {
                "game_id": "0022400002",
                "season": "2024-25",
                "game_date": "2024-11-12",
                "home_team": "Detroit Pistons",
                "away_team": "Miami Heat",
                "home_win_probability": 0.51,
                "away_win_probability": 0.49,
                "predicted_label": 1,
                "actual_home_team_won": 0,
                "prediction_correct": False,
            },
            {
                "game_id": "0022400003",
                "season": "2024-25",
                "game_date": "2024-11-15",
                "home_team": "Boston Celtics",
                "away_team": "Charlotte Hornets",
                "home_win_probability": 0.30,
                "away_win_probability": 0.70,
                "predicted_label": 0,
                "actual_home_team_won": 1,
                "prediction_correct": False,
            },
        ]
    )


def _live_df() -> pd.DataFrame:
    rows = []
    for game_id, home, away, date in [
        ("0022400001", "Boston Celtics", "Atlanta Hawks", "2024-11-12"),
        ("0022400002", "Detroit Pistons", "Miami Heat", "2024-11-12"),
    ]:
        for event_num, prob in [(1, 0.55), (2, 0.70), (3, 0.80)]:
            rows.append(
                {
                    "game_id": game_id,
                    "event_num": event_num,
                    "season": "2024-25",
                    "game_date": date,
                    "home_team": home,
                    "away_team": away,
                    "period": 1,
                    "pctimestring": "PT06M00.00S",
                    "home_score": 0,
                    "away_score": 0,
                    "event_type_label": "made_shot",
                    "home_win_probability": prob,
                    "away_win_probability": 1.0 - prob,
                    "predicted_label": 1,
                    "actual_home_team_won": 1,
                    "prediction_correct": prob >= 0.5,
                }
            )
    return pd.DataFrame(rows)


def test_load_optional_csv_missing(tmp_path):
    assert load_optional_csv(tmp_path / "missing.csv") is None


def test_load_optional_csv_present(tmp_path):
    path = tmp_path / "data.csv"
    pd.DataFrame([{"a": 1}]).to_csv(path, index=False)
    df = load_optional_csv(path)
    assert df is not None
    assert len(df) == 1


def test_pregame_summary_calculations():
    summary, buckets = summarize_pregame_predictions(_pregame_df())
    values = dict(zip(summary["metric"], summary["value"]))

    assert values["total_predictions"] == 3
    assert values["labeled_predictions"] == 3
    assert values["prediction_accuracy"] == pytest.approx(1 / 3)
    assert values["correct_predictions"] == 1
    assert values["incorrect_predictions"] == 2
    assert values["home_pick_rate"] == pytest.approx(2 / 3)
    assert values["away_pick_rate"] == pytest.approx(1 / 3)
    assert "most_confident_prediction" in values
    assert "closest_prediction_to_50_50" in values
    assert not buckets.empty


def test_live_event_level_summary():
    summary = summarize_live_predictions(_live_df())
    event = summary.loc[summary["section"] == "event_level"]
    values = dict(zip(event["metric"], event["value"]))

    assert values["total_event_predictions"] == 6
    assert values["unique_games"] == 2
    assert values["prediction_accuracy"] == pytest.approx(1.0)
    assert values["home_pick_rate"] == pytest.approx(1.0)


def test_final_event_summary_uses_max_event_num():
    final_events = extract_live_final_events(_live_df())
    assert len(final_events) == 2
    assert all(final_events["event_num"] == 3)

    summary = summarize_live_final_events(_live_df())
    final = summary.loc[summary["section"] == "final_event"]
    values = dict(zip(final["metric"], final["value"]))

    assert values["total_games"] == 2
    assert values["final_event_accuracy"] == pytest.approx(1.0)
    assert values["number_home_final_predictions"] == 2


def test_final_event_summary_no_duplicate_game_ids():
    final_events = extract_live_final_events(_live_df())
    assert final_events["game_id"].nunique() == len(final_events)


def test_biggest_momentum_swings_sorted_by_absolute_change():
    swings = calculate_biggest_momentum_swings(_live_df(), top_n=10)
    assert not swings.empty
    assert list(swings.columns) == [
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
    changes = swings["absolute_probability_change"].tolist()
    assert changes == sorted(changes, reverse=True)
    # Largest swing in sample is 0.70 - 0.55 = 0.15 on event 2 (after event 1).
    assert swings.iloc[0]["absolute_probability_change"] == pytest.approx(0.15)


def test_manual_override_summary_empty():
    summary = summarize_manual_overrides(None, None)
    assert summary.loc[0, "value"] == 0


def test_manual_override_mismatch_count():
    manual = pd.DataFrame(
        [
            {
                "game_id": "0022400001",
                "home_score": 120,
                "away_score": 110,
                "winner": "Boston Celtics",
                "source": "manual",
                "confirmed_at": "2024-11-13",
                "notes": "",
            }
        ]
    )
    official = pd.DataFrame(
        [
            {
                "game_id": "0022400001",
                "home_team": "Boston Celtics",
                "away_team": "Atlanta Hawks",
                "home_score": 116.0,
                "away_score": 117.0,
                "winner": "Atlanta Hawks",
            }
        ]
    )
    summary = summarize_manual_overrides(manual, official)
    values = dict(zip(summary["metric"], summary["value"]))
    assert values["manual_override_count"] == 1
    assert values["source_manual"] == 1
    assert values["mismatch_vs_game_results"] == 1


def test_evaluation_summary_required_columns():
    pregame_summary, _ = summarize_pregame_predictions(_pregame_df())
    live_event = summarize_live_predictions(_live_df())
    live_final = summarize_live_final_events(_live_df())
    manual = summarize_manual_overrides(None, None)

    pregame_metrics = pd.DataFrame(
        [{"accuracy": 0.70, "roc_auc": 0.76, "n_train": 980, "n_test": 245, "n_features": 22}]
    )
    live_metrics = pd.DataFrame(
        [{"accuracy": 0.75, "roc_auc": 0.84, "train_games": 980, "test_games": 245, "n_features": 17}]
    )

    summary = build_evaluation_summary(
        pregame_metrics=pregame_metrics,
        live_metrics=live_metrics,
        live_phase_metrics=None,
        pregame_summary=pregame_summary,
        live_event_summary=live_event,
        live_final_summary=live_final,
        manual_summary=manual,
        game_results_df=pd.DataFrame([{"game_id": "0022400001"}]),
        pregame_predictions=_pregame_df(),
        live_predictions=_live_df(),
    )

    assert list(summary.columns) == ["section", "metric", "value", "notes"]
    assert not summary.empty
    assert "pregame_model_training" in summary["section"].values
    assert "live_predictions_event_level" in summary["section"].values


def test_load_metric_value():
    metrics = pd.DataFrame([{"accuracy": 0.702, "roc_auc": 0.763}])
    assert load_metric_value(metrics, "accuracy") == pytest.approx(0.702)
    assert load_metric_value(metrics, "missing") is None
    assert load_metric_value(None, "accuracy") is None


def test_calculate_probability_buckets():
    buckets = calculate_probability_buckets(_pregame_df())
    assert "probability_bucket" in buckets.columns
    assert buckets["row_count"].sum() == 3
