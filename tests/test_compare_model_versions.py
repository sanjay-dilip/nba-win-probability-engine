"""Tests for model version comparison helpers.

Pure-function tests only — no nba_api, training, or live report generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.compare_model_versions import (  # noqa: E402
    HIGHER_IS_BETTER,
    LOWER_IS_BETTER,
    build_calibration_comparison_summary,
    build_model_comparison_summary,
    build_phase_comparison_summary,
    compare_metric_rows,
    load_calibration_report,
    load_metric_report,
    load_phase_metrics,
    normalize_metric_report,
    run_model_comparison,
)


def _write_metrics(path: Path, accuracy: float, roc_auc: float) -> None:
    pd.DataFrame(
        [
            {
                "model_type": "logistic_regression",
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "log_loss": 0.5,
                "brier_score": 0.2,
                "true_positive": 10,
                "true_negative": 8,
                "false_positive": 2,
                "false_negative": 1,
            }
        ]
    ).to_csv(path, index=False)


def test_load_metric_report_adds_labels(tmp_path):
    path = tmp_path / "metrics.csv"
    _write_metrics(path, 0.70, 0.76)
    df = load_metric_report(path, "pregame", "single-season setup")
    assert df is not None
    assert df.iloc[0]["model"] == "pregame"
    assert df.iloc[0]["setup"] == "single-season setup"
    assert df.iloc[0]["source_report"] == "metrics.csv"


def test_normalize_metric_report_long_format(tmp_path):
    path = tmp_path / "metrics.csv"
    _write_metrics(path, 0.70, 0.76)
    wide = load_metric_report(path, "pregame", "setup")
    long_df = normalize_metric_report(wide, "pregame", "setup", "2024-25", "2024-25")
    assert "accuracy" in long_df["metric"].values
    assert "roc_auc" in long_df["metric"].values
    assert long_df.loc[long_df["metric"] == "accuracy", "value"].iloc[0] == pytest.approx(0.70)


def test_compare_metric_rows_difference():
    single = pd.DataFrame([{"accuracy": 0.70, "roc_auc": 0.76, "log_loss": 0.58, "brier_score": 0.19}])
    multi = pd.DataFrame([{"accuracy": 0.68, "roc_auc": 0.74, "log_loss": 0.60, "brier_score": 0.21}])
    result = compare_metric_rows(single, multi, "pregame")
    acc_row = result.loc[result["metric"] == "accuracy"].iloc[0]
    assert acc_row["difference"] == pytest.approx(-0.02)
    assert "higher is better" in acc_row["interpretation"].lower()


def test_higher_is_better_interpretation():
    single = pd.DataFrame([{"accuracy": 0.60}])
    multi = pd.DataFrame([{"accuracy": 0.65}])
    row = compare_metric_rows(single, multi, "live").iloc[0]
    assert row["difference"] == pytest.approx(0.05)
    assert "higher" in row["interpretation"].lower()


def test_lower_is_better_interpretation():
    single = pd.DataFrame([{"log_loss": 0.50}])
    multi = pd.DataFrame([{"log_loss": 0.55}])
    row = compare_metric_rows(single, multi, "live").iloc[0]
    assert row["difference"] == pytest.approx(0.05)
    assert "lower is better" in row["interpretation"].lower()
    assert "log_loss" in LOWER_IS_BETTER
    assert "accuracy" in HIGHER_IS_BETTER


def test_missing_optional_report_handled_gracefully():
    phase_df = build_phase_comparison_summary(None, None)
    assert not phase_df.empty
    assert "unavailable" in phase_df.iloc[0]["interpretation"].lower()

    cal_df = build_calibration_comparison_summary(None, None, None, None)
    assert not cal_df.empty


def test_phase_comparison_with_fake_metrics(tmp_path):
    single_path = tmp_path / "phase_single.csv"
    multi_path = tmp_path / "phase_multi.csv"
    pd.DataFrame(
        [
            {"phase": "early_game", "accuracy": 0.60, "log_loss": 0.67, "brier_score": 0.24},
            {"phase": "late_game", "accuracy": 0.89, "log_loss": 0.29, "brier_score": 0.09},
        ]
    ).to_csv(single_path, index=False)
    pd.DataFrame(
        [
            {"phase": "early_game", "accuracy": 0.62, "log_loss": 0.66, "brier_score": 0.23},
            {"phase": "late_game", "accuracy": 0.88, "log_loss": 0.32, "brier_score": 0.10},
        ]
    ).to_csv(multi_path, index=False)

    single = load_phase_metrics(single_path, "single")
    multi = load_phase_metrics(multi_path, "multi")
    result = build_phase_comparison_summary(single, multi)
    early = result.loc[
        (result["phase"] == "early_game") & (result["metric"] == "accuracy")
    ].iloc[0]
    assert early["difference"] == pytest.approx(0.02)


def test_calibration_comparison_with_fake_bins(tmp_path):
    cal_path = tmp_path / "cal.csv"
    pd.DataFrame(
        [
            {
                "probability_bucket": "0.4-0.5",
                "row_count": 10,
                "avg_predicted_probability": 0.45,
                "actual_home_win_rate": 0.50,
            }
        ]
    ).to_csv(cal_path, index=False)
    cal = load_calibration_report(cal_path, "pregame", "setup")
    assert cal is not None
    assert cal.iloc[0]["bin"] == "0.4-0.5"
    assert cal.iloc[0]["calibration_error"] == pytest.approx(0.05)

    stacked = build_calibration_comparison_summary(cal, None, None, None)
    assert len(stacked) == 1
    assert stacked.iloc[0]["model"] == "pregame"


def test_build_model_comparison_summary(tmp_path, monkeypatch):
    single_pre = tmp_path / "pre_single.csv"
    multi_pre = tmp_path / "pre_multi.csv"
    single_live = tmp_path / "live_single.csv"
    multi_live = tmp_path / "live_multi.csv"
    _write_metrics(single_pre, 0.70, 0.76)
    _write_metrics(multi_pre, 0.68, 0.74)
    _write_metrics(single_live, 0.75, 0.84)
    _write_metrics(multi_live, 0.74, 0.83)

    summary = build_model_comparison_summary(
        load_metric_report(single_pre, "pregame", "single"),
        load_metric_report(multi_pre, "pregame", "multi"),
        load_metric_report(single_live, "live", "single"),
        load_metric_report(multi_live, "live", "multi"),
    )
    assert set(summary["model"]) == {"pregame", "live"}
    assert len(summary) >= 8


def test_compare_models_dry_run_does_not_write_reports(tmp_path, monkeypatch):
    from src import config

    out_summary = tmp_path / "summary.csv"
    monkeypatch.setattr(config, "MODEL_COMPARISON_SUMMARY_PATH", out_summary)
    rc = run_model_comparison(verbose=False, dry_run=True)
    assert rc == 0
    assert not out_summary.exists()
