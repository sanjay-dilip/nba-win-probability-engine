"""Tests for dashboard display formatting helpers (Product UI Polish).

Pure-function tests only — Streamlit is never launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    format_game_clock,
    format_iso_duration_to_clock,
    format_large_number,
    format_metric_value,
    format_pct,
    format_status_label,
    format_status_text,
    metric_display_name,
    page_intro_subtitle,
    safe_metric_delta,
    section_header_text,
    shorten_text,
)


def test_iso_duration_formats_to_clock():
    assert format_iso_duration_to_clock("PT04M20.00S") == "4:20"
    assert format_iso_duration_to_clock("PT06M00.00S") == "6:00"
    assert format_iso_duration_to_clock("PT00M05.50S") == "0:05"
    assert format_iso_duration_to_clock("PT1H02M03S") == "62:03"


def test_game_clock_passthrough_for_mm_ss():
    assert format_game_clock("4:20") == "4:20"
    assert format_game_clock("PT04M20.00S") == "4:20"


def test_format_pct():
    assert format_pct(0.702) == "70.2%"
    assert format_pct(0.702, decimals=0) == "70%"
    assert format_pct(None) == "—"


def test_format_metric_value_accuracy():
    assert format_metric_value(0.7473, "accuracy") == "74.7%"
    assert format_metric_value(0.4874, "log_loss") == "0.4874"
    assert format_metric_value(None, "accuracy") == "—"


def test_format_large_number():
    assert format_large_number(258394) == "258.4K"
    assert format_large_number(1_500_000) == "1.5M"
    assert format_large_number(42) == "42"


def test_format_status_text_readable():
    assert format_status_text("ok") == "Ready"
    assert format_status_text("stale") == "Needs refresh"
    assert "Ready" in format_status_label("ok")


def test_safe_metric_delta_handles_missing():
    assert safe_metric_delta(None) is None
    assert safe_metric_delta(float("nan")) is None
    assert safe_metric_delta(-0.0196) == pytest.approx(-0.0196)


def test_shorten_text():
    assert shorten_text("hello world", max_chars=20) == "hello world"
    assert shorten_text("a" * 100, max_chars=10).endswith("…")
    assert shorten_text(None) == ""


def test_metric_display_name():
    assert metric_display_name("roc_auc") == "ROC-AUC"
    assert metric_display_name("custom_metric") == "Custom Metric"


def test_page_and_section_helpers():
    assert page_intro_subtitle("  Subtitle text  ") == "Subtitle text"
    title, caption = section_header_text("Title", "Caption")
    assert title == "Title"
    assert caption == "Caption"


def test_model_role_display_names():
    from dashboard_utils import (
        BASELINE_MODEL_LABEL,
        PRIMARY_MODEL_LABEL,
        comparison_column_labels,
        model_role_display_name,
    )

    assert model_role_display_name("single_season") == BASELINE_MODEL_LABEL
    assert model_role_display_name("multiseason") == PRIMARY_MODEL_LABEL
    assert model_role_display_name("baseline") == BASELINE_MODEL_LABEL
    assert model_role_display_name("primary") == PRIMARY_MODEL_LABEL
    baseline, primary = comparison_column_labels()
    assert baseline == BASELINE_MODEL_LABEL
    assert primary == PRIMARY_MODEL_LABEL


def test_resolve_pregame_predictions_prefers_processed(tmp_path, monkeypatch):
    from src import config
    from dashboard_utils import resolve_pregame_predictions_source

    processed = tmp_path / "processed.csv"
    deploy = tmp_path / "deploy.csv"
    processed.write_text("game_id\n1\n", encoding="utf-8")
    deploy.write_text("game_id\n2\n", encoding="utf-8")
    monkeypatch.setattr(config, "PREGAME_PREDICTIONS_PATH", processed)
    monkeypatch.setattr(config, "DEPLOY_PREGAME_DEMO_PREDICTIONS_PATH", deploy)

    path, is_demo = resolve_pregame_predictions_source()
    assert path == processed
    assert is_demo is False


def test_resolve_pregame_predictions_falls_back_to_deploy(tmp_path, monkeypatch):
    from src import config
    from dashboard_utils import resolve_pregame_predictions_source

    deploy = tmp_path / "deploy.csv"
    deploy.write_text("game_id\n2\n", encoding="utf-8")
    monkeypatch.setattr(config, "PREGAME_PREDICTIONS_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(config, "DEPLOY_PREGAME_DEMO_PREDICTIONS_PATH", deploy)

    path, is_demo = resolve_pregame_predictions_source()
    assert path == deploy
    assert is_demo is True


def test_format_public_performance_summary():
    from dashboard_utils import format_public_performance_summary

    df = pd.DataFrame(
        [
            {
                "model": "Pre-game model",
                "evaluation_setup": "2022-23 to 2024-25 train / 2025-26 test",
                "accuracy": 0.6824,
                "roc_auc": 0.7357,
                "log_loss": 0.5979,
                "brier_score": 0.2056,
                "notes": "Predicts winner before tip-off",
            }
        ]
    )
    table = format_public_performance_summary(df)
    assert "Model" in table.columns
    assert table.iloc[0]["Model"] == "Pre-game model"
