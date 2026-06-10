"""Tests for season configuration helpers (Build 17)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.season_config import (  # noqa: E402
    DEFAULT_SEASON,
    SUPPORTED_SEASONS,
    parse_pipeline_season_args,
    validate_season_format,
    validate_season_list,
)


def test_valid_season_format_accepted():
    assert validate_season_format("2024-25") == "2024-25"
    assert validate_season_format("2022-23") == "2022-23"


def test_invalid_season_format_rejected():
    with pytest.raises(ValueError, match="Invalid season format"):
        validate_season_format("2024")
    with pytest.raises(ValueError, match="Invalid season format"):
        validate_season_format("24-25")


def test_validate_season_list():
    assert validate_season_list(["2022-23", "2023-24"]) == ["2022-23", "2023-24"]


def test_parse_pipeline_defaults_to_2024_25_and_supported_multi():
    single, multi = parse_pipeline_season_args()
    assert single == DEFAULT_SEASON
    assert multi == SUPPORTED_SEASONS


def test_parse_pipeline_single_season():
    single, multi = parse_pipeline_season_args(season="2023-24")
    assert single == "2023-24"
    assert multi == ["2023-24"]


def test_parse_pipeline_multi_seasons():
    single, multi = parse_pipeline_season_args(seasons=["2022-23", "2024-25"])
    assert single == "2022-23"
    assert multi == ["2022-23", "2024-25"]


def test_parse_pipeline_rejects_both_season_and_seasons():
    with pytest.raises(ValueError, match="not both"):
        parse_pipeline_season_args(season="2024-25", seasons=["2023-24"])
