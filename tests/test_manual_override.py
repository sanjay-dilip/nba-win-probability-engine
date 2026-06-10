"""Tests for manual post-game override helpers (Build 12).

Pure-function tests only — Streamlit is never launched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.manual_override import (  # noqa: E402
    build_manual_result_record,
    compare_manual_to_game_results,
    compare_result_for_game,
    compute_winner_from_scores,
    get_existing_result,
    load_postgame_results,
    normalize_source,
    upsert_manual_result,
    validate_postgame_results_dataframe,
    validate_scores,
    validate_source,
)


def test_validate_scores_accepts_valid_integers():
    validate_scores(110, 102)


def test_validate_scores_rejects_negative():
    with pytest.raises(ValueError, match="non-negative"):
        validate_scores(-1, 100)


def test_validate_scores_rejects_tie():
    with pytest.raises(ValueError, match="Tie scores"):
        validate_scores(100, 100)


def test_normalize_source():
    assert normalize_source(" Manual ") == "manual"
    assert normalize_source("CORRECTED_MANUAL") == "corrected_manual"


@pytest.mark.parametrize(
    "source",
    ["manual", "corrected_manual", "api"],
)
def test_validate_source_accepts_allowed(source):
    assert validate_source(source) == source


def test_validate_source_rejects_invalid():
    with pytest.raises(ValueError, match="Invalid source"):
        validate_source("play_by_play")


def test_compute_winner_from_scores_home_win():
    winner = compute_winner_from_scores("Boston Celtics", "Atlanta Hawks", 120, 110)
    assert winner == "Boston Celtics"


def test_compute_winner_from_scores_away_win():
    winner = compute_winner_from_scores("Boston Celtics", "Atlanta Hawks", 110, 120)
    assert winner == "Atlanta Hawks"


def test_build_manual_result_record_columns():
    record = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=116,
        away_score=117,
        source="manual",
        notes="test note",
        confirmed_at="2024-11-13T00:00:00+00:00",
    )
    assert record["game_id"] == "0022400001"
    assert record["home_score"] == 116
    assert record["away_score"] == 117
    assert record["winner"] == "Atlanta Hawks"
    assert record["source"] == "manual"
    assert record["confirmed_at"] == "2024-11-13T00:00:00+00:00"
    assert record["notes"] == "test note"


def test_upsert_creates_new_csv_if_missing(tmp_path):
    path = tmp_path / "postgame_results.csv"
    record = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=120,
        away_score=110,
        source="manual",
    )
    written = upsert_manual_result(record, path=path)
    assert path.exists()
    assert len(written) == 1
    assert written.loc[0, "game_id"] == "0022400001"


def test_upsert_refuses_overwrite_by_default(tmp_path):
    path = tmp_path / "postgame_results.csv"
    record = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=120,
        away_score=110,
        source="manual",
    )
    upsert_manual_result(record, path=path)

    updated = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=115,
        away_score=112,
        source="corrected_manual",
    )
    with pytest.raises(ValueError, match="already exists"):
        upsert_manual_result(updated, path=path, allow_overwrite=False)


def test_upsert_overwrites_when_allowed(tmp_path):
    path = tmp_path / "postgame_results.csv"
    record = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=120,
        away_score=110,
        source="manual",
    )
    upsert_manual_result(record, path=path)

    updated = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=115,
        away_score=112,
        source="corrected_manual",
    )
    written = upsert_manual_result(updated, path=path, allow_overwrite=True)
    assert len(written) == 1
    assert written.loc[0, "home_score"] == 115
    assert written.loc[0, "source"] == "corrected_manual"


def test_game_id_stays_string(tmp_path):
    path = tmp_path / "postgame_results.csv"
    record = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=120,
        away_score=110,
        source="manual",
    )
    upsert_manual_result(record, path=path)
    loaded = pd.read_csv(path, dtype={"game_id": str})
    assert loaded.loc[0, "game_id"] == "0022400001"
    assert isinstance(loaded.loc[0, "game_id"], str)


def test_load_postgame_results_adds_notes_column_for_legacy_file(tmp_path):
    path = tmp_path / "postgame_results.csv"
    legacy = pd.DataFrame(
        [
            {
                "game_id": "0022400001",
                "home_score": 120,
                "away_score": 110,
                "winner": "Boston Celtics",
                "source": "manual",
                "confirmed_at": "2024-11-13T00:00:00+00:00",
            }
        ]
    )
    legacy.to_csv(path, index=False)

    df = load_postgame_results(path)
    assert "notes" in df.columns
    assert df.loc[0, "notes"] == ""


def test_get_existing_result(tmp_path):
    path = tmp_path / "postgame_results.csv"
    record = build_manual_result_record(
        game_id="0022400001",
        home_team="Boston Celtics",
        away_team="Atlanta Hawks",
        home_score=120,
        away_score=110,
        source="manual",
    )
    upsert_manual_result(record, path=path)
    row = get_existing_result("0022400001", path=path)
    assert row is not None
    assert row["winner"] == "Boston Celtics"
    assert get_existing_result("9999999999", path=path) is None


def _official_row(home_score=116, away_score=117):
    return pd.Series(
        {
            "game_id": "0022400001",
            "home_team": "Boston Celtics",
            "away_team": "Atlanta Hawks",
            "home_score": float(home_score),
            "away_score": float(away_score),
            "winner": "Atlanta Hawks" if away_score > home_score else "Boston Celtics",
        }
    )


def _manual_row(home_score=116, away_score=117):
    return pd.Series(
        {
            "game_id": "0022400001",
            "home_score": home_score,
            "away_score": away_score,
            "winner": "Atlanta Hawks" if away_score > home_score else "Boston Celtics",
            "source": "manual",
        }
    )


def test_compare_result_for_game_match():
    status = compare_result_for_game(
        "0022400001",
        _manual_row(),
        _official_row(),
    )
    assert status == "match"


def test_compare_result_for_game_mismatch():
    status = compare_result_for_game(
        "0022400001",
        _manual_row(home_score=120, away_score=110),
        _official_row(),
    )
    assert status == "mismatch"


def test_compare_result_for_game_no_manual():
    status = compare_result_for_game("0022400001", None, _official_row())
    assert status == "no_manual_result_available"


def test_compare_result_for_game_no_official():
    status = compare_result_for_game("0022400001", _manual_row(), None)
    assert status == "no_official_result_available"


def test_compare_manual_to_game_results(tmp_path):
    manual_df = pd.DataFrame([_manual_row()])
    official_df = pd.DataFrame([_official_row()])
    comparison = compare_manual_to_game_results(manual_df, official_df)
    assert comparison.loc[0, "comparison_status"] == "match"


def test_validate_postgame_results_dataframe_catches_missing_columns():
    df = pd.DataFrame([{"game_id": "0022400001"}])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_postgame_results_dataframe(df)
