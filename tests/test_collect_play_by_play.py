"""Tests for the play-by-play collection helpers.

All tests exercise pure helper logic and never call the NBA API.
``collect_play_by_play.py`` imports ``nba_api`` lazily inside
``fetch_play_by_play()``, so importing the module here is safe and offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.collect_play_by_play import (  # noqa: E402
    COVERAGE_REPORT_COLUMNS,
    DEFAULT_LIMIT,
    FAILURE_REPORT_COLUMNS,
    REQUIRED_PBP_COLUMNS,
    build_coverage_record,
    build_failure_record,
    filter_games_for_collection,
    get_already_collected_game_ids,
    load_eligible_games,
    merge_failure_records,
    normalize_play_by_play_dataframe,
    validate_play_by_play_dataframe,
)
from src.utils import append_or_update_csv  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures / factories
# ---------------------------------------------------------------------------

def _games_csv(tmp_path: Path, rows: list[dict] | None = None) -> Path:
    """Write a minimal games.csv to tmp_path and return its path."""
    if rows is None:
        rows = [
            {
                "game_id": "0022400001",
                "season": "2024-25",
                "game_date": "2024-11-12",
                "home_team": "Boston Celtics",
                "away_team": "Atlanta Hawks",
                "home_team_id": 1610612738,
                "away_team_id": 1610612737,
                "status": "final",
                "game_type": "regular_season",
            },
            {
                "game_id": "0022400002",
                "season": "2024-25",
                "game_date": "2024-11-12",
                "home_team": "Detroit Pistons",
                "away_team": "Miami Heat",
                "home_team_id": 1610612765,
                "away_team_id": 1610612748,
                "status": "final",
                "game_type": "regular_season",
            },
            {
                "game_id": "0022300001",
                "season": "2023-24",
                "game_date": "2023-10-24",
                "home_team": "Denver Nuggets",
                "away_team": "LA Lakers",
                "home_team_id": 1610612743,
                "away_team_id": 1610612747,
                "status": "final",
                "game_type": "regular_season",
            },
        ]
    path = tmp_path / "games.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _game_row(
    game_id: str = "0022400001",
    season: str = "2024-25",
    game_date: str = "2024-11-12",
    home_team: str = "Boston Celtics",
    away_team: str = "Atlanta Hawks",
) -> pd.Series:
    """Return a single game row as a Series (mimics one row from games.csv)."""
    return pd.Series(
        {
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "home_team": home_team,
            "away_team": away_team,
        }
    )


def _raw_v3_rows(game_id: str = "0022400001") -> pd.DataFrame:
    """Return a tiny mock PlayByPlayV3 DataFrame with a few representative rows."""
    return pd.DataFrame(
        [
            # period start (neutral)
            {
                "gameId": game_id,
                "actionNumber": 2,
                "clock": "PT12M00.00S",
                "period": 1,
                "teamId": 0,
                "teamTricode": "",
                "personId": 0,
                "playerName": "",
                "playerNameI": "",
                "location": "",
                "description": "Start of 1st Period",
                "actionType": "period",
                "subType": "start",
                "scoreHome": "0",
                "scoreAway": "0",
                "isFieldGoal": 0,
                "videoAvailable": 0,
                "shotValue": 0,
                "actionId": 1,
            },
            # away team scored
            {
                "gameId": game_id,
                "actionNumber": 21,
                "clock": "PT10M50.00S",
                "period": 1,
                "teamId": 1610612737,
                "teamTricode": "ATL",
                "personId": 1631096,
                "playerName": "Johnson",
                "playerNameI": "T. Johnson",
                "location": "v",
                "description": "Johnson 26' 3PT Jump Shot (3 PTS)",
                "actionType": "Made Shot",
                "subType": "Jump Shot",
                "scoreHome": "0",
                "scoreAway": "3",
                "isFieldGoal": 1,
                "videoAvailable": 1,
                "shotValue": 3,
                "actionId": 2,
            },
            # home team scored
            {
                "gameId": game_id,
                "actionNumber": 23,
                "clock": "PT10M35.00S",
                "period": 1,
                "teamId": 1610612738,
                "teamTricode": "BOS",
                "personId": 201143,
                "playerName": "Brown",
                "playerNameI": "J. Brown",
                "location": "h",
                "description": "Brown 27' 3PT Pullup Jump Shot (3 PTS)",
                "actionType": "Made Shot",
                "subType": "Pullup Jump shot",
                "scoreHome": "3",
                "scoreAway": "3",
                "isFieldGoal": 1,
                "videoAvailable": 1,
                "shotValue": 3,
                "actionId": 3,
            },
            # row with no score yet (empty string scores)
            {
                "gameId": game_id,
                "actionNumber": 5,
                "clock": "PT11M45.00S",
                "period": 1,
                "teamId": 1610612738,
                "teamTricode": "BOS",
                "personId": 201143,
                "playerName": "Brown",
                "playerNameI": "J. Brown",
                "location": "h",
                "description": "Brown Missed Shot",
                "actionType": "Missed Shot",
                "subType": "Jump Shot",
                "scoreHome": "",
                "scoreAway": "",
                "isFieldGoal": 1,
                "videoAvailable": 0,
                "shotValue": 2,
                "actionId": 4,
            },
        ]
    )


# ---------------------------------------------------------------------------
# load_eligible_games
# ---------------------------------------------------------------------------

def test_load_eligible_games_preserves_game_id_as_string(tmp_path):
    path = _games_csv(tmp_path)
    df = load_eligible_games(path)
    # pandas 2.x may return StringDtype instead of object — both are string types.
    assert pd.api.types.is_string_dtype(df["game_id"]), "game_id should be a string dtype"
    assert df["game_id"].iloc[0] == "0022400001"


def test_load_eligible_games_filters_only_final(tmp_path):
    rows = [
        {"game_id": "0022400001", "season": "2024-25", "game_date": "2024-11-12",
         "home_team": "BOS", "away_team": "ATL", "home_team_id": 1, "away_team_id": 2,
         "status": "final", "game_type": "regular_season"},
        {"game_id": "0022400002", "season": "2024-25", "game_date": "2024-11-12",
         "home_team": "DEN", "away_team": "LAL", "home_team_id": 3, "away_team_id": 4,
         "status": "scheduled", "game_type": "regular_season"},
    ]
    path = _games_csv(tmp_path, rows)
    df = load_eligible_games(path)
    assert len(df) == 1
    assert df.iloc[0]["game_id"] == "0022400001"


def test_load_eligible_games_filters_by_season(tmp_path):
    path = _games_csv(tmp_path)
    df = load_eligible_games(path, season="2023-24")
    assert len(df) == 1
    assert df.iloc[0]["season"] == "2023-24"


def test_load_eligible_games_raises_if_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_eligible_games(tmp_path / "nonexistent.csv")


# ---------------------------------------------------------------------------
# filter_games_for_collection
# ---------------------------------------------------------------------------

def test_filter_by_game_id(tmp_path):
    games = load_eligible_games(_games_csv(tmp_path))
    result = filter_games_for_collection(games, game_id="0022400001", limit=100)
    assert len(result) == 1
    assert result.iloc[0]["game_id"] == "0022400001"


def test_filter_skips_already_collected(tmp_path):
    games = load_eligible_games(_games_csv(tmp_path))
    result = filter_games_for_collection(
        games,
        already_collected={"0022400001"},
        limit=100,
    )
    assert "0022400001" not in result["game_id"].tolist()
    assert len(result) == 2  # 2 remaining games (0022400002, 0022300001)


def test_filter_respects_limit(tmp_path):
    games = load_eligible_games(_games_csv(tmp_path))
    result = filter_games_for_collection(games, limit=1)
    assert len(result) == 1


def test_filter_limit_zero_returns_all(tmp_path):
    games = load_eligible_games(_games_csv(tmp_path))
    result = filter_games_for_collection(games, limit=0)
    assert len(result) == 3


def test_filter_default_limit_is_10(tmp_path):
    # Create 15 games to confirm default limit caps at DEFAULT_LIMIT.
    rows = [
        {"game_id": f"00224{str(i).zfill(5)}", "season": "2024-25",
         "game_date": "2024-11-12", "home_team": "BOS", "away_team": "ATL",
         "home_team_id": 1, "away_team_id": 2,
         "status": "final", "game_type": "regular_season"}
        for i in range(1, 16)
    ]
    games = load_eligible_games(_games_csv(tmp_path, rows))
    result = filter_games_for_collection(games)
    assert len(result) == DEFAULT_LIMIT


# ---------------------------------------------------------------------------
# get_already_collected_game_ids
# ---------------------------------------------------------------------------

def test_get_already_collected_returns_empty_set_when_no_file(tmp_path):
    result = get_already_collected_game_ids(tmp_path / "play_by_play.csv")
    assert result == set()


def test_get_already_collected_reads_existing_ids(tmp_path):
    pbp_path = tmp_path / "play_by_play.csv"
    pd.DataFrame({"game_id": ["0022400001", "0022400001", "0022400002"],
                  "event_num": [1, 2, 1]}).to_csv(pbp_path, index=False)
    result = get_already_collected_game_ids(pbp_path)
    assert result == {"0022400001", "0022400002"}


def test_get_already_collected_preserves_leading_zeros(tmp_path):
    pbp_path = tmp_path / "play_by_play.csv"
    pd.DataFrame({"game_id": ["0022400001"]}).to_csv(pbp_path, index=False)
    ids = get_already_collected_game_ids(pbp_path)
    assert "0022400001" in ids


# ---------------------------------------------------------------------------
# normalize_play_by_play_dataframe
# ---------------------------------------------------------------------------

def test_normalize_has_required_columns():
    raw = _raw_v3_rows()
    row = _game_row()
    out = normalize_play_by_play_dataframe(raw, row)
    for col in REQUIRED_PBP_COLUMNS:
        assert col in out.columns, f"Missing required column: {col}"


def test_normalize_game_id_is_string():
    raw = _raw_v3_rows()
    out = normalize_play_by_play_dataframe(raw, _game_row())
    # pandas 2.x may return StringDtype instead of object — both are string types.
    assert pd.api.types.is_string_dtype(out["game_id"])
    assert out["game_id"].iloc[0] == "0022400001"


def test_normalize_description_routing():
    """home_description / visitor_description / neutral_description should be
    populated from the V3 description column based on location."""
    raw = _raw_v3_rows()
    out = normalize_play_by_play_dataframe(raw, _game_row())

    neutral_row = out[out["event_num"] == 2].iloc[0]
    assert neutral_row["neutral_description"] == "Start of 1st Period"
    assert neutral_row["home_description"] == ""
    assert neutral_row["visitor_description"] == ""

    away_row = out[out["event_num"] == 21].iloc[0]
    assert away_row["visitor_description"] == "Johnson 26' 3PT Jump Shot (3 PTS)"
    assert away_row["home_description"] == ""

    home_row = out[out["event_num"] == 23].iloc[0]
    assert home_row["home_description"] == "Brown 27' 3PT Pullup Jump Shot (3 PTS)"
    assert home_row["visitor_description"] == ""


def test_normalize_score_format():
    raw = _raw_v3_rows()
    out = normalize_play_by_play_dataframe(raw, _game_row())
    # Row with scores
    scored = out[out["event_num"] == 23].iloc[0]
    assert scored["score"] == "3 - 3"
    # Row without scores
    no_score = out[out["event_num"] == 5].iloc[0]
    assert no_score["score"] == ""


def test_normalize_score_margin():
    raw = _raw_v3_rows()
    out = normalize_play_by_play_dataframe(raw, _game_row())
    # Away scored 3, home 0 → margin = 0 - 3 = -3
    away_scored = out[out["event_num"] == 21].iloc[0]
    assert away_scored["score_margin"] == -3
    # Both 3 → margin = 0
    tied = out[out["event_num"] == 23].iloc[0]
    assert tied["score_margin"] == 0
    # No score → empty
    no_score = out[out["event_num"] == 5].iloc[0]
    assert no_score["score_margin"] == ""


def test_normalize_attaches_game_metadata():
    raw = _raw_v3_rows()
    row = _game_row(home_team="Boston Celtics", away_team="Atlanta Hawks",
                    season="2024-25", game_date="2024-11-12")
    out = normalize_play_by_play_dataframe(raw, row)
    assert (out["home_team"] == "Boston Celtics").all()
    assert (out["away_team"] == "Atlanta Hawks").all()
    assert (out["season"] == "2024-25").all()
    assert (out["game_date"] == "2024-11-12").all()


def test_normalize_maps_v3_columns():
    raw = _raw_v3_rows()
    out = normalize_play_by_play_dataframe(raw, _game_row())
    assert out["event_num"].tolist()[0] == 2     # actionNumber
    assert out["event_msg_type"].tolist()[0] == "period"   # actionType
    assert out["event_msg_action_type"].tolist()[0] == "start"  # subType
    assert out["pctimestring"].tolist()[0] == "PT12M00.00S"    # clock


# ---------------------------------------------------------------------------
# validate_play_by_play_dataframe
# ---------------------------------------------------------------------------

def test_validate_splits_valid_and_invalid():
    raw = _raw_v3_rows()
    df = normalize_play_by_play_dataframe(raw, _game_row())
    # Inject an invalid row (null event_num)
    bad = df.iloc[0].copy()
    bad["event_num"] = None
    combined = pd.concat([df, bad.to_frame().T], ignore_index=True)
    valid, invalid = validate_play_by_play_dataframe(combined)
    assert len(valid) == len(df)
    assert len(invalid) == 1


def test_validate_raises_on_missing_column():
    df = pd.DataFrame({"game_id": ["0022400001"]})
    with pytest.raises(ValueError):
        validate_play_by_play_dataframe(df)


# ---------------------------------------------------------------------------
# build_failure_record
# ---------------------------------------------------------------------------

def test_build_failure_record_has_required_columns():
    row = _game_row()
    record = build_failure_record(row, "PlayByPlayV3", "empty_response", "Zero rows.")
    assert list(record.keys()) == FAILURE_REPORT_COLUMNS
    assert record["game_id"] == "0022400001"
    assert record["reason_failed"] == "empty_response"
    assert record["endpoint"] == "PlayByPlayV3"


def test_build_failure_record_game_id_is_string():
    row = _game_row(game_id="0022400001")
    record = build_failure_record(row, "PlayByPlayV3", "api_error", "timeout")
    assert isinstance(record["game_id"], str)
    assert record["game_id"] == "0022400001"


# ---------------------------------------------------------------------------
# De-duplication by game_id + event_num (via append_or_update_csv)
# ---------------------------------------------------------------------------

def test_dedup_by_game_id_and_event_num(tmp_path):
    out_path = tmp_path / "play_by_play.csv"
    raw = _raw_v3_rows()
    df1 = normalize_play_by_play_dataframe(raw, _game_row())
    append_or_update_csv(df1, out_path, key_columns=["game_id", "event_num"])

    # Re-collect the same game with one description changed.
    raw2 = _raw_v3_rows()
    raw2.loc[raw2["actionNumber"] == 2, "description"] = "Updated description"
    df2 = normalize_play_by_play_dataframe(raw2, _game_row())
    final = append_or_update_csv(df2, out_path, key_columns=["game_id", "event_num"])

    # No duplicates
    assert len(final) == len(df1)
    # Latest version wins
    event_2 = final[final["event_num"].astype(str) == "2"].iloc[0]
    assert event_2["neutral_description"] == "Updated description"


# ---------------------------------------------------------------------------
# --limit 0 means unlimited (data expansion patch)
# ---------------------------------------------------------------------------

def test_limit_zero_means_unlimited(tmp_path):
    # 15 eligible games, limit 0 -> all 15 selected (no cap).
    rows = [
        {"game_id": f"00224{str(i).zfill(5)}", "season": "2024-25",
         "game_date": "2024-11-12", "home_team": "BOS", "away_team": "ATL",
         "home_team_id": 1, "away_team_id": 2,
         "status": "final", "game_type": "regular_season"}
        for i in range(1, 16)
    ]
    games = load_eligible_games(_games_csv(tmp_path, rows))
    result = filter_games_for_collection(games, limit=0)
    assert len(result) == 15


def test_already_collected_skipped_with_limit_zero(tmp_path):
    rows = [
        {"game_id": f"00224{str(i).zfill(5)}", "season": "2024-25",
         "game_date": "2024-11-12", "home_team": "BOS", "away_team": "ATL",
         "home_team_id": 1, "away_team_id": 2,
         "status": "final", "game_type": "regular_season"}
        for i in range(1, 6)
    ]
    games = load_eligible_games(_games_csv(tmp_path, rows))
    already = {"0022400001", "0022400002"}
    result = filter_games_for_collection(games, already_collected=already, limit=0)
    assert len(result) == 3
    assert not set(result["game_id"]).intersection(already)


def test_all_games_already_collected_returns_empty(tmp_path):
    rows = [
        {"game_id": "0022400001", "season": "2024-25", "game_date": "2024-11-12",
         "home_team": "BOS", "away_team": "ATL", "home_team_id": 1, "away_team_id": 2,
         "status": "final", "game_type": "regular_season"},
    ]
    games = load_eligible_games(_games_csv(tmp_path, rows))
    result = filter_games_for_collection(
        games, already_collected={"0022400001"}, limit=0
    )
    assert result.empty


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def test_coverage_record_schema():
    record = build_coverage_record("2024-25", eligible_games=100, collected_games=56, total_event_rows=26249)
    assert list(record.keys()) == COVERAGE_REPORT_COLUMNS


def test_coverage_percentage_calculation():
    record = build_coverage_record("2024-25", eligible_games=200, collected_games=50, total_event_rows=1000)
    assert record["coverage_pct"] == 25.0
    assert record["remaining_games"] == 150
    assert record["collected_games"] == 50


def test_coverage_percentage_handles_zero_eligible():
    record = build_coverage_record("2024-25", eligible_games=0, collected_games=0, total_event_rows=0)
    assert record["coverage_pct"] == 0.0
    assert record["remaining_games"] == 0


def test_coverage_record_season_none_becomes_all():
    record = build_coverage_record(None, eligible_games=10, collected_games=10, total_event_rows=5)
    assert record["season"] == "all"
    assert record["coverage_pct"] == 100.0


# ---------------------------------------------------------------------------
# Failure report does not duplicate the same failed game
# ---------------------------------------------------------------------------

def _failure(game_id: str, reason: str = "api_error", msg: str = "boom") -> dict:
    return build_failure_record(_game_row(game_id=game_id), "PlayByPlayV3", reason, msg)


def test_merge_failures_no_existing():
    merged = merge_failure_records(None, [_failure("0022400001")], succeeded_game_ids=[])
    assert len(merged) == 1
    assert merged.iloc[0]["game_id"] == "0022400001"


def test_merge_failures_does_not_duplicate_same_game():
    existing = pd.DataFrame([_failure("0022400001", msg="first error")], columns=FAILURE_REPORT_COLUMNS)
    merged = merge_failure_records(existing, [_failure("0022400001", msg="second error")], succeeded_game_ids=[])
    # The same failed game appears only once, with the latest error.
    assert len(merged) == 1
    assert merged.iloc[0]["error_message"] == "second error"


def test_merge_failures_drops_now_succeeded_games():
    existing = pd.DataFrame(
        [_failure("0022400001"), _failure("0022400002")], columns=FAILURE_REPORT_COLUMNS
    )
    merged = merge_failure_records(existing, [], succeeded_game_ids=["0022400001"])
    assert merged["game_id"].tolist() == ["0022400002"]


def test_merge_failures_keeps_game_id_as_string():
    existing = pd.DataFrame([_failure("0022400001")], columns=FAILURE_REPORT_COLUMNS)
    merged = merge_failure_records(existing, [_failure("0022400003")], succeeded_game_ids=[])
    assert all(isinstance(v, str) for v in merged["game_id"])
    assert "0022400001" in merged["game_id"].tolist()


# ---------------------------------------------------------------------------
# game_id stays a string across repeated load/save cycles
# ---------------------------------------------------------------------------

def test_game_id_string_after_repeated_save_load(tmp_path):
    out_path = tmp_path / "play_by_play.csv"
    df = normalize_play_by_play_dataframe(_raw_v3_rows(), _game_row())
    append_or_update_csv(df, out_path, key_columns=["game_id", "event_num"])
    # Re-collect the same game twice more.
    for _ in range(2):
        append_or_update_csv(df, out_path, key_columns=["game_id", "event_num"])
    ids = get_already_collected_game_ids(out_path)
    assert "0022400001" in ids
    reloaded = pd.read_csv(out_path, dtype={"game_id": str})
    assert reloaded["game_id"].iloc[0] == "0022400001"
