"""Tests for the live feature builder helpers.

All tests exercise pure helper logic on small, hand-built DataFrames and never
call the NBA API.  ``build_live_features.py`` does not import ``nba_api`` at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.build_live_features import (  # noqa: E402
    ISSUE_REPORT_COLUMNS,
    REQUIRED_GAME_RESULTS_COLUMNS,
    REQUIRED_LIVE_FEATURE_COLUMNS,
    build_issue_record,
    build_live_feature_rows,
    compute_event_indicator_flags,
    dedupe_events,
    extract_game_result,
    forward_fill_scores,
    map_event_type_label,
    parse_pctimestring_to_seconds,
    seconds_remaining_in_game,
    validate_game_results_dataframe,
    validate_live_features_dataframe,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _pbp_one_game(game_id: str = "0022400001") -> pd.DataFrame:
    """A small but realistic single-game play-by-play (normalized V3 schema)."""
    rows = [
        # event_num, event_msg_type, action, period, clock, scoreHome, scoreAway
        (2, "period", "start", 1, "PT12M00.00S", 0.0, 0.0),
        (4, "Jump Ball", "", 1, "PT12M00.00S", np.nan, np.nan),
        (10, "Made Shot", "Jump Shot", 1, "PT11M38.00S", 2.0, 0.0),
        (12, "Missed Shot", "", 1, "PT11M00.00S", np.nan, np.nan),
        (20, "Made Shot", "3PT", 2, "PT10M00.00S", 2.0, 3.0),
        (30, "Free Throw", "", 4, "PT00M30.00S", 5.0, 3.0),
    ]
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "event_num": en,
                "event_msg_type": emt,
                "event_msg_action_type": act,
                "period": per,
                "pctimestring": clock,
                "scoreHome": sh,
                "scoreAway": sa,
                "home_team": "Boston Celtics",
                "away_team": "Atlanta Hawks",
                "season": "2024-25",
                "game_date": "2024-11-12",
            }
            for (en, emt, act, per, clock, sh, sa) in rows
        ]
    )


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "0022400001",
                "season": "2024-25",
                "game_date": "2024-11-12",
                "home_team": "Boston Celtics",
                "away_team": "Atlanta Hawks",
                "home_team_id": "100",
                "away_team_id": "200",
                "status": "final",
                "game_type": "regular_season",
            }
        ]
    )


# ---------------------------------------------------------------------------
# parse_pctimestring_to_seconds
# ---------------------------------------------------------------------------

def test_parse_iso_minutes_seconds():
    assert parse_pctimestring_to_seconds("PT10M50.00S") == pytest.approx(650.0)


def test_parse_iso_seconds_only():
    assert parse_pctimestring_to_seconds("PT45.00S") == pytest.approx(45.0)


def test_parse_iso_full_period():
    assert parse_pctimestring_to_seconds("PT12M00.00S") == pytest.approx(720.0)


def test_parse_mmss_format():
    assert parse_pctimestring_to_seconds("10:50") == pytest.approx(650.0)


def test_parse_invalid_returns_nan():
    assert np.isnan(parse_pctimestring_to_seconds("not-a-time"))
    assert np.isnan(parse_pctimestring_to_seconds(""))
    assert np.isnan(parse_pctimestring_to_seconds(np.nan))
    assert np.isnan(parse_pctimestring_to_seconds(None))


# ---------------------------------------------------------------------------
# seconds_remaining_in_game
# ---------------------------------------------------------------------------

def test_seconds_remaining_regulation_periods():
    # Start of each quarter (720s left in the period).
    assert seconds_remaining_in_game(1, 720) == 3 * 720 + 720
    assert seconds_remaining_in_game(2, 720) == 2 * 720 + 720
    assert seconds_remaining_in_game(3, 720) == 1 * 720 + 720
    assert seconds_remaining_in_game(4, 720) == 0 * 720 + 720
    # End of regulation.
    assert seconds_remaining_in_game(4, 0) == 0


def test_seconds_remaining_overtime_is_negative():
    # Documented MVP convention: overtime is expressed as negative seconds past
    # the end of regulation.
    assert seconds_remaining_in_game(5, 300) == 0      # tip-off of OT1
    assert seconds_remaining_in_game(5, 120) == -180   # 2:00 left in OT1
    assert seconds_remaining_in_game(6, 300) == -300   # tip-off of OT2


def test_seconds_remaining_nan_inputs():
    assert np.isnan(seconds_remaining_in_game(np.nan, 100))
    assert np.isnan(seconds_remaining_in_game(1, np.nan))


# ---------------------------------------------------------------------------
# map_event_type_label
# ---------------------------------------------------------------------------

def test_map_event_label_string_types():
    assert map_event_type_label("Made Shot") == "made_shot"
    assert map_event_type_label("Missed Shot") == "missed_shot"
    assert map_event_type_label("Free Throw") == "free_throw"
    assert map_event_type_label("Rebound") == "rebound"
    assert map_event_type_label("Turnover") == "turnover"
    assert map_event_type_label("Jump Ball") == "jump_ball"


def test_map_event_label_period_uses_action():
    assert map_event_type_label("period", "start") == "start_period"
    assert map_event_type_label("period", "end") == "end_period"


def test_map_event_label_numeric_codes():
    assert map_event_type_label(1) == "made_shot"
    assert map_event_type_label(4) == "rebound"
    assert map_event_type_label("5") == "turnover"
    assert map_event_type_label(13) == "end_period"


def test_map_event_label_unknown_and_missing():
    assert map_event_type_label("Some New Thing") == "unknown"
    assert map_event_type_label(np.nan) == "unknown"
    assert map_event_type_label(None) == "unknown"


# ---------------------------------------------------------------------------
# compute_event_indicator_flags
# ---------------------------------------------------------------------------

def test_event_flags_for_made_shot():
    flags = compute_event_indicator_flags("made_shot")
    assert flags["is_field_goal_attempt"] is True
    assert flags["is_free_throw"] is False
    assert flags["is_turnover"] is False


def test_event_flags_for_free_throw_and_turnover():
    assert compute_event_indicator_flags("free_throw")["is_free_throw"] is True
    assert compute_event_indicator_flags("turnover")["is_turnover"] is True
    assert compute_event_indicator_flags("foul")["is_foul"] is True
    assert compute_event_indicator_flags("rebound")["is_rebound"] is True
    assert compute_event_indicator_flags("timeout")["is_timeout"] is True
    assert compute_event_indicator_flags("missed_shot")["is_field_goal_attempt"] is True


# ---------------------------------------------------------------------------
# forward_fill_scores
# ---------------------------------------------------------------------------

def test_scores_forward_filled_within_game():
    g, has_any = forward_fill_scores(_pbp_one_game())
    by_event = g.set_index("event_num")
    # e4 has no raw score and sits between 0-0 and 2-0 -> fills to 0-0.
    assert by_event.loc[4, "home_score"] == 0
    assert by_event.loc[4, "away_score"] == 0
    # e12 has no raw score and follows 2-0 -> fills forward to 2-0.
    assert by_event.loc[12, "home_score"] == 2
    assert by_event.loc[12, "away_score"] == 0
    assert has_any is True


def test_pre_score_rows_filled_with_zero():
    # A leading row with no score (and nothing before it) must become 0-0.
    df = _pbp_one_game().copy()
    df.loc[df["event_num"] == 2, ["scoreHome", "scoreAway"]] = [np.nan, np.nan]
    g, _ = forward_fill_scores(df)
    first = g.sort_values("event_num").iloc[0]
    assert first["home_score"] == 0
    assert first["away_score"] == 0


def test_score_margin_calculated():
    g, _ = forward_fill_scores(_pbp_one_game())
    by_event = g.set_index("event_num")
    assert by_event.loc[10, "score_margin_home"] == 2     # 2 - 0
    assert by_event.loc[20, "score_margin_home"] == -1    # 2 - 3
    assert by_event.loc[20, "abs_score_margin"] == 1


def test_is_scoring_event_detected_from_score_change():
    g, _ = forward_fill_scores(_pbp_one_game())
    by_event = g.set_index("event_num")
    assert bool(by_event.loc[10, "is_scoring_event"]) is True   # 0 -> 2
    assert bool(by_event.loc[12, "is_scoring_event"]) is False  # no change
    assert bool(by_event.loc[2, "is_scoring_event"]) is False   # first row


def test_forward_fill_no_scores_sets_has_any_false():
    df = _pbp_one_game().copy()
    df["scoreHome"] = np.nan
    df["scoreAway"] = np.nan
    _g, has_any = forward_fill_scores(df)
    assert has_any is False


# ---------------------------------------------------------------------------
# extract_game_result
# ---------------------------------------------------------------------------

def _meta() -> dict:
    return {
        "game_id": "0022400001", "season": "2024-25", "game_date": "2024-11-12",
        "home_team": "Boston Celtics", "away_team": "Atlanta Hawks",
        "home_team_id": "100", "away_team_id": "200",
    }


def test_extract_final_result():
    g, has_any = forward_fill_scores(_pbp_one_game())
    result = extract_game_result(g, has_any, _meta())
    assert result["home_score"] == 5
    assert result["away_score"] == 3
    assert result["home_team_won"] == 1
    assert result["winner"] == "Boston Celtics"
    assert result["source"] == "play_by_play"


def test_extract_result_none_without_scores():
    df = _pbp_one_game().copy()
    df["scoreHome"] = np.nan
    df["scoreAway"] = np.nan
    g, has_any = forward_fill_scores(df)
    assert extract_game_result(g, has_any, _meta()) is None


def _pbp_with_out_of_order_trailing_event(game_id: str = "0022400259") -> pd.DataFrame:
    """A game whose true final is home 5-3, plus a stale trailing replay event.

    Mirrors a real PlayByPlayV3 quirk: an ``instant_replay`` review is appended
    with the *highest* event_num but the score from an *earlier* moment (2-0).
    The naive "last event_num wins" logic would wrongly report 2-0.
    """
    base = _pbp_one_game(game_id=game_id)
    stale = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "event_num": 999,            # highest event_num...
                "event_msg_type": "Instant Replay",
                "event_msg_action_type": "",
                "period": 2,                 # ...but from an earlier period
                "pctimestring": "PT01M31.00S",
                "scoreHome": 2.0,            # stale, lower score
                "scoreAway": 0.0,
                "home_team": "Boston Celtics",
                "away_team": "Atlanta Hawks",
                "season": "2024-25",
                "game_date": "2024-11-12",
            }
        ]
    )
    return pd.concat([base, stale], ignore_index=True)


def test_scores_are_monotonic_despite_out_of_order_trailing_event():
    g, has_any = forward_fill_scores(_pbp_with_out_of_order_trailing_event())
    assert has_any is True
    # The trailing replay row (event 999) must not rewind the running score.
    by_event = g.set_index("event_num")
    assert by_event.loc[999, "home_score"] == 5
    assert by_event.loc[999, "away_score"] == 3
    # Running scores never decrease.
    ordered = g.sort_values("event_num")
    assert (ordered["home_score"].diff().dropna() >= 0).all()
    assert (ordered["away_score"].diff().dropna() >= 0).all()


def test_extract_final_result_ignores_out_of_order_trailing_event():
    g, has_any = forward_fill_scores(_pbp_with_out_of_order_trailing_event())
    result = extract_game_result(g, has_any, _meta())
    # True final is 5-3 (home win), not the stale 2-0 trailing replay value.
    assert result["home_score"] == 5
    assert result["away_score"] == 3
    assert result["home_team_won"] == 1
    assert result["winner"] == "Boston Celtics"


# ---------------------------------------------------------------------------
# dedupe_events
# ---------------------------------------------------------------------------

def test_duplicate_events_removed_keep_last():
    df = _pbp_one_game()
    dup = df[df["event_num"] == 10].copy()
    dup["event_msg_action_type"] = "DUPLICATE"
    combined = pd.concat([df, dup], ignore_index=True)
    deduped, removed = dedupe_events(combined)
    assert removed == 1
    assert len(deduped) == len(df)
    # keep="last" -> the duplicate (added last) wins.
    kept = deduped[deduped["event_num"] == 10].iloc[0]
    assert kept["event_msg_action_type"] == "DUPLICATE"


# ---------------------------------------------------------------------------
# build_live_feature_rows — full build
# ---------------------------------------------------------------------------

def test_full_build_required_columns_and_string_id():
    features, results, issues, dup = build_live_feature_rows(_pbp_one_game(), _games())
    assert list(features.columns) == REQUIRED_LIVE_FEATURE_COLUMNS
    assert list(results.columns) == REQUIRED_GAME_RESULTS_COLUMNS
    assert features["game_id"].iloc[0] == "0022400001"
    assert all(isinstance(v, str) for v in features["game_id"])
    assert dup == 0


def test_full_build_target_repeated_on_every_event():
    features, results, _issues, _dup = build_live_feature_rows(_pbp_one_game(), _games())
    assert (features["home_team_won"] == 1).all()
    assert len(results) == 1
    assert results.iloc[0]["home_team_won"] == 1


def test_full_build_time_and_labels():
    features, _r, _i, _d = build_live_feature_rows(_pbp_one_game(), _games())
    by_event = features.set_index("event_num")
    # e2: period 1, full clock -> 3*720 + 720.
    assert by_event.loc[2, "seconds_remaining_game"] == 2880
    # e30: period 4, 30s left -> 30s left in the game.
    assert by_event.loc[30, "seconds_remaining_game"] == 30
    assert by_event.loc[2, "event_type_label"] == "start_period"
    assert by_event.loc[10, "event_type_label"] == "made_shot"
    assert bool(by_event.loc[10, "is_field_goal_attempt"]) is True
    assert bool(by_event.loc[30, "is_free_throw"]) is True


def test_full_build_team_ids_from_games():
    features, _r, _i, _d = build_live_feature_rows(_pbp_one_game(), _games())
    assert (features["home_team_id"] == "100").all()
    assert (features["away_team_id"] == "200").all()


def test_missing_game_creates_build_issue():
    pbp = _pbp_one_game(game_id="0022499999")  # not present in _games()
    _features, _results, issues, _dup = build_live_feature_rows(pbp, _games())
    issue_types = {i["issue_type"] for i in issues}
    assert "missing_from_games" in issue_types


def test_no_score_game_marked_invalid_target():
    pbp = _pbp_one_game()
    pbp["scoreHome"] = np.nan
    pbp["scoreAway"] = np.nan
    features, results, issues, _dup = build_live_feature_rows(pbp, _games())
    # No result row and home_team_won is null -> rows are invalid (not fabricated).
    assert results.empty
    assert features["home_team_won"].isna().all()
    assert "missing_scores" in {i["issue_type"] for i in issues}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_validate_live_features_splits_invalid():
    features, _r, _i, _d = build_live_feature_rows(_pbp_one_game(), _games())
    bad = features.iloc[0].copy()
    bad["home_team_won"] = np.nan
    combined = pd.concat([features, bad.to_frame().T], ignore_index=True)
    valid, invalid = validate_live_features_dataframe(combined)
    assert len(invalid) == 1
    assert len(valid) == len(features)


def test_validate_live_features_raises_on_missing_column():
    with pytest.raises(ValueError):
        validate_live_features_dataframe(pd.DataFrame({"game_id": ["x"]}))


def test_validate_game_results_required_columns():
    _f, results, _i, _d = build_live_feature_rows(_pbp_one_game(), _games())
    valid, invalid = validate_game_results_dataframe(results)
    assert list(results.columns) == REQUIRED_GAME_RESULTS_COLUMNS
    assert len(valid) == 1
    assert invalid.empty


def test_validate_game_results_raises_on_missing_column():
    with pytest.raises(ValueError):
        validate_game_results_dataframe(pd.DataFrame({"game_id": ["x"]}))


# ---------------------------------------------------------------------------
# build_issue_record
# ---------------------------------------------------------------------------

def test_build_issue_record_columns():
    rec = build_issue_record("0022400001", "missing_scores", "no scores", 5)
    assert list(rec.keys()) == ISSUE_REPORT_COLUMNS
    assert rec["game_id"] == "0022400001"
    assert rec["row_count"] == 5
