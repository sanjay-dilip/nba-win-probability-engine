"""Playoff and NBA Finals case-study pipeline (Build 20.6).

Applies the trained primary multi-season model to playoff games as an
out-of-distribution case study. All outputs are kept separate from the
regular-season pipeline.

Does **not** train models or call ``nba_api`` inside pure helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.build_live_features import build_live_features  # noqa: E402
from src.build_pregame_features import (  # noqa: E402
    build_pregame_feature_rows,
    load_game_results,
    load_games_for_features,
    merge_game_results,
    sort_games,
)
from src.collect_games import collect_games  # noqa: E402
from src.collect_play_by_play import collect_play_by_play  # noqa: E402
from src.predict_live import predict_live  # noqa: E402
from src.predict_pregame import (  # noqa: E402
    build_pregame_prediction_output,
    generate_pregame_predictions,
    load_pregame_prediction_inputs,
    normalize_pregame_feature_columns,
    validate_pregame_prediction_inputs,
)
from src.manual_override import load_postgame_results  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402
import joblib  # noqa: E402

PLAYOFF_SEASON_TYPE = "Playoffs"
FINALS_MIN_GAMES = 4
FINALS_MAX_GAMES = 7
CASE_STUDY_FOCUS_SEASON = "2025-26"
PLAYOFF_GAME_ID_LENGTH = 10
NBA_FINALS_ROUND_CODE = "004"

CASE_STUDY_COLUMNS = [
    "season",
    "game_id",
    "finals_game_number",
    "game_date",
    "home_team",
    "away_team",
    "game_status",
    "series_status_if_available",
    "prediction_available",
    "replay_available",
    "final_result_available",
    "home_win_probability_start",
    "home_win_probability_final",
    "predicted_winner_start",
    "predicted_winner_final",
    "final_winner",
    "start_prediction_correct_if_final",
    "final_prediction_correct_if_final",
    "notes",
]

UPCOMING_PREDICTIONS_COLUMNS = [
    "season",
    "game_id",
    "finals_game_number",
    "game_date",
    "home_team",
    "away_team",
    "game_status",
    "pregame_prediction_available",
    "home_win_probability_pregame",
    "away_win_probability_pregame",
    "predicted_winner_pregame",
    "prediction_confidence",
    "final_result_available",
    "final_winner",
    "pregame_prediction_correct_if_final",
    "replay_available",
    "notes",
]

COVERAGE_COLUMNS = [
    "season",
    "playoff_games",
    "games_with_pbp",
    "pbp_coverage_pct",
    "finals_games_detected",
    "missing_pbp_games",
    "notes",
]

FINALS_SCHEDULE_OVERRIDE_COLUMNS = [
    "season",
    "finals_game_number",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "status",
    "season_type",
    "source",
    "notes",
]

FORBIDDEN_OVERRIDE_COLUMNS = [
    "home_score",
    "away_score",
    "winner",
    "final_winner",
    "home_team_won",
    "actual_home_team_won",
    "prediction_correct",
]

ALLOWED_OVERRIDE_STATUSES = {
    "scheduled",
    "if_necessary",
    "if_necessary_scheduled",
    "not_available_yet",
}
ALLOWED_OVERRIDE_SOURCES = {"manual_schedule", "api_schedule", "placeholder"}

FINALS_WINS_TO_CLINCH = 4

PROJECTED_SERIES_COLUMNS = [
    "season",
    "finals_game_number",
    "game_id",
    "game_date",
    "home_team",
    "away_team",
    "game_status",
    "actual_winner",
    "actual_result_source",
    "predicted_winner_pregame",
    "projected_winner_used",
    "projection_type",
    "series_team_a",
    "series_team_b",
    "team_a_wins_before",
    "team_b_wins_before",
    "team_a_wins_after",
    "team_b_wins_after",
    "game_needed_under_projection",
    "series_over_after_game",
    "notes",
]

REQUIRED_SCHEDULE_FIELDS = [
    "season",
    "finals_game_number",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "status",
    "season_type",
]


def normalize_season_type(value: object) -> str:
    """Normalize season type labels to ``Playoffs`` or ``Regular Season``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Regular Season"
    text = str(value).strip().lower()
    if text in {"playoffs", "playoff", "postseason", "post season"}:
        return PLAYOFF_SEASON_TYPE
    if text == "playoffs":
        return PLAYOFF_SEASON_TYPE
    if "playoff" in text:
        return PLAYOFF_SEASON_TYPE
    return "Regular Season"


def _load_csv(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path, dtype={"game_id": str})
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def load_finals_schedule_overrides(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load manual Finals schedule overrides; return empty DataFrame if missing."""
    path = path or config.FINALS_SCHEDULE_OVERRIDES_PATH
    df = _load_csv(path)
    if df is None:
        return pd.DataFrame(columns=FINALS_SCHEDULE_OVERRIDE_COLUMNS)
    return df


def validate_finals_schedule_overrides(df: pd.DataFrame) -> None:
    """Validate override schema; raise ``ValueError`` on forbidden or invalid rows."""
    if df is None or df.empty:
        return

    forbidden = [c for c in df.columns if c in FORBIDDEN_OVERRIDE_COLUMNS]
    if forbidden:
        raise ValueError(
            f"Finals schedule overrides must not contain result columns: {forbidden}"
        )

    extra = [c for c in df.columns if c not in FINALS_SCHEDULE_OVERRIDE_COLUMNS]
    if extra:
        raise ValueError(f"Unexpected columns in finals schedule overrides: {extra}")

    missing_cols = [c for c in FINALS_SCHEDULE_OVERRIDE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in finals schedule overrides: {missing_cols}")

    dupes = df.duplicated(subset=["season", "finals_game_number"], keep=False)
    if dupes.any():
        bad = df.loc[dupes, ["season", "finals_game_number"]]
        raise ValueError(
            "Duplicate season + finals_game_number override rows: "
            f"{bad.to_dict('records')}"
        )

    for _, row in df.iterrows():
        status = str(row.get("status", "")).strip().lower()
        if status and status not in ALLOWED_OVERRIDE_STATUSES:
            raise ValueError(
                f"Invalid status '{status}' for Finals game {row.get('finals_game_number')}. "
                f"Allowed: {sorted(ALLOWED_OVERRIDE_STATUSES)}"
            )
        source = str(row.get("source", "")).strip().lower()
        if source and source not in ALLOWED_OVERRIDE_SOURCES:
            raise ValueError(
                f"Invalid source '{source}' for Finals game {row.get('finals_game_number')}. "
                f"Allowed: {sorted(ALLOWED_OVERRIDE_SOURCES)}"
            )
        if status in {"scheduled", "if_necessary_scheduled"}:
            for field in REQUIRED_SCHEDULE_FIELDS:
                val = row.get(field)
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    raise ValueError(
                        f"status={status} requires non-null '{field}' "
                        f"(game {row.get('finals_game_number')})"
                    )
                if field != "finals_game_number" and str(val).strip() == "":
                    raise ValueError(
                        f"status={status} requires non-empty '{field}' "
                        f"(game {row.get('finals_game_number')})"
                    )


def normalize_finals_schedule_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize override dtypes and string fields."""
    if df is None or df.empty:
        return pd.DataFrame(columns=FINALS_SCHEDULE_OVERRIDE_COLUMNS)

    out = df.copy()
    out["finals_game_number"] = pd.to_numeric(out["finals_game_number"], errors="coerce")
    out["game_id"] = out["game_id"].apply(
        lambda x: "" if pd.isna(x) or str(x).strip().lower() in {"", "nan", "none"} else str(x).strip()
    )
    for col in ["home_team_id", "away_team_id"]:
        out[col] = out[col].apply(
            lambda x: "" if pd.isna(x) or str(x).strip().lower() in {"", "nan", "none"} else str(x).strip()
        )
    out["status"] = out["status"].astype(str).str.strip().str.lower()
    out["source"] = out["source"].astype(str).str.strip().str.lower()
    if "season_type" in out.columns:
        out["season_type"] = out["season_type"].apply(
            lambda x: PLAYOFF_SEASON_TYPE if str(x).strip().lower() in {"playoffs", "playoff"} else str(x)
        )
    return out


def _field_is_blank(value: object) -> bool:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return True
    return str(value).strip() == ""


def _has_complete_pregame_metadata(row: pd.Series) -> bool:
    """Return True when a Finals row has enough metadata for pre-game feature building."""
    status = str(row.get("status", "")).strip().lower()
    if status == "not_available_yet":
        return False
    if status in {"final", "completed"}:
        return not _field_is_blank(row.get("game_id"))
    if status not in {"scheduled", "if_necessary_scheduled"}:
        return False
    for field in REQUIRED_SCHEDULE_FIELDS:
        if _field_is_blank(row.get(field)):
            return False
    return True


def _synthetic_finals_game_id(season: str, game_num: int) -> str:
    """Assign a stable synthetic id when schedule metadata lacks a real game_id."""
    season_key = str(season).replace("-", "")
    return f"finals_schedule_{season_key}_g{game_num}"


def _resolve_game_id(row: pd.Series) -> str:
    """Return a non-empty game_id, synthesizing one when only schedule metadata exists."""
    gid = row.get("game_id")
    if not _field_is_blank(gid):
        return str(gid).strip()
    if _has_complete_pregame_metadata(row):
        return _synthetic_finals_game_id(str(row["season"]), int(row["finals_game_number"]))
    return ""


def _override_row_to_game(override: pd.Series) -> pd.Series:
    """Convert one override CSV row into a playoff-games-compatible Series."""
    game_num = int(override["finals_game_number"])
    row = pd.Series(
        {
            "game_id": override.get("game_id", ""),
            "season": override["season"],
            "game_date": override.get("game_date"),
            "home_team": override.get("home_team"),
            "away_team": override.get("away_team"),
            "home_team_id": override.get("home_team_id"),
            "away_team_id": override.get("away_team_id"),
            "status": override.get("status"),
            "game_type": "playoffs",
            "season_type": override.get("season_type", PLAYOFF_SEASON_TYPE),
            "finals_game_number": game_num,
            "playoff_round": "NBA Finals",
            "schedule_source": override.get("source", ""),
            "schedule_notes": override.get("notes", ""),
        }
    )
    row["game_id"] = _resolve_game_id(row)
    return row


def merge_finals_metadata_with_overrides(
    local_finals_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
) -> Dict[int, pd.Series]:
    """Merge local Finals metadata with manual overrides (local wins when present)."""
    finals_by_number: Dict[int, pd.Series] = {}

    if local_finals_df is not None and not local_finals_df.empty:
        for _, game in local_finals_df.iterrows():
            num = game.get("finals_game_number")
            if pd.isna(num):
                continue
            finals_by_number[int(num)] = game

    if overrides_df is not None and not overrides_df.empty:
        season_overrides = overrides_df.loc[overrides_df["season"] == focus_season]
        for _, override in season_overrides.iterrows():
            num = int(override["finals_game_number"])
            if num in finals_by_number:
                continue
            finals_by_number[num] = _override_row_to_game(override)

    return finals_by_number


def build_merged_finals_games_df(
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
    games_path: Optional[Path] = None,
    overrides_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Return merged Finals metadata rows for Games 1–7 (local + overrides)."""
    games_path = games_path or config.PLAYOFF_GAMES_PATH
    overrides_path = overrides_path or config.FINALS_SCHEDULE_OVERRIDES_PATH

    games = _load_csv(games_path)
    local_finals = pd.DataFrame()
    if games is not None:
        labeled = add_finals_game_numbers(games)
        local_finals = labeled.loc[
            (labeled["season"] == focus_season) & (labeled["playoff_round"] == "NBA Finals")
        ].copy()

    overrides_raw = load_finals_schedule_overrides(overrides_path)
    if not overrides_raw.empty:
        validate_finals_schedule_overrides(overrides_raw)
        overrides = normalize_finals_schedule_overrides(overrides_raw)
    else:
        overrides = overrides_raw

    merged = merge_finals_metadata_with_overrides(local_finals, overrides, focus_season)
    if not merged and local_finals.empty:
        return pd.DataFrame()

    rows: List[pd.Series] = []
    for game_num in range(1, FINALS_MAX_GAMES + 1):
        if game_num in merged:
            rows.append(merged[game_num])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def ensure_playoff_season_type_column(games_df: pd.DataFrame) -> pd.DataFrame:
    """Add or normalize ``season_type`` on a games DataFrame."""
    out = games_df.copy()
    out["game_id"] = out["game_id"].astype(str)
    if "season_type" in out.columns:
        out["season_type"] = out["season_type"].map(normalize_season_type)
    elif "game_type" in out.columns:
        out["season_type"] = out["game_type"].map(
            lambda x: PLAYOFF_SEASON_TYPE if str(x).lower() == "playoffs" else "Regular Season"
        )
    else:
        out["season_type"] = PLAYOFF_SEASON_TYPE
    return out


def filter_playoff_games(games_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Return only playoff game rows."""
    if games_df is None or games_df.empty:
        return pd.DataFrame()
    work = ensure_playoff_season_type_column(games_df)
    playoff_mask = work["season_type"] == PLAYOFF_SEASON_TYPE
    if "game_type" in work.columns:
        playoff_mask = playoff_mask | (work["game_type"].astype(str).str.lower() == "playoffs")
    return work.loc[playoff_mask].copy()


def _team_pair_key(home_team: str, away_team: str) -> Tuple[str, str]:
    pair = sorted([str(home_team), str(away_team)])
    return pair[0], pair[1]


def extract_playoff_round_from_game_id(game_id: object) -> Optional[str]:
    """Extract the three-digit playoff round code from an NBA ``game_id``.

    Playoff IDs follow ``004YYRRRGG`` (10 chars): prefix ``004``, season ``YY``,
    round ``RRR`` (``001``–``004``), game-in-series ``GG``. Round ``004`` is the
    NBA Finals.

    Returns ``None`` when the ID cannot be parsed.
    """
    if game_id is None or (isinstance(game_id, float) and pd.isna(game_id)):
        return None
    gid = str(game_id).strip()
    if len(gid) != PLAYOFF_GAME_ID_LENGTH or not gid.isdigit():
        return None
    if not gid.startswith("004"):
        return None
    round_code = gid[5:8]
    if not round_code.isdigit():
        return None
    return round_code


def is_nba_finals_game_id(game_id: object) -> bool:
    """Return True when ``game_id`` encodes NBA Finals (round ``004``)."""
    return extract_playoff_round_from_game_id(game_id) == NBA_FINALS_ROUND_CODE


def _identify_finals_by_team_pair_heuristic(season_games: pd.DataFrame) -> pd.DataFrame:
    """Fallback: pick the latest recurring team-pair series as Finals."""
    season_games = season_games.copy()
    season_games["game_date"] = pd.to_datetime(season_games["game_date"], errors="coerce")
    season_games = season_games.sort_values("game_date")

    pair_counts: Dict[Tuple[str, str], List[pd.Series]] = {}
    for _, row in season_games.iterrows():
        key = _team_pair_key(row["home_team"], row["away_team"])
        pair_counts.setdefault(key, []).append(row)

    best_pair = None
    best_score = -1
    for key, pair_rows in pair_counts.items():
        pair_games = pd.DataFrame(pair_rows)
        count = len(pair_games)
        if count < FINALS_MIN_GAMES:
            continue
        latest_date = pair_games["game_date"].max()
        score = count * 1_000_000 + (latest_date.timestamp() if pd.notna(latest_date) else 0)
        if score > best_score:
            best_score = score
            best_pair = key

    if best_pair is None:
        for key, pair_rows in pair_counts.items():
            pair_games = pd.DataFrame(pair_rows)
            if len(pair_games) >= 2:
                latest_date = pair_games["game_date"].max()
                score = len(pair_games) * 1_000_000 + (
                    latest_date.timestamp() if pd.notna(latest_date) else 0
                )
                if score > best_score:
                    best_score = score
                    best_pair = key

    if best_pair is None:
        return pd.DataFrame()

    finals_games = pd.DataFrame(pair_counts[best_pair]).copy()
    finals_games["is_finals"] = True
    finals_games["playoff_round"] = "NBA Finals"
    return finals_games.head(FINALS_MAX_GAMES)


def identify_nba_finals_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Identify NBA Finals games per season using ``game_id`` round encoding."""
    playoffs = filter_playoff_games(games_df)
    if playoffs.empty:
        return pd.DataFrame()

    playoffs = playoffs.copy()
    playoffs["game_id"] = playoffs["game_id"].astype(str)

    finals_rows: List[pd.DataFrame] = []
    for season, season_games in playoffs.groupby("season"):
        season_games = season_games.copy()
        round_codes = season_games["game_id"].map(extract_playoff_round_from_game_id)
        parseable = round_codes.notna()
        finals_mask = round_codes == NBA_FINALS_ROUND_CODE

        if finals_mask.any():
            finals_games = season_games.loc[finals_mask].copy()
            finals_games["game_date"] = pd.to_datetime(finals_games["game_date"], errors="coerce")
            finals_games = finals_games.sort_values("game_date").head(FINALS_MAX_GAMES)
            finals_games["is_finals"] = True
            finals_games["playoff_round"] = "NBA Finals"
            finals_rows.append(finals_games)
            continue

        if parseable.any():
            # Round codes parsed but no Finals round — do not guess via heuristic.
            continue

        fallback = _identify_finals_by_team_pair_heuristic(season_games)
        if not fallback.empty:
            finals_rows.append(fallback)

    if not finals_rows:
        return pd.DataFrame()
    return pd.concat(finals_rows, ignore_index=True)


def add_playoff_round_labels(games_df: pd.DataFrame) -> pd.DataFrame:
    """Label identified Finals games; other playoff games stay unlabeled."""
    out = ensure_playoff_season_type_column(games_df).copy()
    out["playoff_round"] = ""
    finals = identify_nba_finals_games(out)
    if finals.empty:
        return out
    finals_ids = set(finals["game_id"].astype(str))
    out.loc[out["game_id"].isin(finals_ids), "playoff_round"] = "NBA Finals"
    return out


def add_finals_game_numbers(games_df: pd.DataFrame) -> pd.DataFrame:
    """Assign Finals game numbers 1–7 within each season."""
    out = add_playoff_round_labels(games_df).copy()
    out["finals_game_number"] = pd.NA
    finals = out.loc[out["playoff_round"] == "NBA Finals"].copy()
    if finals.empty:
        return out

    for season, season_finals in finals.groupby("season"):
        ordered = season_finals.sort_values("game_date")
        numbers = list(range(1, len(ordered) + 1))
        out.loc[ordered.index, "finals_game_number"] = numbers[:FINALS_MAX_GAMES]
    return out


def summarize_playoff_coverage(
    games_df: Optional[pd.DataFrame],
    pbp_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build per-season playoff coverage summary rows."""
    playoffs = filter_playoff_games(games_df)
    if playoffs.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)

    pbp_game_ids: set[str] = set()
    if pbp_df is not None and not pbp_df.empty and "game_id" in pbp_df.columns:
        pbp_game_ids = set(pbp_df["game_id"].astype(str).unique())

    finals = identify_nba_finals_games(playoffs)
    rows: List[dict] = []
    for season, season_games in playoffs.groupby("season"):
        game_ids = season_games["game_id"].astype(str).tolist()
        with_pbp = [gid for gid in game_ids if gid in pbp_game_ids]
        missing = [gid for gid in game_ids if gid not in pbp_game_ids]
        finals_count = int(finals.loc[finals["season"] == season].shape[0]) if not finals.empty else 0
        total = len(game_ids)
        pct = round(100.0 * len(with_pbp) / total, 1) if total else 0.0
        rows.append(
            {
                "season": season,
                "playoff_games": total,
                "games_with_pbp": len(with_pbp),
                "pbp_coverage_pct": pct,
                "finals_games_detected": finals_count,
                "missing_pbp_games": len(missing),
                "notes": f"missing: {', '.join(missing[:5])}" + ("..." if len(missing) > 5 else ""),
            }
        )
    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)


def build_playoff_live_features_from_raw(
    seasons: Optional[Sequence[str]] = None,
    pbp_path: Optional[Path] = None,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
) -> int:
    """Build playoff live features using existing live feature logic."""
    pbp_path = pbp_path or config.PLAYOFF_PLAY_BY_PLAY_PATH
    games_path = games_path or config.PLAYOFF_GAMES_PATH
    output_path = output_path or config.PLAYOFF_LIVE_FEATURES_PATH
    results_path = results_path or config.PLAYOFF_GAME_RESULTS_PATH

    if not pbp_path.exists() or not games_path.exists():
        print("ERROR: Playoff games and/or play-by-play files are missing.")
        return 1

    rc = build_live_features(
        pbp_path=pbp_path,
        games_path=games_path,
        output_path=output_path,
        results_path=results_path,
    )
    if rc != 0:
        return rc

    features = _load_csv(output_path)
    if features is not None:
        features = ensure_playoff_season_type_column(
            features.rename(columns={"game_type": "season_type"}, errors="ignore")
        )
        if "season_type" not in features.columns:
            features["season_type"] = PLAYOFF_SEASON_TYPE
        labeled = add_finals_game_numbers(
            features[["game_id", "season", "game_date", "home_team", "away_team"]].drop_duplicates("game_id")
        )
        features = features.merge(
            labeled[["game_id", "playoff_round", "finals_game_number"]],
            on="game_id",
            how="left",
        )
        save_csv(features, output_path)
    return 0


def predict_playoff_live_games(
    features_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> int:
    """Score playoff live features with the primary multi-season live model."""
    return predict_live(
        features_path=features_path or config.PLAYOFF_LIVE_FEATURES_PATH,
        model_path=config.LIVE_MODEL_MULTISEASON_PATH,
        feature_columns_path=config.LIVE_FEATURE_COLUMNS_MULTISEASON_PATH,
        output_path=output_path or config.PLAYOFF_LIVE_PREDICTIONS_PATH,
    )


def _game_status_label(row: pd.Series) -> str:
    status = str(row.get("status", row.get("game_status", "unknown"))).strip().lower()
    if status in {"final", "completed"}:
        return "completed"
    if status in {"scheduled", "not started", "not_started"}:
        return "scheduled"
    if status in {"if_necessary_scheduled", "if necessary scheduled"}:
        return "if_necessary_scheduled"
    if status in {"if_necessary", "if necessary"}:
        return "if_necessary"
    if status == "not_available_yet":
        return "not_available_yet"
    if pd.isna(row.get("game_date")):
        return "not_available_yet"
    return status or "unknown"


def _has_play_by_play(game_id: str, pbp_df: Optional[pd.DataFrame]) -> bool:
    if pbp_df is None or pbp_df.empty or "game_id" not in pbp_df.columns:
        return False
    return str(game_id) in set(pbp_df["game_id"].astype(str))


def _winner_matches_prediction(
    predicted_winner: Optional[str],
    final_winner: Optional[str],
) -> Optional[bool]:
    if not predicted_winner or not final_winner:
        return None
    return str(predicted_winner).strip() == str(final_winner).strip()


def _prediction_snapshot(
    game_id: str,
    predictions: Optional[pd.DataFrame],
) -> dict:
    """Return start/final prediction fields for one Finals game."""
    empty = {
        "prediction_available": False,
        "home_win_probability_start": None,
        "home_win_probability_final": None,
        "predicted_winner_start": None,
        "predicted_winner_final": None,
    }
    if predictions is None or predictions.empty:
        return empty
    game_preds = predictions.loc[predictions["game_id"].astype(str) == str(game_id)].copy()
    if game_preds.empty:
        return empty
    if "event_num" in game_preds.columns:
        game_preds["event_num"] = pd.to_numeric(game_preds["event_num"], errors="coerce")
        game_preds = game_preds.sort_values("event_num")
    start_row = game_preds.iloc[0]
    final_row = game_preds.iloc[-1]
    return {
        "prediction_available": True,
        "home_win_probability_start": float(start_row["home_win_probability"]),
        "home_win_probability_final": float(final_row["home_win_probability"]),
        "predicted_winner_start": str(start_row.get("predicted_winner", "")).strip() or None,
        "predicted_winner_final": str(final_row.get("predicted_winner", "")).strip() or None,
    }


def _placeholder_note(game_num: int, status: str = "") -> str:
    status = str(status).strip().lower()
    if status == "if_necessary" or game_num >= 6:
        return "If necessary — awaiting official schedule/metadata."
    if game_num in (4, 5):
        return "Finals game not yet collected or not available in local playoff metadata."
    return "Finals game not yet scheduled or collected."


def _build_finals_game_row(
    focus_season: str,
    game_num: int,
    game: Optional[pd.Series],
    predictions: Optional[pd.DataFrame],
    pbp_df: Optional[pd.DataFrame],
    results: Optional[pd.DataFrame],
) -> dict:
    """Build one Finals case-study row for an existing or placeholder game."""
    if game is None:
        return {
            "season": focus_season,
            "game_id": "",
            "finals_game_number": game_num,
            "game_date": "",
            "home_team": "",
            "away_team": "",
            "game_status": "not_available_yet",
            "series_status_if_available": "",
            "prediction_available": False,
            "replay_available": False,
            "final_result_available": False,
            "home_win_probability_start": None,
            "home_win_probability_final": None,
            "predicted_winner_start": None,
            "predicted_winner_final": None,
            "final_winner": None,
            "start_prediction_correct_if_final": None,
            "final_prediction_correct_if_final": None,
            "notes": _placeholder_note(game_num),
        }

    game_id = _resolve_game_id(game)
    status = _game_status_label(game)
    pred = _prediction_snapshot(game_id, predictions) if game_id else _prediction_snapshot("", predictions)
    replay_available = _has_play_by_play(game_id, pbp_df) if game_id else False

    final_winner = None
    if game_id and results is not None and not results.empty and status == "completed":
        match = results.loc[results["game_id"].astype(str) == game_id]
        if not match.empty:
            final_winner = str(match.iloc[0].get("winner", "")).strip() or None

    final_result_available = status == "completed" and bool(final_winner)
    start_correct = (
        _winner_matches_prediction(pred["predicted_winner_start"], final_winner)
        if final_result_available
        else None
    )
    final_correct = (
        _winner_matches_prediction(pred["predicted_winner_final"], final_winner)
        if final_result_available
        else None
    )

    schedule_source = str(game.get("schedule_source", "")).strip().lower()
    notes = str(game.get("schedule_notes", "")).strip()
    if schedule_source == "manual_schedule" and status == "scheduled":
        prefix = "Manual schedule metadata used for pre-game prediction."
        notes = f"{prefix} {notes}".strip() if notes else prefix
    elif schedule_source == "manual_schedule" and status == "if_necessary_scheduled":
        prefix = (
            "If-necessary scheduled game — prediction depends on projected/actual series path."
        )
        notes = f"{prefix} {notes}".strip() if notes else prefix
    elif status == "scheduled":
        notes = notes or (
            "Game scheduled — replay will appear after play-by-play is collected."
        )
    elif status == "if_necessary_scheduled":
        notes = notes or (
            "If-necessary scheduled game — prediction depends on projected/actual series path."
        )
    elif status == "if_necessary":
        notes = notes or "If necessary — awaiting official schedule/metadata."
    elif not replay_available and pred["prediction_available"]:
        notes = notes or "Predictions available; play-by-play replay not yet collected."
    elif replay_available and not pred["prediction_available"]:
        notes = notes or "Play-by-play collected; predictions not yet generated."

    return {
        "season": focus_season,
        "game_id": game_id,
        "finals_game_number": game_num,
        "game_date": game.get("game_date"),
        "home_team": game.get("home_team"),
        "away_team": game.get("away_team"),
        "game_status": status,
        "series_status_if_available": game.get("series_status", ""),
        "prediction_available": pred["prediction_available"],
        "replay_available": replay_available,
        "final_result_available": final_result_available,
        "home_win_probability_start": pred["home_win_probability_start"],
        "home_win_probability_final": pred["home_win_probability_final"],
        "predicted_winner_start": pred["predicted_winner_start"],
        "predicted_winner_final": pred["predicted_winner_final"],
        "final_winner": final_winner,
        "start_prediction_correct_if_final": start_correct,
        "final_prediction_correct_if_final": final_correct,
        "notes": notes,
    }


def build_nba_finals_case_study_summary(
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
    games_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
    pbp_path: Optional[Path] = None,
    overrides_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build Finals case-study rows for Games 1–7 (local metadata + schedule overrides)."""
    games_path = games_path or config.PLAYOFF_GAMES_PATH
    predictions_path = predictions_path or config.PLAYOFF_LIVE_PREDICTIONS_PATH
    results_path = results_path or config.PLAYOFF_GAME_RESULTS_PATH
    pbp_path = pbp_path or config.PLAYOFF_PLAY_BY_PLAY_PATH
    overrides_path = overrides_path or config.FINALS_SCHEDULE_OVERRIDES_PATH

    games = _load_csv(games_path)
    predictions = _load_csv(predictions_path)
    results = _load_csv(results_path)
    pbp = _load_csv(pbp_path)

    if games is None:
        games = pd.DataFrame()

    labeled = add_finals_game_numbers(games) if not games.empty else games
    local_finals = pd.DataFrame()
    if not labeled.empty:
        local_finals = labeled.loc[
            (labeled["season"] == focus_season) & (labeled["playoff_round"] == "NBA Finals")
        ].copy()

    if local_finals.empty:
        overrides_only = load_finals_schedule_overrides(overrides_path)
        if overrides_only.empty or not (
            overrides_only["season"] == focus_season
        ).any():
            return pd.DataFrame(columns=CASE_STUDY_COLUMNS)

    overrides_raw = load_finals_schedule_overrides(overrides_path)
    if not overrides_raw.empty:
        validate_finals_schedule_overrides(overrides_raw)
        overrides = normalize_finals_schedule_overrides(overrides_raw)
    else:
        overrides = overrides_raw

    finals_by_number = merge_finals_metadata_with_overrides(
        local_finals, overrides, focus_season
    )

    rows: List[dict] = []
    for game_num in range(1, FINALS_MAX_GAMES + 1):
        rows.append(
            _build_finals_game_row(
                focus_season,
                game_num,
                finals_by_number.get(game_num),
                predictions,
                pbp,
                results,
            )
        )

    return pd.DataFrame(rows, columns=CASE_STUDY_COLUMNS)


def save_case_study_outputs(
    coverage_df: pd.DataFrame,
    case_study_df: pd.DataFrame,
    coverage_path: Optional[Path] = None,
    case_study_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Write playoff coverage and Finals case-study reports."""
    ensure_directories()
    coverage_path = coverage_path or config.PLAYOFF_COVERAGE_REPORT_PATH
    case_study_path = case_study_path or config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH
    save_csv(coverage_df, coverage_path)
    save_csv(case_study_df, case_study_path)
    return {"coverage": coverage_path, "case_study": case_study_path}


FINALS_LIVE_EXPORT_COLUMNS = [
    "season",
    "game_id",
    "finals_game_number",
    "game_date",
    "home_team",
    "away_team",
    "event_num",
    "period",
    "pctimestring",
    "seconds_remaining_period",
    "seconds_remaining_game",
    "home_score",
    "away_score",
    "score_margin_home",
    "abs_score_margin",
    "event_type_label",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "predicted_label",
    "actual_home_team_won",
    "prediction_correct",
]


def build_finals_live_predictions_export(
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
    games_path: Optional[Path] = None,
    predictions_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Filter full playoff live predictions to Finals rows for the deploy-safe export."""
    games_path = games_path or config.PLAYOFF_GAMES_PATH
    predictions_path = predictions_path or config.PLAYOFF_LIVE_PREDICTIONS_PATH

    games = _load_csv(games_path)
    predictions = _load_csv(predictions_path)
    if games is None or predictions is None:
        return pd.DataFrame(columns=FINALS_LIVE_EXPORT_COLUMNS)

    labeled = add_finals_game_numbers(games)
    finals_games = labeled.loc[
        (labeled["season"] == focus_season) & (labeled["playoff_round"] == "NBA Finals")
    ][["game_id", "finals_game_number"]]
    if finals_games.empty:
        return pd.DataFrame(columns=FINALS_LIVE_EXPORT_COLUMNS)

    merged = predictions.merge(finals_games, on="game_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=FINALS_LIVE_EXPORT_COLUMNS)

    merged["finals_game_number"] = merged["finals_game_number"].astype(int)
    return (
        merged[FINALS_LIVE_EXPORT_COLUMNS]
        .sort_values(["finals_game_number", "event_num"])
        .reset_index(drop=True)
    )


def run_export_finals_live_predictions_for_deploy(verbose: bool = True) -> int:
    """Write the deploy-safe Finals live-replay export."""
    export_df = build_finals_live_predictions_export()
    save_csv(export_df, config.FINALS_LIVE_PREDICTIONS_DEPLOY_PATH)
    if verbose:
        print(f"Finals live predictions deploy export written to {config.FINALS_LIVE_PREDICTIONS_DEPLOY_PATH}")
        print(f"  Rows:  {len(export_df)}")
        games = export_df["game_id"].nunique() if not export_df.empty else 0
        print(f"  Games: {games}")
    return 0


def collect_playoff_games(seasons: Sequence[str]) -> int:
    """Collect playoff game metadata into the playoff games CSV."""
    rc = collect_games(
        seasons=seasons,
        season_type="Playoffs",
        output_path=config.PLAYOFF_GAMES_PATH,
    )
    if rc != 0:
        return rc
    games = _load_csv(config.PLAYOFF_GAMES_PATH)
    if games is not None:
        save_csv(ensure_playoff_season_type_column(games), config.PLAYOFF_GAMES_PATH)
    return 0


def collect_playoff_play_by_play(seasons: Sequence[str]) -> int:
    """Collect playoff play-by-play into the playoff PBP CSV."""
    exit_code = 0
    for season in seasons:
        rc = collect_play_by_play(
            season=season,
            limit=0,
            output_path=config.PLAYOFF_PLAY_BY_PLAY_PATH,
            games_path=config.PLAYOFF_GAMES_PATH,
            coverage_report_path=config.PLAYOFF_COVERAGE_REPORT_PATH,
        )
        if rc != 0:
            exit_code = rc
    return exit_code


def run_check_playoff_coverage(seasons: Sequence[str], verbose: bool = True) -> int:
    """Inspect local playoff files and write coverage report."""
    games = _load_csv(config.PLAYOFF_GAMES_PATH)
    pbp = _load_csv(config.PLAYOFF_PLAY_BY_PLAY_PATH)
    if games is not None:
        games = games.loc[games["season"].isin(seasons)]
    coverage_df = summarize_playoff_coverage(games, pbp)
    save_csv(coverage_df, config.PLAYOFF_COVERAGE_REPORT_PATH)
    if verbose:
        print("Playoff coverage report written to:")
        print(f"  {config.PLAYOFF_COVERAGE_REPORT_PATH}")
        if coverage_df.empty:
            print("  No playoff games found for requested seasons.")
        else:
            print(coverage_df.to_string(index=False))
    return 0


def run_build_finals_case_study(verbose: bool = True) -> int:
    """Build the NBA Finals case-study summary report."""
    case_study_df = build_nba_finals_case_study_summary()
    save_csv(case_study_df, config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH)
    if verbose:
        print(f"Finals case-study report written to {config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH}")
        print(f"  Rows: {len(case_study_df)}")
    return 0


def _load_completed_playoff_games(games_path: Path) -> pd.DataFrame:
    """Load completed playoff games for pre-game history (status == final)."""
    if not games_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(
        games_path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )
    if "status" in df.columns:
        df = df[df["status"] == "final"].copy()
    for col in ["game_id", "game_date", "home_team", "away_team"]:
        if col in df.columns:
            df = df[df[col].notna()].copy()
    if "game_type" not in df.columns:
        df["game_type"] = "playoffs"
    return sort_games(df)


def _load_finals_target_games(
    games_path: Path,
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
    overrides_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load merged Finals target rows (local Round 4 + schedule overrides)."""
    merged = build_merged_finals_games_df(
        focus_season=focus_season,
        games_path=games_path,
        overrides_path=overrides_path,
    )
    if merged.empty:
        return merged
    if "game_type" not in merged.columns:
        merged = merged.copy()
        merged["game_type"] = "playoffs"
    work = merged.copy()
    work["game_id"] = work.apply(_resolve_game_id, axis=1)
    return sort_games(work)


def _build_history_games_for_finals(
    regular_games_path: Path,
    regular_results_path: Path,
    playoff_games_path: Path,
    playoff_results_path: Path,
    exclude_game_ids: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Combine completed regular-season and playoff games for leakage-safe history."""
    exclude = set(str(g) for g in (exclude_game_ids or []))
    frames: List[pd.DataFrame] = []

    if regular_games_path.exists():
        regular = load_games_for_features(regular_games_path)
        results = load_game_results(regular_results_path)
        if results is not None:
            regular = merge_game_results(regular, results)
        if exclude:
            regular = regular.loc[~regular["game_id"].astype(str).isin(exclude)]
        frames.append(regular)

    playoff = _load_completed_playoff_games(playoff_games_path)
    if not playoff.empty:
        results = load_game_results(playoff_results_path)
        if results is not None:
            playoff = merge_game_results(playoff, results)
        if exclude:
            playoff = playoff.loc[~playoff["game_id"].astype(str).isin(exclude)]
        frames.append(playoff)

    if not frames:
        return pd.DataFrame()
    return sort_games(pd.concat(frames, ignore_index=True))


def build_finals_pregame_features(
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
    games_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    overrides_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build leakage-safe pre-game feature rows for Finals games with metadata."""
    games_path = games_path or config.PLAYOFF_GAMES_PATH
    output_path = output_path or config.FINALS_PREGAME_FEATURES_PATH
    overrides_path = overrides_path or config.FINALS_SCHEDULE_OVERRIDES_PATH

    all_targets = _load_finals_target_games(games_path, focus_season, overrides_path)
    if all_targets.empty:
        print("  No Finals target games with metadata found.")
        return pd.DataFrame()

    targets = all_targets.loc[all_targets.apply(_has_complete_pregame_metadata, axis=1)].copy()
    if targets.empty:
        print("  No Finals games with complete pre-game metadata found.")
        return pd.DataFrame()

    targets["game_id"] = targets.apply(_resolve_game_id, axis=1)
    target_ids = targets["game_id"].astype(str).tolist()
    history = _build_history_games_for_finals(
        config.RAW_GAMES_PATH,
        config.GAME_RESULTS_PATH,
        games_path,
        config.PLAYOFF_GAME_RESULTS_PATH,
        exclude_game_ids=target_ids,
    )

    targets = targets.copy()
    playoff_results = load_game_results(config.PLAYOFF_GAME_RESULTS_PATH)
    if playoff_results is not None:
        completed_mask = targets["status"].astype(str).str.lower().isin({"final", "completed"})
        if completed_mask.any():
            completed = merge_game_results(targets.loc[completed_mask].copy(), playoff_results)
            scheduled = targets.loc[~completed_mask].copy()
            targets = sort_games(pd.concat([completed, scheduled], ignore_index=True))

    combined = pd.concat([history, targets], ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    combined = sort_games(combined)

    features, issues = build_pregame_feature_rows(combined)
    finals_features = features.loc[features["game_id"].astype(str).isin(target_ids)].copy()

    if issues:
        print(f"  Feature build notes: {len(issues)} issue(s) logged during history build.")

    ensure_directories()
    if not finals_features.empty:
        save_csv(finals_features, output_path)
        print(f"  Finals pre-game features written: {output_path} ({len(finals_features)} rows)")
    else:
        print("  No Finals pre-game feature rows produced.")

    return finals_features


def predict_finals_pregame_games(
    features_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Score Finals pre-game features with the primary multi-season pre-game model."""
    features_path = features_path or config.FINALS_PREGAME_FEATURES_PATH
    output_path = output_path or config.FINALS_PREGAME_PREDICTIONS_PATH

    if not Path(features_path).exists():
        print(f"  Finals pre-game features not found: {features_path}")
        return pd.DataFrame()

    df = pd.read_csv(
        features_path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )
    if df.empty:
        return pd.DataFrame()

    model_path = config.PREGAME_MODEL_MULTISEASON_PATH
    columns_path = config.PREGAME_FEATURE_COLUMNS_MULTISEASON_PATH
    if not model_path.exists() or not columns_path.exists():
        raise FileNotFoundError(
            "Primary multi-season pre-game model artifacts not found. "
            f"Expected: {model_path} and {columns_path}"
        )

    model = joblib.load(model_path)
    feature_columns = normalize_pregame_feature_columns(joblib.load(columns_path))
    validate_pregame_prediction_inputs(df, feature_columns)
    probabilities = generate_pregame_predictions(model, df, feature_columns)
    predictions = build_pregame_prediction_output(df, probabilities)

    ensure_directories()
    save_csv(predictions, output_path)
    print(f"  Finals pre-game predictions written: {output_path} ({len(predictions)} rows)")
    return predictions


def _pregame_confidence(home_prob: Optional[float]) -> Optional[float]:
    if home_prob is None or pd.isna(home_prob):
        return None
    return round(max(float(home_prob), 1.0 - float(home_prob)), 4)


def build_finals_upcoming_predictions_report(
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
    case_study_path: Optional[Path] = None,
    pregame_predictions_path: Optional[Path] = None,
    pbp_path: Optional[Path] = None,
    overrides_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build one row per Finals game 1–7 with pre-game and replay availability."""
    case_study = build_nba_finals_case_study_summary(
        focus_season=focus_season,
        games_path=config.PLAYOFF_GAMES_PATH,
        predictions_path=config.PLAYOFF_LIVE_PREDICTIONS_PATH,
        results_path=config.PLAYOFF_GAME_RESULTS_PATH,
        pbp_path=pbp_path or config.PLAYOFF_PLAY_BY_PLAY_PATH,
        overrides_path=overrides_path,
    )
    if case_study.empty:
        return pd.DataFrame(columns=UPCOMING_PREDICTIONS_COLUMNS)

    pregame = _load_csv(pregame_predictions_path or config.FINALS_PREGAME_PREDICTIONS_PATH)
    pregame_by_id: Dict[str, pd.Series] = {}
    if pregame is not None and not pregame.empty:
        for _, row in pregame.iterrows():
            pregame_by_id[str(row["game_id"])] = row

    rows: List[dict] = []
    for _, game in case_study.sort_values("finals_game_number").iterrows():
        game_id = str(game.get("game_id", "")).strip()
        game_num = int(game["finals_game_number"])
        status = str(game.get("game_status", "unknown"))
        pred_row = pregame_by_id.get(game_id)

        pregame_available = pred_row is not None and bool(game_id)
        home_prob = None
        away_prob = None
        predicted = None
        confidence = None
        pregame_correct = None

        if pred_row is not None:
            home_prob = float(pred_row["home_win_probability"])
            away_prob = float(pred_row["away_win_probability"])
            predicted = str(pred_row.get("predicted_winner", ""))
            confidence = _pregame_confidence(home_prob)
            if game.get("final_result_available") and pd.notna(pred_row.get("prediction_correct")):
                pregame_correct = bool(pred_row["prediction_correct"])

        notes = str(game.get("notes", "")).strip()
        schedule_source = str(game.get("schedule_source", "")).strip().lower()
        if schedule_source == "manual_schedule" and status == "scheduled":
            notes = notes or "Manual schedule metadata used for pre-game prediction."
        elif status == "if_necessary_scheduled" and pregame_available:
            prefix = (
                "Conditional pre-game prediction available because if-necessary "
                "schedule metadata is known."
            )
            notes = f"{prefix} {notes}".strip() if notes else prefix
        elif status == "if_necessary_scheduled" and not pregame_available:
            notes = notes or (
                "If-necessary scheduled game — prediction depends on projected/actual series path."
            )
        elif not game_id and game_num in (4, 5):
            notes = notes or (
                "Finals game not yet collected or not available in local playoff metadata."
            )
        elif not game_id and game_num >= 6:
            notes = notes or "If necessary — awaiting official schedule/metadata."
        elif game_id and not pregame_available:
            notes = notes or "Pre-game prediction could not be generated for this game."
        elif game_id and pregame_available and not game.get("replay_available"):
            if status in {"scheduled", "if_necessary_scheduled"}:
                notes = notes or (
                    "Pre-game prediction available; live replay appears after "
                    "play-by-play is collected and the pipeline is rerun."
                )

        rows.append(
            {
                "season": focus_season,
                "game_id": game_id,
                "finals_game_number": game_num,
                "game_date": game.get("game_date", ""),
                "home_team": game.get("home_team", ""),
                "away_team": game.get("away_team", ""),
                "game_status": status,
                "pregame_prediction_available": pregame_available,
                "home_win_probability_pregame": home_prob,
                "away_win_probability_pregame": away_prob,
                "predicted_winner_pregame": predicted,
                "prediction_confidence": confidence,
                "final_result_available": bool(game.get("final_result_available")),
                "final_winner": game.get("final_winner"),
                "pregame_prediction_correct_if_final": pregame_correct,
                "replay_available": bool(game.get("replay_available")),
                "notes": notes,
            }
        )

    return pd.DataFrame(rows, columns=UPCOMING_PREDICTIONS_COLUMNS)


def run_build_finals_pregame_predictions(verbose: bool = True) -> int:
    """Build Finals pre-game features, predictions, and upcoming report."""
    ensure_directories()
    if verbose:
        print("Building 2025-26 Finals pre-game predictions (primary multi-season model)...")

    overrides_raw = load_finals_schedule_overrides()
    if verbose:
        if overrides_raw.empty:
            print("  Schedule overrides: none loaded")
        else:
            overrides = normalize_finals_schedule_overrides(overrides_raw)
            season_overrides = overrides.loc[overrides["season"] == CASE_STUDY_FOCUS_SEASON]
            scheduled = int((season_overrides["status"] == "scheduled").sum())
            if_necessary = int((season_overrides["status"] == "if_necessary").sum())
            if_necessary_scheduled = int(
                (season_overrides["status"] == "if_necessary_scheduled").sum()
            )
            print(f"  Schedule overrides loaded: {len(season_overrides)} row(s)")
            print(
                f"    scheduled: {scheduled}, if_necessary: {if_necessary}, "
                f"if_necessary_scheduled: {if_necessary_scheduled}"
            )

    features = build_finals_pregame_features()
    if features.empty:
        if verbose:
            print("  Warning: no Finals feature rows — writing report with placeholders only.")
    else:
        try:
            predict_finals_pregame_games()
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ERROR: {exc}")
            return 1
        if verbose:
            print(f"  Prediction rows generated: {len(features)}")

    report = build_finals_upcoming_predictions_report()
    save_csv(report, config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH)
    if verbose:
        print(f"  Upcoming Finals report: {config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH}")
        print(f"  Rows: {len(report)}")
        available = int(report["pregame_prediction_available"].sum()) if not report.empty else 0
        print(f"  Pre-game predictions available: {available}/{len(report)}")
    return 0


def load_manual_postgame_overrides(
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load manual post-game result overrides (actual results only)."""
    return load_postgame_results(path or config.POSTGAME_RESULTS_PATH)


def get_actual_finals_results(
    upcoming_df: pd.DataFrame,
    playoff_results_path: Optional[Path] = None,
) -> Dict[str, Tuple[str, str]]:
    """Return ``game_id -> (winner, source)`` for completed Finals games."""
    results: Dict[str, Tuple[str, str]] = {}
    playoff_results_path = playoff_results_path or config.PLAYOFF_GAME_RESULTS_PATH
    playoff_results = load_game_results(playoff_results_path)

    if playoff_results is not None and not playoff_results.empty:
        for _, row in playoff_results.iterrows():
            gid = str(row.get("game_id", "")).strip()
            winner = str(row.get("winner", "")).strip()
            if gid and winner:
                results[gid] = (winner, "official/local")

    if upcoming_df is not None and not upcoming_df.empty:
        for _, row in upcoming_df.iterrows():
            gid = str(row.get("game_id", "")).strip()
            if not gid:
                continue
            if bool(row.get("final_result_available")):
                winner = str(row.get("final_winner", "")).strip()
                if winner:
                    results[gid] = (winner, "official/local")

    return results


def apply_manual_overrides_to_finals_results(
    actual_results: Dict[str, Tuple[str, str]],
    manual_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Tuple[str, str]]:
    """Apply manual post-game overrides; manual winner wins over official/local."""
    merged = dict(actual_results)
    manual_df = manual_df if manual_df is not None else load_manual_postgame_overrides()
    if manual_df is None or manual_df.empty:
        return merged
    for _, row in manual_df.iterrows():
        gid = str(row.get("game_id", "")).strip()
        winner = str(row.get("winner", "")).strip()
        if gid and winner:
            merged[gid] = (winner, "manual_override")
    return merged


def _identify_series_teams(games_df: pd.DataFrame) -> Tuple[str, str]:
    """Return two Finals team names in stable sorted order."""
    teams: set[str] = set()
    for col in ["home_team", "away_team"]:
        if col in games_df.columns:
            for value in games_df[col].dropna():
                text = str(value).strip()
                if text:
                    teams.add(text)
    if len(teams) < 2:
        return ("Team A", "Team B")
    ordered = sorted(teams)
    return ordered[0], ordered[1]


def _winner_to_team_slot(winner: str, team_a: str, team_b: str) -> Optional[str]:
    winner = str(winner).strip()
    if winner == team_a:
        return "a"
    if winner == team_b:
        return "b"
    return None


def calculate_series_state_before_game(
    team_a_wins: int,
    team_b_wins: int,
) -> Dict[str, int]:
    """Return a simple wins dict before a Finals game."""
    return {"team_a_wins": team_a_wins, "team_b_wins": team_b_wins}


def _series_over(team_a_wins: int, team_b_wins: int) -> bool:
    return team_a_wins >= FINALS_WINS_TO_CLINCH or team_b_wins >= FINALS_WINS_TO_CLINCH


def project_finals_series_path(
    upcoming_df: pd.DataFrame,
    actual_results: Dict[str, Tuple[str, str]],
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
) -> pd.DataFrame:
    """Simulate a Finals series path using actual results then model projections."""
    if upcoming_df is None or upcoming_df.empty:
        return pd.DataFrame(columns=PROJECTED_SERIES_COLUMNS)

    games = upcoming_df.loc[upcoming_df["season"] == focus_season].copy()
    if games.empty:
        return pd.DataFrame(columns=PROJECTED_SERIES_COLUMNS)

    games = games.sort_values("finals_game_number").reset_index(drop=True)
    team_a, team_b = _identify_series_teams(games)

    team_a_wins = 0
    team_b_wins = 0
    series_over = False
    rows: List[dict] = []

    for _, game in games.iterrows():
        game_num = int(game["finals_game_number"])
        gid = str(game.get("game_id", "")).strip()
        if gid.lower() in {"nan", "none"}:
            gid = ""
        status = str(game.get("game_status", "unknown")).strip().lower()
        home = str(game.get("home_team", "")).strip()
        away = str(game.get("away_team", "")).strip()
        predicted = str(game.get("predicted_winner_pregame", "")).strip() or None
        pregame_available = bool(game.get("pregame_prediction_available"))

        ta_before = team_a_wins
        tb_before = team_b_wins

        actual_winner = None
        actual_source = "none"
        if gid and gid in actual_results:
            actual_winner, actual_source = actual_results[gid]

        game_needed = not series_over
        if game_num == 7 and not (ta_before == 3 and tb_before == 3):
            game_needed = False
        projected_winner: Optional[str] = None
        projection_type = "not_available"
        notes = ""
        is_conditional_status = status in {"if_necessary", "if_necessary_scheduled"}

        if series_over:
            game_needed = False
            projection_type = "not_needed_under_projection"
            notes = "Series already decided under projected path."
        elif actual_winner:
            projected_winner = actual_winner
            projection_type = (
                "actual_manual_override"
                if actual_source == "manual_override"
                else "actual_official"
            )
        elif game_num == 7 and not series_over and not (ta_before == 3 and tb_before == 3):
            projection_type = "if_necessary_pending"
            notes = "Game 7 depends on Game 6 under this path."
        elif status in {"scheduled", "if_necessary", "if_necessary_scheduled", "not_available_yet"}:
            if game_needed and pregame_available and predicted:
                projected_winner = predicted
                if status == "if_necessary_scheduled":
                    projection_type = "conditional_model_projected"
                    notes = (
                        "Conditional model projection — game only needed if the "
                        "series remains alive under this path."
                    )
                else:
                    projection_type = "model_projected"
            elif game_needed and status == "if_necessary":
                projection_type = "needed_but_prediction_unavailable"
                notes = (
                    f"Game {game_num} would be necessary under this path, but a "
                    "pre-game prediction requires schedule metadata."
                )
            elif game_needed:
                projection_type = "needed_but_prediction_unavailable"
                notes = "Pre-game prediction unavailable for this game."
            elif is_conditional_status:
                projection_type = "if_necessary_pending"
                notes = "If necessary — awaiting official schedule/metadata."
            else:
                projection_type = "not_needed_under_projection"
        elif status == "completed" and not actual_winner:
            projection_type = "not_available"
            notes = "Completed game missing actual winner in local data."

        if projected_winner:
            slot = _winner_to_team_slot(projected_winner, team_a, team_b)
            if slot == "a":
                team_a_wins += 1
            elif slot == "b":
                team_b_wins += 1

        if _series_over(team_a_wins, team_b_wins):
            series_over = True

        rows.append(
            {
                "season": focus_season,
                "finals_game_number": game_num,
                "game_id": gid,
                "game_date": game.get("game_date", ""),
                "home_team": home,
                "away_team": away,
                "game_status": status,
                "actual_winner": actual_winner,
                "actual_result_source": actual_source,
                "predicted_winner_pregame": predicted,
                "projected_winner_used": projected_winner,
                "projection_type": projection_type,
                "series_team_a": team_a,
                "series_team_b": team_b,
                "team_a_wins_before": ta_before,
                "team_b_wins_before": tb_before,
                "team_a_wins_after": team_a_wins,
                "team_b_wins_after": team_b_wins,
                "game_needed_under_projection": game_needed,
                "series_over_after_game": series_over,
                "notes": notes,
            }
        )

    return pd.DataFrame(rows, columns=PROJECTED_SERIES_COLUMNS)


def build_finals_projected_series_report(
    focus_season: str = CASE_STUDY_FOCUS_SEASON,
    upcoming_path: Optional[Path] = None,
    manual_path: Optional[Path] = None,
    playoff_results_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build projected Finals series path report from upcoming predictions + actuals."""
    upcoming_path = upcoming_path or config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH
    upcoming = _load_csv(upcoming_path)
    if upcoming is None or upcoming.empty:
        return pd.DataFrame(columns=PROJECTED_SERIES_COLUMNS)

    actual = get_actual_finals_results(upcoming, playoff_results_path=playoff_results_path)
    manual_df = load_manual_postgame_overrides(manual_path)
    actual = apply_manual_overrides_to_finals_results(actual, manual_df)
    return project_finals_series_path(upcoming, actual, focus_season=focus_season)


def run_build_finals_projected_series_path(verbose: bool = True) -> int:
    """Write the projected Finals series path report."""
    ensure_directories()
    report = build_finals_projected_series_report()
    save_csv(report, config.FINALS_PROJECTED_SERIES_PATH)
    if verbose:
        print(f"Finals projected series path written to {config.FINALS_PROJECTED_SERIES_PATH}")
        print(f"  Rows: {len(report)}")
        if not report.empty:
            actual_rows = report.loc[
                report["projection_type"].astype(str).str.startswith("actual_")
            ]
            projected_rows = report.loc[
                report["projection_type"].isin(["model_projected", "conditional_model_projected"])
            ]
            print(f"  Actual results used: {len(actual_rows)}")
            print(f"  Model-projected steps: {len(projected_rows)}")
    return 0
