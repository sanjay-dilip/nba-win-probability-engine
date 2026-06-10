"""Pure helper functions for Streamlit dashboard pages (Build 10+).

These helpers contain no Streamlit imports so they can be unit-tested without
running the app.  Dashboard pages import from here to keep UI code readable.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Columns the Live Replay page requires in live_predictions.csv.
REQUIRED_LIVE_PREDICTION_COLUMNS = [
    "game_id",
    "event_num",
    "season",
    "game_date",
    "home_team",
    "away_team",
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

# Subset shown in the play-by-play event table on the Live Replay page.
EVENT_TABLE_COLUMNS = [
    "event_num",
    "period",
    "pctimestring",
    "home_score",
    "away_score",
    "event_type_label",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "prediction_correct",
]

MOMENTUM_SWING_COLUMNS = [
    "period",
    "pctimestring",
    "home_score",
    "away_score",
    "event_type_label",
    "home_win_probability",
    "probability_change",
]


def missing_required_columns(
    df: pd.DataFrame,
    required: Sequence[str] = REQUIRED_LIVE_PREDICTION_COLUMNS,
) -> List[str]:
    """Return required column names that are absent from ``df``."""
    return [col for col in required if col not in df.columns]


def build_game_label(row: pd.Series) -> str:
    """Build a readable game selector label from one catalog row.

    Example: ``2024-11-12 | Boston Celtics vs Atlanta Hawks``
    """
    return f"{row['game_date']} | {row['home_team']} vs {row['away_team']}"


def build_games_catalog(predictions: pd.DataFrame) -> pd.DataFrame:
    """One row per game with a readable ``label`` for selectors.

    Args:
        predictions: Live prediction rows (may contain many events per game).

    Returns:
        A DataFrame sorted by ``game_date`` then ``game_id`` with columns
        ``game_id``, ``season``, ``game_date``, ``home_team``, ``away_team``,
        ``label``.
    """
    cols = ["game_id", "season", "game_date", "home_team", "away_team"]
    catalog = predictions[cols].drop_duplicates(subset="game_id").copy()
    catalog["label"] = catalog.apply(build_game_label, axis=1)
    return catalog.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def filter_games_catalog(
    catalog: pd.DataFrame,
    season: Optional[str] = None,
    team: Optional[str] = None,
) -> pd.DataFrame:
    """Filter the games catalog by season and/or team name.

    Args:
        catalog: Output of :func:`build_games_catalog`.
        season: If given, keep only this season label.
        team: If given and not ``"All teams"``, keep games where the team is
            home or away.

    Returns:
        A filtered catalog (may be empty).
    """
    out = catalog.copy()
    if season is not None and season != "All seasons":
        out = out[out["season"] == season]
    if team is not None and team != "All teams":
        mask = (out["home_team"] == team) | (out["away_team"] == team)
        out = out[mask]
    return out.reset_index(drop=True)


def sort_game_events(game_df: pd.DataFrame) -> pd.DataFrame:
    """Sort one game's events in chronological replay order."""
    return game_df.sort_values("event_num").reset_index(drop=True)


def compute_game_elapsed_seconds(seconds_remaining_game: pd.Series) -> pd.Series:
    """Convert countdown seconds into elapsed seconds (left-to-right chart axis).

    ``game_elapsed_seconds = max(seconds_remaining_game) - seconds_remaining_game``
    within the game so the chart progresses from tip-off toward the end.
    """
    secs = pd.to_numeric(seconds_remaining_game, errors="coerce")
    return secs.max() - secs


def add_replay_columns(game_df: pd.DataFrame) -> pd.DataFrame:
    """Add ``game_elapsed_seconds`` and probability-change columns for one game."""
    out = sort_game_events(game_df).copy()
    out["game_elapsed_seconds"] = compute_game_elapsed_seconds(out["seconds_remaining_game"])
    out["probability_change"] = out["home_win_probability"].diff().fillna(0.0)
    out["absolute_probability_change"] = out["probability_change"].abs()
    return out


def top_momentum_swings(game_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the top ``n`` events by absolute home-win probability change."""
    enriched = add_replay_columns(game_df)
    cols = [c for c in MOMENTUM_SWING_COLUMNS if c in enriched.columns]
    top = enriched.nlargest(n, "absolute_probability_change")
    return top[cols].reset_index(drop=True)


def get_final_event_row(game_df: pd.DataFrame) -> pd.Series:
    """Return the last event row for a game (by ``event_num``)."""
    return sort_game_events(game_df).iloc[-1]


def format_score(home_score: object, away_score: object) -> str:
    """Format a readable ``HOME - AWAY`` score string."""
    return f"{int(float(home_score))} – {int(float(away_score))}"


def lookup_game_result(
    game_results: Optional[pd.DataFrame],
    game_id: str,
) -> Optional[pd.Series]:
    """Return the game_results row for ``game_id``, or ``None`` if unavailable."""
    if game_results is None or game_results.empty:
        return None
    match = game_results.loc[game_results["game_id"] == str(game_id)]
    if match.empty:
        return None
    return match.iloc[0]


# ---------------------------------------------------------------------------
# Pre-game Predictor helpers (Build 11)
# ---------------------------------------------------------------------------

REQUIRED_PREGAME_PREDICTION_COLUMNS = [
    "game_id",
    "season",
    "game_date",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "game_type",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "predicted_label",
    "actual_home_team_won",
    "prediction_correct",
]

PREGAME_TABLE_COLUMNS = [
    "game_date",
    "home_team",
    "away_team",
    "home_win_probability",
    "away_win_probability",
    "predicted_winner",
    "prediction_correct",
]

# Feature sections shown on the Pre-game Predictor page (label -> column name).
PREGAME_FEATURE_SECTIONS = {
    "Season record before game": {
        "Home games played": "home_games_played_before",
        "Away games played": "away_games_played_before",
        "Home win %": "home_win_pct_before",
        "Away win %": "away_win_pct_before",
        "Win % diff (home)": "win_pct_diff_before",
    },
    "Scoring form before game": {
        "Home pts for avg": "home_points_for_avg_before",
        "Away pts for avg": "away_points_for_avg_before",
        "Pts for diff (home)": "points_for_avg_diff_before",
        "Home pts allowed avg": "home_points_allowed_avg_before",
        "Away pts allowed avg": "away_points_allowed_avg_before",
        "Pts allowed diff (home)": "points_allowed_avg_diff_before",
    },
    "Recent form": {
        "Home recent win %": "home_recent_win_pct_before",
        "Away recent win %": "away_recent_win_pct_before",
        "Recent win % diff (home)": "recent_win_pct_diff_before",
    },
    "Rest advantage": {
        "Home rest days": "home_rest_days",
        "Away rest days": "away_rest_days",
        "Rest days diff (home)": "rest_days_diff",
    },
}

# Metrics compared in the home-vs-away horizontal bar chart.
PREGAME_COMPARISON_METRICS = [
    ("Win %", "home_win_pct_before", "away_win_pct_before"),
    ("Pts for avg", "home_points_for_avg_before", "away_points_for_avg_before"),
    ("Pts allowed avg", "home_points_allowed_avg_before", "away_points_allowed_avg_before"),
    ("Recent win %", "home_recent_win_pct_before", "away_recent_win_pct_before"),
    ("Rest days", "home_rest_days", "away_rest_days"),
]


def build_pregame_feature_sections(feature_row: pd.Series) -> dict[str, dict[str, object]]:
    """Group pre-game feature values into readable sections for display.

    Missing columns or NaN values are omitted from each section.
    """
    sections: dict[str, dict[str, object]] = {}
    for section_name, fields in PREGAME_FEATURE_SECTIONS.items():
        section_data: dict[str, object] = {}
        for label, col in fields.items():
            if col not in feature_row.index:
                continue
            val = feature_row[col]
            if pd.isna(val):
                continue
            section_data[label] = val
        if section_data:
            sections[section_name] = section_data
    return sections


def build_team_comparison_frame(feature_row: pd.Series) -> pd.DataFrame:
    """Build a long-form DataFrame for home-vs-away comparison charts.

    Returns columns ``metric``, ``team``, ``value``.  Skips metrics whose
    columns are missing or NaN.
    """
    home_team = str(feature_row.get("home_team", "Home"))
    away_team = str(feature_row.get("away_team", "Away"))
    rows: list[dict] = []
    for metric_label, home_col, away_col in PREGAME_COMPARISON_METRICS:
        if home_col not in feature_row.index or away_col not in feature_row.index:
            continue
        home_val = feature_row[home_col]
        away_val = feature_row[away_col]
        if pd.isna(home_val) and pd.isna(away_val):
            continue
        if pd.notna(home_val):
            rows.append({"metric": metric_label, "team": home_team, "value": float(home_val)})
        if pd.notna(away_val):
            rows.append({"metric": metric_label, "team": away_team, "value": float(away_val)})
    return pd.DataFrame(rows)


def compute_pregame_filtered_summary(predictions: pd.DataFrame) -> dict:
    """Summarise the currently filtered pre-game prediction rows."""
    summary: dict = {"game_count": len(predictions)}
    if predictions.empty:
        return summary

    summary["avg_home_win_probability"] = float(predictions["home_win_probability"].mean())

    actual = predictions["actual_home_team_won"]
    labeled = predictions.loc[actual.notna()]
    if not labeled.empty and "prediction_correct" in labeled.columns:
        correct = labeled["prediction_correct"]
        valid = correct.notna()
        if valid.any():
            summary["prediction_accuracy"] = float(correct[valid].astype(bool).mean())

    probs = predictions["home_win_probability"]
    confidence = (probs - 0.5).abs()
    most_idx = confidence.idxmax()
    closest_idx = confidence.idxmin()
    summary["most_confident_game"] = build_game_label(predictions.loc[most_idx])
    summary["most_confident_home_prob"] = float(probs.loc[most_idx])
    summary["closest_game"] = build_game_label(predictions.loc[closest_idx])
    summary["closest_home_prob"] = float(probs.loc[closest_idx])
    return summary


def lookup_pregame_features(
    features: Optional[pd.DataFrame],
    game_id: str,
) -> Optional[pd.Series]:
    """Return the pregame_features row for ``game_id``, or ``None``."""
    if features is None or features.empty:
        return None
    match = features.loc[features["game_id"] == str(game_id)]
    if match.empty:
        return None
    return match.iloc[0]


# ---------------------------------------------------------------------------
# Data freshness helpers (Build 14)
# ---------------------------------------------------------------------------

FRESHNESS_CHECK_CMD = "python run_pipeline.py --mode check_data_freshness"

STATUS_ICONS = {
    "ok": "✅",
    "missing": "❌",
    "stale": "⚠️",
    "warning": "⚠️",
    "not_required": "—",
    "unknown": "❓",
}


def load_freshness_summary(path: Path) -> Optional[pd.DataFrame]:
    """Load a saved data freshness summary CSV, or ``None`` if missing."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def get_asset_status(summary_df: Optional[pd.DataFrame], asset_name: str) -> str:
    """Return the status string for one asset from a freshness summary."""
    if summary_df is None or summary_df.empty:
        return "unknown"
    match = summary_df.loc[summary_df["asset"] == asset_name, "status"]
    if match.empty:
        return "unknown"
    return str(match.iloc[0])


# ---------------------------------------------------------------------------
# Display formatting helpers (product UI)
# ---------------------------------------------------------------------------

_ISO_DURATION_RE = re.compile(
    r"^PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?$",
    re.IGNORECASE,
)

METRIC_DISPLAY_NAMES = {
    "accuracy": "Accuracy",
    "roc_auc": "ROC-AUC",
    "log_loss": "Log Loss",
    "brier_score": "Brier Score",
    "true_positive": "True Positives",
    "true_negative": "True Negatives",
    "false_positive": "False Positives",
    "false_negative": "False Negatives",
    "n_train": "Training Games",
    "n_test": "Test Games",
    "train_games": "Training Games",
    "test_games": "Test Games",
    "n_features": "Features",
    "row_count": "Events",
}


def format_pct(value: object, decimals: int = 1) -> str:
    """Format a probability in ``[0, 1]`` as a percentage string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{float(value) * 100:.{decimals}f}%"


def format_metric_value(value: object, metric_name: Optional[str] = None) -> str:
    """Format a metric for dashboard display."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"

    numeric = float(value)
    if metric_name in {"accuracy", "roc_auc"} and 0 <= numeric <= 1:
        return format_pct(numeric, decimals=1)
    if metric_name in {"log_loss", "brier_score"}:
        return f"{numeric:.4f}"
    if 0 <= numeric <= 1:
        return format_pct(numeric)
    if numeric == int(numeric):
        return str(int(numeric))
    return f"{numeric:.4f}"


def format_iso_duration_to_clock(value: object) -> str:
    """Convert ISO-8601 durations like ``PT04M20.00S`` to ``4:20``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"

    text = str(value).strip()
    if not text:
        return "—"
    if re.match(r"^\d+:\d{2}$", text):
        return text

    match = _ISO_DURATION_RE.match(text)
    if not match:
        return text

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(float(match.group("seconds") or 0))
    total_minutes = hours * 60 + minutes
    return f"{total_minutes}:{seconds:02d}"


def format_game_clock(value: object) -> str:
    """Format a game clock value for display."""
    return format_iso_duration_to_clock(value)


def format_large_number(value: object) -> str:
    """Format large counts compactly (e.g. 1.2M, 45.3K)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"

    number = float(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:.1f}M"
    if abs(number) >= 1_000:
        return f"{number / 1_000:.1f}K"
    if number == int(number):
        return str(int(number))
    return f"{number:.1f}"


def format_status_text(status: str) -> str:
    """Return a readable status label without icons."""
    labels = {
        "ok": "Ready",
        "missing": "Not available",
        "stale": "Needs refresh",
        "warning": "Check recommended",
        "not_required": "Optional",
        "unknown": "Unknown",
    }
    return labels.get(status, status.replace("_", " ").title())


def safe_metric_delta(value: object) -> Optional[float]:
    """Return a numeric delta when available, otherwise ``None``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def shorten_text(value: object, max_chars: int = 80) -> str:
    """Truncate long text with an ellipsis."""
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def metric_display_name(metric: str) -> str:
    """Return a human-readable metric label."""
    return METRIC_DISPLAY_NAMES.get(metric, metric.replace("_", " ").title())


def page_intro_subtitle(text: str) -> str:
    """Return subtitle text for a dashboard page header."""
    return text.strip()


def section_header_text(title: str, caption: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Return section title and optional caption for page rendering."""
    return title, caption


# ---------------------------------------------------------------------------
# Model role labels (primary vs baseline)
# ---------------------------------------------------------------------------

PRIMARY_MODEL_LABEL = "Primary multi-season model"
BASELINE_MODEL_LABEL = "Baseline"

PRIMARY_MODEL_TRAIN_SEASONS = "2022-23, 2023-24, 2024-25"
PRIMARY_MODEL_TEST_SEASON = "2025-26"

PRIMARY_MODEL_DESCRIPTION = (
    "Trained on 2022-23 through 2024-25 and evaluated on a 2025-26 future-season holdout — "
    "the primary evaluation target and closer to real deployment."
)
BASELINE_MODEL_DESCRIPTION = (
    "Original single-season model with a 2024-25 chronological hold-out split, "
    "retained to validate the end-to-end pipeline."
)

DEMO_PREDICTION_NOTE = (
    "This page uses prepared prediction outputs for interactive exploration. "
    "Primary model evaluation is shown on the Model Performance page. "
    "For the current 2025-26 NBA Finals prediction showcase, use the **2025-26 NBA Finals** page."
)

FINALS_PREGAME_NOTE = (
    "Pre-game predictions use known schedule/matchup metadata and historical performance "
    "before tip-off. Live replay appears only after play-by-play is collected."
)


def model_role_display_name(role: str) -> str:
    """Map internal role keys to user-facing model labels."""
    mapping = {
        "primary": PRIMARY_MODEL_LABEL,
        "multiseason": PRIMARY_MODEL_LABEL,
        "baseline": BASELINE_MODEL_LABEL,
        "single_season": BASELINE_MODEL_LABEL,
    }
    return mapping.get(role, role.replace("_", " ").title())


def comparison_column_labels() -> Tuple[str, str]:
    """Return (baseline column label, primary column label) for comparison tables."""
    return BASELINE_MODEL_LABEL, PRIMARY_MODEL_LABEL


def format_status_label(status: str) -> str:
    """Return a compact display label such as ``✅ Ready``."""
    icon = STATUS_ICONS.get(status, "❓")
    return f"{icon} {format_status_text(status)}"


def get_status_items(
    summary_df: Optional[pd.DataFrame],
    asset_names: Sequence[str],
) -> List[tuple[str, str]]:
    """Return ``(asset_name, status)`` pairs for compact dashboard panels."""
    return [(name, get_asset_status(summary_df, name)) for name in asset_names]


def format_status_panel_text(
    summary_df: Optional[pd.DataFrame],
    asset_names: Sequence[str],
) -> str:
    """Build one-line status text for a dashboard page header."""
    if summary_df is None:
        return f"System status unavailable — run `{FRESHNESS_CHECK_CMD}`."
    items = get_status_items(summary_df, asset_names)
    return " · ".join(f"{name}: {format_status_label(status)}" for name, status in items)


# ---------------------------------------------------------------------------
# 2025-26 NBA Finals showcase helpers (Build 20.8)
# ---------------------------------------------------------------------------

FINALS_SHOWCASE_SEASON = "2025-26"


def load_optional_csv(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    """Load a CSV if present; return None when missing or empty."""
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


def format_finals_matchup(row: pd.Series) -> str:
    """Format away @ home for a Finals summary row."""
    home = str(row.get("home_team", "")).strip() or "TBD"
    away = str(row.get("away_team", "")).strip() or "TBD"
    if home == "TBD" and away == "TBD":
        return "TBD"
    return f"{away} @ {home}"


def format_finals_game_selector_label(row: pd.Series) -> str:
    """Build a selector label such as ``Game 3: NYK @ SAS (completed)``."""
    num = row.get("finals_game_number", "?")
    status = str(row.get("game_status", "unknown"))
    suffix = ""
    if num >= 6 and status in {"not_available_yet", "if_necessary", "if_necessary_scheduled"}:
        suffix = ", if necessary"
    status_label = format_game_status_for_finals(status)
    return f"Game {num}: {format_finals_matchup(row)} ({status_label}{suffix})"


def build_finals_overview_display(
    summary_df: pd.DataFrame,
    upcoming_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return a compact overview table for Games 1–7."""
    if summary_df.empty:
        return pd.DataFrame()

    work = summary_df.sort_values("finals_game_number").copy()
    if upcoming_df is not None and not upcoming_df.empty:
        up = upcoming_df.set_index("finals_game_number")
        work["pregame"] = work["finals_game_number"].map(
            lambda n: "Yes"
            if n in up.index and bool(up.loc[n].get("pregame_prediction_available"))
            else "No"
        )
        work["pregame_winner"] = work["finals_game_number"].map(
            lambda n: up.loc[n].get("predicted_winner_pregame", "—")
            if n in up.index
            else "—"
        )
    else:
        work["pregame"] = "No"
        work["pregame_winner"] = "—"

    work["matchup"] = work.apply(format_finals_matchup, axis=1)
    work["status"] = work["game_status"].map(format_game_status_for_finals)
    work["replay"] = work["replay_available"].map(lambda x: "Yes" if bool(x) else "No")
    work["start_prediction"] = work["predicted_winner_start"].fillna("—")
    work["final_prediction"] = work["predicted_winner_final"].fillna("—")
    work["winner"] = work["final_winner"].fillna("—")

    columns = {
        "finals_game_number": "Game",
        "game_date": "Date",
        "matchup": "Matchup",
        "status": "Status",
        "pregame": "Pre-game pred.",
        "pregame_winner": "Pre-game pick",
        "replay": "Replay",
        "winner": "Final winner",
        "start_prediction": "Live start",
        "final_prediction": "Live final",
    }
    return work[list(columns.keys())].rename(columns=columns)


def find_next_upcoming_finals_prediction(
    upcoming_df: pd.DataFrame,
) -> Optional[pd.Series]:
    """Return the lowest-numbered Finals game without a result but with pre-game prediction."""
    if upcoming_df is None or upcoming_df.empty:
        return None
    candidates = upcoming_df.loc[
        (~upcoming_df["final_result_available"].astype(bool))
        & (upcoming_df["pregame_prediction_available"].astype(bool))
        & (upcoming_df["game_id"].astype(str).str.strip() != "")
    ].sort_values("finals_game_number")
    if candidates.empty:
        return None
    return candidates.iloc[0]


def merge_finals_display_row(
    summary_row: pd.Series,
    upcoming_row: Optional[pd.Series],
) -> pd.Series:
    """Combine case-study summary and upcoming pre-game fields for one game."""
    out = summary_row.copy()
    if upcoming_row is None:
        return out
    for col in [
        "pregame_prediction_available",
        "home_win_probability_pregame",
        "away_win_probability_pregame",
        "predicted_winner_pregame",
        "prediction_confidence",
        "pregame_prediction_correct_if_final",
    ]:
        if col in upcoming_row.index:
            out[col] = upcoming_row[col]
    return out


def build_finals_win_probability_chart(
    game_df: pd.DataFrame,
    home_team: str,
) -> "go.Figure":
    """Plot home win probability over elapsed game time for a Finals replay."""
    import plotly.graph_objects as go

    enriched = add_replay_columns(game_df)
    hover_cols = ["period", "pctimestring", "home_score", "away_score", "event_type_label"]
    available_hover = [c for c in hover_cols if c in enriched.columns]
    customdata = enriched[available_hover].values if available_hover else None
    hovertemplate = (
        "Period: %{customdata[0]}<br>"
        "Clock: %{customdata[1]}<br>"
        "Score: %{customdata[2]} – %{customdata[3]}<br>"
        "Home win prob: %{y:.1%}<extra></extra>"
        if len(available_hover) >= 4
        else "Home win prob: %{y:.1%}<extra></extra>"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=enriched["game_elapsed_seconds"],
            y=enriched["home_win_probability"],
            mode="lines",
            name=f"{home_team} win probability",
            customdata=customdata,
            hovertemplate=hovertemplate,
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        title=f"{home_team} win probability",
        xaxis_title="Elapsed game time (seconds)",
        yaxis_title="Home win probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


def finals_game_can_show_replay(row: pd.Series) -> bool:
    """True when a Finals row has replay data and a valid game_id."""
    game_id = str(row.get("game_id", "")).strip()
    if not game_id:
        return False
    return bool(row.get("replay_available")) and bool(row.get("prediction_available"))


# ---------------------------------------------------------------------------
# Projected Finals series path helpers (Build 20.9.2)
# ---------------------------------------------------------------------------

PROJECTION_TYPE_LABELS = {
    "actual_official": "Actual result",
    "actual_manual_override": "Actual (manual correction)",
    "model_projected": "Model projection",
    "conditional_model_projected": "Conditional model projection",
    "needed_but_prediction_unavailable": "Needed — prediction unavailable",
    "if_necessary_pending": "If necessary — pending metadata",
    "not_needed_under_projection": "Not needed under projection",
    "not_available": "Not available",
}

FINALS_GAME_STATUS_LABELS = {
    "completed": "Completed",
    "scheduled": "Scheduled",
    "if_necessary": "If necessary",
    "if_necessary_scheduled": "If necessary, scheduled",
    "not_available_yet": "Not available yet",
}


def format_game_status_for_finals(status: str) -> str:
    """Return a readable Finals game status label."""
    key = str(status).strip().lower()
    return FINALS_GAME_STATUS_LABELS.get(key, key.replace("_", " ").title())


def format_projection_type(projection_type: str) -> str:
    """Return a readable label for a projection type code."""
    key = str(projection_type).strip().lower()
    return PROJECTION_TYPE_LABELS.get(key, key.replace("_", " ").title())


def format_series_score(team_a: str, team_b: str, wins_a: int, wins_b: int) -> str:
    """Format a series score such as ``Knicks 2 – 1 Spurs``."""
    return f"{team_a} {int(wins_a)} – {int(wins_b)} {team_b}"


def find_series_leader(
    team_a: str,
    team_b: str,
    wins_a: int,
    wins_b: int,
) -> Tuple[str, int, int]:
    """Return ``(leader_name, leader_wins, trailer_wins)``; tied leaders use empty name."""
    wins_a = int(wins_a)
    wins_b = int(wins_b)
    if wins_a > wins_b:
        return team_a, wins_a, wins_b
    if wins_b > wins_a:
        return team_b, wins_b, wins_a
    return "", wins_a, wins_b


def summarize_projected_series_state(path_df: pd.DataFrame) -> dict:
    """Summarize actual vs projected series state from a projected-path report."""
    empty = {
        "team_a": "",
        "team_b": "",
        "actual_score_text": "—",
        "actual_leader": "",
        "projected_score_text": "—",
        "projected_leader": "",
        "last_actual_game": None,
        "next_projected_game": None,
    }
    if path_df is None or path_df.empty:
        return empty

    work = path_df.sort_values("finals_game_number").copy()
    team_a = str(work.iloc[0].get("series_team_a", ""))
    team_b = str(work.iloc[0].get("series_team_b", ""))

    actual_a = 0
    actual_b = 0
    last_actual = None
    for _, row in work.iterrows():
        ptype = str(row.get("projection_type", ""))
        if not ptype.startswith("actual_"):
            continue
        winner = str(row.get("actual_winner", "")).strip()
        if winner == team_a:
            actual_a += 1
        elif winner == team_b:
            actual_b += 1
        last_actual = int(row["finals_game_number"])

    proj_a = int(work.iloc[-1].get("team_a_wins_after", 0))
    proj_b = int(work.iloc[-1].get("team_b_wins_after", 0))
    actual_leader, _, _ = find_series_leader(team_a, team_b, actual_a, actual_b)
    proj_leader, _, _ = find_series_leader(team_a, team_b, proj_a, proj_b)

    next_proj = work.loc[
        (work["projection_type"].isin(["model_projected", "conditional_model_projected"]))
        & (~work["series_over_after_game"].astype(bool))
    ]
    next_game = None
    if not next_proj.empty:
        next_game = int(next_proj.iloc[0]["finals_game_number"])

    return {
        "team_a": team_a,
        "team_b": team_b,
        "actual_score_text": format_series_score(team_a, team_b, actual_a, actual_b),
        "actual_leader": actual_leader,
        "projected_score_text": format_series_score(team_a, team_b, proj_a, proj_b),
        "projected_leader": proj_leader,
        "last_actual_game": last_actual,
        "next_projected_game": next_game,
    }


def build_projected_path_display_table(path_df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact display table for the projected series path section."""
    if path_df is None or path_df.empty:
        return pd.DataFrame()

    work = path_df.sort_values("finals_game_number").copy()
    work["Path label"] = work["projection_type"].map(format_projection_type)
    work["Score before"] = work.apply(
        lambda r: f"{int(r['team_a_wins_before'])}–{int(r['team_b_wins_before'])}",
        axis=1,
    )
    work["Score after"] = work.apply(
        lambda r: f"{int(r['team_a_wins_after'])}–{int(r['team_b_wins_after'])}",
        axis=1,
    )
    work["Matchup"] = work.apply(
        lambda r: f"{r.get('away_team', 'TBD')} @ {r.get('home_team', 'TBD')}",
        axis=1,
    )
    columns = {
        "finals_game_number": "Game",
        "Matchup": "Matchup",
        "Path label": "Path type",
        "actual_winner": "Actual winner",
        "predicted_winner_pregame": "Pre-game pick",
        "projected_winner_used": "Used in path",
        "Score before": "Score before",
        "Score after": "Score after",
        "game_needed_under_projection": "Needed",
        "notes": "Notes",
    }
    return work[list(columns.keys())].rename(columns=columns)

