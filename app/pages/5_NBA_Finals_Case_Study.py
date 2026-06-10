"""NBA Finals Case Study page — primary model on playoff games."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    EVENT_TABLE_COLUMNS,
    PRIMARY_MODEL_LABEL,
    PRIMARY_MODEL_TRAIN_SEASONS,
    add_replay_columns,
    format_game_clock,
    format_pct,
    DEPLOY_FINALS_REPLAY_NOTE,
    format_score,
    format_status_text,
    page_intro_subtitle,
    resolve_playoff_live_predictions_source,
)
from src import config  # noqa: E402
from src.playoff_case_study import CASE_STUDY_FOCUS_SEASON  # noqa: E402

st.set_page_config(page_title="NBA Finals Case Study", page_icon="🏆", layout="wide")

st.title("🏆 NBA Finals Case Study")
st.markdown(
    page_intro_subtitle(
        "See how the primary multi-season model behaves on playoff and Finals games."
    )
)

st.markdown(
    f"The **{PRIMARY_MODEL_LABEL}** is trained on regular-season games "
    f"({PRIMARY_MODEL_TRAIN_SEASONS}) and applied here to playoffs as an "
    "**out-of-distribution case study**. Playoff performance is **not** the same "
    "benchmark as regular-season holdout evaluation on the Model Performance page."
)


@st.cache_data(show_spinner=False)
def load_optional_csv(path: str) -> Optional[pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        return pd.read_csv(csv_path, dtype={"game_id": str})
    except pd.errors.EmptyDataError:
        return None


def asset_status(path: Path) -> str:
    if not path.exists():
        return "missing"
    df = load_optional_csv(str(path))
    if df is None or df.empty:
        return "empty"
    return "ok"


def build_probability_chart(game_df: pd.DataFrame, home_team: str) -> go.Figure:
    enriched = add_replay_columns(game_df)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=enriched["game_elapsed_seconds"],
            y=enriched["home_win_probability"],
            mode="lines",
            name=f"{home_team} win probability",
            customdata=enriched[
                ["period", "pctimestring", "home_score", "away_score", "event_type_label"]
            ].values,
            hovertemplate=(
                "Period: %{customdata[0]}<br>"
                "Clock: %{customdata[1]}<br>"
                "Score: %{customdata[2]} – %{customdata[3]}<br>"
                "Home win prob: %{y:.1%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.6)
    fig.update_layout(
        title=f"{home_team} win probability (playoffs)",
        xaxis_title="Elapsed game time (seconds)",
        yaxis_title="Home win probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


games_df = load_optional_csv(str(config.PLAYOFF_GAMES_PATH))
pbp_df = load_optional_csv(str(config.PLAYOFF_PLAY_BY_PLAY_PATH))
predictions_path, using_finals_deploy = resolve_playoff_live_predictions_source()
predictions_df = load_optional_csv(str(predictions_path))
case_study_df = load_optional_csv(str(config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH))

if using_finals_deploy and predictions_df is not None and not predictions_df.empty:
    st.caption(DEPLOY_FINALS_REPLAY_NOTE)

st.markdown("### Coverage & status")
status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    st.metric("Playoff metadata", format_status_text(asset_status(config.PLAYOFF_GAMES_PATH)))
with status_col2:
    st.metric("Playoff play-by-play", format_status_text(asset_status(config.PLAYOFF_PLAY_BY_PLAY_PATH)))
with status_col3:
    st.metric("Playoff predictions", format_status_text(asset_status(predictions_path)))

if case_study_df is None or case_study_df.empty:
    focus_games = pd.DataFrame()
    if games_df is not None and not games_df.empty:
        from src.playoff_case_study import add_finals_game_numbers, filter_playoff_games

        playoffs = filter_playoff_games(games_df)
        labeled = add_finals_game_numbers(playoffs)
        focus_games = labeled.loc[
            (labeled["season"] == CASE_STUDY_FOCUS_SEASON)
            & (labeled["playoff_round"] == "NBA Finals")
        ].copy()
else:
    focus_games = case_study_df.loc[
        case_study_df["season"] == CASE_STUDY_FOCUS_SEASON
    ].copy()

if focus_games.empty:
    st.info(
        f"No {CASE_STUDY_FOCUS_SEASON} NBA Finals games found yet. "
        "Collect playoff metadata and run the case-study pipeline:\n\n"
        "`python run_pipeline.py --mode collect_playoff_games --seasons 2022-23 2023-24 2024-25 2025-26`\n\n"
        "`python run_pipeline.py --mode run_playoff_case_study_pipeline --seasons 2022-23 2023-24 2024-25 2025-26`"
    )
    st.stop()

focus_games = focus_games.sort_values("finals_game_number")
game_options = []
for _, row in focus_games.iterrows():
    num = row.get("finals_game_number", "?")
    home = row.get("home_team", "TBD")
    away = row.get("away_team", "TBD")
    status = row.get("game_status", "unknown")
    game_options.append((f"Game {num}: {away} @ {home} ({status})", row))

st.markdown(f"### {CASE_STUDY_FOCUS_SEASON} NBA Finals")
labels = [opt[0] for opt in game_options]
selected_label = st.selectbox("Select a Finals game", labels)
selected_row = dict(next(row for label, row in game_options if label == selected_label))

game_id = str(selected_row.get("game_id", "")).strip()
game_status = str(selected_row.get("game_status", "unknown"))

summary_col1, summary_col2 = st.columns(2)
with summary_col1:
    st.markdown("**Teams**")
    home = selected_row.get("home_team") or "TBD"
    away = selected_row.get("away_team") or "TBD"
    st.write(f"{away} @ {home}")
    st.markdown("**Date / status**")
    st.write(selected_row.get("game_date") or "Not scheduled")
    st.write(format_status_text(game_status))
with summary_col2:
    st.markdown("**Availability**")
    pred_avail = bool(selected_row.get("prediction_available", False))
    replay_avail = bool(selected_row.get("replay_available", False))
    st.write(f"Predictions: {'Yes' if pred_avail else 'No'}")
    st.write(f"Replay: {'Yes' if replay_avail else 'No'}")
    if selected_row.get("final_result_available"):
        st.write(f"Final winner: {selected_row.get('final_winner', '—')}")
        start_prob = selected_row.get("home_win_probability_start")
        final_prob = selected_row.get("home_win_probability_final")
        if pd.notna(start_prob):
            st.write(f"Start probability (home): {format_pct(start_prob)}")
        if pd.notna(final_prob):
            st.write(f"Final probability (home): {format_pct(final_prob)}")

notes = str(selected_row.get("notes", "")).strip()
if notes:
    st.caption(notes)

if game_status in {"scheduled", "not_available_yet", "unknown"} or not game_id:
    st.warning(
        "This game is not available for replay yet. "
        "Scheduled and if-necessary games will appear after metadata and play-by-play are collected."
    )
    st.stop()

if predictions_df is None or predictions_df.empty:
    if not using_finals_deploy:
        st.warning(
            "Playoff live predictions are not available. Run:\n\n"
            "`python run_pipeline.py --mode run_playoff_case_study_pipeline "
            "--seasons 2022-23 2023-24 2024-25 2025-26`"
        )
    st.stop()

game_preds = predictions_df.loc[predictions_df["game_id"].astype(str) == game_id].copy()
if game_preds.empty:
    st.warning("No prediction rows found for this game.")
    st.stop()

game_preds = game_preds.sort_values("event_num")
home_team = str(game_preds.iloc[0]["home_team"])

st.markdown("### Live win probability")
st.plotly_chart(build_probability_chart(game_preds, home_team), use_container_width=True)

with st.expander("Event table"):
    display_cols = [c for c in EVENT_TABLE_COLUMNS if c in game_preds.columns]
    table = add_replay_columns(game_preds)
    if "game_clock_display" not in table.columns and "pctimestring" in table.columns:
        table["game_clock_display"] = table["pctimestring"].map(format_game_clock)
    if {"home_score", "away_score"}.issubset(table.columns):
        table["score_display"] = table.apply(
            lambda r: format_score(r["home_score"], r["away_score"]), axis=1
        )
    show_cols = [c for c in display_cols if c in table.columns]
    st.dataframe(table[show_cols], use_container_width=True, hide_index=True)
