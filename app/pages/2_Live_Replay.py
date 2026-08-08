"""Live Replay page — interactive win-probability replay."""

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
    BASELINE_MODEL_LABEL,
    DEMO_PREDICTION_NOTE,
    DEPLOY_LIVE_DEMO_NOTE,
    EVENT_TABLE_COLUMNS,
    add_replay_columns,
    build_games_catalog,
    filter_games_catalog,
    format_game_clock,
    format_pct,
    format_score,
    format_status_panel_text,
    get_final_event_row,
    load_freshness_summary,
    lookup_game_result,
    missing_required_columns,
    page_intro_subtitle,
    resolve_live_predictions_source,
    top_momentum_swings,
)
from src import config  # noqa: E402

st.set_page_config(page_title="Live Replay", page_icon="📈", layout="wide")

st.title("📈 Live Win Probability Replay")
st.markdown(
    page_intro_subtitle(
        "Interactive live replay demo — see how win probability evolves during a game."
    )
)
st.caption(DEMO_PREDICTION_NOTE)
st.caption("Historical demo view — prepared baseline outputs from the 2024-25 season pipeline.")
with st.expander("Data status", expanded=False):
    st.caption(
        format_status_panel_text(
            load_freshness_summary(config.DATA_FRESHNESS_REPORT_PATH),
            ["live_predictions", "live_model_metrics"],
        )
    )


@st.cache_data(show_spinner="Loading live predictions…")
def load_live_predictions(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str})


@st.cache_data(show_spinner="Loading game results…")
def load_game_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str})


@st.cache_data(show_spinner="Loading model metrics…")
def load_metrics_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading game list…")
def get_games_catalog_from_path(path: str) -> pd.DataFrame:
    cols = ["game_id", "season", "game_date", "home_team", "away_team"]
    df = pd.read_csv(path, dtype={"game_id": str}, usecols=cols)
    return build_games_catalog(df)


def build_win_probability_chart(
    game_df: pd.DataFrame,
    home_team: str,
    selected_event_num: Optional[int] = None,
) -> go.Figure:
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
                "Home win prob: %{y:.1%}<br>"
                "Event: %{customdata[4]}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.6)

    if selected_event_num is not None:
        marker = enriched.loc[enriched["event_num"] == selected_event_num]
        if not marker.empty:
            fig.add_trace(
                go.Scatter(
                    x=marker["game_elapsed_seconds"],
                    y=marker["home_win_probability"],
                    mode="markers",
                    name="Selected event",
                    marker=dict(size=12, color="red"),
                )
            )

    fig.update_layout(
        title=f"{home_team} win probability",
        xaxis_title="Elapsed game time (seconds)",
        yaxis_title="Home win probability",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


predictions_path, using_deploy_demo = resolve_live_predictions_source()

if not predictions_path.exists():
    st.warning(
        "Live predictions are not available yet.\n\n"
        "Generate them with:\n\n"
        "`python run_pipeline.py --mode predict_live`"
    )
    st.stop()

if using_deploy_demo:
    st.caption(DEPLOY_LIVE_DEMO_NOTE)

predictions = load_live_predictions(str(predictions_path))

if predictions.empty:
    st.error("Live predictions file is empty.")
    st.stop()

missing = missing_required_columns(predictions)
if missing:
    st.error(f"Live predictions are missing required columns: {missing}")
    st.stop()

game_results: Optional[pd.DataFrame] = None
if config.GAME_RESULTS_PATH.exists():
    game_results = load_game_results(str(config.GAME_RESULTS_PATH))

metrics_df: Optional[pd.DataFrame] = None
phase_df: Optional[pd.DataFrame] = None
if config.LIVE_MODEL_METRICS_PATH.exists():
    metrics_df = load_metrics_csv(str(config.LIVE_MODEL_METRICS_PATH))
if config.LIVE_MODEL_PHASE_METRICS_PATH.exists():
    phase_df = load_metrics_csv(str(config.LIVE_MODEL_PHASE_METRICS_PATH))

catalog = get_games_catalog_from_path(str(predictions_path))

with st.sidebar:
    st.header("Filters")

    seasons = sorted(catalog["season"].dropna().unique().tolist())
    selected_season = st.selectbox("Season", ["All seasons"] + seasons)

    all_teams = sorted(
        set(catalog["home_team"].tolist()) | set(catalog["away_team"].tolist())
    )
    selected_team = st.selectbox("Team", ["All teams"] + all_teams)

    filtered_catalog = filter_games_catalog(
        catalog,
        season=selected_season,
        team=selected_team,
    )

    if filtered_catalog.empty:
        st.warning("No games match the current filters.")
        st.stop()

    game_labels = filtered_catalog["label"].tolist()
    selected_label = st.selectbox("Game", game_labels)
    selected_game_id = filtered_catalog.loc[
        filtered_catalog["label"] == selected_label, "game_id"
    ].iloc[0]

    st.divider()
    st.subheader(f"{BASELINE_MODEL_LABEL} metrics")
    st.caption("Hold-out metrics from the baseline model used for demo predictions.")
    if metrics_df is not None and not metrics_df.empty:
        m = metrics_df.iloc[0]
        col_a, col_b = st.columns(2)
        col_a.metric("Accuracy", format_pct(m["accuracy"]))
        col_b.metric("ROC-AUC", f"{float(m['roc_auc']):.3f}")
    else:
        st.info("Model metrics not available.")

    if phase_df is not None and not phase_df.empty:
        with st.expander("Accuracy by game phase"):
            st.dataframe(phase_df, hide_index=True, use_container_width=True)

game_df = predictions.loc[predictions["game_id"] == selected_game_id].copy()
if game_df.empty:
    st.error("No prediction rows found for the selected game.")
    st.stop()

game_events = add_replay_columns(game_df)
home_team = game_events["home_team"].iloc[0]
away_team = game_events["away_team"].iloc[0]
game_date = game_events["game_date"].iloc[0]

result_row = lookup_game_result(game_results, selected_game_id)
final_event = get_final_event_row(game_events)

st.subheader(f"{home_team} vs {away_team}")
st.caption(f"{game_date} · {selected_game_id}")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Matchup", f"{home_team} vs {away_team}")

with col2:
    if result_row is not None:
        st.metric(
            "Final score",
            format_score(result_row["home_score"], result_row["away_score"]),
            help=f"Winner: {result_row['winner']}",
        )
    else:
        st.metric(
            "Final score",
            format_score(final_event["home_score"], final_event["away_score"]),
            help="From last prediction row",
        )

with col3:
    st.metric(
        "Final home win probability",
        format_pct(final_event["home_win_probability"]),
        help=f"Predicted: {final_event['predicted_winner']}",
    )

with col4:
    actual = final_event.get("actual_home_team_won")
    if pd.notna(actual):
        actual_winner = home_team if int(actual) == 1 else away_team
        correct = final_event.get("prediction_correct")
        delta = None
        if pd.notna(correct):
            delta = "Correct" if bool(correct) else "Incorrect"
        st.metric("Actual winner", actual_winner, delta=delta)
    else:
        st.metric("Actual winner", "Unavailable")

st.divider()
st.subheader("Replay")

event_indices = list(range(len(game_events)))
selected_idx = st.slider(
    "Event",
    min_value=0,
    max_value=len(game_events) - 1,
    value=len(game_events) - 1,
    help="Scrub through the game event by event.",
)

current = game_events.iloc[selected_idx]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Period", int(current["period"]))
c2.metric("Time remaining", format_game_clock(current["pctimestring"]))
c3.metric("Score", format_score(current["home_score"], current["away_score"]))
c4.metric("Home win probability", format_pct(current["home_win_probability"]))
c5.metric("Away win probability", format_pct(current["away_win_probability"]))

margin = float(current["score_margin_home"])
leader = home_team if margin > 0 else away_team if margin < 0 else "Tied"
st.caption(
    f"Score margin (home): {margin:+.0f} · Predicted leader: {current['predicted_winner']} · "
    f"Score leader: {leader} · Event type: {current.get('event_type_label', '—')}"
)

st.subheader("Win probability chart")
fig = build_win_probability_chart(
    game_df,
    home_team,
    selected_event_num=int(current["event_num"]),
)
st.plotly_chart(fig, use_container_width=True)

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Biggest momentum swings")
    swings = top_momentum_swings(game_df, n=10)
    swings_display = swings.copy()
    if "pctimestring" in swings_display.columns:
        swings_display["Time"] = swings_display["pctimestring"].map(format_game_clock)
    swings_display["Home win probability"] = swings_display["home_win_probability"].map(
        format_pct
    )
    swings_display["Change"] = swings_display["probability_change"].map(
        lambda x: f"{float(x):+.1%}"
    )
    show_cols = [
        c
        for c in ["period", "Time", "home_score", "away_score", "event_type_label", "Home win probability", "Change"]
        if c in swings_display.columns
    ]
    st.dataframe(swings_display[show_cols], hide_index=True, use_container_width=True)

with col_right:
    with st.expander("Play-by-play events"):
        table_cols = [c for c in EVENT_TABLE_COLUMNS if c in game_events.columns]
        event_table = game_events[table_cols].copy()
        if "pctimestring" in event_table.columns:
            event_table["Time remaining"] = event_table["pctimestring"].map(format_game_clock)
        if "home_win_probability" in event_table.columns:
            event_table["Home win probability"] = event_table["home_win_probability"].map(format_pct)
        if "away_win_probability" in event_table.columns:
            event_table["Away win probability"] = event_table["away_win_probability"].map(format_pct)
        display_cols = [
            c
            for c in [
                "event_num",
                "period",
                "Time remaining",
                "home_score",
                "away_score",
                "event_type_label",
                "Home win probability",
                "Away win probability",
                "predicted_winner",
            ]
            if c in event_table.columns
        ]
        st.dataframe(
            event_table[display_cols],
            hide_index=True,
            use_container_width=True,
            height=400,
        )

st.caption(f"{len(game_events):,} events in this game")
