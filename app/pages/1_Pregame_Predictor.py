"""Pre-game Predictor page — interactive pre-game prediction dashboard."""

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
    PREGAME_TABLE_COLUMNS,
    REQUIRED_PREGAME_PREDICTION_COLUMNS,
    build_games_catalog,
    build_pregame_feature_sections,
    build_team_comparison_frame,
    compute_pregame_filtered_summary,
    filter_games_catalog,
    format_pct,
    format_status_panel_text,
    load_freshness_summary,
    lookup_pregame_features,
    missing_required_columns,
    page_intro_subtitle,
)
from src import config  # noqa: E402

st.set_page_config(page_title="Pre-game Predictor", page_icon="📊", layout="wide")

st.title("📊 Pre-game Predictor")
st.markdown(
    page_intro_subtitle(
        "Interactive pre-game prediction demo — explore win probabilities and team form."
    )
)
st.caption(DEMO_PREDICTION_NOTE)
st.caption("Historical demo view — prepared baseline outputs from the 2024-25 season pipeline.")
with st.expander("Data status", expanded=False):
    st.caption(
        format_status_panel_text(
            load_freshness_summary(config.DATA_FRESHNESS_REPORT_PATH),
            ["pregame_predictions", "pregame_model_metrics"],
        )
    )


@st.cache_data(show_spinner="Loading pre-game predictions…")
def load_pregame_predictions(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )


@st.cache_data(show_spinner="Loading pre-game features…")
def load_pregame_features(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={"game_id": str, "home_team_id": str, "away_team_id": str},
    )


@st.cache_data(show_spinner="Loading model metrics…")
def load_metrics_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def build_probability_bar_chart(
    home_team: str,
    away_team: str,
    home_prob: float,
    away_prob: float,
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            y=[home_team, away_team],
            x=[home_prob, away_prob],
            orientation="h",
            text=[format_pct(home_prob), format_pct(away_prob)],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Win probability",
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        yaxis_title="",
        margin=dict(l=40, r=20, t=40, b=20),
        height=220,
    )
    return fig


def build_comparison_chart(comparison_df: pd.DataFrame) -> Optional[go.Figure]:
    if comparison_df.empty:
        return None
    fig = go.Figure()
    for team in comparison_df["team"].unique():
        team_df = comparison_df[comparison_df["team"] == team]
        fig.add_trace(
            go.Bar(
                name=team,
                y=team_df["metric"],
                x=team_df["value"],
                orientation="h",
            )
        )
    fig.update_layout(
        barmode="group",
        title="Team comparison before tip-off",
        xaxis_title="Value",
        margin=dict(l=40, r=20, t=40, b=20),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


predictions_path = config.PREGAME_PREDICTIONS_PATH

if not predictions_path.exists():
    st.warning(
        "Pre-game predictions are not available yet.\n\n"
        "Generate them with:\n\n"
        "`python run_pipeline.py --mode predict_pregame`"
    )
    st.stop()

predictions = load_pregame_predictions(str(predictions_path))

if predictions.empty:
    st.error("Pre-game predictions file is empty.")
    st.stop()

missing = missing_required_columns(predictions, REQUIRED_PREGAME_PREDICTION_COLUMNS)
if missing:
    st.error(f"Pre-game predictions are missing required columns: {missing}")
    st.stop()

features: Optional[pd.DataFrame] = None
if config.PREGAME_FEATURES_PATH.exists():
    features = load_pregame_features(str(config.PREGAME_FEATURES_PATH))

metrics_df: Optional[pd.DataFrame] = None
calibration_df: Optional[pd.DataFrame] = None
if config.PREGAME_MODEL_METRICS_PATH.exists():
    metrics_df = load_metrics_csv(str(config.PREGAME_MODEL_METRICS_PATH))
if config.PREGAME_MODEL_CALIBRATION_PATH.exists():
    calibration_df = load_metrics_csv(str(config.PREGAME_MODEL_CALIBRATION_PATH))

catalog = build_games_catalog(predictions)

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
        col_c, col_d = st.columns(2)
        col_c.metric("Log loss", f"{float(m['log_loss']):.3f}")
        col_d.metric("Brier score", f"{float(m['brier_score']):.3f}")
    else:
        st.info("Model metrics not available.")

    if calibration_df is not None and not calibration_df.empty:
        with st.expander("Calibration detail"):
            st.dataframe(calibration_df, hide_index=True, use_container_width=True)

filtered_predictions = predictions.loc[
    predictions["game_id"].isin(filtered_catalog["game_id"])
].copy()

summary = compute_pregame_filtered_summary(filtered_predictions)
sum_cols = st.columns(4)
sum_cols[0].metric("Games", summary.get("game_count", 0))
if "avg_home_win_probability" in summary:
    sum_cols[1].metric("Avg home win %", format_pct(summary["avg_home_win_probability"]))
if "prediction_accuracy" in summary:
    sum_cols[2].metric("Accuracy", format_pct(summary["prediction_accuracy"]))
if "most_confident_game" in summary:
    sum_cols[3].metric(
        "Most confident",
        format_pct(summary["most_confident_home_prob"]),
        help=summary["most_confident_game"],
    )

row = predictions.loc[predictions["game_id"] == selected_game_id]
if row.empty:
    st.error("No prediction found for the selected game.")
    st.stop()
row = row.iloc[0]

home_team = row["home_team"]
away_team = row["away_team"]
home_prob = float(row["home_win_probability"])
away_prob = float(row["away_win_probability"])

st.divider()
st.subheader(f"{home_team} vs {away_team}")
st.caption(f"{row['game_date']} · {selected_game_id}")

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{home_team}", format_pct(home_prob), help="Home win probability")
c2.metric(f"{away_team}", format_pct(away_prob), help="Away win probability")
c3.metric("Predicted winner", str(row["predicted_winner"]))

actual = row.get("actual_home_team_won")
with c4:
    if pd.notna(actual):
        actual_winner = home_team if int(actual) == 1 else away_team
        correct = row.get("prediction_correct")
        delta = "Correct" if pd.notna(correct) and bool(correct) else "Incorrect"
        st.metric("Actual winner", actual_winner, delta=delta)
    else:
        st.metric("Actual winner", "Unavailable")

chart_col, compare_col = st.columns(2)
with chart_col:
    st.plotly_chart(
        build_probability_bar_chart(home_team, away_team, home_prob, away_prob),
        use_container_width=True,
    )

feature_row = lookup_pregame_features(features, selected_game_id)
with compare_col:
    if feature_row is not None:
        comparison = build_team_comparison_frame(feature_row)
        comp_fig = build_comparison_chart(comparison)
        if comp_fig is not None:
            st.plotly_chart(comp_fig, use_container_width=True)
        else:
            st.info("Not enough feature data for comparison chart.")
    else:
        st.info("Feature details unavailable for this game.")

if feature_row is not None:
    with st.expander("Feature inputs for this game"):
        sections = build_pregame_feature_sections(feature_row)
        sec_cols = st.columns(min(len(sections), 4) or 1)
        for idx, (section_name, fields) in enumerate(sections.items()):
            with sec_cols[idx % len(sec_cols)]:
                st.markdown(f"**{section_name}**")
                display = pd.Series(fields, name="value").to_frame()
                st.dataframe(display, use_container_width=True)

with st.expander(f"All filtered games ({len(filtered_predictions)})"):
    table_cols = [c for c in PREGAME_TABLE_COLUMNS if c in filtered_predictions.columns]
    display_table = filtered_predictions[table_cols].copy()
    rename_map = {
        "home_win_probability": "Home win %",
        "away_win_probability": "Away win %",
        "predicted_winner": "Predicted winner",
        "prediction_correct": "Correct",
        "game_date": "Date",
        "home_team": "Home",
        "away_team": "Away",
    }
    display_table = display_table.rename(columns=rename_map)
    for col in ("Home win %", "Away win %"):
        if col in display_table.columns:
            display_table[col] = display_table[col].map(lambda x: format_pct(x))
    st.dataframe(display_table, hide_index=True, use_container_width=True, height=360)

if "closest_game" in summary:
    st.caption(
        f"Closest call: {format_pct(summary['closest_home_prob'])} — {summary['closest_game']}"
    )
