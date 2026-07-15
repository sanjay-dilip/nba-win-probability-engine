"""2025-26 NBA Finals retrospective: how the model did vs what happened.

The 2025-26 Finals ended on June 13, 2026. This page is frozen: every
number is read from committed prediction outputs; nothing is regenerated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    EVENT_TABLE_COLUMNS,
    FINALS_SHOWCASE_SEASON,
    PRIMARY_MODEL_LABEL,
    PRIMARY_MODEL_TRAIN_SEASONS,
    add_replay_columns,
    build_finals_win_probability_chart,
    format_finals_matchup,
    format_game_clock,
    format_pct,
    format_score,
    format_series_score,
    load_optional_csv,
    merge_finals_display_row,
    page_intro_subtitle,
    resolve_playoff_live_predictions_source,
)
from src import config  # noqa: E402

st.set_page_config(page_title="2025-26 NBA Finals", page_icon="🏆", layout="wide")

st.title("🏆 2025-26 NBA Finals: Retrospective")
st.markdown(
    page_intro_subtitle(
        "The series is over. This page compares what the model predicted with what happened."
    )
)

st.markdown(
    f"The **{PRIMARY_MODEL_LABEL}** ({PRIMARY_MODEL_TRAIN_SEASONS}) generated a "
    "**pre-game prediction** before each Finals game and a **live win-probability "
    "replay** from play-by-play. All values below come from the prediction files "
    "committed while the series was in progress; nothing has been re-scored."
)
st.info(
    "Playoff results are an **out-of-distribution case study**, not the "
    "regular-season holdout benchmark reported on the Model Performance page."
)


def safe_bool(value) -> bool:
    """Convert CSV-loaded boolean-like values safely."""
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    value_text = str(value).strip().lower()

    if value_text in {"true", "1", "yes", "y"}:
        return True

    return False


summary_df = load_optional_csv(config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH)
pregame_df = load_optional_csv(config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH)
live_predictions_path, _is_deploy_export = resolve_playoff_live_predictions_source()
predictions_df = load_optional_csv(live_predictions_path)

if summary_df is None or summary_df.empty:
    st.warning(
        "Finals case-study summary not found. Generate outputs with:\n\n"
        "`python run_pipeline.py --mode build_finals_case_study`"
    )
    st.stop()

finals_df = summary_df.loc[summary_df["season"] == FINALS_SHOWCASE_SEASON].copy()
if finals_df.empty:
    st.warning(f"No {FINALS_SHOWCASE_SEASON} Finals rows in the case-study summary.")
    st.stop()

completed_df = (
    finals_df.loc[finals_df["final_result_available"].map(safe_bool)]
    .sort_values("finals_game_number")
    .reset_index(drop=True)
)
if completed_df.empty:
    st.warning("No completed Finals games found in the saved outputs.")
    st.stop()

pregame_completed = None
if pregame_df is not None:
    pregame_completed = (
        pregame_df.loc[pregame_df["final_result_available"].map(safe_bool)]
        .sort_values("finals_game_number")
        .reset_index(drop=True)
    )

st.markdown("### Series result")
win_counts = completed_df["final_winner"].value_counts()
champion = win_counts.idxmax()
runner_up = win_counts.idxmin() if len(win_counts) > 1 else "—"
last_game_date = str(completed_df["game_date"].max())
rcol1, rcol2 = st.columns(2)
with rcol1:
    st.metric("Champion", champion)
    st.caption(f"Series ended {last_game_date}")
with rcol2:
    st.metric(
        "Final series score",
        format_series_score(
            champion,
            runner_up,
            int(win_counts.get(champion, 0)),
            int(win_counts.get(runner_up, 0)),
        ),
    )

st.markdown("### Pre-game predictions vs results")
if pregame_completed is not None and not pregame_completed.empty:
    pre = pregame_completed.copy()
    pre["pick_correct"] = pre["predicted_winner_pregame"] == pre["final_winner"]
    display = pd.DataFrame(
        {
            "Game": pre["finals_game_number"].astype(int),
            "Date": pre["game_date"],
            "Matchup": pre.apply(format_finals_matchup, axis=1),
            "Pre-game pick": pre["predicted_winner_pregame"],
            "Confidence": pre["prediction_confidence"].map(format_pct),
            "Actual winner": pre["final_winner"],
            "Correct": pre["pick_correct"].map(lambda x: "Yes" if x else "No"),
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)
    n_correct = int(pre["pick_correct"].sum())
    st.caption(
        f"Pre-game winner picks: {n_correct}/{len(pre)} correct across the series."
    )
else:
    st.info("Saved pre-game prediction report not found.")

st.markdown("### What the model got right and wrong")
live = completed_df.copy()
live_start_correct = int(live["start_prediction_correct_if_final"].map(safe_bool).sum())
live_final_correct = int(live["final_prediction_correct_if_final"].map(safe_bool).sum())
n_games = len(live)
scol1, scol2, scol3 = st.columns(3)
with scol1:
    if pregame_completed is not None and not pregame_completed.empty:
        st.metric("Pre-game picks correct", f"{n_correct}/{len(pregame_completed)}")
        st.caption("Winner picked before tip-off")
with scol2:
    st.metric("Live opening leans correct", f"{live_start_correct}/{n_games}")
    st.caption("Model lean at the first play-by-play event")
with scol3:
    st.metric("Live final leans correct", f"{live_final_correct}/{n_games}")
    st.caption("Model lean at the last play-by-play event")
st.markdown(
    """
- **Right:** in-game updating worked. The live model's final lean matched the
  eventual winner in every completed game.
- **Wrong:** pre-game picks were near coin-flip confidence, and the live
  opening lean rarely matched the winner. Playoff games are
  out-of-distribution for a regular-season model.
- All figures above are computed from the committed prediction files, including
  the games the model got wrong.
"""
)

st.markdown("### Live win-probability replay")
selector_labels = [
    f"Game {int(row['finals_game_number'])}: {format_finals_matchup(row)} "
    f"({row['final_winner']} won)"
    for _, row in completed_df.iterrows()
]
selected_label = st.selectbox("Finals game", selector_labels)
selected_idx = selector_labels.index(selected_label)
selected_summary = completed_df.iloc[selected_idx]
selected_pregame = None
if pregame_completed is not None and not pregame_completed.empty:
    match = pregame_completed.loc[
        pregame_completed["finals_game_number"]
        == selected_summary["finals_game_number"]
    ]
    if not match.empty:
        selected_pregame = match.iloc[0]
selected = merge_finals_display_row(selected_summary, selected_pregame)

game_id = str(selected.get("game_id", "")).strip()

dcol1, dcol2 = st.columns(2)
with dcol1:
    st.write(f"**Final winner:** {selected.get('final_winner', '—')}")
    if pd.notna(selected.get("predicted_winner_pregame")):
        st.write(f"Pre-game pick: {selected.get('predicted_winner_pregame')}")
        conf = selected.get("prediction_confidence")
        if pd.notna(conf):
            st.caption(f"Confidence: {format_pct(conf)}")
with dcol2:
    start_pred = selected.get("predicted_winner_start")
    final_pred = selected.get("predicted_winner_final")
    if pd.notna(start_pred) and start_pred:
        st.write(f"Live opening lean: {start_pred}")
    if pd.notna(final_pred) and final_pred:
        st.write(f"Live final lean: {final_pred}")

game_preds = pd.DataFrame()
if predictions_df is not None and game_id:
    game_preds = predictions_df.loc[
        predictions_df["game_id"].astype(str) == game_id
    ].copy()

if not game_preds.empty:
    if "event_num" in game_preds.columns:
        game_preds = game_preds.sort_values("event_num")
    home_team = str(game_preds.iloc[0]["home_team"])
    st.plotly_chart(
        build_finals_win_probability_chart(game_preds, home_team),
        use_container_width=True,
    )
    with st.expander("Event table"):
        table = add_replay_columns(game_preds)
        if "game_clock_display" not in table.columns and "pctimestring" in table.columns:
            table["game_clock_display"] = table["pctimestring"].map(format_game_clock)
        if {"home_score", "away_score"}.issubset(table.columns):
            table["score_display"] = table.apply(
                lambda r: format_score(r["home_score"], r["away_score"]), axis=1
            )
        show_cols = [c for c in EVENT_TABLE_COLUMNS if c in table.columns]
        st.dataframe(table[show_cols], use_container_width=True, hide_index=True)
else:
    st.info(
        "Live replay data for this game is not part of the committed replay export."
    )

with st.expander("Notes & limitations"):
    st.markdown(
        """
- This page is a frozen retrospective. The scheduled cloud refresh that updated
  it during the series has been retired; the values are the final committed outputs.
- Pre-game predictions used known schedule/matchup metadata and historical
  performance before tip-off. Completed-game predictions were locked at
  generation time and never overwritten by later refreshes.
- The model is trained on regular-season games; playoffs are an
  out-of-distribution case study, not a holdout benchmark.
- Live replay uses play-by-play win probabilities exported to
  `data/deploy/finals_live_predictions.csv` while the series was live.
        """
    )
