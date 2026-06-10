"""2025-26 NBA Finals showcase — pre-game predictions and live replay."""

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
    FINALS_PREGAME_NOTE,
    FINALS_SHOWCASE_SEASON,
    PRIMARY_MODEL_LABEL,
    PRIMARY_MODEL_TRAIN_SEASONS,
    add_replay_columns,
    build_finals_overview_display,
    build_finals_win_probability_chart,
    build_projected_path_display_table,
    finals_game_can_show_replay,
    find_next_upcoming_finals_prediction,
    format_finals_game_selector_label,
    format_finals_matchup,
    format_game_clock,
    format_game_status_for_finals,
    format_pct,
    format_projection_type,
    format_score,
    format_status_text,
    load_optional_csv,
    merge_finals_display_row,
    page_intro_subtitle,
    summarize_projected_series_state,
)
from src import config  # noqa: E402

st.set_page_config(page_title="2025-26 NBA Finals", page_icon="🏆", layout="wide")

st.title("🏆 2025-26 NBA Finals")
st.markdown(
    page_intro_subtitle(
        "Pre-game predictions for upcoming Finals games and live replay for completed games."
    )
)

st.markdown(
    f"The **{PRIMARY_MODEL_LABEL}** ({PRIMARY_MODEL_TRAIN_SEASONS}) generates "
    "**pre-game predictions** for upcoming Finals games and **live replay** for "
    "completed games when play-by-play is available."
)
st.info(
    "Playoff results are a **case study**, not the regular-season holdout benchmark. "
    + FINALS_PREGAME_NOTE
)
st.caption(
    "Games 6 and 7 are scheduled if necessary. When schedule metadata is available, "
    "the model can calculate conditional pre-game predictions, but those games only "
    "become part of the projected path if the series remains alive."
)

summary_df = load_optional_csv(config.NBA_FINALS_CASE_STUDY_SUMMARY_PATH)
upcoming_df = load_optional_csv(config.FINALS_UPCOMING_PREDICTIONS_REPORT_PATH)
projected_df = load_optional_csv(config.FINALS_PROJECTED_SERIES_PATH)
predictions_df = load_optional_csv(config.PLAYOFF_LIVE_PREDICTIONS_PATH)

if summary_df is None or summary_df.empty:
    st.warning(
        "Finals case-study summary not found. Generate outputs with:\n\n"
        "`python run_pipeline.py --mode build_finals_case_study`\n\n"
        "`python run_pipeline.py --mode build_finals_pregame_predictions`"
    )
    st.stop()

finals_df = summary_df.loc[summary_df["season"] == FINALS_SHOWCASE_SEASON].copy()
if finals_df.empty:
    st.warning(f"No {FINALS_SHOWCASE_SEASON} Finals rows in the case-study summary.")
    st.stop()

finals_df = finals_df.sort_values("finals_game_number").reset_index(drop=True)
if upcoming_df is not None:
    upcoming_df = upcoming_df.sort_values("finals_game_number").reset_index(drop=True)

st.markdown("### Series overview")
overview = build_finals_overview_display(finals_df, upcoming_df)
st.dataframe(overview, use_container_width=True, hide_index=True)

st.markdown("### Model-projected series path")
st.caption(
    "Completed games use actual results, including manual corrections when supplied. "
    "Future games use the model's current pre-game predictions. If a manual result is "
    "entered later, the actual series state updates and the remaining path is recalculated."
)
if projected_df is not None and not projected_df.empty:
    series_state = summarize_projected_series_state(projected_df)
    scol1, scol2 = st.columns(2)
    with scol1:
        st.metric("Actual series state", series_state["actual_score_text"])
        if series_state["actual_leader"]:
            st.caption(f"{series_state['actual_leader']} lead")
    with scol2:
        st.metric("Projected path after upcoming games", series_state["projected_score_text"])
        if series_state["projected_leader"]:
            st.caption(f"{series_state['projected_leader']} lead under projection")
    path_table = build_projected_path_display_table(projected_df)
    st.dataframe(path_table, use_container_width=True, hide_index=True)
    model_steps = projected_df.loc[
        projected_df["projection_type"].isin(["model_projected", "conditional_model_projected"])
    ]
    if not model_steps.empty:
        parts = []
        for _, r in model_steps.iterrows():
            label = format_projection_type(r["projection_type"])
            parts.append(
                f"Game {int(r['finals_game_number'])} → {r['projected_winner_used']} ({label})"
            )
        st.caption("Model projections: " + ", ".join(parts))
    pending = projected_df.loc[
        projected_df["projection_type"] == "needed_but_prediction_unavailable"
    ]
    for _, row in pending.iterrows():
        st.info(str(row.get("notes", "")).strip() or format_projection_type(row["projection_type"]))
else:
    st.info(
        "Projected series path not generated yet. Run:\n\n"
        "`python run_pipeline.py --mode build_finals_projected_series_path`"
    )

st.markdown("### Official scheduled predictions")
if upcoming_df is not None:
    scheduled_preds = upcoming_df.loc[
        (upcoming_df["game_status"] == "scheduled")
        & (upcoming_df["pregame_prediction_available"].astype(bool))
        & (~upcoming_df["final_result_available"].astype(bool))
    ].sort_values("finals_game_number")
    if not scheduled_preds.empty:
        for _, row in scheduled_preds.iterrows():
            st.write(
                f"**Game {int(row['finals_game_number'])}** — "
                f"{format_finals_matchup(row)}: "
                f"{row.get('predicted_winner_pregame', '—')}"
            )
    else:
        st.caption("No official scheduled pre-game predictions are available yet.")

st.markdown("### Conditional scheduled predictions")
if upcoming_df is not None:
    conditional_preds = upcoming_df.loc[
        (upcoming_df["game_status"] == "if_necessary_scheduled")
        & (upcoming_df["pregame_prediction_available"].astype(bool))
    ].sort_values("finals_game_number")
    if not conditional_preds.empty:
        for _, row in conditional_preds.iterrows():
            st.write(
                f"**Game {int(row['finals_game_number'])}** — "
                f"{format_finals_matchup(row)}: "
                f"{row.get('predicted_winner_pregame', '—')} "
                "(conditional — only if the series remains alive)"
            )
            note = str(row.get("notes", "")).strip()
            if note:
                st.caption(note)
    else:
        st.caption(
            "Conditional predictions for Games 6–7 appear when if-necessary schedule "
            "metadata is supplied in the manual override file."
        )

st.markdown("### Next available Finals prediction")
next_game = find_next_upcoming_finals_prediction(upcoming_df) if upcoming_df is not None else None
if next_game is not None:
    ncol1, ncol2, ncol3 = st.columns(3)
    with ncol1:
        st.metric("Game", f"Game {int(next_game['finals_game_number'])}")
        st.write(format_finals_matchup(next_game))
    with ncol2:
        st.metric("Predicted winner", str(next_game.get("predicted_winner_pregame", "—")))
        conf = next_game.get("prediction_confidence")
        if pd.notna(conf):
            st.caption(f"Confidence: {format_pct(conf)}")
    with ncol3:
        home_p = next_game.get("home_win_probability_pregame")
        if pd.notna(home_p):
            st.metric("Home win probability", format_pct(home_p))
        st.caption(str(next_game.get("game_date", "")))
    st.caption(
        "Live replay appears after play-by-play is collected and the local case-study pipeline is rerun."
    )
else:
    st.info("No upcoming Finals game prediction is available from local metadata yet.")

st.markdown("### Select a game")
selector_labels = [format_finals_game_selector_label(row) for _, row in finals_df.iterrows()]
selected_label = st.selectbox("Finals game", selector_labels)
selected_idx = selector_labels.index(selected_label)
selected_summary = finals_df.iloc[selected_idx]
selected_upcoming = None
if upcoming_df is not None and not upcoming_df.empty:
    match = upcoming_df.loc[
        upcoming_df["finals_game_number"] == selected_summary["finals_game_number"]
    ]
    if not match.empty:
        selected_upcoming = match.iloc[0]
selected = merge_finals_display_row(selected_summary, selected_upcoming)

game_id = str(selected.get("game_id", "")).strip()
game_status = str(selected.get("game_status", "unknown"))
game_num = int(selected.get("finals_game_number", 0))
can_replay = finals_game_can_show_replay(selected)
pregame_avail = bool(selected.get("pregame_prediction_available", False))

st.markdown("#### Pre-game prediction")
if pregame_avail:
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        st.write(f"**Predicted winner:** {selected.get('predicted_winner_pregame', '—')}")
        home_p = selected.get("home_win_probability_pregame")
        away_p = selected.get("away_win_probability_pregame")
        if pd.notna(home_p):
            st.write(f"Home ({selected.get('home_team', 'TBD')}): {format_pct(home_p)}")
        if pd.notna(away_p):
            st.write(f"Away ({selected.get('away_team', 'TBD')}): {format_pct(away_p)}")
    with pcol2:
        if selected.get("final_result_available"):
            pre_ok = selected.get("pregame_prediction_correct_if_final")
            if pd.notna(pre_ok):
                st.write(f"Pre-game prediction correct: {'Yes' if pre_ok else 'No'}")
            st.write(f"**Final winner:** {selected.get('final_winner', '—')}")
else:
    if not game_id:
        if game_status == "if_necessary_scheduled":
            st.warning(
                "If-necessary scheduled game — schedule metadata is known, but a "
                "pre-game prediction could not be generated."
            )
        elif game_num >= 6 or game_status == "if_necessary":
            st.warning("If necessary — awaiting official schedule/metadata.")
        elif game_num in (4, 5):
            st.warning(
                "This game is expected in the series context, but schedule metadata is not "
                "available locally yet. Add rows to "
                "`data/manual/finals_schedule_overrides.csv` or re-run playoff metadata collection."
            )
        else:
            st.warning("This game is not available yet.")
    elif game_status == "if_necessary_scheduled":
        st.info(
            "Conditional pre-game prediction — this game is only part of the projected "
            "path if the series remains alive after earlier games."
        )
    else:
        st.info("Pre-game prediction is not available for this game.")

st.markdown("#### Game details")
dcol1, dcol2 = st.columns(2)
with dcol1:
    st.write(format_finals_matchup(selected))
    st.write(selected.get("game_date") or "Not scheduled")
    st.write(format_game_status_for_finals(game_status))
with dcol2:
    st.write(f"Live replay: {'Available' if can_replay else 'Not available'}")
    if can_replay and selected.get("final_result_available"):
        start_pred = selected.get("predicted_winner_start")
        final_pred = selected.get("predicted_winner_final")
        if pd.notna(start_pred) and start_pred:
            st.write(f"Live start lean: {start_pred}")
            start_ok = selected.get("start_prediction_correct_if_final")
            if pd.notna(start_ok):
                st.write(f"Live start correct: {'Yes' if start_ok else 'No'}")
        if pd.notna(final_pred) and final_pred:
            st.write(f"Live final lean: {final_pred}")
            final_ok = selected.get("final_prediction_correct_if_final")
            if pd.notna(final_ok):
                st.write(f"Live final correct: {'Yes' if final_ok else 'No'}")

notes = str(selected.get("notes", "")).strip()
if notes:
    st.caption(notes)

if can_replay and predictions_df is not None and game_id:
    game_preds = predictions_df.loc[predictions_df["game_id"].astype(str) == game_id].copy()
    if not game_preds.empty:
        if "event_num" in game_preds.columns:
            game_preds = game_preds.sort_values("event_num")
        home_team = str(game_preds.iloc[0]["home_team"])
        st.markdown("#### Live win probability replay")
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
elif game_status == "scheduled" and game_id:
    st.info(
        "Replay will appear once play-by-play is collected and the local case-study pipeline is rerun."
    )

with st.expander("Notes & limitations"):
    st.markdown(
        """
- Pre-game predictions use known schedule/matchup metadata and historical performance before tip-off.
- Manual schedule rows in `data/manual/finals_schedule_overrides.csv` supply matchup metadata only — not results.
- Manual post-game overrides in `data/manual/postgame_results.csv` update actual series state only — never model projections.
- The projected path uses model pre-game picks for unplayed games; it is not the same as the actual future.
- Live replay uses play-by-play and appears only after collection.
- The model is trained on regular-season games; playoffs are an out-of-distribution case study.
- Games 6 and 7 can be listed as if-necessary scheduled rows when matchup/date metadata is known; predictions are conditional on the series path.
        """
    )
