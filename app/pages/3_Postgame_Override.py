"""Postgame Override page — manual post-game result entry."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    build_games_catalog,
    filter_games_catalog,
    format_score,
    format_status_panel_text,
    load_freshness_summary,
    lookup_game_result,
    page_intro_subtitle,
)
from src import config  # noqa: E402
from src.manual_override import (  # noqa: E402
    VALID_SOURCES,
    build_manual_result_record,
    compare_result_for_game,
    get_existing_result,
    load_postgame_results,
    upsert_manual_result,
    validate_postgame_results_dataframe,
)

st.set_page_config(page_title="Postgame Override", page_icon="📝", layout="wide")

POSTGAME_TABLE_COLUMNS = [
    "game_id",
    "home_score",
    "away_score",
    "winner",
    "source",
    "confirmed_at",
    "notes",
]

COMPARISON_LABELS = {
    "match": "Results match",
    "mismatch": "Results differ",
    "no_official_result_available": "No play-by-play result available",
    "no_manual_result_available": "No manual entry recorded",
}


@st.cache_data(show_spinner="Loading game results…")
def load_game_results(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"game_id": str})


@st.cache_data(show_spinner="Loading manual results…")
def load_manual_results_cached(path: str) -> pd.DataFrame:
    return load_postgame_results(Path(path))


@st.cache_data(show_spinner="Loading game list…")
def get_games_catalog_from_results(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"game_id": str})
    return build_games_catalog(df)


st.title("📝 Postgame Result Override")
st.markdown(
    page_intro_subtitle(
        "Record or correct a final score when the generated result is missing or incorrect. "
        "Manual entries are kept separate from play-by-play results and are used for "
        "tracking and evaluation only."
    )
)
st.caption(
    "Portfolio demo page — entries here run against the live app instance and "
    "are not saved back to the project's data; they reset on the next deploy."
)
with st.expander("Data status", expanded=False):
    st.caption(
        format_status_panel_text(
            load_freshness_summary(config.DATA_FRESHNESS_REPORT_PATH),
            ["postgame_results", "game_results"],
        )
    )

game_results_df: Optional[pd.DataFrame] = None
games_catalog: Optional[pd.DataFrame] = None

if config.GAME_RESULTS_PATH.exists():
    try:
        game_results_df = load_game_results(str(config.GAME_RESULTS_PATH))
        games_catalog = get_games_catalog_from_results(str(config.GAME_RESULTS_PATH))
    except Exception as exc:
        st.warning(f"Could not load game results: {exc}")
else:
    st.info("Official game results are not available. You can still enter a game ID manually.")

manual_df = load_manual_results_cached(str(config.POSTGAME_RESULTS_PATH))

st.subheader("Select game")

selected_game_id: Optional[str] = None
selected_home_team = ""
selected_away_team = ""

if games_catalog is not None and not games_catalog.empty:
    seasons = ["All seasons"] + sorted(games_catalog["season"].dropna().unique().tolist())
    teams = sorted(
        set(games_catalog["home_team"].tolist() + games_catalog["away_team"].tolist())
    )
    team_options = ["All teams"] + teams

    col_season, col_team = st.columns(2)
    season_filter = col_season.selectbox("Season", options=seasons, index=0)
    team_filter = col_team.selectbox("Team", options=team_options, index=0)

    filtered_catalog = filter_games_catalog(
        games_catalog,
        season=season_filter,
        team=team_filter,
    )

    if filtered_catalog.empty:
        st.warning("No games match the current filters.")
    else:
        label_to_id = dict(zip(filtered_catalog["label"], filtered_catalog["game_id"]))
        selected_label = st.selectbox(
            "Game",
            options=filtered_catalog["label"].tolist(),
        )
        selected_game_id = label_to_id[selected_label]
        game_row = filtered_catalog.loc[filtered_catalog["game_id"] == selected_game_id].iloc[0]
        selected_home_team = str(game_row["home_team"])
        selected_away_team = str(game_row["away_team"])
else:
    selected_game_id = st.text_input(
        "Game ID",
        value="",
        placeholder="e.g. 0022400001",
    ).strip()
    if selected_game_id:
        selected_home_team = st.text_input("Home team", value="")
        selected_away_team = st.text_input("Away team", value="")

if selected_game_id:
    st.subheader("Compare results")

    official_row = lookup_game_result(game_results_df, selected_game_id)
    existing_manual = get_existing_result(selected_game_id, config.POSTGAME_RESULTS_PATH)
    comparison = compare_result_for_game(selected_game_id, existing_manual, official_row)

    col_official, col_manual, col_compare = st.columns(3)

    with col_official:
        st.markdown("**Play-by-play result**")
        if official_row is not None:
            st.write(f"{official_row['home_team']} vs {official_row['away_team']}")
            st.write(
                f"Score: {format_score(official_row['home_score'], official_row['away_score'])}"
            )
            st.write(f"Winner: {official_row['winner']}")
        else:
            st.write("Not available")

    with col_manual:
        st.markdown("**Manual entry**")
        if existing_manual is not None:
            st.write(
                f"Score: {format_score(existing_manual['home_score'], existing_manual['away_score'])}"
            )
            st.write(f"Winner: {existing_manual['winner']}")
            if existing_manual.get("notes"):
                st.caption(f"Notes: {existing_manual['notes']}")
        else:
            st.write("None recorded")

    with col_compare:
        st.markdown("**Status**")
        if comparison == "match":
            st.success(COMPARISON_LABELS[comparison])
        elif comparison == "mismatch":
            st.warning(COMPARISON_LABELS[comparison])
        else:
            st.info(COMPARISON_LABELS[comparison])

    if comparison == "mismatch":
        st.warning(
            "The manual entry differs from the play-by-play result. "
            "You can still save a correction — the difference will remain visible here."
        )

    st.subheader("Save manual result")

    default_home = int(existing_manual["home_score"]) if existing_manual is not None else 0
    default_away = int(existing_manual["away_score"]) if existing_manual is not None else 0
    default_source = "corrected_manual" if existing_manual is not None else "manual"
    default_notes = str(existing_manual.get("notes", "")) if existing_manual is not None else ""

    source_options = sorted(VALID_SOURCES)
    default_source_index = source_options.index(default_source) if default_source in source_options else 0

    with st.form("postgame_override_form"):
        st.caption(f"Game: {selected_game_id}")
        if selected_home_team and selected_away_team:
            st.caption(f"{selected_home_team} (home) vs {selected_away_team} (away)")

        score_col1, score_col2 = st.columns(2)
        home_score = score_col1.number_input(
            f"Home score ({selected_home_team or 'home'})",
            min_value=0,
            step=1,
            value=default_home,
        )
        away_score = score_col2.number_input(
            f"Away score ({selected_away_team or 'away'})",
            min_value=0,
            step=1,
            value=default_away,
        )

        source = st.selectbox("Source", options=source_options, index=default_source_index)
        notes = st.text_area("Notes (optional)", value=default_notes)

        allow_overwrite = False
        if existing_manual is not None:
            allow_overwrite = st.checkbox(
                "Replace existing manual entry",
                value=False,
            )

        submitted = st.form_submit_button("Save result")

    if submitted:
        if not selected_home_team or not selected_away_team:
            st.error("Home and away team names are required.")
        else:
            try:
                record = build_manual_result_record(
                    game_id=selected_game_id,
                    home_team=selected_home_team,
                    away_team=selected_away_team,
                    home_score=int(home_score),
                    away_score=int(away_score),
                    source=source,
                    notes=notes,
                )
                upsert_manual_result(
                    record,
                    path=config.POSTGAME_RESULTS_PATH,
                    allow_overwrite=allow_overwrite,
                )
                st.success(
                    f"Saved — {record['winner']} "
                    f"({record['home_score']}–{record['away_score']})"
                )
                load_manual_results_cached.clear()
            except (ValueError, TypeError) as exc:
                st.error(str(exc))

with st.expander("All manual entries"):
    if manual_df.empty:
        st.info("No manual entries recorded yet.")
    else:
        try:
            validate_postgame_results_dataframe(manual_df)
            display_df = manual_df[POSTGAME_TABLE_COLUMNS].copy()
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        except ValueError as exc:
            st.error(str(exc))
