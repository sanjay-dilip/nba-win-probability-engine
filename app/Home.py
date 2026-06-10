"""Streamlit entry point for the NBA Win Probability Engine."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dashboard_utils import (  # noqa: E402
    BASELINE_MODEL_DESCRIPTION,
    BASELINE_MODEL_LABEL,
    FRESHNESS_CHECK_CMD,
    PRIMARY_MODEL_DESCRIPTION,
    PRIMARY_MODEL_LABEL,
    PRIMARY_MODEL_TEST_SEASON,
    PRIMARY_MODEL_TRAIN_SEASONS,
    format_large_number,
    format_status_label,
    format_status_text,
    load_freshness_summary,
    page_intro_subtitle,
)
from src import config  # noqa: E402
from src.data_freshness import compute_overall_status  # noqa: E402

st.set_page_config(page_title="NBA Win Probability Engine", page_icon="🏀", layout="wide")

st.title("🏀 NBA Win Probability Engine")
st.markdown(
    page_intro_subtitle(
        "A multi-season NBA prediction system trained on historical seasons "
        "and evaluated on a future-season holdout."
    )
)

st.markdown("### Models at a glance")

model_col1, model_col2 = st.columns(2)

with model_col1:
    st.markdown(f"#### {PRIMARY_MODEL_LABEL}")
    st.write(PRIMARY_MODEL_DESCRIPTION)
    st.caption(f"Train: {PRIMARY_MODEL_TRAIN_SEASONS} · Test: {PRIMARY_MODEL_TEST_SEASON}")

with model_col2:
    st.markdown(f"#### {BASELINE_MODEL_LABEL}")
    st.write(BASELINE_MODEL_DESCRIPTION)
    st.caption("Used for pipeline validation and side-by-side comparison.")

st.markdown("### What this app does")

feature_col1, feature_col2, feature_col3 = st.columns(3)

with feature_col1:
    st.markdown("#### Pre-game Predictor")
    st.write(
        "Interactive demo — browse pre-game win probabilities, compare team form, "
        "and inspect features behind each prediction."
    )

with feature_col2:
    st.markdown("#### Live Win Probability Replay")
    st.write(
        "Interactive demo — replay how win probability shifts play-by-play with "
        "charts, momentum swings, and event detail."
    )

with feature_col3:
    st.markdown("#### Model Performance")
    st.write(
        "Primary multi-season model evaluation, baseline comparison, calibration, "
        "and phase breakdowns."
    )

st.markdown("#### NBA Finals Case Study")
st.write(
    "Out-of-distribution check — see how the primary model behaves on playoff "
    "and Finals games, including the current season when data is available."
)

st.markdown("#### 2025-26 NBA Finals")
st.write(
    "Focused case study showing how the primary model behaves during the current "
    "Finals series — Games 1–7 with live replay when play-by-play is available."
)

st.divider()

data_mode = config.get_data_mode()
st.subheader("Data mode")
if data_mode == "sample":
    st.info("Using **sample data** for quick exploration. Full-season files are used when available.")
else:
    st.info("Using **full local data** from collected CSV files.")

st.subheader("System & data status")

FRESHNESS_CMD_DISPLAY = f"`{FRESHNESS_CHECK_CMD}`"
freshness_df = load_freshness_summary(config.DATA_FRESHNESS_REPORT_PATH)

if freshness_df is None:
    st.warning(
        f"Status report not found. Refresh local status with:\n\n{FRESHNESS_CMD_DISPLAY}"
    )
else:
    overall = compute_overall_status(freshness_df)
    st.markdown(f"**Overall:** {format_status_label(overall)}")

    highlight_assets = [
        ("pregame_model_multiseason", "Primary pre-game model"),
        ("live_model_multiseason", "Primary live model"),
        ("pregame_model", "Baseline pre-game model"),
        ("live_model", "Baseline live model"),
        ("pregame_predictions", "Demo pre-game predictions"),
        ("live_predictions", "Demo live predictions"),
        ("evaluation_summary", "Evaluation"),
        ("postgame_results", "Manual results"),
    ]

    cols = st.columns(4)
    for idx, (asset_key, label) in enumerate(highlight_assets):
        row = freshness_df.loc[freshness_df["asset"] == asset_key]
        with cols[idx % 4]:
            if row.empty:
                st.metric(label, "Unknown")
            else:
                item = row.iloc[0]
                status = str(item["status"])
                detail = format_status_text(status)
                if pd.notna(item.get("row_count")) and item["row_count"] != "":
                    detail = format_large_number(item["row_count"])
                st.metric(label, format_status_text(status), detail)

    with st.expander("Full status report"):
        st.dataframe(freshness_df, use_container_width=True, hide_index=True)

st.caption(
    "Use the sidebar to open Pre-game Predictor, Live Replay, Postgame Override, "
    "Model Performance, NBA Finals Case Study, or 2025-26 NBA Finals."
)
