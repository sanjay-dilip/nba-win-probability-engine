"""Model Performance page — evaluation metrics and model comparison."""

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
    BASELINE_MODEL_DESCRIPTION,
    BASELINE_MODEL_LABEL,
    PRIMARY_MODEL_DESCRIPTION,
    PRIMARY_MODEL_LABEL,
    PRIMARY_MODEL_TEST_SEASON,
    PRIMARY_MODEL_TRAIN_SEASONS,
    comparison_column_labels,
    format_game_clock,
    format_large_number,
    format_metric_value,
    format_pct,
    format_evaluation_public_summary,
    format_public_performance_summary,
    format_status_panel_text,
    load_freshness_summary,
    metric_display_name,
    page_intro_subtitle,
    safe_metric_delta,
)
from src import config  # noqa: E402

st.set_page_config(page_title="Model Performance", page_icon="✅", layout="wide")

st.title("✅ Model Performance")
st.markdown(
    page_intro_subtitle(
        "Primary multi-season model evaluation, baseline comparison, and calibration."
    )
)
with st.expander("Data status", expanded=False):
    st.caption(
        format_status_panel_text(
            load_freshness_summary(config.DATA_FRESHNESS_REPORT_PATH),
            [
                "pregame_model_metrics_multiseason",
                "live_model_metrics_multiseason",
                "model_comparison_summary",
            ],
        )
    )

EVALUATE_CMD = "`python run_pipeline.py --mode evaluate`"
COMPARE_CMD = "`python run_pipeline.py --mode compare_models`"


@st.cache_data(show_spinner="Loading report…")
def load_report_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _report_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _summary_value(df: pd.DataFrame, metric: str, section: Optional[str] = None) -> Optional[object]:
    if df.empty:
        return None
    subset = df.loc[df["metric"] == metric]
    if section is not None and "section" in subset.columns:
        subset = subset.loc[subset["section"] == section]
    if subset.empty:
        return None
    return subset.iloc[0]["value"]


def _metrics_table(metrics_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    available = [c for c in columns if c in metrics_df.columns]
    if not available:
        return metrics_df
    row = metrics_df.iloc[0]
    return pd.DataFrame(
        [
            {
                "Metric": metric_display_name(col),
                "Value": format_metric_value(row[col], metric_name=col),
            }
            for col in available
        ]
    )


def _format_comparison_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    if comp_df.empty:
        return comp_df
    baseline_label, primary_label = comparison_column_labels()
    rows = []
    for _, row in comp_df.iterrows():
        metric = str(row["metric"])
        diff = safe_metric_delta(row["difference"])
        if diff is None:
            diff_str = "—"
        elif metric in {"accuracy", "roc_auc"}:
            diff_str = f"{diff * 100:+.1f} pp"
        elif metric in {"log_loss", "brier_score"}:
            diff_str = f"{diff:+.4f}"
        else:
            diff_str = format_large_number(diff)
        rows.append(
            {
                "Metric": metric_display_name(metric),
                baseline_label: format_metric_value(
                    row["single_season_value"], metric_name=metric
                ),
                primary_label: format_metric_value(
                    row["multiseason_value"], metric_name=metric
                ),
                "Difference (primary − baseline)": diff_str,
            }
        )
    return pd.DataFrame(rows)


def _calibration_chart(calibration_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if calibration_df is None or calibration_df.empty:
        return fig

    x = calibration_df["avg_predicted_probability"]
    y = calibration_df["actual_home_win_rate"]
    sizes = calibration_df["row_count"]

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers+lines",
            name="Observed rate",
            marker=dict(size=sizes.clip(upper=40) / max(sizes.max(), 1) * 20 + 6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line=dict(dash="dash", color="gray"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted home win probability",
        yaxis_title="Actual home win rate",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _calibration_bar_chart(calibration_df: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if calibration_df is None or calibration_df.empty:
        return fig

    fig.add_trace(
        go.Bar(
            x=calibration_df["probability_bucket"],
            y=calibration_df["row_count"],
            name="Games",
        )
    )
    fig.update_layout(title=title, xaxis_title="Probability bucket", yaxis_title="Count", height=320)
    return fig


evaluation_ready = _report_exists(config.EVALUATION_SUMMARY_PATH)
evaluation_public_ready = _report_exists(config.EVALUATION_PUBLIC_SUMMARY_PATH)

if not evaluation_ready:
    if evaluation_public_ready:
        evaluation_public_df = load_report_csv(str(config.EVALUATION_PUBLIC_SUMMARY_PATH))
        st.subheader("Evaluation overview")
        st.caption(
            "Deployment summary: full local evaluation reports are not committed, "
            "but the key model results are shown here."
        )
        st.dataframe(
            format_evaluation_public_summary(evaluation_public_df),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning(f"Evaluation reports not found. Generate them with:\n\n{EVALUATE_CMD}")

# ---------------------------------------------------------------------------
# 1. Primary model performance
# ---------------------------------------------------------------------------

st.subheader(f"{PRIMARY_MODEL_LABEL} performance")
st.caption(PRIMARY_MODEL_DESCRIPTION)
st.caption(f"Train: {PRIMARY_MODEL_TRAIN_SEASONS} · Test: {PRIMARY_MODEL_TEST_SEASON}")

public_summary_df: Optional[pd.DataFrame] = None
if _report_exists(config.MODEL_PERFORMANCE_PUBLIC_SUMMARY_PATH):
    public_summary_df = load_report_csv(str(config.MODEL_PERFORMANCE_PUBLIC_SUMMARY_PATH))

has_full_pregame = _report_exists(config.PREGAME_MODEL_METRICS_MULTISEASON_PATH)
has_full_live = _report_exists(config.LIVE_MODEL_METRICS_MULTISEASON_PATH)
has_public_summary = public_summary_df is not None and not public_summary_df.empty

if not has_full_pregame and not has_full_live and has_public_summary:
    st.dataframe(
        format_public_performance_summary(public_summary_df),
        use_container_width=True,
        hide_index=True,
    )
else:
    primary_col1, primary_col2 = st.columns(2)

    with primary_col1:
        st.markdown("**Pre-game model**")
        if has_full_pregame:
            pregame_primary = load_report_csv(str(config.PREGAME_MODEL_METRICS_MULTISEASON_PATH))
            st.table(
                _metrics_table(
                    pregame_primary,
                    ["accuracy", "roc_auc", "log_loss", "brier_score", "n_train", "n_test", "n_features"],
                )
            )
        elif has_public_summary:
            pregame_public = public_summary_df.loc[
                public_summary_df["model"].astype(str).str.contains("Pre-game", case=False, na=False)
            ]
            if not pregame_public.empty:
                st.dataframe(
                    format_public_performance_summary(pregame_public),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Primary pre-game metrics not available.")

    with primary_col2:
        st.markdown("**Live model**")
        if has_full_live:
            live_primary = load_report_csv(str(config.LIVE_MODEL_METRICS_MULTISEASON_PATH))
            st.table(
                _metrics_table(
                    live_primary,
                    ["accuracy", "roc_auc", "log_loss", "brier_score", "train_games", "test_games", "n_features"],
                )
            )
        elif has_public_summary:
            live_public = public_summary_df.loc[
                public_summary_df["model"].astype(str).str.contains("Live", case=False, na=False)
            ]
            if not live_public.empty:
                st.dataframe(
                    format_public_performance_summary(live_public),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("Primary live metrics not available.")

st.info(
    "The multi-season model is the primary evaluation target. "
    "The future-season holdout is stricter and closer to real deployment — "
    "lower pre-game accuracy under that holdout should be interpreted in context."
)

# ---------------------------------------------------------------------------
# 2. Baseline comparison
# ---------------------------------------------------------------------------

st.subheader("Primary model vs baseline")
st.caption(
    f"**{BASELINE_MODEL_LABEL}:** {BASELINE_MODEL_DESCRIPTION} "
    "Metrics should not be compared as identical experiments."
)

comparison_ready = _report_exists(config.MODEL_COMPARISON_SUMMARY_PATH)

if comparison_ready:
    comparison_df = load_report_csv(str(config.MODEL_COMPARISON_SUMMARY_PATH))
    core_metrics = ["accuracy", "roc_auc", "log_loss", "brier_score"]

    comp_col1, comp_col2 = st.columns(2)

    with comp_col1:
        st.markdown("**Pre-game model**")
        pre_comp = comparison_df.loc[
            (comparison_df["model"] == "pregame") & (comparison_df["metric"].isin(core_metrics))
        ]
        if not pre_comp.empty:
            st.dataframe(_format_comparison_table(pre_comp), use_container_width=True, hide_index=True)
        else:
            st.info("No pre-game comparison available.")

    with comp_col2:
        st.markdown("**Live model**")
        live_comp = comparison_df.loc[
            (comparison_df["model"] == "live") & (comparison_df["metric"].isin(core_metrics))
        ]
        if not live_comp.empty:
            st.dataframe(_format_comparison_table(live_comp), use_container_width=True, hide_index=True)
        else:
            st.info("No live comparison available.")

    with st.expander(f"{BASELINE_MODEL_LABEL} hold-out metrics (detail)"):
        baseline_col1, baseline_col2 = st.columns(2)
        with baseline_col1:
            if _report_exists(config.PREGAME_MODEL_METRICS_PATH):
                pregame_baseline = load_report_csv(str(config.PREGAME_MODEL_METRICS_PATH))
                st.markdown("**Pre-game baseline**")
                st.table(
                    _metrics_table(
                        pregame_baseline,
                        ["accuracy", "roc_auc", "log_loss", "brier_score", "n_train", "n_test"],
                    )
                )
        with baseline_col2:
            if _report_exists(config.LIVE_MODEL_METRICS_PATH):
                live_baseline = load_report_csv(str(config.LIVE_MODEL_METRICS_PATH))
                st.markdown("**Live baseline**")
                st.table(
                    _metrics_table(
                        live_baseline,
                        ["accuracy", "roc_auc", "log_loss", "brier_score", "train_games", "test_games"],
                    )
                )
else:
    st.info(f"Comparison reports not found. Generate them with:\n\n{COMPARE_CMD}")

# ---------------------------------------------------------------------------
# 3. Live phase performance
# ---------------------------------------------------------------------------

st.subheader("Live phase performance")

phase_comparison_df = (
    load_report_csv(str(config.PHASE_COMPARISON_SUMMARY_PATH))
    if _report_exists(config.PHASE_COMPARISON_SUMMARY_PATH)
    else pd.DataFrame()
)

phase_col1, phase_col2 = st.columns(2)

with phase_col1:
    st.markdown(f"**{PRIMARY_MODEL_LABEL}**")
    if _report_exists(config.LIVE_MODEL_PHASE_METRICS_MULTISEASON_PATH):
        phase_primary = load_report_csv(str(config.LIVE_MODEL_PHASE_METRICS_MULTISEASON_PATH))
        phase_display = phase_primary.copy()
        if "accuracy" in phase_display.columns:
            phase_display["accuracy"] = phase_display["accuracy"].map(lambda x: format_pct(x))
        st.dataframe(phase_display, use_container_width=True, hide_index=True)
    else:
        st.info("Primary phase metrics not available.")

with phase_col2:
    st.markdown(f"**{BASELINE_MODEL_LABEL}**")
    if _report_exists(config.LIVE_MODEL_PHASE_METRICS_PATH):
        phase_baseline = load_report_csv(str(config.LIVE_MODEL_PHASE_METRICS_PATH))
        phase_display = phase_baseline.copy()
        if "accuracy" in phase_display.columns:
            phase_display["accuracy"] = phase_display["accuracy"].map(lambda x: format_pct(x))
        st.dataframe(phase_display, use_container_width=True, hide_index=True)
    else:
        st.info("Baseline phase metrics not available.")

if not phase_comparison_df.empty and "phase" in phase_comparison_df.columns:
    valid_phases = phase_comparison_df.loc[phase_comparison_df["phase"] != "all"]
    if not valid_phases.empty:
        with st.expander("Phase comparison (primary vs baseline)"):
            baseline_label, primary_label = comparison_column_labels()
            phase_rows = []
            for _, row in valid_phases.iterrows():
                metric = str(row["metric"])
                diff = safe_metric_delta(row.get("difference"))
                if diff is None:
                    diff_str = "—"
                elif metric in {"accuracy", "roc_auc"}:
                    diff_str = f"{diff * 100:+.1f} pp"
                else:
                    diff_str = f"{diff:+.4f}"
                phase_rows.append(
                    {
                        "Phase": row["phase"],
                        "Metric": metric_display_name(metric),
                        baseline_label: format_metric_value(row["single_season_value"], metric),
                        primary_label: format_metric_value(row["multiseason_value"], metric),
                        "Difference": diff_str,
                    }
                )
            st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 4. Calibration
# ---------------------------------------------------------------------------

st.subheader("Calibration")
st.caption("How predicted probabilities align with actual outcomes on each model's hold-out test set.")

cal_col1, cal_col2 = st.columns(2)

with cal_col1:
    st.markdown(f"**{PRIMARY_MODEL_LABEL}**")
    if _report_exists(config.PREGAME_MODEL_CALIBRATION_MULTISEASON_PATH):
        pre_cal_primary = load_report_csv(str(config.PREGAME_MODEL_CALIBRATION_MULTISEASON_PATH))
        st.plotly_chart(
            _calibration_chart(pre_cal_primary, "Pre-game — primary model"),
            use_container_width=True,
        )
    else:
        st.info("Primary pre-game calibration not available.")
    if _report_exists(config.LIVE_MODEL_CALIBRATION_MULTISEASON_PATH):
        live_cal_primary = load_report_csv(str(config.LIVE_MODEL_CALIBRATION_MULTISEASON_PATH))
        st.plotly_chart(
            _calibration_chart(live_cal_primary, "Live — primary model"),
            use_container_width=True,
        )

with cal_col2:
    st.markdown(f"**{BASELINE_MODEL_LABEL}**")
    if _report_exists(config.PREGAME_MODEL_CALIBRATION_PATH):
        pre_cal_baseline = load_report_csv(str(config.PREGAME_MODEL_CALIBRATION_PATH))
        st.plotly_chart(
            _calibration_chart(pre_cal_baseline, "Pre-game — baseline"),
            use_container_width=True,
        )
    if _report_exists(config.LIVE_MODEL_CALIBRATION_PATH):
        live_cal_baseline = load_report_csv(str(config.LIVE_MODEL_CALIBRATION_PATH))
        st.plotly_chart(
            _calibration_chart(live_cal_baseline, "Live — baseline"),
            use_container_width=True,
        )

if _report_exists(config.CALIBRATION_COMPARISON_SUMMARY_PATH):
    calibration_comparison_df = load_report_csv(str(config.CALIBRATION_COMPARISON_SUMMARY_PATH))
    if not calibration_comparison_df.empty and "model" in calibration_comparison_df.columns:
        with st.expander("Calibration bins (both models)"):
            st.dataframe(calibration_comparison_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# 5. Full-dataset reference summaries
# ---------------------------------------------------------------------------

st.subheader("Full-dataset reference summaries")
st.caption(
    "These summarize prepared demo prediction outputs across the full dataset. "
    "They differ from hold-out test metrics above and are for dashboard context only."
)

if evaluation_ready:
    eval_summary = load_report_csv(str(config.EVALUATION_SUMMARY_PATH))

    pre_col, live_event_col, live_final_col = st.columns(3)

    with pre_col:
        st.markdown("**Pre-game predictions**")
        for metric, label in [
            ("total_predictions", "Total predictions"),
            ("prediction_accuracy", "Accuracy"),
            ("avg_confidence", "Avg confidence"),
            ("home_pick_rate", "Home pick rate"),
        ]:
            val = _summary_value(eval_summary, metric, section="pregame_predictions_full_dataset")
            if val is not None:
                display = format_large_number(val) if metric == "total_predictions" else format_metric_value(val, metric)
                st.metric(label, display)

    with live_event_col:
        st.markdown("**Live — every event**")
        st.caption("Event-weighted across all play-by-play rows.")
        for metric, label in [
            ("total_event_predictions", "Total events"),
            ("unique_games", "Unique games"),
            ("prediction_accuracy", "Accuracy"),
            ("avg_confidence", "Avg confidence"),
        ]:
            val = _summary_value(eval_summary, metric, section="live_predictions_event_level")
            if val is not None:
                display = format_large_number(val) if "total" in metric or "unique" in metric else format_metric_value(val, metric)
                st.metric(label, display)

    with live_final_col:
        st.markdown("**Live — final event per game**")
        for metric, label in [
            ("total_games", "Games"),
            ("final_event_accuracy", "Accuracy"),
            ("avg_final_confidence", "Avg confidence"),
        ]:
            val = _summary_value(eval_summary, metric, section="live_predictions_final_event")
            if val is not None:
                st.metric(label, format_metric_value(val, metric))

    if _report_exists(config.PREGAME_PREDICTION_SUMMARY_PATH):
        with st.expander("Pre-game prediction detail"):
            st.dataframe(
                load_report_csv(str(config.PREGAME_PREDICTION_SUMMARY_PATH)),
                use_container_width=True,
                hide_index=True,
            )

    if _report_exists(config.LIVE_PREDICTION_SUMMARY_PATH):
        with st.expander("Live prediction detail"):
            st.dataframe(
                load_report_csv(str(config.LIVE_PREDICTION_SUMMARY_PATH)),
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info(f"Run {EVALUATE_CMD} to generate prediction summaries.")

# ---------------------------------------------------------------------------
# Supplementary reports
# ---------------------------------------------------------------------------

if _report_exists(config.BIGGEST_MOMENTUM_SWINGS_PATH):
    with st.expander("Biggest momentum swings (demo predictions)"):
        swings = load_report_csv(str(config.BIGGEST_MOMENTUM_SWINGS_PATH))
        display_cols = [
            "game_date",
            "home_team",
            "away_team",
            "period",
            "pctimestring",
            "home_score",
            "away_score",
            "event_type_label",
            "home_win_probability",
            "probability_change",
        ]
        show_cols = [c for c in display_cols if c in swings.columns]
        swings_show = swings[show_cols].head(10).copy()
        if "pctimestring" in swings_show.columns:
            swings_show["Time"] = swings_show["pctimestring"].map(format_game_clock)
        if "home_win_probability" in swings_show.columns:
            swings_show["Home win probability"] = swings_show["home_win_probability"].map(format_pct)
        if "probability_change" in swings_show.columns:
            swings_show["Change"] = swings_show["probability_change"].map(lambda x: f"{float(x):+.1%}")
        display_final = [
            c
            for c in [
                "game_date", "home_team", "away_team", "period", "Time",
                "home_score", "away_score", "event_type_label", "Home win probability", "Change",
            ]
            if c in swings_show.columns
        ]
        st.dataframe(swings_show[display_final], use_container_width=True, hide_index=True)

if evaluation_ready:
    with st.expander("Manual result entries"):
        manual_rows = eval_summary.loc[eval_summary["section"] == "manual_overrides"]
        if manual_rows.empty:
            st.info("No manual entry summary available.")
        else:
            manual_col1, manual_col2, manual_col3 = st.columns(3)
            count_val = manual_rows.loc[manual_rows["metric"] == "manual_override_count", "value"]
            manual_col1.metric(
                "Manual entries",
                int(count_val.iloc[0]) if not count_val.empty else 0,
            )
            mismatch_val = manual_rows.loc[
                manual_rows["metric"] == "mismatch_vs_game_results", "value"
            ]
            if not mismatch_val.empty:
                manual_col2.metric("Differ from play-by-play", int(mismatch_val.iloc[0]))
            match_val = manual_rows.loc[manual_rows["metric"] == "match_vs_game_results", "value"]
            if not match_val.empty:
                manual_col3.metric("Match play-by-play", int(match_val.iloc[0]))

            source_rows = manual_rows.loc[manual_rows["metric"].str.startswith("source_", na=False)]
            if not source_rows.empty:
                st.markdown("**Entry sources**")
                source_table = source_rows.copy()
                source_table["source"] = source_table["metric"].str.replace("source_", "", regex=False)
                st.dataframe(
                    source_table[["source", "value"]].rename(columns={"value": "count"}),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("Data coverage"):
        coverage = eval_summary.loc[eval_summary["section"] == "data_coverage"]
        if not coverage.empty:
            st.dataframe(coverage, use_container_width=True, hide_index=True)

if _report_exists(config.EVALUATION_BY_TEAM_PATH):
    with st.expander("Pre-game accuracy by team"):
        st.dataframe(
            load_report_csv(str(config.EVALUATION_BY_TEAM_PATH)),
            use_container_width=True,
            hide_index=True,
        )
