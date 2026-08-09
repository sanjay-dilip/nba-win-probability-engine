"""Multi-season model version comparison.

Reads existing single-season baseline and multi-season holdout metric reports,
then writes comparison CSVs. Does **not** train models, collect data, or call
``nba_api``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import config  # noqa: E402
from src.utils import ensure_directories, save_csv  # noqa: E402

SINGLE_SEASON_SETUP = "2024-25 chronological hold-out (single-season baseline)"
MULTISEASON_SETUP = (
    "Future-season holdout: train 2022-23/2023-24/2024-25, test 2025-26"
)
SETUP_DIFFERENCE_NOTE = (
    "Evaluation setups differ — single-season uses within-season chronological "
    "split; multi-season uses future-season holdout (closer to deployment)."
)

SINGLE_SEASON_TRAIN_SEASONS = "2024-25"
SINGLE_SEASON_TEST_SEASON = "2024-25 (chronological 20% hold-out)"
MULTISEASON_TRAIN_SEASONS = "2022-23, 2023-24, 2024-25"
MULTISEASON_TEST_SEASON = "2025-26"

CORE_METRICS = ["accuracy", "roc_auc", "log_loss", "brier_score"]
CONFUSION_METRICS = [
    "true_positive",
    "true_negative",
    "false_positive",
    "false_negative",
]
PHASE_METRICS = ["accuracy", "log_loss", "brier_score"]

HIGHER_IS_BETTER = {"accuracy", "roc_auc"}
LOWER_IS_BETTER = {"log_loss", "brier_score"}


def _load_csv_if_present(path: Union[str, Path]) -> Optional[pd.DataFrame]:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return None
    if df.empty:
        return None
    return df


def load_metric_report(
    path: Union[str, Path],
    model_name: str,
    setup_name: str,
) -> Optional[pd.DataFrame]:
    """Load a one-row training metrics CSV and attach model/setup labels."""
    df = _load_csv_if_present(path)
    if df is None:
        return None
    out = df.copy()
    out["model"] = model_name
    out["setup"] = setup_name
    out["source_report"] = Path(path).name
    return out


def normalize_metric_report(
    df: Optional[pd.DataFrame],
    model_name: str,
    setup_name: str,
    train_seasons: str,
    test_season: str,
) -> pd.DataFrame:
    """Convert a wide metrics row into long-form detail rows."""
    columns = [
        "source_report",
        "model",
        "setup",
        "metric",
        "value",
        "train_seasons",
        "test_season",
        "notes",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    row = df.iloc[0]
    source = str(row.get("source_report", ""))
    metrics = CORE_METRICS + CONFUSION_METRICS
    rows: List[dict] = []
    for metric in metrics:
        if metric not in row.index or pd.isna(row[metric]):
            continue
        rows.append(
            {
                "source_report": source,
                "model": model_name,
                "setup": setup_name,
                "metric": metric,
                "value": float(row[metric]),
                "train_seasons": train_seasons,
                "test_season": test_season,
                "notes": SETUP_DIFFERENCE_NOTE,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def load_phase_metrics(
    path: Union[str, Path],
    setup_name: str,
) -> Optional[pd.DataFrame]:
    """Load live phase metrics and attach setup label."""
    df = _load_csv_if_present(path)
    if df is None:
        return None
    if "phase" not in df.columns:
        return None
    out = df.copy()
    out["setup"] = setup_name
    return out


def load_calibration_report(
    path: Union[str, Path],
    model_name: str,
    setup_name: str,
) -> Optional[pd.DataFrame]:
    """Load a calibration CSV and normalize column names."""
    df = _load_csv_if_present(path)
    if df is None:
        return None

    rename_map = {
        "probability_bucket": "bin",
        "avg_predicted_probability": "mean_predicted_probability",
        "actual_home_win_rate": "observed_rate",
        "row_count": "count",
    }
    out = df.rename(columns=rename_map)
    if "bin" not in out.columns:
        return None

    out["model"] = model_name
    out["setup"] = setup_name
    if "mean_predicted_probability" not in out.columns:
        out["mean_predicted_probability"] = pd.NA
    if "observed_rate" not in out.columns:
        out["observed_rate"] = pd.NA
    if "count" not in out.columns:
        out["count"] = pd.NA

    pred = pd.to_numeric(out["mean_predicted_probability"], errors="coerce")
    obs = pd.to_numeric(out["observed_rate"], errors="coerce")
    out["calibration_error"] = (pred - obs).abs()
    return out


def _metric_value(df: Optional[pd.DataFrame], metric: str) -> Optional[float]:
    if df is None or df.empty or metric not in df.columns:
        return None
    value = df.iloc[0][metric]
    if pd.isna(value):
        return None
    return float(value)


def _interpret_difference(metric: str, difference: Optional[float]) -> str:
    if difference is None or pd.isna(difference):
        return SETUP_DIFFERENCE_NOTE

    direction = ""
    if metric in HIGHER_IS_BETTER:
        if difference > 0:
            direction = "Multi-season value is higher (higher is better for this metric)."
        elif difference < 0:
            direction = "Multi-season value is lower (higher is better for this metric)."
        else:
            direction = "Values are equal on this metric."
    elif metric in LOWER_IS_BETTER:
        if difference > 0:
            direction = "Multi-season value is higher (lower is better for this metric)."
        elif difference < 0:
            direction = "Multi-season value is lower (lower is better for this metric)."
        else:
            direction = "Values are equal on this metric."
    elif metric in CONFUSION_METRICS:
        direction = (
            "Confusion counts reflect different test-set sizes — "
            "compare rates, not raw counts."
        )
    else:
        direction = "Compare with caution — evaluation setups differ."

    return f"{direction} {SETUP_DIFFERENCE_NOTE}"


def compare_metric_rows(
    single_df: Optional[pd.DataFrame],
    multi_df: Optional[pd.DataFrame],
    model_name: str,
) -> pd.DataFrame:
    """Build comparison rows for one model's core and confusion metrics."""
    columns = [
        "model",
        "metric",
        "single_season_value",
        "multiseason_value",
        "difference",
        "single_season_setup",
        "multiseason_setup",
        "interpretation",
    ]
    rows: List[dict] = []
    for metric in CORE_METRICS + CONFUSION_METRICS:
        single_val = _metric_value(single_df, metric)
        multi_val = _metric_value(multi_df, metric)
        if single_val is None and multi_val is None:
            continue
        diff = None
        if single_val is not None and multi_val is not None:
            diff = multi_val - single_val
        rows.append(
            {
                "model": model_name,
                "metric": metric,
                "single_season_value": single_val,
                "multiseason_value": multi_val,
                "difference": diff,
                "single_season_setup": SINGLE_SEASON_SETUP,
                "multiseason_setup": MULTISEASON_SETUP,
                "interpretation": _interpret_difference(metric, diff),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_model_comparison_summary(
    pregame_single: Optional[pd.DataFrame],
    pregame_multi: Optional[pd.DataFrame],
    live_single: Optional[pd.DataFrame],
    live_multi: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Compare pre-game and live metrics side by side."""
    parts = [
        compare_metric_rows(pregame_single, pregame_multi, "pregame"),
        compare_metric_rows(live_single, live_multi, "live"),
    ]
    return pd.concat(parts, ignore_index=True)


def build_phase_comparison_summary(
    single_phase: Optional[pd.DataFrame],
    multi_phase: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Compare live phase metrics where both setups have data."""
    columns = [
        "phase",
        "metric",
        "single_season_value",
        "multiseason_value",
        "difference",
        "interpretation",
    ]
    if single_phase is None and multi_phase is None:
        return pd.DataFrame(
            [
                {
                    "phase": "all",
                    "metric": "n/a",
                    "single_season_value": None,
                    "multiseason_value": None,
                    "difference": None,
                    "interpretation": "Phase comparison unavailable — no phase reports found.",
                }
            ],
            columns=columns,
        )

    if single_phase is None or multi_phase is None:
        missing = "single-season" if single_phase is None else "multi-season"
        return pd.DataFrame(
            [
                {
                    "phase": "all",
                    "metric": "n/a",
                    "single_season_value": None,
                    "multiseason_value": None,
                    "difference": None,
                    "interpretation": (
                        f"Phase comparison incomplete — {missing} phase report missing."
                    ),
                }
            ],
            columns=columns,
        )

    rows: List[dict] = []
    phases = sorted(set(single_phase["phase"]).union(set(multi_phase["phase"])))
    for phase in phases:
        single_row = single_phase.loc[single_phase["phase"] == phase]
        multi_row = multi_phase.loc[multi_phase["phase"] == phase]
        for metric in PHASE_METRICS:
            single_val = None
            multi_val = None
            if not single_row.empty and metric in single_row.columns:
                val = single_row.iloc[0][metric]
                if pd.notna(val):
                    single_val = float(val)
            if not multi_row.empty and metric in multi_row.columns:
                val = multi_row.iloc[0][metric]
                if pd.notna(val):
                    multi_val = float(val)
            if single_val is None and multi_val is None:
                continue
            diff = None
            if single_val is not None and multi_val is not None:
                diff = multi_val - single_val
            rows.append(
                {
                    "phase": phase,
                    "metric": metric,
                    "single_season_value": single_val,
                    "multiseason_value": multi_val,
                    "difference": diff,
                    "interpretation": _interpret_difference(metric, diff),
                }
            )

    if not rows:
        return pd.DataFrame(
            [
                {
                    "phase": "all",
                    "metric": "n/a",
                    "single_season_value": None,
                    "multiseason_value": None,
                    "difference": None,
                    "interpretation": "Phase reports exist but share no comparable metrics.",
                }
            ],
            columns=columns,
        )
    return pd.DataFrame(rows, columns=columns)


def build_calibration_comparison_summary(
    pregame_single: Optional[pd.DataFrame],
    pregame_multi: Optional[pd.DataFrame],
    live_single: Optional[pd.DataFrame],
    live_multi: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Stack calibration bins from both setups for side-by-side review."""
    columns = [
        "model",
        "setup",
        "bin",
        "mean_predicted_probability",
        "observed_rate",
        "count",
        "calibration_error",
    ]
    parts: List[pd.DataFrame] = []
    for model_name, single_df, multi_df in [
        ("pregame", pregame_single, pregame_multi),
        ("live", live_single, live_multi),
    ]:
        if single_df is not None and not single_df.empty:
            parts.append(single_df[columns])
        if multi_df is not None and not multi_df.empty:
            parts.append(multi_df[columns])

    if not parts:
        return pd.DataFrame(
            [
                {
                    "model": "n/a",
                    "setup": "n/a",
                    "bin": "n/a",
                    "mean_predicted_probability": None,
                    "observed_rate": None,
                    "count": None,
                    "calibration_error": None,
                }
            ],
            columns=columns,
        )
    return pd.concat(parts, ignore_index=True)


def build_model_comparison_detail(
    pregame_single: Optional[pd.DataFrame],
    pregame_multi: Optional[pd.DataFrame],
    live_single: Optional[pd.DataFrame],
    live_multi: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Build long-form detail rows for all compared metrics."""
    parts = [
        normalize_metric_report(
            pregame_single,
            "pregame",
            SINGLE_SEASON_SETUP,
            SINGLE_SEASON_TRAIN_SEASONS,
            SINGLE_SEASON_TEST_SEASON,
        ),
        normalize_metric_report(
            pregame_multi,
            "pregame",
            MULTISEASON_SETUP,
            MULTISEASON_TRAIN_SEASONS,
            MULTISEASON_TEST_SEASON,
        ),
        normalize_metric_report(
            live_single,
            "live",
            SINGLE_SEASON_SETUP,
            SINGLE_SEASON_TRAIN_SEASONS,
            SINGLE_SEASON_TEST_SEASON,
        ),
        normalize_metric_report(
            live_multi,
            "live",
            MULTISEASON_SETUP,
            MULTISEASON_TRAIN_SEASONS,
            MULTISEASON_TEST_SEASON,
        ),
    ]
    return pd.concat(parts, ignore_index=True)


def save_comparison_reports(
    summary_df: pd.DataFrame,
    detail_df: pd.DataFrame,
    phase_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> Dict[str, Path]:
    """Write all comparison report CSVs and return output paths."""
    ensure_directories()
    outputs = {
        "model_comparison_summary": config.MODEL_COMPARISON_SUMMARY_PATH,
        "model_comparison_detail": config.MODEL_COMPARISON_DETAIL_PATH,
        "live_phase_comparison_summary": config.PHASE_COMPARISON_SUMMARY_PATH,
        "calibration_comparison_summary": config.CALIBRATION_COMPARISON_SUMMARY_PATH,
    }
    save_csv(summary_df, outputs["model_comparison_summary"])
    save_csv(detail_df, outputs["model_comparison_detail"])
    save_csv(phase_df, outputs["live_phase_comparison_summary"])
    save_csv(calibration_df, outputs["calibration_comparison_summary"])
    return outputs


def _load_all_reports() -> Dict[str, Optional[pd.DataFrame]]:
    """Load single-season and multi-season metric/calibration/phase reports."""
    return {
        "pregame_single": load_metric_report(
            config.PREGAME_MODEL_METRICS_PATH, "pregame", SINGLE_SEASON_SETUP
        ),
        "pregame_multi": load_metric_report(
            config.PREGAME_MODEL_METRICS_MULTISEASON_PATH,
            "pregame",
            MULTISEASON_SETUP,
        ),
        "live_single": load_metric_report(
            config.LIVE_MODEL_METRICS_PATH, "live", SINGLE_SEASON_SETUP
        ),
        "live_multi": load_metric_report(
            config.LIVE_MODEL_METRICS_MULTISEASON_PATH, "live", MULTISEASON_SETUP
        ),
        "phase_single": load_phase_metrics(
            config.LIVE_MODEL_PHASE_METRICS_PATH, SINGLE_SEASON_SETUP
        ),
        "phase_multi": load_phase_metrics(
            config.LIVE_MODEL_PHASE_METRICS_MULTISEASON_PATH, MULTISEASON_SETUP
        ),
        "cal_pregame_single": load_calibration_report(
            config.PREGAME_MODEL_CALIBRATION_PATH, "pregame", SINGLE_SEASON_SETUP
        ),
        "cal_pregame_multi": load_calibration_report(
            config.PREGAME_MODEL_CALIBRATION_MULTISEASON_PATH,
            "pregame",
            MULTISEASON_SETUP,
        ),
        "cal_live_single": load_calibration_report(
            config.LIVE_MODEL_CALIBRATION_PATH, "live", SINGLE_SEASON_SETUP
        ),
        "cal_live_multi": load_calibration_report(
            config.LIVE_MODEL_CALIBRATION_MULTISEASON_PATH, "live", MULTISEASON_SETUP
        ),
    }


def get_comparison_io_paths() -> Tuple[List[Path], List[Path]]:
    """Return (input paths, output paths) for compare_models dry-run."""
    inputs = [
        config.PREGAME_MODEL_METRICS_PATH,
        config.PREGAME_MODEL_METRICS_MULTISEASON_PATH,
        config.LIVE_MODEL_METRICS_PATH,
        config.LIVE_MODEL_METRICS_MULTISEASON_PATH,
        config.LIVE_MODEL_PHASE_METRICS_PATH,
        config.LIVE_MODEL_PHASE_METRICS_MULTISEASON_PATH,
        config.PREGAME_MODEL_CALIBRATION_PATH,
        config.PREGAME_MODEL_CALIBRATION_MULTISEASON_PATH,
        config.LIVE_MODEL_CALIBRATION_PATH,
        config.LIVE_MODEL_CALIBRATION_MULTISEASON_PATH,
    ]
    outputs = [
        config.MODEL_COMPARISON_SUMMARY_PATH,
        config.MODEL_COMPARISON_DETAIL_PATH,
        config.PHASE_COMPARISON_SUMMARY_PATH,
        config.CALIBRATION_COMPARISON_SUMMARY_PATH,
    ]
    return inputs, outputs


def _print_headline(summary_df: pd.DataFrame) -> None:
    """Print a short pre-game and live comparison for CLI output."""
    if summary_df.empty:
        print("No comparison metrics available.")
        return

    print("\nComparison headline (interpret with setup differences in mind):")
    for model in ["pregame", "live"]:
        subset = summary_df.loc[
            (summary_df["model"] == model) & (summary_df["metric"] == "accuracy")
        ]
        if subset.empty:
            continue
        row = subset.iloc[0]
        single_val = row["single_season_value"]
        multi_val = row["multiseason_value"]
        diff = row["difference"]
        print(
            f"  {model}: single-season accuracy={single_val:.4f}, "
            f"multi-season accuracy={multi_val:.4f}, diff={diff:+.4f}"
        )
    print(f"\n  Note: {SETUP_DIFFERENCE_NOTE}")


def run_model_comparison(verbose: bool = True, dry_run: bool = False) -> int:
    """Build comparison reports from existing metric CSVs."""
    inputs, outputs = get_comparison_io_paths()

    if dry_run:
        if verbose:
            print("\n=== Mode: compare_models (DRY-RUN) ===")
            print("Would read:")
            for path in inputs:
                status = "found" if path.exists() else "missing (optional)"
                print(f"  [{status}] {path.relative_to(config.ROOT_DIR)}")
            print("Would write:")
            for path in outputs:
                print(f"  {path.relative_to(config.ROOT_DIR)}")
            print("\n[dry-run] Skipping execution for mode: compare_models")
        return 0

    reports = _load_all_reports()
    if reports["pregame_single"] is None and reports["pregame_multi"] is None:
        if reports["live_single"] is None and reports["live_multi"] is None:
            if verbose:
                print("ERROR: No metric reports found for comparison.")
            return 1

    summary_df = build_model_comparison_summary(
        reports["pregame_single"],
        reports["pregame_multi"],
        reports["live_single"],
        reports["live_multi"],
    )
    detail_df = build_model_comparison_detail(
        reports["pregame_single"],
        reports["pregame_multi"],
        reports["live_single"],
        reports["live_multi"],
    )
    phase_df = build_phase_comparison_summary(
        reports["phase_single"], reports["phase_multi"]
    )
    calibration_df = build_calibration_comparison_summary(
        reports["cal_pregame_single"],
        reports["cal_pregame_multi"],
        reports["cal_live_single"],
        reports["cal_live_multi"],
    )

    written = save_comparison_reports(summary_df, detail_df, phase_df, calibration_df)

    if verbose:
        print("Model comparison complete. Reports written to outputs/reports/")
        for name, path in written.items():
            print(f"  - {path.name}")
        _print_headline(summary_df)

    return 0


def main() -> int:
    """CLI entry point for ``python src/compare_model_versions.py``."""
    return run_model_comparison(verbose=True, dry_run=False)


if __name__ == "__main__":
    sys.exit(main())
