# Architecture

This document describes the high-level architecture of the NBA Win
Probability Engine as it stands today: an end-to-end pipeline from NBA API
collection through trained models to a deployed Streamlit dashboard.

## Layers

1. **Data ingestion** — collect games, play-by-play events, and team stats
   from the `nba_api` package (`src/collect_games.py`,
   `src/collect_play_by_play.py`, `src/collect_team_stats.py`). Raw data
   lands in `data/raw/`, gitignored except for lightweight placeholders.
2. **Feature engineering** — build leakage-safe pre-game and per-event live
   feature tables (`src/build_pregame_features.py`,
   `src/build_live_features.py`) into `data/processed/`.
3. **Model training** — fit baseline logistic-regression pipelines for both
   the single-season baseline and the primary multi-season model
   (`src/train_pregame_model.py`, `src/train_live_model.py`,
   `src/multiseason_training.py`). Trained artifacts live under `models/`.
4. **Prediction** — score saved feature tables into dashboard-ready CSVs
   (`src/predict_pregame.py`, `src/predict_live.py`).
5. **Manual post-game result override** — record/correct results into
   `data/manual/postgame_results.csv` (`src/manual_override.py`), kept
   separate from model features to avoid leakage.
6. **Evaluation** — score models, compare the baseline and primary
   multi-season models, and summarize calibration (`src/evaluate.py`,
   `src/compare_model_versions.py`).
7. **Playoff / Finals case study** — apply the trained primary model to
   playoff games as an out-of-distribution case study, kept under a
   separate `data/playoffs/` path (`src/playoff_case_study.py`).
8. **QA and freshness** — verify expected files, pipeline registration, and
   file staleness before a report is trusted (`src/project_qa.py`,
   `src/data_freshness.py`).
9. **Dashboard** — a Streamlit multipage app (`app/`) that reads only saved
   outputs; it never trains models or calls `nba_api` at runtime.

## Data flow

```
nba_api -> data/raw -> data/processed -> features -> models -> predictions -> dashboard
```

Playoff data follows the same shape under `data/playoffs/raw` and
`data/playoffs/processed`, kept separate from the regular-season pipeline.

## Modules

- `src/config.py` — central path and mode configuration (pathlib-based);
  every file path used elsewhere in the project is defined once here.
- `src/utils.py` — directory creation, CSV read/write/upsert helpers, and
  small cross-cutting helpers (rate limiting, safe sleep).
- `src/data_validation.py` — schema, probability, and leakage checks.
- `src/season_config.py` — season label validation shared by collectors and
  training modes.
- `run_pipeline.py` — CLI entry point. Run `python run_pipeline.py
  --list-modes` for the full list of individual and grouped modes.
- `app/` — Streamlit multipage dashboard; `app/dashboard_utils.py` holds
  pure helper functions with no Streamlit imports, so they're unit-testable
  without launching the app.

## Data modes

Controlled by the `USE_SAMPLE_DATA` environment variable. `sample` mode runs
entirely on the committed files in `data/sample/`, making the project a
self-contained portfolio demo that needs no `nba_api` calls or local data
collection to explore.
