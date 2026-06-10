# Architecture

This document describes the high-level architecture of the NBA Win Probability
Engine. The current build implements the **foundation and sample-data shell**
only.

## Layers

1. **Data ingestion** — collect games, play-by-play events, and team stats
   (NBA API, planned for a later build). Raw data lands in `data/raw/`,
   cleaned data in `data/processed/`.
2. **Feature engineering** — build pre-game and live feature tables.
3. **Pre-game prediction** — predict the winner before tip-off.
4. **Live win probability replay** — recompute win probability per play.
5. **Manual post-game result override** — record/correct results into
   `data/manual/postgame_results.csv`.
6. **Evaluation** — score models and summarize calibration.
7. **Dashboard** — a Streamlit app (`app/`) that surfaces all of the above.

## Data flow (target state)

```
NBA API -> data/raw -> data/processed -> features -> models -> predictions -> dashboard
```

## Modules

- `src/config.py` — central path and mode configuration (pathlib-based).
- `src/utils.py` — directory creation and CSV helpers.
- `src/data_validation.py` — schema, probability, and leakage checks.
- `run_pipeline.py` — CLI entry point (`setup`, `sample`).
- `app/` — Streamlit multipage dashboard.

## Data modes

Controlled by the `USE_SAMPLE_DATA` environment variable. `sample` mode runs
entirely on the committed files in `data/sample/`, making the project a
self-contained portfolio demo.
