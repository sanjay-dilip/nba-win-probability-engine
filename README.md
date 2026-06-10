# NBA Win Probability Engine

> **NBA Win Probability Engine** — pre-game prediction, live win probability replay,
> and model evaluation for NBA games. The Streamlit dashboard reads saved predictions,
> features, and evaluation reports from local CSV files.

## Overview

A multi-season NBA win probability system that estimates which team will win a
game — both **before tip-off** and **live**, updating win probability
play-by-play. The **primary model** is trained on 2022-23 through 2024-25 and
evaluated on a **2025-26 future-season holdout**. An original **single-season
baseline** (2024-25 chronological split) is retained for pipeline validation and
comparison.

The project first validates an end-to-end single-season pipeline, then evaluates
the intended multi-season model on a stricter future-season holdout. The Streamlit
app includes interactive demo pages (pre-game and live replay) plus model
evaluation and comparison views.

## Project Goal

Demonstrate an end-to-end, well-structured sports analytics project: data
ingestion, feature engineering, modeling, evaluation, and a dashboard — built
incrementally with clean, readable code.

## Scope

1. Pre-game winner prediction
2. Live win probability replay from play-by-play data
3. Manual post-game result override
4. Model evaluation
5. Streamlit dashboard
6. Sample-data mode for offline demos

The project includes folder structure, config, utilities, validation, sample data,
the dashboard, a CLI pipeline, and tests.

## Project Status

The core pipeline is complete:

- Project folder structure
- Sample data mode
- Streamlit dashboard
- Manual override support
- Config and utility helpers
- Validation helpers
- Test suite

## Features

- Central, `pathlib`-based configuration (`src/config.py`).
- CSV utilities: load, save, append-or-update, and an API attempt logger.
- Data validation: required columns, probability ranges, leakage checks.
- Multipage Streamlit dashboard (sample-data driven).
- Sample-vs-full data mode toggle via `USE_SAMPLE_DATA`.
- Foundation test suite.

## Architecture

See [`docs/architecture.md`](docs/architecture.md). High level:

```
NBA API -> data/raw -> data/processed -> features -> models -> predictions -> dashboard
```

Core modules live in `src/`; the dashboard in `app/`; the CLI is
`run_pipeline.py`.

## Data Sources

- **Planned:** NBA play-by-play and team stats via `nba_api` (later build).
- **Now:** committed fictional sample data in `data/sample/`.

## Sample Data Mode

When `USE_SAMPLE_DATA=true` (the default in `.env.example`), the app runs
entirely on `data/sample/`. This makes the repository a self-contained demo that
works without any API access. See `config.get_data_mode()`.

## How to Run Locally

> **Recommended Python version: 3.10 or 3.11.**
> Several data science / ML packages in `requirements.txt` (e.g. `nba_api`,
> `xgboost`, `scikit-learn`) may not fully support the newest Python releases
> yet. If you are on Python 3.14.x, prefer creating the virtual environment with
> Python 3.11 (or 3.10) to avoid install/compatibility issues. **Python 3.10 is
> acceptable if Python 3.11 is unavailable.** This is only a recommendation — you
> do not need to change your system Python installation.

### Recommended setup (Windows PowerShell)

```powershell
cd D:/Projects/nba-win-probability-engine
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python --version
pip install -r requirements.txt
```

If `py -3.11` is not available, you can use `py -3.10` instead.

### Full local run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Set up environment variables
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux

# 4. Create the folder structure
python run_pipeline.py --mode setup

# 5. Confirm sample data
python run_pipeline.py --mode sample

# 6. Launch the dashboard
streamlit run app/Home.py

# 7. Run tests
pytest
```

## Pipeline Commands

List all available modes:

```bash
python run_pipeline.py --list-modes
```

Preview a grouped mode without executing it:

```bash
python run_pipeline.py --mode dashboard_ready --dry-run
```

### Individual modes

| Mode | Purpose |
|------|---------|
| `collect_games` | Collect game schedule into `data/raw/games.csv` |
| `collect_play_by_play` | Collect play-by-play for up to 10 games |
| `collect_team_stats` | Collect team season stats |
| `build_pregame_features` | Build leakage-safe pre-game features |
| `build_live_features` | Build live features + `game_results.csv` |
| `train_pregame_model` | Train pre-game Logistic Regression baseline |
| `train_live_model` | Train live Logistic Regression baseline |
| `predict_all` | Generate pre-game and live prediction CSVs |
| `evaluate` | Write evaluation reports from saved predictions |
| `check_data_freshness` | Inspect files and write freshness summary |

Other individual modes: `setup`, `sample`, `predict_pregame`, `predict_live`,
`collect_play_by_play_full_season`.

### Grouped local modes (no new API data collection)

| Mode | Steps |
|------|-------|
| `build_features` | pre-game features → live features → freshness check |
| `train_models` | train pre-game → train live → freshness check |
| `score_outputs` | predict_all → evaluate → freshness check |
| `dashboard_ready` | `build_features` → `train_models` → `score_outputs` |
| `qa` | `check_data_freshness` → project QA report (no API, no training) |

`dashboard_ready` uses **existing local raw data only** — it does not call the
NBA API. Use it to prepare the dashboard after raw data is already on disk.

Run **`qa`** after pipeline changes to verify files, reports, and command docs.

### API-heavy modes

| Mode | Notes |
|------|-------|
| `collect_play_by_play_full_season` | Collect all remaining season play-by-play |
| `refresh_local_data` | `collect_games` → `collect_team_stats` |
| `refresh_full_season_play_by_play` | full PBP collect → rebuild live → predict_live → evaluate → freshness |

`refresh_full_season_play_by_play` may take **30–45+ minutes** and makes many
NBA API calls. Preview first:

```bash
python run_pipeline.py --mode refresh_full_season_play_by_play --dry-run
```

### Safety notes

- There is **no generic `all` mode** — use explicit grouped mode names.
- Use `--dry-run` to preview grouped steps before running them.
- `dashboard_ready` does not collect new NBA data.
- After pipeline changes, run `check_data_freshness` (included in grouped modes).

### Multi-season Data Support

Default behavior is unchanged: single-season **`2024-25`** when no season flag is passed.

| Flag | Purpose |
|------|---------|
| `--season 2024-25` | One season for individual modes |
| `--seasons 2022-23 2023-24 2024-25` | Multiple seasons for grouped multi-season modes |

You cannot use `--season` and `--seasons` together. Supported train seasons:
`2022-23`, `2023-24`, `2024-25`. **`2025-26`** is reserved as a future test target only.

**API-light metadata refresh** (games + team stats + freshness + QA — no play-by-play):

```bash
python run_pipeline.py --mode refresh_multi_season_metadata --seasons 2022-23 2023-24 2024-25 --dry-run
python run_pipeline.py --mode refresh_multi_season_metadata --seasons 2022-23 2023-24 2024-25
```

**API-heavy play-by-play** (explicit, slow — may take several hours across seasons):

```bash
python run_pipeline.py --mode collect_play_by_play_multi_season --seasons 2022-23 2023-24 2024-25 --dry-run
```

**Local feature rebuild** (no API calls; uses combined raw CSVs if present):

```bash
python run_pipeline.py --mode build_features_multi_season --dry-run
python run_pipeline.py --mode build_features_multi_season
```

Multi-season play-by-play collection is **slow and explicit by design** — it is not
included in `dashboard_ready` or `refresh_multi_season_metadata`.

### Multi-season Model Training

Build 18 adds **explicit season-based training and testing**. It does not collect
data automatically. Multi-season training requires processed feature files to
contain every requested train season and the hold-out test season. If data is
incomplete, readiness checks **fail** and training is blocked.

**Existing single-season model artifacts are preserved.** Multi-season models and
reports are saved to separate `*_multiseason*` paths.

| Flag | Purpose |
|------|---------|
| `--train-seasons 2022-23 2023-24` | Seasons used for training |
| `--test-season 2024-25` | Hold-out test season (must not overlap train) |

Both flags are **required** for multi-season training modes. There is no silent
default split.

Readiness check (no training):

```bash
python run_pipeline.py --mode check_multiseason_training_readiness --train-seasons 2022-23 2023-24 --test-season 2024-25
```

Train with 2024-25 as temporary hold-out (after features are rebuilt):

```bash
python run_pipeline.py --mode train_models_multiseason --train-seasons 2022-23 2023-24 --test-season 2024-25 --dry-run
python run_pipeline.py --mode train_models_multiseason --train-seasons 2022-23 2023-24 --test-season 2024-25
```

Future intended split (when 2025-26 data is available):

```bash
python run_pipeline.py --mode train_models_multiseason --train-seasons 2022-23 2023-24 2024-25 --test-season 2025-26
```

Do not run training until multi-season play-by-play has been collected and
`build_features_multi_season` has been run successfully.

### Multi-season Coverage and Runbook

Check which seasons are present in local raw and processed files:

```bash
python run_pipeline.py --mode check_multiseason_coverage --seasons 2022-23 2023-24 2024-25
```

After explicit play-by-play collection, rebuild features and re-check locally:

```bash
python run_pipeline.py --mode prepare_multiseason_training_data --seasons 2022-23 2023-24 2024-25 --train-seasons 2022-23 2023-24 --test-season 2024-25
```

See **`MULTISEASON_RUNBOOK.md`** for the full safe sequence (metadata refresh,
API-heavy PBP collection, readiness, training). Play-by-play collection is
**explicit and slow**. Training should only happen after **coverage** and
**readiness** both pass.

### Model Version Comparison

Compare the **baseline** single-season reports with the **primary** multi-season
future-season holdout reports (no training, no data collection):

```bash
python run_pipeline.py --mode compare_models --dry-run
python run_pipeline.py --mode compare_models
```

This mode:

- Reads existing baseline and primary model metric reports
- Writes comparison CSVs to `outputs/reports/`
- Does **not** train models or call the NBA API

The **Model Performance** page presents the primary multi-season model first,
with the baseline retained for side-by-side comparison. Interactive demo pages
(Pre-game Predictor, Live Replay) use prepared prediction outputs — primary
model evaluation is on the Model Performance page.

Interpret results carefully — the baseline uses a within-season chronological
split while the primary model uses a future-season holdout closer to deployment.

### NBA Finals Case Study

The **primary multi-season model** is trained on regular-season games. Playoffs
are applied as an **out-of-distribution case study** — separate from regular-season
training, testing, and baseline evaluation.

Historical playoff evaluation covers **2022-23, 2023-24, and 2024-25** playoffs
(including the NBA Finals). The current case study supports **2025-26** playoffs
and **Finals Games 1–7** when metadata and play-by-play are available.

All playoff outputs live under `data/playoffs/` and are **not** mixed into
regular-season processed files. Collection is explicit and user-triggered:

```bash
python run_pipeline.py --mode collect_playoff_games --seasons 2022-23 2023-24 2024-25 2025-26 --dry-run
python run_pipeline.py --mode collect_playoff_games --seasons 2022-23 2023-24 2024-25 2025-26

python run_pipeline.py --mode collect_playoff_play_by_play --seasons 2022-23 2023-24 2024-25 2025-26 --dry-run
python run_pipeline.py --mode collect_playoff_play_by_play --seasons 2022-23 2023-24 2024-25 2025-26

python run_pipeline.py --mode run_playoff_case_study_pipeline --seasons 2022-23 2023-24 2024-25 2025-26
```

Open the dashboard:

```bash
streamlit run app/Home.py
```

Use **NBA Finals Case Study** for the broader playoff case study, or **2025-26 NBA Finals**
for a focused showcase with **pre-game predictions** for upcoming games and **live replay**
for completed games.

Generate Finals pre-game predictions after playoff metadata is available:

```bash
python run_pipeline.py --mode build_finals_case_study
python run_pipeline.py --mode build_finals_pregame_predictions
```

Upcoming Finals games can also be supplied through `data/manual/finals_schedule_overrides.csv`.
This file should contain **schedule/matchup metadata only** — not scores, winners, or results.
Games 6–7 can use status `if_necessary_scheduled` when matchup, date, and team IDs are known;
these rows are **conditional schedule metadata**, not game results.

After updating the override file (including Games 6–7 when metadata is known), rerun:

```bash
python run_pipeline.py --mode build_finals_case_study
python run_pipeline.py --mode build_finals_pregame_predictions
python run_pipeline.py --mode build_finals_projected_series_path
streamlit run app/Home.py
```

Pre-game prediction only needs known matchup, date, and home/away team IDs.
Live replay requires play-by-play and is available only after the game is played and collected.

Build the projected series path after pre-game predictions:

```bash
python run_pipeline.py --mode build_finals_projected_series_path
```

### Optional GitHub Actions refresh

An optional workflow at `.github/workflows/finals_refresh.yml` can refresh Finals
reports on a schedule or via **workflow_dispatch**. It does **not** train models.
Large playoff play-by-play files remain gitignored by default; the workflow commits
small deploy-safe Finals CSV reports when they change.

Refresh the full local case-study pipeline:

```bash
python run_pipeline.py --mode run_playoff_case_study_pipeline --seasons 2022-23 2023-24 2024-25 2025-26
```

Examples:

```bash
python run_pipeline.py --mode dashboard_ready --dry-run
python run_pipeline.py --mode dashboard_ready
python run_pipeline.py --mode check_data_freshness
```

## Project QA

Run a local stability check without collecting data, training models, or calling
the NBA API:

```bash
python run_pipeline.py --mode qa --dry-run
python run_pipeline.py --mode qa
```

The `qa` mode runs `check_data_freshness`, then inspects required data files,
model artifacts, reports, dashboard app files, pipeline mode registration,
README commands, `CONTEXT.md`, and freshness status. It writes
`outputs/reports/project_qa_summary.csv`.

Preview steps with `--dry-run`. Overall status is **pass**, **warning**, or
**fail** (exit code 1 on fail).

### Game Schedule Collection

Collect the real NBA game schedule into the master schedule file:

```bash
python run_pipeline.py --mode collect_games
```

You can also run the collector directly with optional arguments:

```bash
python src/collect_games.py --seasons 2023-24 2024-25 --season-type "Regular Season"
```

Notes:

- This creates `data/raw/games.csv` (one row per game).
- It uses `nba_api` (the `LeagueGameFinder` endpoint).
- Every API attempt is logged to `data/logs/data_refresh_log.csv`.
- Seasons default to the `NBA_SEASONS` environment variable (see `.env.example`).
- This build collects **schedule / game metadata only** — not play-by-play and
  not team stats. Those come in later builds.

The schedule collector creates a skipped games report when raw NBA API rows
cannot be safely converted into one home/away game row. This prevents silent
data loss and helps diagnose API or parsing issues. When that happens, the
report is written to `outputs/reports/skipped_games_report.csv`; if no games are
skipped, no report file is created.

### Play-by-Play Collection

**Development collection** (safe default — collects up to 10 new games per run):

```bash
python run_pipeline.py --mode collect_play_by_play
```

**Full-season collection** (collects ALL remaining 2024-25 regular-season games):

```bash
python run_pipeline.py --mode collect_play_by_play_full_season
```

Then rebuild the live feature dataset from the newly collected play-by-play:

```bash
python run_pipeline.py --mode build_live_features
```

You can also run the collector directly with optional arguments:

```bash
python src/collect_play_by_play.py --limit 5
python src/collect_play_by_play.py --season 2024-25 --limit 20
python src/collect_play_by_play.py --season 2024-25 --limit 0   # full season, no cap
python src/collect_play_by_play.py --game-id 0022400001
```

Notes:

- This reads completed games from `data/raw/games.csv`.
- This writes event-level data to `data/raw/play_by_play.csv`.
- **Default limit is 10 games per run** to prevent accidental full-season pulls.
  **`--limit 0` means "no cap"** — it collects every remaining eligible game
  (use with care — a full season is 1 200+ games at roughly 500 events each).
- **Full-season collection may take a while** (one API call per game, rate
  limited). Progress is saved to disk after every batch (`--batch-size`,
  default 50), so an interruption never discards already-collected games.
- Collection is **idempotent**: already-collected games are automatically
  skipped, so re-running never duplicates event rows. If everything eligible is
  already collected, it prints a clear message and exits cleanly.
- A live progress plan and summary are printed (eligible / already collected /
  remaining / selected this run / successes / failures / totals).
- A coverage report is written after every run to
  `outputs/reports/play_by_play_coverage_report.csv`
  (season, eligible/collected/remaining games, coverage %, event rows).
- Every API attempt is logged to `data/logs/data_refresh_log.csv`.
- Failures and empty responses are recorded in
  `outputs/reports/play_by_play_collection_failures.csv` (a failed game is never
  duplicated; entries for games that later succeed are removed).
- This collects **raw event data only**. Feature engineering happens in
  `build_live_features` / `build_pregame_features`.
- **Implementation note:** `PlayByPlayV2` is deprecated and no longer returns
  data from the NBA API. This module uses `PlayByPlayV3` directly.

### Team Stats Collection

Collect raw team-level season statistics into the master team-stats file:

```bash
python run_pipeline.py --mode collect_team_stats
```

You can also run the collector directly with optional arguments:

```bash
python src/collect_team_stats.py --seasons 2023-24 2024-25
python src/collect_team_stats.py --season-type "Regular Season"
python src/collect_team_stats.py --season-type "Playoffs"
```

Notes:

- This writes raw team-level season statistics to `data/raw/team_stats.csv`.
- It uses `nba_api` (the `LeagueDashTeamStats` endpoint).
- It collects **raw team-level season statistics only** — it does **not** create
  pre-game features yet, train models, or merge stats into games.
- Seasons default to the `NBA_SEASONS` environment variable (or `2024-25` when
  unset). Season type defaults to `"Regular Season"`.
- `team_id` is preserved as a string; rows are de-duplicated by
  `season` + `season_type` + `team_id` (latest collection wins).
- Every season-level API attempt is logged to `data/logs/data_refresh_log.csv`.
- Failed or empty season pulls are saved to
  `outputs/reports/team_stats_collection_failures.csv`; any rows that fail
  validation are saved to `outputs/reports/invalid_team_stats_rows.csv`.
- **Leakage warning:** these are full-season aggregates. Later pre-game feature
  engineering must use only statistics available *before* each game's date.
  Do **not** use these full-season stats directly for historical pre-game
  predictions, as that would leak future (rest-of-season) information into a
  prediction made before tip-off.

> **Note on `data/raw/team_stats.csv`:** this file contains **full-season
> aggregates** and must **not** be used directly to build historical pre-game
> model-training features — it already "knows" how the full season turned out.
> It may be used later for current/future game prediction or as a reference
> table, but historical features must be generated using only data available
> *before* each game (see the Pre-game Feature Builder below).

### Pre-game Feature Builder

Build the leakage-safe pre-game feature dataset from the schedule:

```bash
python run_pipeline.py --mode build_pregame_features
```

You can also run the builder directly with optional arguments:

```bash
python src/build_pregame_features.py --recent-window 5
python src/build_pregame_features.py --season 2024-25
```

Notes:

- This reads from `data/raw/games.csv` and writes one row per completed game to
  `data/processed/pregame_features.csv`.
- Features are **leakage-safe**: for each game, every rolling stat is computed
  using only that team's games with a *strictly earlier* `game_date`. The
  current game (and any same-date game) is never used in its own features.
- It does **not** use the full-season `data/raw/team_stats.csv` for historical
  rows (that would leak the future), and it does **not** train a model yet.
- **First-game priors:** for a team's first game of the season,
  `games_played_before = 0`, `wins_before = 0`, `losses_before = 0`,
  `win_pct_before = 0.5` and `recent_win_pct_before = 0.5` (neutral coin-flip
  prior), and `rest_days` falls back to `3`. Point averages stay `NaN` until
  prior scoring data exists.
- **Missing scores:** the current `games.csv` is schedule metadata only and has
  **no score columns**. The builder still produces date-based features
  (`games_played_before`, `rest_days`), but it cannot compute wins/losses,
  point averages, or the `home_team_won` target. Rather than fabricate results,
  it leaves those values empty and records every affected game in
  `outputs/reports/pregame_feature_build_issues.csv`. The score-based logic is
  fully implemented and will activate automatically once `games.csv` carries
  scores (e.g. `home_score` / `away_score`).
- Structurally invalid rows (missing `game_id` / `game_date` / team names) are
  written to `outputs/reports/invalid_pregame_feature_rows.csv`.

### Live Feature Builder

Turn raw play-by-play into model-ready live game-state rows, and derive
game-level final results:

```bash
python run_pipeline.py --mode build_live_features
```

You can also run the builder directly with optional arguments:

```bash
python src/build_live_features.py --season 2024-25
python src/build_live_features.py --game-id 0022400001
```

Notes:

- This reads from `data/raw/play_by_play.csv` and `data/raw/games.csv`.
- It writes one row per play-by-play event to
  `data/processed/live_features.csv`, describing the game state after each
  event (running score, score margin, seconds remaining, event type/flags).
- It also writes `data/processed/game_results.csv` — one row per game with the
  final score, `winner`, and `home_team_won`, derived from play-by-play scores
  (`source = play_by_play`).
- **Leakage-safe:** running scores are forward-filled only from *earlier*
  events, so no future information enters an earlier event's row. The final
  `home_team_won` is a **training label** repeated on every event row — it must
  **not** be used as a model input feature later.
- **Score parsing:** `scoreHome` / `scoreAway` are made numeric, forward-filled
  within each game (sorted by `event_num`), and any rows before the first score
  are set to `0`. `score_margin_home = home_score - away_score`.
- **Time parsing:** `pctimestring` is parsed from ISO-8601 durations (e.g.
  `PT10M50.00S`) or `MM:SS`; `seconds_remaining_game` is
  `(4 - period) * 720 + seconds_remaining_period` for regulation.
- **Overtime is simplified for this MVP:** overtime periods are expressed as
  *negative* seconds past the end of regulation. This is intentionally basic and
  may be refined later.
- It does **not** train a model yet. Games without usable scores are not
  fabricated — they are recorded in
  `outputs/reports/live_feature_build_issues.csv`, and their event rows (with a
  null target) go to `outputs/reports/invalid_live_feature_rows.csv`.

> **Future note:** `data/processed/game_results.csv` can later be used to rerun
> the pre-game feature builder so that `home_team_won` and score-based rolling
> features can be created **without** using the full-season
> `data/raw/team_stats.csv`. (That rerun is intentionally **not** done in this
> build.)

### Pre-game Model Training

Train the baseline pre-game winner-prediction model from the pre-game features:

```bash
python run_pipeline.py --mode train_pregame_model
```

You can also run the trainer directly with an optional test-size argument:

```bash
python src/train_pregame_model.py
python src/train_pregame_model.py --test-size 0.2
```

Notes:

- This reads from `data/processed/pregame_features.csv`.
- It trains a **Logistic Regression baseline** inside a scikit-learn `Pipeline`
  (median imputation + scaling for numeric features, most-frequent imputation +
  one-hot encoding for `game_type`).
- It uses a **chronological train/test split** (sorted by `game_date`, no
  shuffling) — the first 80% of games train and the most recent 20% are held out
  for testing, so the model is always evaluated on *later* games.
- Rows with a missing `home_team_won` target are dropped from training.
- `home_team_won` is the **label only** and is **never** used as an input
  feature. Final scores and any same-game outcome columns are likewise excluded
  (a leakage guard hard-fails if a forbidden column reaches the feature set).
- It saves the fitted pipeline to `models/pregame_model.pkl` and the exact
  feature-column list to `models/pregame_feature_columns.pkl` (via `joblib`).
- It writes evaluation metrics (accuracy, ROC-AUC, log loss, Brier score, and
  confusion-matrix counts) to `outputs/reports/pregame_model_metrics.csv`, and a
  10-bucket probability calibration summary to
  `outputs/reports/pregame_model_calibration.csv`.
- It does **not** train the live model, build prediction scripts, or modify the
  dashboard. It does **not** use the full-season `data/raw/team_stats.csv`.

### Live Model Training

Train the baseline live win-probability model from the per-event live features:

```bash
python run_pipeline.py --mode train_live_model
```

You can also run the trainer directly with an optional test-size argument:

```bash
python src/train_live_model.py
python src/train_live_model.py --test-size 0.2
```

Notes:

- This reads from `data/processed/live_features.csv` (one row per play-by-play
  event).
- It trains a **Logistic Regression baseline** inside a scikit-learn `Pipeline`
  (median imputation + scaling for numeric features, most-frequent imputation +
  one-hot encoding for categorical features).
- It uses a **chronological game-level split**: unique games are ordered by
  `game_date`, the first 80% of games train and the most recent 20% are held
  out for testing. **Every event row of a game stays on the same side**, and a
  validation check guarantees no `game_id` appears in both train and test (no
  event-level leakage).
- Allowed features are live game-state columns known *as of each event* (clock,
  running score/margin, event type and flags). The running `home_score` /
  `away_score` are legitimate live state, not final scores.
- `home_team_won` is the **label only** and is **never** used as an input
  feature. Final-outcome columns (`winner`, final scores) are excluded too.
- `event_msg_type` / `event_msg_action_type` are handled safely whether the
  source stores them as numbers or strings (string values are treated as
  categorical).
- It saves the fitted pipeline to `models/live_model.pkl` and a feature-column
  **dictionary** (`numeric_features`, `categorical_features`, `all_features`) to
  `models/live_feature_columns.pkl` (via `joblib`).
- It writes metrics (accuracy, ROC-AUC, log loss, Brier score, confusion-matrix
  counts, plus game/row split counts) to
  `outputs/reports/live_model_metrics.csv`, a 10-bucket calibration summary to
  `outputs/reports/live_model_calibration.csv`, and a by-phase breakdown
  (early/mid/late game) to `outputs/reports/live_model_metrics_by_phase.csv`.
- It does **not** retrain the pre-game model or modify the dashboard.

### Prediction Layer

Generate dashboard-ready prediction CSVs from saved model artifacts (no training):

```bash
python run_pipeline.py --mode predict_pregame
python run_pipeline.py --mode predict_live
python run_pipeline.py --mode predict_all
```

You can also run each script directly:

```bash
python src/predict_pregame.py
python src/predict_live.py
```

Notes:

- This **loads** saved model artifacts (`models/pregame_model.pkl`,
  `models/live_model.pkl`) and feature-column files — it does **not** train
  models or call `nba_api`.
- Pre-game predictions are read from `data/processed/pregame_features.csv` and
  written to `data/processed/pregame_predictions.csv`.
- Live predictions are read from `data/processed/live_features.csv` and written
  to `data/processed/live_predictions.csv`.
- `home_team_won` is **never** used as a model input. It may appear in the
  output as `actual_home_team_won` for evaluation/reference, along with
  `prediction_correct` when labels exist.
- Output columns include `home_win_probability`, `away_win_probability`,
  `predicted_label`, and `predicted_winner` for dashboard use.

## Dashboard Pages

- **Home** — project overview, active data mode, status table.
- **Pregame Predictor** — interactive pre-game win probability and feature explorer.
- **Live Replay** — interactive win-probability replay from saved live predictions.
- **Postgame Override** — manage manual/corrected final results in `data/manual/postgame_results.csv`; compares against `game_results.csv` without merging into model features.
- **Model Performance** — model test metrics, full-dataset prediction summaries, calibration, and momentum swings from evaluation reports.

### Live Replay Dashboard

Launch the app:

```bash
streamlit run app/Home.py
```

Open **Live Replay** in the sidebar. The page reads from
`data/processed/live_predictions.csv` (and optionally `game_results.csv` for
final scores and the model metrics reports for context).

If predictions are missing, generate them first:

```bash
python run_pipeline.py --mode predict_live
```

The Live Replay page includes:

- Season / team / game selectors with readable labels (no raw `game_id` required)
- Game summary card (matchup, final score, winner, final probability)
- Event replay slider with current game-state metrics
- Plotly win-probability chart over elapsed game time
- Top 10 momentum swings table
- Play-by-play event table for the selected game
- Compact model metrics panel (test-set metrics from training)

No model training or pickle loading happens inside Streamlit — only CSV files.

### Pre-game Predictor Dashboard

Open **Pre-game Predictor** in the sidebar after launching:

```bash
streamlit run app/Home.py
```

The page reads from `data/processed/pregame_predictions.csv` and optionally
`pregame_features.csv` for the selected game's feature inputs.

If predictions are missing:

```bash
python run_pipeline.py --mode predict_pregame
```

The Pre-game Predictor page includes:

- Season / team / game selectors with readable labels
- Prediction summary card (probabilities, predicted winner, actual result)
- Win-probability bar chart and home-vs-away feature comparison chart
- Grouped pre-game feature panel (record, scoring, recent form, rest)
- Filtered prediction table and small filter summary (accuracy, most confident)
- Compact model metrics panel (chronological test-set metrics from training)

No model training happens inside Streamlit.

### Model Performance Dashboard

Open **Model Performance** in the sidebar after launching:

```bash
streamlit run app/Home.py
```

The page reads evaluation reports from `outputs/reports/` and existing training
metric/calibration CSVs. If evaluation reports are missing, generate them first:

```bash
python run_pipeline.py --mode evaluate
```

The Model Performance page includes:

- Pre-game and live **chronological test-set** training metrics
- Live model phase breakdown (early / mid / late game)
- **Model version comparison** (primary multi-season vs baseline) when comparison reports exist
- Full-dataset prediction summaries (reference only — not holdout performance)
- Calibration charts from training reports
- Top 10 biggest momentum swings
- Manual override summary (count, sources, mismatches)

No model training happens inside Streamlit.

### Manual UI check

After UI changes, open each page and confirm layout, labels, and status panels:

```bash
streamlit run app/Home.py
```

Pages to inspect: Home, Pre-game Predictor, Live Replay, Postgame Override, Model Performance.

## Evaluation Layer

Summarize saved predictions and model reports without retraining:

```bash
python run_pipeline.py --mode evaluate
```

Or run directly:

```bash
python src/evaluate.py
```

This mode:

- Reads `data/processed/pregame_predictions.csv` and `live_predictions.csv`
- Reads optional `game_results.csv` and `data/manual/postgame_results.csv`
- Reads existing training metric reports under `outputs/reports/`
- Does **not** train models or regenerate predictions
- Writes evaluation reports to `outputs/reports/`:
  - `evaluation_summary.csv` — compact key-value summary
  - `pregame_prediction_summary.csv` — full-dataset pre-game stats + probability buckets
  - `live_prediction_summary.csv` — event-level and final-event summaries
  - `biggest_momentum_swings.csv` — top 50 probability swings
  - `evaluation_by_team.csv` — optional pre-game accuracy by team

**Important:** Full-dataset prediction accuracy (shown on dashboard pages) is
**not** the same as chronological test-set model performance from training.
Test-set metrics come from the hold-out split during `train_*_model`; reference
metrics come from scoring the entire saved prediction CSV.

## Modeling Approach

Planned (not in this build): logistic regression baselines and gradient boosting
(XGBoost) for both pre-game and live win probability. See
[`docs/model_card.md`](docs/model_card.md).

## Manual Post-Game Result Override

Launch the app:

```bash
streamlit run app/Home.py
```

Open **Postgame Override** in the sidebar. The page reads and writes
`data/manual/postgame_results.csv` and can compare manual entries against
`data/processed/game_results.csv` (play-by-play-derived official results).

The Postgame Override page includes:

- Season / team / game selectors (when `game_results.csv` is available) or manual game ID entry
- Side-by-side display of official vs manual results with comparison status
- Validated score entry form with source selector (`manual`, `corrected_manual`, `api`) and notes
- Overwrite protection — existing manual results require an explicit checkbox to replace
- Mismatch warning when a manual result conflicts with the play-by-play result
- Table of all recorded manual overrides

Validation rules:

- Scores must be non-negative integers; tie scores are rejected
- Winner is calculated from team names and scores (not typed manually)
- `game_id` is preserved as a string (leading zeros kept)
- `confirmed_at` is set automatically on save

Manual overrides are **not** used as model inputs in this build. They are
intended for result tracking, evaluation, and future safe data enrichment only.
They do not modify `game_results.csv`, retrain models, or regenerate prediction files.

## Evaluation Plan

Implemented in Build 13: the evaluation layer reports Accuracy, ROC-AUC, Log
loss, Brier score, and calibration summaries. Training metrics use chronological
hold-out splits; full-dataset prediction summaries are labeled as reference-only.
Run `python run_pipeline.py --mode evaluate` to regenerate reports.

## Data Freshness and System Status

Check whether data files, model artifacts, prediction outputs, and reports
exist and are up to date:

```bash
python run_pipeline.py --mode check_data_freshness
```

Or run directly:

```bash
python src/data_freshness.py
```

This mode:

- Inspects raw, processed, prediction, model, report, manual, and log files
- Does **not** collect data, train models, or generate predictions
- Writes `outputs/reports/data_freshness_summary.csv`
- Prints counts of ok / warning / missing / stale assets and top issues

The **Home** page displays an overall status summary and key asset indicators.
Dashboard pages may show compact status lines for their dependencies (e.g.
predictions and metrics on Pregame Predictor and Live Replay).

Status values: `ok`, `missing`, `stale`, `warning`, `not_required`, `unknown`.

Stale detection is simple: prediction files older than their model artifact, or
evaluation reports older than prediction files, are marked `stale`.

## Repository Structure

```
nba-win-probability-engine/
├── app/                 # Streamlit dashboard (Home + pages/)
├── data/
│   ├── sample/          # committed demo CSVs
│   ├── raw/             # raw collected data (gitignored)
│   ├── processed/       # cleaned data (gitignored)
│   ├── manual/          # manual post-game results
│   └── logs/            # data refresh logs (gitignored)
├── models/sample/       # placeholder for sample models
├── notebooks/           # exploration & experimentation
├── outputs/             # charts/reports/predictions (gitignored)
├── src/                 # config, utils, data_validation
├── tests/               # foundation tests
├── docs/                # architecture, data dictionary, model card, limitations
├── assets/
├── run_pipeline.py
├── requirements.txt
├── .env.example
└── .gitignore
```

## Limitations

See [`docs/limitations.md`](docs/limitations.md). In short: no models, no live
data, no database, CSV-only, sample data is fictional, and this is **not** for
betting.

## Future Improvements

- Wire in the NBA API for real games and play-by-play.
- Train and calibrate pre-game and live models.
- Real evaluation dashboard with calibration curves.
- Optional: scheduled data refresh, packaging, and deployment.
