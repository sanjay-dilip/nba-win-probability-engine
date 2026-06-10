# Multi-season Data Runbook

Internal step-by-step guide for expanding from a 2024-25 single-season baseline
to multi-season data (2022-23 through 2024-25) and unblocking multi-season model
training.

## Purpose

Build 17 added season-aware collection; Build 18 added readiness checks and
separate multi-season model artifacts. Processed feature files may still contain
only 2024-25 until older-season play-by-play is collected and features rebuilt.

This runbook is the safe manual sequence. **Nothing here runs automatically.**

## Current Intended Split

**Temporary (while 2025-26 is unavailable):**

- Train: 2022-23, 2023-24
- Test: 2024-25

**Future:**

- Train: 2022-23, 2023-24, 2024-25
- Test: 2025-26

## Safety Rules

- Do **not** run API-heavy play-by-play collection accidentally — always `--dry-run` first.
- Do **not** train until **coverage** and **readiness** both pass.
- Existing single-season models (`pregame_model.pkl`, `live_model.pkl`) are **preserved**.
- Multi-season models save to separate `*_multiseason*` paths only.
- If collection is interrupted, rerun the same command (idempotent append, no duplicate `game_id` rows).

## Step 1: Check Current Coverage

Inspect local files only — no API calls:

```bash
python run_pipeline.py --mode check_multiseason_coverage --seasons 2022-23 2023-24 2024-25
```

Dry-run:

```bash
python run_pipeline.py --mode check_multiseason_coverage --seasons 2022-23 2023-24 2024-25 --dry-run
```

Report: `outputs/reports/multiseason_coverage_report.csv`

## Step 2: Refresh Metadata if Needed

If `games.csv` is missing seasons or `team_stats.csv` is stale:

```bash
python run_pipeline.py --mode refresh_multi_season_metadata --seasons 2022-23 2023-24 2024-25 --dry-run
python run_pipeline.py --mode refresh_multi_season_metadata --seasons 2022-23 2023-24 2024-25
```

API-light — games + team stats only (no play-by-play).

## Step 3: Collect Play-by-Play (API-heavy)

Collect older seasons first if 2024-25 PBP is already complete:

Dry-run:

```bash
python run_pipeline.py --mode collect_play_by_play_multi_season --seasons 2022-23 2023-24 --dry-run
```

Live (may take **hours**; may timeout — rerun if interrupted):

```bash
python run_pipeline.py --mode collect_play_by_play_multi_season --seasons 2022-23 2023-24
```

Notes:

- Appends/updates `data/raw/play_by_play.csv` without duplicating existing `game_id` rows.
- NBA API timeouts are common — rerun the same command to resume.
- Do not include this step in any local-only grouped mode.

## Step 4: Rebuild Features Locally

After PBP collection completes for the needed seasons:

```bash
python run_pipeline.py --mode prepare_multiseason_training_data --seasons 2022-23 2023-24 2024-25 --train-seasons 2022-23 2023-24 --test-season 2024-25 --dry-run
python run_pipeline.py --mode prepare_multiseason_training_data --seasons 2022-23 2023-24 2024-25 --train-seasons 2022-23 2023-24 --test-season 2024-25
```

Runs: rebuild pre-game + live features → coverage check → readiness check.
No API. No training.

## Step 5: Confirm Readiness

```bash
python run_pipeline.py --mode check_multiseason_training_readiness --train-seasons 2022-23 2023-24 --test-season 2024-25
```

Report: `outputs/reports/multiseason_training_readiness.csv`

Training is blocked until this passes.

## Step 6: Train Multi-season Models

**Only if readiness passes:**

```bash
python run_pipeline.py --mode train_models_multiseason --train-seasons 2022-23 2023-24 --test-season 2024-25 --dry-run
python run_pipeline.py --mode train_models_multiseason --train-seasons 2022-23 2023-24 --test-season 2024-25
```

Artifacts: `models/*_multiseason.pkl` and `outputs/reports/*_multiseason.csv`

## Step 7: Run QA

```bash
python run_pipeline.py --mode qa
```

## Expected Failure Modes

| Issue | What to do |
|-------|------------|
| NBA API timeout | Rerun the same PBP collection command |
| Missing PBP season | Run `collect_play_by_play_multi_season` for that season |
| Feature files not rebuilt | Run `prepare_multiseason_training_data` |
| Readiness fails (missing season) | Complete PBP + feature rebuild for that season |
| Freshness mtime warning on `pregame_model_metrics` | Usually safe to ignore (timestamp ordering) |

## What Not To Do

- Do **not** start Build 19 evaluation until multi-season models exist.
- Do **not** overwrite single-season artifacts with multi-season training.
- Do **not** treat partial seasons as full training data.
- Do **not** run `train_models_multiseason` when coverage or readiness fails.
