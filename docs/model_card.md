# Model Card

## Models

### 1. Pre-game winner model (primary, multi-season)
- **Task:** Binary classification — will the home team win before tip-off?
- **Inputs:** Leakage-safe pre-game features (win pct, net rating, rest,
  recent form differentials), built strictly from games before the target
  game's date (`src/build_pregame_features.py`).
- **Algorithm:** Logistic regression inside a scikit-learn `Pipeline`.
- **Training / test split:** Trained on 2022-23, 2023-24, 2024-25 regular
  seasons; evaluated on 2025-26 as a future-season holdout.
- **Result:** 68.2% accuracy, 0.7357 ROC-AUC on 4,910 games
  (`outputs/reports/pregame_model_metrics_multiseason.csv`).

### 2. Live win probability model (primary, multi-season)
- **Task:** Binary classification per play-by-play event — will the home
  team win, given the current game state?
- **Inputs:** Time remaining, score margin, period, and related live
  game-state features (`src/build_live_features.py`).
- **Algorithm:** Logistic regression inside a scikit-learn `Pipeline`.
- **Training / test split:** Same 2022-23 to 2024-25 train / 2025-26 test
  split as the pre-game model.
- **Result:** 74.7% accuracy, 0.8300 ROC-AUC on 2.3M+ play-by-play event
  rows (`outputs/reports/live_model_metrics_multiseason.csv`). These are
  **event-level** metrics — a single game contributes hundreds of rows, so
  this is not comparable to one-row-per-game accuracy.

### Baseline models (single-season)
Both models are also retained in a single-season 2024-25 chronological
hold-out form, used for pipeline validation and as a comparison point
against the primary multi-season models
(`outputs/reports/model_comparison_summary.csv`).

## Evaluation
- Accuracy, ROC-AUC, log loss, Brier score (`src/evaluate.py`).
- Calibration curve / reliability summary
  (`outputs/reports/pregame_model_calibration_multiseason.csv`,
  `outputs/reports/live_model_calibration_multiseason.csv`).
- Chronological train/test split by season to avoid leakage across time.
- Playoff / NBA Finals games are scored as an **out-of-distribution case
  study**, not part of training or the primary evaluation
  (`src/playoff_case_study.py`).

## Leakage guardrails
- Final scores and `home_team_won` are never used as pre-game features.
- Enforced by `src/data_validation.validate_no_target_leakage` and the
  strictly-earlier-games construction in `src/build_pregame_features.py`.

## Ethical / usage notes
- For educational and portfolio purposes only.
- Not intended for betting or any real-money decision making.
