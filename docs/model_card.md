# Model Card

> Status: **No models exist yet.** This card documents the intended models so
> the contract is clear before they are built.

## Intended models

### 1. Pre-game winner model
- **Task:** Binary classification — will the home team win?
- **Inputs:** `sample_pregame_features.csv`-style features (win pct, net rating,
  rest, recent form differentials).
- **Output:** Probability the home team wins.
- **Candidate algorithms:** Logistic regression (baseline), XGBoost.

### 2. Live win probability model
- **Task:** Binary classification per play — will the home team win, given the
  current game state?
- **Inputs:** Time remaining, score margin, period, etc.
- **Output:** Live P(home win) that updates each event.
- **Candidate algorithms:** Gradient boosting / logistic regression.

## Evaluation (planned)
- Accuracy, ROC-AUC, log loss, Brier score.
- Calibration curve / reliability summary.
- Train/validation/test split by season to avoid leakage across time.

## Leakage guardrails
- Final scores and `home_team_won` must never be used as pre-game features.
- See `src/data_validation.validate_no_target_leakage`.

## Ethical / usage notes
- For educational and portfolio purposes only.
- Not intended for betting or any real-money decision making.
