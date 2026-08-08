# Limitations

## Model and data

- The primary and baseline models are logistic-regression baselines. More
  advanced models may improve performance but would need careful
  leakage/calibration evaluation before adoption.
- Primary models are trained on regular-season games only. Playoff and NBA
  Finals outputs are an out-of-distribution case study, not a claim of
  playoff-level accuracy — see `docs/model_card.md`.
- Live model metrics are event-level (hundreds of rows per game) and should
  not be interpreted as one prediction per game.
- Full raw and processed NBA datasets are not committed to GitHub — only
  small sample CSVs (`data/sample/`), deploy-safe reports
  (`outputs/reports/`), and lightweight model artifacts needed for the
  dashboard and scheduled refresh are.
- Manual post-game overrides (`data/manual/postgame_results.csv`) are
  stored separately and are **not** fed back into model features, to avoid
  accidental leakage.

## Deployment and freshness

- The public dashboard reads deploy-safe CSVs from Streamlit Community
  Cloud's free tier, which sleeps the app after inactivity — see the
  README's Live Demo section.
- The 2025-26 NBA Finals page is a frozen retrospective (series ended
  2026-06-13); it no longer reflects live/in-progress predictions and won't
  update further unless a new playoff series starts.
- Sports outcomes are inherently noisy; even good models will be wrong
  often. Calibration can drift across seasons due to rule and roster
  changes.

## Not for betting

This project is for learning and portfolio demonstration only. It is not
financial advice and must not be used for gambling decisions.
