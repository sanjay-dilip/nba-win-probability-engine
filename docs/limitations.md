# Limitations

## Current build (foundation + sample-data shell)

- **No models.** Nothing is trained; the dashboard displays sample data and
  placeholders only.
- **No live NBA data.** The NBA API is not called yet. All data is the small,
  fictional sample set in `data/sample/`.
- **No database.** Everything is stored in CSV files.
- **No deployment infra.** No Docker, no FastAPI, no cloud setup.
- **Sample data is fictional.** IDs like `SAMPLE_GAME_001` and the feature
  values are illustrative, not real results.

## General / future limitations

- Sports outcomes are inherently noisy; even good models will be wrong often.
- Win probability estimates depend heavily on data quality and feature design.
- Calibration can drift across seasons due to rule and roster changes.
- Manual post-game overrides are stored separately and are **not** fed back
  into features, to avoid accidental leakage.

## Not for betting

This project is for learning and portfolio demonstration only. It is not
financial advice and must not be used for gambling decisions.
