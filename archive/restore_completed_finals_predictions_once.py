from __future__ import annotations

from pathlib import Path

import pandas as pd


OLD_PATH = Path("old_finals_upcoming_predictions.csv")
CURRENT_PATH = Path("outputs/reports/finals_upcoming_predictions.csv")


PREDICTION_TOKENS = (
    "pred",
    "prob",
    "model",
    "confidence",
    "favorite",
    "pick",
)

RESULT_TOKENS = (
    "actual",
    "final",
    "score",
    "status",
    "result",
    "series",
)


def find_game_id_column(df: pd.DataFrame) -> str:
    if "game_id" in df.columns:
        return "game_id"

    for col in df.columns:
        if col.lower() == "gameid":
            return col

    raise ValueError("Could not find a game_id column.")


def prediction_columns(df: pd.DataFrame) -> list[str]:
    cols = []

    for col in df.columns:
        lower = col.lower()

        if lower in {"game_id", "gameid"}:
            continue

        is_prediction_col = any(token in lower for token in PREDICTION_TOKENS)
        is_result_col = any(token in lower for token in RESULT_TOKENS)

        if is_prediction_col and not is_result_col:
            cols.append(col)

    return cols


def completed_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    for col in df.columns:
        lower = col.lower()

        if lower in {"actual_winner", "winner_actual", "actual_result"}:
            mask = mask | df[col].notna() & (df[col].astype(str).str.strip() != "")

        if lower in {"game_status", "status"}:
            mask = mask | df[col].astype(str).str.lower().str.contains("final|completed", na=False)

        if lower in {"actual_home_score", "actual_away_score", "home_score_actual", "away_score_actual"}:
            mask = mask | df[col].notna()

    return mask


def main() -> None:
    if not OLD_PATH.exists():
        raise FileNotFoundError(f"Missing old file: {OLD_PATH}")

    if not CURRENT_PATH.exists():
        raise FileNotFoundError(f"Missing current file: {CURRENT_PATH}")

    old_df = pd.read_csv(OLD_PATH)
    current_df = pd.read_csv(CURRENT_PATH)

    old_id = find_game_id_column(old_df)
    current_id = find_game_id_column(current_df)

    lock_cols = [
        col for col in prediction_columns(current_df)
        if col in old_df.columns
    ]

    if not lock_cols:
        raise ValueError("No prediction columns found to restore.")

    old_lookup = old_df.set_index(old_id)
    done_mask = completed_mask(current_df)

    restored_cells = 0

    for col in lock_cols:
        for idx, game_id in current_df.loc[done_mask, current_id].items():
            if game_id not in old_lookup.index:
                continue

            old_value = old_lookup.at[game_id, col]

            if pd.isna(old_value) or str(old_value).strip() == "":
                continue

            current_df.at[idx, col] = old_value
            restored_cells += 1

    current_df.to_csv(CURRENT_PATH, index=False)

    print(f"Restored {restored_cells} locked prediction cell(s).")
    print(f"Updated: {CURRENT_PATH}")


if __name__ == "__main__":
    main()