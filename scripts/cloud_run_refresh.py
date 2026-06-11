from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path("/tmp/nba-win-probability-engine")

FINALS_UPCOMING_REPORT = REPO_DIR / "outputs/reports/finals_upcoming_predictions.csv"
FINALS_UPCOMING_SNAPSHOT = Path("/tmp/finals_upcoming_predictions_before_refresh.csv")
PLAYOFF_GAMES_PATH = REPO_DIR / "data/playoffs/raw/playoff_games.csv"
PLAYOFF_PLAY_BY_PLAY_PATH = REPO_DIR / "data/playoffs/raw/playoff_play_by_play.csv"


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


def find_game_id_column(df):
    if "game_id" in df.columns:
        return "game_id"

    for col in df.columns:
        if col.lower() == "gameid":
            return col

    raise ValueError("Could not find a game_id column.")


def prediction_columns(df):
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


def completed_mask(df):
    import pandas as pd

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


def restore_completed_pregame_predictions(snapshot_path: Path, current_path: Path) -> None:
    import pandas as pd

    if not snapshot_path.exists():
        print("No prior Finals prediction snapshot found. Skipping prediction lock.", flush=True)
        return

    if not current_path.exists():
        print("No current Finals prediction report found. Skipping prediction lock.", flush=True)
        return

    old_df = pd.read_csv(snapshot_path)
    current_df = pd.read_csv(current_path)

    old_id = find_game_id_column(old_df)
    current_id = find_game_id_column(current_df)

    lock_cols = [
        col for col in prediction_columns(current_df)
        if col in old_df.columns
    ]

    if not lock_cols:
        print("No prediction columns found for locking.", flush=True)
        return

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

    current_df.to_csv(current_path, index=False)

    print(
        f"Locked completed-game pre-game prediction values. Restored {restored_cells} cell(s).",
        flush=True,
    )


def run(command: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(f"\n$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def run_with_retries(command: list[str], cwd: Path, attempts: int = 3) -> None:
    last_status = 1

    for attempt in range(1, attempts + 1):
        print(f"\nAttempt {attempt}/{attempts}", flush=True)
        last_status = run(command, cwd=cwd, check=False)

        if last_status == 0:
            print("Step succeeded.", flush=True)
            return

        print(f"Step failed with exit code {last_status}.", flush=True)

    raise RuntimeError(f"Command failed after {attempts} attempts: {' '.join(command)}")


def git_has_changes(cwd: Path) -> bool:
    result = subprocess.run(
        ["git", "diff", "--staged", "--quiet"],
        cwd=cwd,
        text=True,
    )
    return result.returncode != 0


def main() -> int:
    github_token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    github_repository = os.environ.get(
        "GITHUB_REPOSITORY",
        "sanjay-dilip/nba-win-probability-engine",
    )
    git_branch = os.environ.get("GIT_BRANCH", "main")

    if not github_token:
        print("Missing required environment variable: GITHUB_TOKEN", file=sys.stderr)
        return 1

    clone_url = f"https://x-access-token:{github_token}@github.com/{github_repository}.git"

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)

    run(["git", "clone", "--branch", git_branch, "--depth", "1", clone_url, str(REPO_DIR)])

    run(["git", "config", "user.name", "cloud-run-refresh[bot]"], cwd=REPO_DIR)
    run(["git", "config", "user.email", "cloud-run-refresh-bot@example.com"], cwd=REPO_DIR)

    if FINALS_UPCOMING_REPORT.exists():
        shutil.copy2(FINALS_UPCOMING_REPORT, FINALS_UPCOMING_SNAPSHOT)
        print(
            f"Saved pre-refresh Finals prediction snapshot: {FINALS_UPCOMING_SNAPSHOT}",
            flush=True
        )
    else:
        print("No existing Finals upcoming prediction report found before refresh.", flush=True)

    games_status = run(
        ["python", "run_pipeline.py", "--mode", "collect_playoff_games", "--seasons", "2025-26"],
        cwd=REPO_DIR,
        check=False,
    )

    if games_status != 0:
        print("First playoff metadata attempt failed. Retrying...", flush=True)
        games_status = run(
            ["python", "run_pipeline.py", "--mode", "collect_playoff_games", "--seasons", "2025-26"],
            cwd=REPO_DIR,
            check=False,
        )

    if not PLAYOFF_GAMES_PATH.exists():
        print(
            "Playoff metadata file was not created. "
            "Skipping play-by-play collection and keeping existing deploy-safe files.",
            flush=True,
        )

    play_by_play_status = run(
        ["python", "run_pipeline.py", "--mode", "collect_playoff_play_by_play", "--seasons", "2025-26"],
        cwd=REPO_DIR,
        check=False,
    )


    if play_by_play_status != 0:
        print("First playoff play-by-play attempt failed. Retrying...", flush=True)
        play_by_play_status = run(
            ["python", "run_pipeline.py", "--mode", "collect_playoff_play_by_play", "--seasons", "2025-26"],
            cwd=REPO_DIR,
            check=False,
        )

    if not PLAYOFF_PLAY_BY_PLAY_PATH.exists():
        print(
            "Playoff play-by-play file was not created. "
            "Skipping case-study rebuild and keeping existing deploy-safe files.",
            flush=True,
        )
        return 0

    case_study_status = run(
        ["python", "run_pipeline.py", "--mode", "run_playoff_case_study_pipeline", "--seasons", "2025-26"],
        cwd=REPO_DIR,
        check=False,
    )

    if case_study_status != 0:
        print(
            "Case-study grouped pipeline exited non-zero. "
            "Continuing because deploy-safe Finals outputs may have been generated before QA.",
            flush=True
        )

    # The grouped playoff pipeline writes full playoff predictions under data/playoffs/.
    # The deployed Streamlit app reads the lightweight export under data/deploy/.
    # Always export the Finals live replay file after the case-study pipeline.
    run(
        ["python", "run_pipeline.py", "--mode", "export_finals_live_predictions_for_deploy"],
        cwd=REPO_DIR,
        check=False,
    )

    # Rebuild deploy-safe Finals reports after export so GitHub and Streamlit receive the latest files.
    run(
        ["python", "run_pipeline.py", "--mode", "build_finals_pregame_predictions"],
        cwd=REPO_DIR,
        check=False,
    )

    restore_completed_pregame_predictions(
        FINALS_UPCOMING_SNAPSHOT,
        FINALS_UPCOMING_REPORT
    )

    run(
        ["python", "run_pipeline.py", "--mode", "build_finals_projected_series_path"],
        cwd=REPO_DIR,
        check=False,
    )

    files_to_commit = [
        "data/deploy/finals_live_predictions.csv",
        "data/deploy/pregame_demo_predictions.csv",
        "data/deploy/live_demo_predictions.csv",
        "outputs/reports/finals_upcoming_predictions.csv",
        "outputs/reports/finals_projected_series_path.csv",
        "outputs/reports/nba_finals_case_study_summary.csv",
        "outputs/reports/playoff_coverage_report.csv",
        "outputs/reports/project_qa_summary.csv",
        "outputs/reports/data_freshness_summary.csv",
        "outputs/reports/model_performance_public_summary.csv",
        "outputs/reports/evaluation_public_summary.csv",
        "data/manual/finals_schedule_overrides.csv",
    ]

    run(["git", "add", *files_to_commit], cwd=REPO_DIR, check=False)

    if not git_has_changes(REPO_DIR):
        print("No deploy-safe Finals report changes to commit.", flush=True)
        return 0

    run(["git", "commit", "-m", "chore: refresh Finals live replay [skip ci]"], cwd=REPO_DIR)
    run(["git", "push", "origin", git_branch], cwd=REPO_DIR)

    print("Cloud refresh completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())