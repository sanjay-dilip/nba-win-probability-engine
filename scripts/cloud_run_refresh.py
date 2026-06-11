from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_DIR = Path("/tmp/nba-win-probability-engine")


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

    run_with_retries(
        ["python", "run_pipeline.py", "--mode", "collect_playoff_games", "--seasons", "2025-26"],
        cwd=REPO_DIR,
        attempts=3,
    )

    run_with_retries(
        ["python", "run_pipeline.py", "--mode", "collect_playoff_play_by_play", "--seasons", "2025-26"],
        cwd=REPO_DIR,
        attempts=3,
    )

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