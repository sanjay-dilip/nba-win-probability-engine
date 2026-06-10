"""Season configuration and validation helpers (Build 17).

Centralizes season labels used by the pipeline and collectors. Does not call
``nba_api`` or read/write data files.
"""

from __future__ import annotations

import re
from typing import List, Optional, Sequence

# Default single-season behavior (unchanged from pre-multi-season builds).
DEFAULT_SEASON = "2024-25"

# Earliest training season in the planned multi-season workflow.
MULTI_SEASON_TRAIN_START = "2022-23"

# Seasons supported for explicit multi-season collection/feature rebuilds.
SUPPORTED_SEASONS = ["2022-23", "2023-24", "2024-25"]

# Planned future hold-out season — not collected automatically in this build.
FUTURE_TEST_SEASON = "2025-26"

DEFAULT_MULTI_SEASONS = list(SUPPORTED_SEASONS)

_SEASON_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def validate_season_format(season: str) -> str:
    """Validate and return a season string like ``2024-25``.

    Raises:
        ValueError: If the format is invalid.
    """
    season = str(season).strip()
    if not _SEASON_PATTERN.match(season):
        raise ValueError(
            f"Invalid season format '{season}'. Expected YYYY-YY, e.g. 2024-25."
        )
    start_year = int(season[:4])
    end_suffix = int(season[5:])
    if end_suffix != (start_year + 1) % 100:
        raise ValueError(
            f"Invalid season years in '{season}'. End year must follow start year."
        )
    return season


def validate_season_list(seasons: Sequence[str]) -> List[str]:
    """Validate a non-empty list of season strings."""
    if not seasons:
        raise ValueError("At least one season is required.")
    return [validate_season_format(s) for s in seasons]


def parse_pipeline_season_args(
    season: Optional[str] = None,
    seasons: Optional[Sequence[str]] = None,
) -> tuple[str, List[str]]:
    """Parse ``--season`` / ``--seasons`` CLI args for the pipeline.

    Returns:
        ``(single_season, multi_season_list)`` where ``single_season`` is used
        for individual modes (defaults to :data:`DEFAULT_SEASON`) and
        ``multi_season_list`` is used for grouped multi-season modes (defaults
        to :data:`SUPPORTED_SEASONS`).

    Raises:
        ValueError: If both arguments are provided or formats are invalid.
    """
    if season is not None and seasons is not None:
        raise ValueError("Use either --season or --seasons, not both.")

    if seasons is not None:
        validated = validate_season_list(seasons)
        return validated[0], validated

    if season is not None:
        single = validate_season_format(season)
        return single, [single]

    return DEFAULT_SEASON, list(SUPPORTED_SEASONS)


def summarize_seasons_in_csv(path) -> str:
    """Return a short note listing unique seasons in a CSV, or empty string."""
    from pathlib import Path

    import pandas as pd

    csv_path = Path(path)
    if not csv_path.exists():
        return ""
    try:
        if "season" not in pd.read_csv(csv_path, nrows=0).columns:
            return ""
        seasons = pd.read_csv(csv_path, usecols=["season"])["season"].dropna().unique()
    except (pd.errors.EmptyDataError, ValueError):
        return ""
    if len(seasons) == 0:
        return ""
    listed = ", ".join(sorted(str(s) for s in seasons))
    return f"seasons ({len(seasons)}): {listed}"
