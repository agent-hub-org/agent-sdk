"""Shared financial pipeline utilities."""
from __future__ import annotations

from datetime import datetime, timezone


def format_date_context(as_of_date: str | None = None) -> str:
    """Return a date-context string for injection into phase system prompts."""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if as_of_date:
        year = as_of_date[:4]
        return (
            f"\n\nTODAY'S DATE: {today}. HISTORICAL REFERENCE DATE: {as_of_date}\n"
            f"Always include the historical year ({year}) in search queries to get results "
            f"relevant to that specific point in time."
        )
    year = datetime.now(timezone.utc).year
    return (
        f"\n\nTODAY'S DATE: {today}\n"
        f"Always include the current year ({year}) in search queries to get up-to-date results."
    )
