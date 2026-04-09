import re
import string
import logging
from datetime import datetime, timezone
from typing import Any, Optional, Tuple, List

from agent_sdk.database.memory import get_memories

logger = logging.getLogger("agent_sdk.utils.context")

_TRIVIAL_FOLLOWUP_PATTERN = re.compile(
    r'^(yes|no|sure|ok|okay|please|proceed|go\s*ahead|continue|yeah|yep|thanks|thank\s*you|got\s*it|tell\s*me\s*more|no\s*thanks)$',
    re.IGNORECASE
)

def is_trivial_followup(query: str) -> bool:
    """Check if a query is a trivial confirmation or follow-up."""
    normalized = query.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    return bool(_TRIVIAL_FOLLOWUP_PATTERN.match(normalized))

async def build_dynamic_context(
    session_id: str,
    query: str,
    response_format: Optional[str] = None,
    user_id: Optional[str] = None,
    as_of_date: Optional[str] = None,
    watchlist_id: Optional[str] = None,
    format_instructions: Optional[dict] = None
) -> Tuple[str, Optional[str]]:
    """
    Build dynamic context block (date, memories, format instructions) to prepend to the user query.
    Returns a tuple of (context_block, mem_error).
    """
    mem_key = user_id or session_id  # prefer stable user_id for Mem0
    mem_error: Optional[str] = None

    # Skip Mem0 search for trivial follow-ups
    if not is_trivial_followup(query) and len(query.strip()) > 10:
        memories, mem_error = get_memories(user_id=mem_key, query=query)
    else:
        memories = []

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    year = today[:4]

    parts = []

    # Point-in-time context injection
    if as_of_date:
        parts.append(
            f"IMPORTANT: The user is requesting a historical analysis as of {as_of_date}. "
            f"While yfinance provides current data, please interpret all reported metrics "
            f"within the context of that specific date. Acknowledge that fundamental data "
            f"may be the most recent available rather than a true point-in-time snapshot."
        )
    else:
        parts.append(
            f"Today's date: {today}. When using tavily_quick_search include the current year "
            f"({year}) in search queries (e.g. 'HDFC Bank Q4 {year} results')."
        )

    # Watchlist context injection (this requires a DB call, which we handle if watchlist_id is provided)
    # Note: In a fully modular SDK, the DB call might be passed in as a pre-fetched value
    # but for now we keep the logic consistent with the agent implementation.
    # We'll assume the caller handles the MongoDB call if needed, or we can import it here.
    # To avoid circular imports or heavy dependencies in utils, we expect the caller to provide any
    # specific watchlist content if they want it injected, or we'll keep the logic as is.
    # For this migration, we'll move the core string building logic.

    if memories:
        memory_lines = "\n".join(f"- {m}" for m in memories)
        parts.append(f"User context (long-term memory):\n{memory_lines}")
        logger.info("Injected %d memories into context for session='%s'", len(memories), session_id)

    if mem_error:
        parts.append(f"Note: {mem_error}")
        logger.warning("Mem0 degradation for session='%s': %s", session_id, mem_error)

    # Response format instructions
    if format_instructions and response_format:
        instruction = format_instructions.get(response_format, "")
        if instruction:
            parts.append(instruction.strip())
            logger.info("Applied response format '%s' for session='%s'", response_format, session_id)

    context_block = "\n\n".join(parts)
    formatted_block = f"[CONTEXT]\n{context_block}\n[/CONTEXT]\n\n" if context_block else ""

    return formatted_block, mem_error
